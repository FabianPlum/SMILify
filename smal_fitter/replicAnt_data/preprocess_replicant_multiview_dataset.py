#!/usr/bin/env python3
"""Preprocess a flat-directory replicAnt multi-camera dataset into the multi-view
HDF5 layout consumed by `SLEAPMultiViewDataset` and `train_multiview_regressor.py`.

Reads the per-frame, per-camera JSON+JPG+ID-mask+optional-depth files produced by
the replicAnt Unreal Engine pipeline, calls `load_SMIL_Unreal_multiview_sample`
once per frame (in worker processes), and streams the result to an HDF5 file with
the same group/dataset names as the SLEAP multi-view path. The trainer reads
both files interchangeably via `SLEAPMultiViewDataset`.

Conventions baked in by this preprocessor:

- **Scale unification (Phase 1b)**. Calls the loader with `translation_factor=0.1`
  by default. All world-frame translations (`root_loc`, `cam_trans_per_view`,
  `keypoints_3d`, `canonical_to_world_t`) come out of the loader already scaled
  into model-world units. HDF5 metadata advertises `world_scale=1.0`; no further
  scaling at trainer load.
- **OpenCV-form extrinsics**. The loader returns extrinsics in PyTorch3D-mirrored
  row-vector form. We invert that here and store OpenCV-form `(R_cv, t_cv)` so
  that `SLEAPMultiViewDataset._sleap_to_pytorch3d_camera` re-derives the
  PyTorch3D form at load time, byte-equivalent to the loader output.
  See MULTIVIEW_REPLICANT_INTEGRATION_DESIGN.md §Scale Unification.
- **No depth in HDF5**. Depth-buffer self-occlusion runs inside the loader at
  preprocess time; the resulting visibility is what ends up in
  `/multiview_keypoints/keypoint_visibility`. The four depth thresholds are
  recorded in `/metadata` attrs for reproducibility.
- **Per-frame failure handling**. A frame that fails to load is logged with its
  reason, skipped, and its source index recorded in
  `/metadata.skipped_frame_indices`. `num_samples` is the count of successfully
  preprocessed frames; the HDF5 axis has no gaps.

Usage:
    python preprocess_replicant_multiview_dataset.py \\
        --input_dir /mnt/c/replicAnt-dataset-multi-cam-mice \\
        --output_hdf5 replicant_multiview.h5 \\
        --smal_file 3D_model_prep/SMILy_Mouse_static_joints_Falkner_conv_repose_hind_legs.pkl \\
        --max_frames 500
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "smal_fitter"))
sys.path.insert(0, str(_REPO / "smal_fitter" / "neuralSMIL"))

import config  # noqa: E402
from neuralSMIL.configs.config_utils import apply_smal_file_override  # noqa: E402


# Rz_180 = diag(-1, -1, 1). SLEAPMultiViewDataset._sleap_to_pytorch3d_camera
# uses this matrix to map OpenCV (R_cv, t_cv) -> PyTorch3D (R_p3d, T_p3d) at
# dataset-load time:
#     R_p3d = R_cv.T @ Rz_180
#     T_p3d = Rz_180 @ t_cv
# To round-trip the multi-view loader's PyTorch3D-mirrored output through that
# class without changing its conversion, we apply the inverse here:
#     R_cv  = Rz_180 @ R_p3d.T
#     t_cv  = Rz_180 @ T_p3d           (Rz_180 is its own inverse)
_RZ_180 = np.array(
    [[-1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


def _pytorch3d_to_opencv_camera(R_p3d: np.ndarray, T_p3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R_cv = (_RZ_180 @ R_p3d.T).astype(np.float32)
    t_cv = (_RZ_180 @ T_p3d).astype(np.float32)
    return R_cv, t_cv


def _build_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _encode_jpeg_rgb(image_rgb: np.ndarray, quality: int) -> bytes:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode JPEG failed")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Worker-side state and entry point.
# ---------------------------------------------------------------------------

_WORKER_LOADER = None  # populated by _worker_init


def _worker_init(smal_file: Optional[str], shape_family: Optional[int]) -> None:
    """ProcessPoolExecutor initializer. Runs ONCE per worker process before
    any task is dispatched. Applies the SMAL pickle override (so
    `config.dd["J_names"]` matches the dataset) and imports the multi-view
    loader into a module-level slot."""
    if smal_file:
        apply_smal_file_override(smal_file, shape_family=shape_family)
    global _WORKER_LOADER
    from Unreal2Pytorch3D import load_SMIL_Unreal_multiview_sample
    _WORKER_LOADER = load_SMIL_Unreal_multiview_sample


def _process_frame(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entry: load one frame, encode JPEGs, build OpenCV-form camera
    params, return a typed dict the main process can stream to HDF5."""
    frame_idx = int(task["frame_idx"])
    try:
        x, y = _WORKER_LOADER(
            data_path=task["data_path"],
            frame_index=frame_idx,
            camera_indices=task.get("camera_subset"),
            propagate_scaling=task["propagate_scaling"],
            translation_factor=task["translation_factor"],
            load_images=True,
            canonical_frame=True,
            depth_occlusion_check=task["depth_occlusion_check"],
            depth_max_cm=task["depth_max_cm"],
            depth_tolerance_cm=task["depth_tolerance_cm"],
            depth_neighborhood=task["depth_neighborhood"],
        )
    except Exception as exc:
        return {
            "frame_idx": frame_idx,
            "ok": False,
            "reason": f"loader: {type(exc).__name__}: {exc}",
        }

    num_views = int(x["num_views"])
    images = x["image_data"]
    if any(img is None for img in images):
        return {
            "frame_idx": frame_idx,
            "ok": False,
            "reason": "loader returned None image for one or more views",
        }

    # Per-view supervision validity from the loader. False slots have empty
    # `subject Data` (animal absent from this view's render); we keep the
    # slot (image + camera geometry still valid) but mark view_mask=False so
    # the trainer skips supervision for it.
    view_valid_per_view = list(y.get("view_valid_per_view", [True] * num_views))
    num_valid_views = int(sum(view_valid_per_view))
    min_views = int(task["min_views_per_sample"])
    if num_valid_views < min_views:
        return {
            "frame_idx": frame_idx,
            "ok": False,
            "reason": (
                f"only {num_valid_views}/{num_views} views have valid subject data "
                f"(min_views={min_views})"
            ),
        }

    H, W = int(images[0].shape[0]), int(images[0].shape[1])

    jpeg_blobs: List[bytes] = []
    K_list: List[np.ndarray] = []
    R_cv_list: List[np.ndarray] = []
    t_cv_list: List[np.ndarray] = []
    image_sizes: List[np.ndarray] = []

    try:
        for v in range(num_views):
            jpeg_blobs.append(_encode_jpeg_rgb(images[v], task["jpeg_quality"]))

            fx = float(y["fx_per_view"][v])
            fy = float(y["fy_per_view"][v])
            cx = float(y["cx_per_view"][v])
            cy = float(y["cy_per_view"][v])
            K_list.append(_build_K(fx, fy, cx, cy))

            R_p3d_t = y["cam_rot_per_view"][v]
            T_p3d_t = y["cam_trans_per_view"][v]
            R_p3d = R_p3d_t.detach().cpu().numpy().astype(np.float32) if hasattr(R_p3d_t, "detach") else np.asarray(R_p3d_t, dtype=np.float32)
            T_p3d = T_p3d_t.detach().cpu().numpy().astype(np.float32) if hasattr(T_p3d_t, "detach") else np.asarray(T_p3d_t, dtype=np.float32)

            R_cv, t_cv = _pytorch3d_to_opencv_camera(R_p3d, T_p3d)
            R_cv_list.append(R_cv)
            t_cv_list.append(t_cv)

            image_sizes.append(np.array([W, H], dtype=np.int32))
    except Exception as exc:
        return {
            "frame_idx": frame_idx,
            "ok": False,
            "reason": f"camera/jpeg pack: {type(exc).__name__}: {exc}",
        }

    keypoints_2d = np.stack(y["keypoints_2d_per_view"], axis=0).astype(np.float32)
    keypoint_visibility = np.stack(y["keypoint_visibility_per_view"], axis=0).astype(np.float32)
    keypoints_3d = np.asarray(y["keypoints_3d"], dtype=np.float32)

    camera_ids = list(x["camera_ids"])

    return {
        "frame_idx": frame_idx,
        "ok": True,
        "num_views": num_views,
        "image_h": H,
        "image_w": W,
        "camera_ids": camera_ids,
        "camera_names": [f"CAM{c}" for c in camera_ids],
        "jpeg_blobs": jpeg_blobs,
        "view_valid_per_view": view_valid_per_view,
        "keypoints_2d": keypoints_2d,
        "keypoint_visibility": keypoint_visibility,
        "camera_intrinsics": np.stack(K_list, axis=0),
        "camera_extrinsics_R": np.stack(R_cv_list, axis=0),
        "camera_extrinsics_t": np.stack(t_cv_list, axis=0),
        "image_sizes": np.stack(image_sizes, axis=0),
        "keypoints_3d": keypoints_3d,
        "joint_angles": np.asarray(y["joint_angles"], dtype=np.float32),
        "root_rot": np.asarray(y["root_rot"], dtype=np.float32),
        "root_loc": np.asarray(y["root_loc"], dtype=np.float32),
        "shape_betas": np.asarray(y["shape_betas"], dtype=np.float32),
        "canonical_to_world_R": np.asarray(y["canonical_to_world"][0], dtype=np.float32),
        "canonical_to_world_t": np.asarray(y["canonical_to_world"][1], dtype=np.float32),
        "canonical_cam_id": int(y["canonical_cam_id"]),
    }


# ---------------------------------------------------------------------------
# Preprocessor.
# ---------------------------------------------------------------------------


class replicAntMultiViewPreprocessor:
    """Streams one HDF5 multi-view dataset from a flat-directory replicAnt
    multi-camera dataset. Output schema is identical to the layout produced by
    `preprocess_sleap_multiview_dataset.py` (modulo replicAnt-specific extras
    under /auxiliary/), so `SLEAPMultiViewDataset` reads either file unchanged.
    """

    def __init__(
        self,
        target_resolution: int = 512,
        backbone_name: str = "vit_large_patch16_224",
        jpeg_quality: int = 95,
        chunk_size: int = 8,
        compression: str = "gzip",
        compression_level: int = 6,
        frame_skip: int = 1,
        camera_subset: Optional[List[int]] = None,
        min_views_per_sample: int = 2,
        depth_occlusion_check: bool = True,
        depth_max_cm: float = 1000.0,
        depth_tolerance_cm: float = 5.0,
        depth_neighborhood: int = 1,
        propagate_scaling: bool = True,
        translation_factor: float = 0.1,
        debug: bool = False,
    ):
        if frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")
        if not (1 <= jpeg_quality <= 100):
            raise ValueError(f"jpeg_quality must be in [1, 100], got {jpeg_quality}")
        if min_views_per_sample < 1:
            raise ValueError(
                f"min_views_per_sample must be >= 1, got {min_views_per_sample}"
            )

        self.target_resolution = int(target_resolution)
        self.backbone_name = backbone_name
        self.jpeg_quality = int(jpeg_quality)
        self.chunk_size = int(chunk_size)
        self.compression = compression
        self.compression_level = int(compression_level)
        self.frame_skip = int(frame_skip)
        self.camera_subset = list(camera_subset) if camera_subset is not None else None
        self.min_views_per_sample = int(min_views_per_sample)
        self.depth_occlusion_check = bool(depth_occlusion_check)
        self.depth_max_cm = float(depth_max_cm)
        self.depth_tolerance_cm = float(depth_tolerance_cm)
        self.depth_neighborhood = int(depth_neighborhood)
        self.propagate_scaling = bool(propagate_scaling)
        self.translation_factor = float(translation_factor)
        self.debug = bool(debug)

    # ---------- dataset discovery ------------------------------------------

    def _detect_dataset_structure(self, data_path: Path) -> Tuple[str, List[int], List[int]]:
        """Returns (dataset_name, canonical_camera_subset, frame_indices)."""
        batch_files = list(data_path.glob("_BatchData_*.json"))
        if not batch_files:
            raise FileNotFoundError(f"No _BatchData_*.json found in {data_path}")
        batch_file = batch_files[0]
        dataset_name = batch_file.stem.replace("_BatchData_", "")

        # Discover cameras present at frame 0
        cam_files_00000 = list(data_path.glob(f"{dataset_name}_00000_CAM*.json"))
        all_cam_ids = sorted(
            {int(re.search(r"CAM(\d+)", f.name).group(1)) for f in cam_files_00000}
        )
        if not all_cam_ids:
            raise FileNotFoundError(
                f"No {dataset_name}_00000_CAM*.json files under {data_path}"
            )

        if self.camera_subset is not None:
            missing = set(self.camera_subset) - set(all_cam_ids)
            if missing:
                raise ValueError(
                    f"camera_subset {sorted(self.camera_subset)} requested but cameras "
                    f"{sorted(missing)} not present at frame 00000 (have {all_cam_ids})"
                )
            camera_subset = sorted(self.camera_subset)
        else:
            camera_subset = all_cam_ids

        # Discover frames present (use the FIRST canonical cam as the index).
        canonical_cam = camera_subset[0]
        cam_files = list(
            data_path.glob(f"{dataset_name}_*_CAM{canonical_cam}.json")
        )
        frame_indices: List[int] = []
        pattern = re.compile(
            rf"{re.escape(dataset_name)}_(\d+)_CAM{canonical_cam}\.json$"
        )
        for cf in cam_files:
            m = pattern.search(cf.name)
            if m:
                frame_indices.append(int(m.group(1)))
        if not frame_indices:
            raise FileNotFoundError(
                f"No CAM{canonical_cam} frame JSON files matched under {data_path}"
            )
        frame_indices.sort()

        return dataset_name, camera_subset, frame_indices

    # ---------- HDF5 setup --------------------------------------------------

    def _create_hdf5_layout(
        self,
        f: h5py.File,
        num_alloc: int,
        max_views: int,
        n_joints: int,
        n_pose: int,
        n_betas: int,
    ) -> Dict[str, Any]:
        """Pre-allocate all fixed-shape datasets at `num_alloc` rows. Returns a
        dict of handles for streaming writes."""
        images_g = f.create_group("multiview_images")
        kp_g = f.create_group("multiview_keypoints")
        par_g = f.create_group("parameters")
        aux_g = f.create_group("auxiliary")
        meta_g = f.create_group("metadata")

        ck_n = max(1, min(self.chunk_size, num_alloc))
        comp = dict(compression=self.compression, compression_opts=self.compression_level)

        # Variable-length JPEG bytes, one dataset per view slot.
        vlen_u8 = h5py.vlen_dtype(np.uint8)
        image_jpeg_datasets = []
        for v in range(max_views):
            ds = images_g.create_dataset(
                f"image_jpeg_view_{v}",
                shape=(num_alloc,),
                maxshape=(None,),
                dtype=vlen_u8,
            )
            image_jpeg_datasets.append(ds)

        view_mask = images_g.create_dataset(
            "view_mask",
            shape=(num_alloc, max_views),
            maxshape=(None, max_views),
            dtype=bool,
            chunks=(ck_n, max_views),
            **comp,
        )

        keypoints_2d = kp_g.create_dataset(
            "keypoints_2d",
            shape=(num_alloc, max_views, n_joints, 2),
            maxshape=(None, max_views, n_joints, 2),
            dtype=np.float32,
            chunks=(ck_n, max_views, n_joints, 2),
            **comp,
        )
        keypoint_visibility = kp_g.create_dataset(
            "keypoint_visibility",
            shape=(num_alloc, max_views, n_joints),
            maxshape=(None, max_views, n_joints),
            dtype=np.float32,
            chunks=(ck_n, max_views, n_joints),
            **comp,
        )
        camera_indices = kp_g.create_dataset(
            "camera_indices",
            shape=(num_alloc, max_views),
            maxshape=(None, max_views),
            dtype=np.int32,
            chunks=(ck_n, max_views),
            **comp,
        )
        camera_intrinsics = kp_g.create_dataset(
            "camera_intrinsics",
            shape=(num_alloc, max_views, 3, 3),
            maxshape=(None, max_views, 3, 3),
            dtype=np.float32,
            chunks=(ck_n, max_views, 3, 3),
            **comp,
        )
        camera_extrinsics_R = kp_g.create_dataset(
            "camera_extrinsics_R",
            shape=(num_alloc, max_views, 3, 3),
            maxshape=(None, max_views, 3, 3),
            dtype=np.float32,
            chunks=(ck_n, max_views, 3, 3),
            **comp,
        )
        camera_extrinsics_t = kp_g.create_dataset(
            "camera_extrinsics_t",
            shape=(num_alloc, max_views, 3),
            maxshape=(None, max_views, 3),
            dtype=np.float32,
            chunks=(ck_n, max_views, 3),
            **comp,
        )
        image_sizes = kp_g.create_dataset(
            "image_sizes",
            shape=(num_alloc, max_views, 2),
            maxshape=(None, max_views, 2),
            dtype=np.int32,
            chunks=(ck_n, max_views, 2),
            **comp,
        )
        keypoints_3d = kp_g.create_dataset(
            "keypoints_3d",
            shape=(num_alloc, n_joints, 3),
            maxshape=(None, n_joints, 3),
            dtype=np.float32,
            chunks=(ck_n, n_joints, 3),
            **comp,
        )

        global_rot = par_g.create_dataset(
            "global_rot",
            shape=(num_alloc, 3),
            maxshape=(None, 3),
            dtype=np.float32,
            chunks=(ck_n, 3),
            **comp,
        )
        joint_rot = par_g.create_dataset(
            "joint_rot",
            shape=(num_alloc, n_pose + 1, 3),
            maxshape=(None, n_pose + 1, 3),
            dtype=np.float32,
            chunks=(ck_n, n_pose + 1, 3),
            **comp,
        )
        betas = par_g.create_dataset(
            "betas",
            shape=(num_alloc, n_betas),
            maxshape=(None, n_betas),
            dtype=np.float32,
            chunks=(ck_n, n_betas),
            **comp,
        )
        trans = par_g.create_dataset(
            "trans",
            shape=(num_alloc, 3),
            maxshape=(None, 3),
            dtype=np.float32,
            chunks=(ck_n, 3),
            **comp,
        )

        has_3d_data = aux_g.create_dataset(
            "has_3d_data",
            shape=(num_alloc,),
            maxshape=(None,),
            dtype=bool,
            chunks=(ck_n,),
            **comp,
        )
        has_gt_betas = aux_g.create_dataset(
            "has_ground_truth_betas",
            shape=(num_alloc,),
            maxshape=(None,),
            dtype=bool,
            chunks=(ck_n,),
            **comp,
        )
        num_views_ds = aux_g.create_dataset(
            "num_views",
            shape=(num_alloc,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(ck_n,),
            **comp,
        )
        frame_idx_ds = aux_g.create_dataset(
            "frame_idx",
            shape=(num_alloc,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(ck_n,),
            **comp,
        )
        canonical_to_world_R_ds = aux_g.create_dataset(
            "canonical_to_world_R",
            shape=(num_alloc, 3, 3),
            maxshape=(None, 3, 3),
            dtype=np.float32,
            chunks=(ck_n, 3, 3),
            **comp,
        )
        canonical_to_world_t_ds = aux_g.create_dataset(
            "canonical_to_world_t",
            shape=(num_alloc, 3),
            maxshape=(None, 3),
            dtype=np.float32,
            chunks=(ck_n, 3),
            **comp,
        )
        canonical_cam_id_ds = aux_g.create_dataset(
            "canonical_cam_id",
            shape=(num_alloc,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(ck_n,),
            **comp,
        )
        vlen_str = h5py.string_dtype(encoding="utf-8")
        session_name_ds = aux_g.create_dataset(
            "session_name",
            shape=(num_alloc,),
            maxshape=(None,),
            dtype=vlen_str,
        )
        camera_names_ds = aux_g.create_dataset(
            "camera_names",
            shape=(num_alloc,),
            maxshape=(None,),
            dtype=vlen_str,
        )

        return {
            "metadata_group": meta_g,
            "image_jpeg_datasets": image_jpeg_datasets,
            "view_mask": view_mask,
            "keypoints_2d": keypoints_2d,
            "keypoint_visibility": keypoint_visibility,
            "camera_indices": camera_indices,
            "camera_intrinsics": camera_intrinsics,
            "camera_extrinsics_R": camera_extrinsics_R,
            "camera_extrinsics_t": camera_extrinsics_t,
            "image_sizes": image_sizes,
            "keypoints_3d": keypoints_3d,
            "global_rot": global_rot,
            "joint_rot": joint_rot,
            "betas": betas,
            "trans": trans,
            "has_3d_data": has_3d_data,
            "has_gt_betas": has_gt_betas,
            "num_views": num_views_ds,
            "frame_idx": frame_idx_ds,
            "canonical_to_world_R": canonical_to_world_R_ds,
            "canonical_to_world_t": canonical_to_world_t_ds,
            "canonical_cam_id": canonical_cam_id_ds,
            "session_name": session_name_ds,
            "camera_names": camera_names_ds,
        }

    def _resize_datasets(self, handles: Dict[str, Any], num_written: int, max_views: int) -> None:
        for ds in handles["image_jpeg_datasets"]:
            ds.resize((num_written,))
        # All non-image datasets have the sample axis as axis 0; resize keeps
        # the remaining axes intact.
        for key in (
            "view_mask",
            "keypoints_2d",
            "keypoint_visibility",
            "camera_indices",
            "camera_intrinsics",
            "camera_extrinsics_R",
            "camera_extrinsics_t",
            "image_sizes",
            "keypoints_3d",
            "global_rot",
            "joint_rot",
            "betas",
            "trans",
            "has_3d_data",
            "has_gt_betas",
            "num_views",
            "frame_idx",
            "canonical_to_world_R",
            "canonical_to_world_t",
            "canonical_cam_id",
            "session_name",
            "camera_names",
        ):
            ds = handles[key]
            new_shape = (num_written,) + ds.shape[1:]
            ds.resize(new_shape)

    # ---------- main entry --------------------------------------------------

    def preprocess(
        self,
        input_dir: str,
        output_hdf5: str,
        num_workers: int = 8,
        max_frames: Optional[int] = None,
        smal_file: Optional[str] = None,
        shape_family: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        data_path = Path(input_dir)
        if not data_path.is_dir():
            raise NotADirectoryError(f"input_dir does not exist: {data_path}")

        dataset_name, camera_subset, all_frame_indices = self._detect_dataset_structure(data_path)
        canonical_camera_order = [f"CAM{c}" for c in camera_subset]
        max_views = len(camera_subset)
        canonical_camera_names_str = ",".join(canonical_camera_order)

        # Apply frame_skip then max_frames.
        frame_indices = all_frame_indices[:: self.frame_skip]
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        num_alloc = len(frame_indices)
        if num_alloc == 0:
            raise ValueError("No frames left after frame_skip/max_frames filtering")

        # n_joints / n_pose / n_betas — derive from config AFTER the SMAL pickle
        # override (so the main-process state matches what workers will compute).
        if smal_file:
            apply_smal_file_override(smal_file, shape_family=shape_family)
        n_joints = int(len(config.dd["J_names"]))
        n_pose = int(config.N_POSE)
        n_betas = int(config.N_BETAS)

        if verbose:
            print(f"Dataset: {dataset_name}")
            print(f"  flat-dir path:           {data_path}")
            print(f"  cameras (subset):        {camera_subset}")
            print(f"  canonical_camera_order:  {canonical_camera_order}")
            print(f"  total frames:            {len(all_frame_indices)}")
            print(f"  frames after skip/max:   {num_alloc} (frame_skip={self.frame_skip}, max_frames={max_frames})")
            print(f"  n_joints={n_joints}, n_pose={n_pose}, n_betas={n_betas}")
            print(f"  translation_factor:      {self.translation_factor}")
            print(f"  depth_occlusion_check:   {self.depth_occlusion_check}")
            print(f"  jpeg_quality:            {self.jpeg_quality}")
            print(f"  num_workers:             {num_workers}")
            print()

        worker_tasks = [
            {
                "frame_idx": fi,
                "data_path": str(data_path),
                "camera_subset": list(camera_subset),
                "propagate_scaling": self.propagate_scaling,
                "translation_factor": self.translation_factor,
                "jpeg_quality": self.jpeg_quality,
                "min_views_per_sample": self.min_views_per_sample,
                "depth_occlusion_check": self.depth_occlusion_check,
                "depth_max_cm": self.depth_max_cm,
                "depth_tolerance_cm": self.depth_tolerance_cm,
                "depth_neighborhood": self.depth_neighborhood,
            }
            for fi in frame_indices
        ]

        start_time = time.time()
        num_written = 0
        skipped: List[Tuple[int, str]] = []

        with h5py.File(output_hdf5, "w") as f:
            handles = self._create_hdf5_layout(
                f, num_alloc, max_views, n_joints, n_pose, n_betas
            )
            cam_idx_map = {cam_id: idx for idx, cam_id in enumerate(camera_subset)}

            executor_kwargs = dict(
                max_workers=num_workers,
                initializer=_worker_init,
                initargs=(smal_file, shape_family),
            )
            with ProcessPoolExecutor(**executor_kwargs) as pool:
                results_iter = pool.map(_process_frame, worker_tasks, chunksize=1)
                progress = tqdm(
                    results_iter,
                    total=len(worker_tasks),
                    desc="Preprocess",
                    disable=not verbose,
                )
                for result in progress:
                    if not result.get("ok"):
                        skipped.append((int(result["frame_idx"]), str(result.get("reason", "unknown"))))
                        if self.debug and verbose:
                            tqdm.write(
                                f"  SKIP frame {result['frame_idx']}: {result.get('reason')}"
                            )
                        continue

                    i = num_written
                    nv = int(result["num_views"])
                    view_valid_per_view = list(
                        result.get("view_valid_per_view", [True] * nv)
                    )
                    # Per-view image blobs and view_mask.
                    # The JPG image and camera geometry (K/R/t) are written for
                    # every slot regardless of subject-data validity — the image
                    # itself is real, the camera params still describe the rig.
                    # `view_mask` and `camera_indices` reflect per-view supervision
                    # validity: invalid views get view_mask=False + camera_indices=-1
                    # so SLEAPMultiViewDataset drops the slot at __getitem__ time.
                    view_mask_row = np.zeros((max_views,), dtype=bool)
                    cam_idx_row = np.full((max_views,), -1, dtype=np.int32)
                    for v in range(nv):
                        # Slot index: the v-th camera in this frame should land in
                        # the canonical slot for its camera id. With a fixed
                        # camera_subset and canonical_frame=True, the loader
                        # returns views in the same order — assert it.
                        cam_id = result["camera_ids"][v]
                        slot = cam_idx_map.get(int(cam_id), v)
                        handles["image_jpeg_datasets"][slot][i] = np.frombuffer(
                            result["jpeg_blobs"][v], dtype=np.uint8
                        )
                        if view_valid_per_view[v]:
                            view_mask_row[slot] = True
                            cam_idx_row[slot] = slot
                        # else: leave view_mask_row[slot]=False and
                        # cam_idx_row[slot]=-1 (the "this slot is not a real
                        # supervisable view" convention).

                    # Fixed-shape per-view fields. Stored in slot order (matches
                    # canonical_camera_order); padded slots stay at zeros.
                    kp2d_row = np.zeros((max_views, n_joints, 2), dtype=np.float32)
                    vis_row = np.zeros((max_views, n_joints), dtype=np.float32)
                    K_row = np.zeros((max_views, 3, 3), dtype=np.float32)
                    R_row = np.zeros((max_views, 3, 3), dtype=np.float32)
                    t_row = np.zeros((max_views, 3), dtype=np.float32)
                    sz_row = np.zeros((max_views, 2), dtype=np.int32)
                    for v in range(nv):
                        cam_id = result["camera_ids"][v]
                        slot = cam_idx_map.get(int(cam_id), v)
                        kp2d_row[slot] = result["keypoints_2d"][v]
                        vis_row[slot] = result["keypoint_visibility"][v]
                        K_row[slot] = result["camera_intrinsics"][v]
                        R_row[slot] = result["camera_extrinsics_R"][v]
                        t_row[slot] = result["camera_extrinsics_t"][v]
                        sz_row[slot] = result["image_sizes"][v]

                    handles["view_mask"][i] = view_mask_row
                    handles["camera_indices"][i] = cam_idx_row
                    handles["keypoints_2d"][i] = kp2d_row
                    handles["keypoint_visibility"][i] = vis_row
                    handles["camera_intrinsics"][i] = K_row
                    handles["camera_extrinsics_R"][i] = R_row
                    handles["camera_extrinsics_t"][i] = t_row
                    handles["image_sizes"][i] = sz_row

                    handles["keypoints_3d"][i] = result["keypoints_3d"]
                    handles["global_rot"][i] = result["root_rot"]
                    handles["joint_rot"][i] = result["joint_angles"]
                    handles["betas"][i] = result["shape_betas"]
                    handles["trans"][i] = result["root_loc"]

                    handles["has_3d_data"][i] = True
                    handles["has_gt_betas"][i] = True
                    # `num_views` stores the count of valid (supervisable) views,
                    # matching the SLEAP convention and the trainer's expectation.
                    handles["num_views"][i] = int(sum(view_valid_per_view))
                    handles["frame_idx"][i] = result["frame_idx"]
                    handles["canonical_to_world_R"][i] = result["canonical_to_world_R"]
                    handles["canonical_to_world_t"][i] = result["canonical_to_world_t"]
                    handles["canonical_cam_id"][i] = result["canonical_cam_id"]
                    handles["session_name"][i] = dataset_name
                    handles["camera_names"][i] = canonical_camera_names_str

                    num_written += 1

            # Resize down to actual successful count.
            if num_written < num_alloc:
                self._resize_datasets(handles, num_written, max_views)

            # Metadata attrs.
            meta = handles["metadata_group"]
            meta.attrs["num_samples"] = int(num_written)
            meta.attrs["max_views"] = int(max_views)
            meta.attrs["n_joints"] = n_joints
            meta.attrs["n_pose"] = n_pose
            meta.attrs["n_betas"] = n_betas
            meta.attrs["target_resolution"] = int(self.target_resolution)
            meta.attrs["backbone_name"] = self.backbone_name
            meta.attrs["jpeg_quality"] = int(self.jpeg_quality)
            meta.attrs["dataset_type"] = "replicant_multiview"
            meta.attrs["is_multiview"] = True
            meta.attrs["has_camera_parameters"] = True
            meta.attrs["has_3d_keypoints"] = True
            meta.attrs["load_3d_data"] = True
            meta.attrs["world_scale"] = 1.0
            meta.attrs["canonical_camera_order"] = json.dumps(canonical_camera_order)
            meta.attrs["dataset_name"] = dataset_name
            meta.attrs["frame_skip"] = int(self.frame_skip)
            meta.attrs["translation_factor"] = float(self.translation_factor)
            meta.attrs["propagate_scaling"] = bool(self.propagate_scaling)
            meta.attrs["depth_occlusion_check"] = bool(self.depth_occlusion_check)
            meta.attrs["depth_max_cm"] = float(self.depth_max_cm)
            meta.attrs["depth_tolerance_cm"] = float(self.depth_tolerance_cm)
            meta.attrs["depth_neighborhood"] = int(self.depth_neighborhood)
            meta.attrs["min_views_per_sample"] = int(self.min_views_per_sample)
            meta.attrs["camera_extrinsics_convention"] = "opencv"
            # Skipped-frame ledger.
            meta.attrs["num_skipped_frames"] = len(skipped)
            if skipped:
                meta.attrs["skipped_frame_indices"] = np.array(
                    [s[0] for s in skipped], dtype=np.int32
                )
                meta.attrs["skipped_frame_reasons"] = json.dumps(
                    {int(idx): reason for idx, reason in skipped}
                )

        elapsed = time.time() - start_time
        if verbose:
            print()
            print(f"Wrote {num_written} samples to {output_hdf5} in {elapsed:.1f}s")
            if skipped:
                print(f"Skipped {len(skipped)} frames:")
                for idx, reason in skipped[:10]:
                    print(f"  frame {idx}: {reason}")
                if len(skipped) > 10:
                    print(f"  … and {len(skipped) - 10} more")

        return {
            "num_written": num_written,
            "num_skipped": len(skipped),
            "skipped": skipped,
            "elapsed_seconds": elapsed,
            "output_hdf5": str(output_hdf5),
        }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_camera_subset(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess a flat-directory replicAnt multi-camera dataset into HDF5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir", required=True, help="Flat-directory replicAnt dataset")
    parser.add_argument("--output_hdf5", required=True, help="Output HDF5 path")
    parser.add_argument("--smal_file", default=None, help="SMAL/SMIL .pkl matching the dataset skeleton")
    parser.add_argument("--shape_family", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument(
        "--camera_subset",
        type=str,
        default=None,
        help='Comma-separated camera IDs (e.g. "1,2,3"). Default: all cameras at frame 0.',
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Cap number of frames (smoke test)")
    parser.add_argument("--min_views", type=int, default=2,
                        help="Minimum valid views per sample. Frames with fewer "
                             "cameras containing valid subject data are discarded; "
                             "frames with >= this many keep all camera slots but "
                             "set view_mask=False on the invalid ones. Default: 2.")
    parser.add_argument("--translation_factor", type=float, default=0.1,
                        help="Loader-side uniform world scale (default 0.1 unifies mesh + data units)")
    parser.add_argument("--depth_occlusion_check", dest="depth_occlusion_check",
                        action="store_true", default=True)
    parser.add_argument("--no_depth_occlusion_check", dest="depth_occlusion_check",
                        action="store_false")
    parser.add_argument("--depth_max_cm", type=float, default=1000.0)
    parser.add_argument("--depth_tolerance_cm", type=float, default=5.0)
    parser.add_argument("--depth_neighborhood", type=int, default=1)
    parser.add_argument("--target_resolution", type=int, default=512,
                        help="Stored image resolution (default: 512, native). Dataset class resizes at load.")
    parser.add_argument("--backbone_name", type=str, default="vit_large_patch16_224")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.smal_file:
        smal_path = Path(args.smal_file)
        if not smal_path.is_file():
            print(f"Error: --smal_file does not exist: {smal_path}", file=sys.stderr)
            sys.exit(1)

    output_dir = os.path.dirname(args.output_hdf5)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    preprocessor = replicAntMultiViewPreprocessor(
        target_resolution=args.target_resolution,
        backbone_name=args.backbone_name,
        jpeg_quality=args.jpeg_quality,
        chunk_size=args.chunk_size,
        frame_skip=args.frame_skip,
        camera_subset=_parse_camera_subset(args.camera_subset),
        min_views_per_sample=args.min_views,
        depth_occlusion_check=args.depth_occlusion_check,
        depth_max_cm=args.depth_max_cm,
        depth_tolerance_cm=args.depth_tolerance_cm,
        depth_neighborhood=args.depth_neighborhood,
        translation_factor=args.translation_factor,
        debug=args.debug,
    )

    try:
        stats = preprocessor.preprocess(
            input_dir=args.input_dir,
            output_hdf5=args.output_hdf5,
            num_workers=args.num_workers,
            max_frames=args.max_frames,
            smal_file=args.smal_file,
            shape_family=args.shape_family,
            verbose=True,
        )
    except Exception as exc:
        print(f"\nError during preprocessing: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    print(f"\nDone. {stats['num_written']} samples written; {stats['num_skipped']} skipped.")


if __name__ == "__main__":
    main()
