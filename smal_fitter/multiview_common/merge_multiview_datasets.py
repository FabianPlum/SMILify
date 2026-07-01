#!/usr/bin/env python3
"""Merge multiple multi-view SMIL HDF5 datasets (SLEAP-format + replicAnt-format)
into one uniform-convention HDF5.

Why and what
------------

Both `SLEAPMultiViewDataset`-format HDF5s (produced by
`smal_fitter/sleap_data/preprocess_sleap_multiview_dataset.py`) and the
replicAnt multi-view format (produced by
`smal_fitter/replicAnt_data/preprocess_replicant_multiview_dataset.py`)
share an on-disk schema. They differ in three load-bearing ways:

1. **Camera/3D coordinate frame.** replicAnt stores cameras in a
   canonical-camera-frame OpenCV form whose reader-time inverse hands the
   trainer cam-0 = (I, 0) in PyTorch3D. SLEAP stores the raw rig-world
   OpenCV calibration, which gives an arbitrary per-session cam-0 in the
   trainer's frame. Mixing the two without normalisation breaks the
   shared `trans` head because the same image features map to disjoint
   trans values across sources.
2. **`world_scale`.** SLEAP omits the attribute and the reader infers
   `0.001` from `||t|| > 50` (mm → m). replicAnt writes `1.0` and the
   loader has already baked `translation_factor=0.1` into stored coords.
3. **Stored JPEG resolution vs `image_sizes` / `K`.** SLEAP's stored
   JPEG is at `target_resolution` but `K`, `image_sizes`, and 2D
   keypoints are in the original calibration frame. replicAnt is
   self-consistent at native resolution.

The merger resolves all three at write time and produces a single output
HDF5 in which every sample lives in the same convention as a replicAnt
sample (cam-0 OpenCV = `Rz_180`, world_scale = 1.0, K/image_sizes/JPEG
all at `--jpeg_resolution`). The trainer reads it via the existing
`SLEAPMultiViewDataset` unchanged.

Convention references
---------------------

Per-sample canonical-camera-frame storage follows the convention used by:

- HMR (Kanazawa et al., CVPR 2018) — canonical body frame + weak perspective.
- Expose (Choutas et al., ECCV 2020) — explicit per-sample canonical camera.
- AGORA (Patel et al., CVPR 2021) — synth multi-view, canonical reference cam.
- 4D-Humans / SLAHMR (Goel et al., ICCV 2023) — multi-view canonical-frame
  training.
- FreeMan (Wang et al., CVPR 2024) — multi-view in-the-wild 3D pose,
  per-sample canonical reference camera.
- EFT (Joo et al., 3DV 2021) — explicit cross-dataset camera-convention
  normalisation before mixing training data.

OpenCV column-vector convention + camera centre algebra: Hartley &
Zisserman, "Multiple View Geometry in Computer Vision" (2nd ed., 2004,
ch. 6) and OpenCV's `cv2.projectPoints` docs.

Per-camera-resilience (variable `view_mask`) follows the design landed
on the multiview-replicant-integration branch (see
MULTIVIEW_REPLICANT_INTEGRATION_DESIGN.md, Phase 3 + 7).

Usage
-----

    python smal_fitter/multiview_common/merge_multiview_datasets.py \\
        --inputs SMILymice_3D_6_cam_undistort.h5 replicant_multiview_mice.h5 \\
        --output merged_multiview.h5 \\
        --jpeg_resolution 512 \\
        [--jpeg_quality 95] [--chunk_size 8]

`n_joints`, `n_pose`, and `n_betas` are hard-asserted equal across all
inputs (they're SMAL-pickle-bound; no per-sample reconciliation is
possible). All inputs must have `is_multiview=True`. SLEAP samples
with `has_3d_data=False` are kept with cameras canonicalised but
keypoints_3d left at the zero sentinel.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# Shared geometry helpers (see canonical_frame.py for the conventions doc).
_THIS_DIR = Path(__file__).resolve().parent
from smal_fitter.multiview_common.canonical_frame import (  # noqa: E402
    RZ_180,
    align_to_pytorch3d_reader_convention,
    canonicalize_sample,
    infer_world_scale,
)


# ---------------------------------------------------------------------------
# Input descriptor.
# ---------------------------------------------------------------------------


@dataclass
class InputInfo:
    """Per-input metadata gathered in the validate pass."""
    path: Path
    dataset_type: str           # 'sleap_multiview', 'replicant_multiview', 'merged_multiview'
    num_samples: int
    max_views: int
    n_joints: int
    n_pose: int
    n_betas: int
    world_scale: float
    target_resolution: int
    needs_canonicalization: bool   # True iff cameras are not already in replicAnt-storage form
    decoded_jpeg_hw: Tuple[int, int]   # (H, W) of the stored JPEG for view 0 sample 0


def _infer_per_input_world_scale(hf: h5py.File) -> float:
    """Return the world_scale to apply to t and kp3d at merge time. If the
    file declares one, use it; otherwise run the SLEAP reader heuristic
    on sample 0 cam 0."""
    md = hf["metadata"].attrs
    if "world_scale" in md:
        return float(md["world_scale"])
    if "multiview_keypoints" in hf and "camera_extrinsics_t" in hf["multiview_keypoints"]:
        t0 = hf["multiview_keypoints/camera_extrinsics_t"][0]  # (max_views, 3)
        mask0 = hf["multiview_images/view_mask"][0]
        return float(infer_world_scale(t0, mask0))
    return 1.0


def _median_subject_extent(hf: h5py.File, world_scale: float, sample_size: int = 400) -> float:
    """Median bbox diagonal of valid keypoints_3d (in reader units = stored
    units x world_scale), used to co-scale heterogeneous sources to one
    physical scale. Returns nan if no valid 3D keypoints are present."""
    n = int(hf["metadata"].attrs["num_samples"])
    if "multiview_keypoints" not in hf or "keypoints_3d" not in hf["multiview_keypoints"]:
        return float("nan")
    idx = np.linspace(0, n - 1, min(n, sample_size)).astype(int)
    kp3d = hf["multiview_keypoints/keypoints_3d"][idx].astype(np.float64) * float(world_scale)
    extents = []
    for kp in kp3d:
        ok = np.isfinite(kp).all(axis=1) & (np.abs(kp).sum(axis=1) > 1e-9)
        if ok.sum() >= 2:
            extents.append(float(np.linalg.norm(kp[ok].max(0) - kp[ok].min(0))))
    return float(np.median(extents)) if extents else float("nan")


def _peek_jpeg_hw(hf: h5py.File) -> Tuple[int, int]:
    """Decode sample 0 view 0's JPEG to learn its actual on-disk H, W. We
    need this because SLEAP's `image_sizes` records the calibration frame,
    not the stored JPEG dims."""
    blob = hf["multiview_images/image_jpeg_view_0"][0]
    img = cv2.imdecode(np.asarray(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Could not decode sample 0 view 0 JPEG to determine stored resolution")
    return int(img.shape[0]), int(img.shape[1])  # (H, W)


def validate_inputs(input_paths: List[Path]) -> List[InputInfo]:
    """Pass 1: hard-assert skeleton compatibility, collect per-input info."""
    infos: List[InputInfo] = []
    n_joints = n_pose = n_betas = None
    for p in input_paths:
        if not p.is_file():
            raise FileNotFoundError(f"Input HDF5 not found: {p}")
        with h5py.File(p, "r") as hf:
            md = hf["metadata"].attrs
            if not bool(md.get("is_multiview", False)):
                raise ValueError(f"{p}: not a multi-view HDF5 (is_multiview != True)")
            dtype_attr = md.get("dataset_type", b"")
            if isinstance(dtype_attr, bytes):
                dtype_attr = dtype_attr.decode("utf-8")
            dtype_attr = str(dtype_attr)

            this_nj = int(md["n_joints"])
            this_np = int(md["n_pose"])
            this_nb = int(md["n_betas"])
            if n_joints is None:
                n_joints, n_pose, n_betas = this_nj, this_np, this_nb
            else:
                if (this_nj, this_np, this_nb) != (n_joints, n_pose, n_betas):
                    raise ValueError(
                        f"{p}: skeleton mismatch — "
                        f"got (n_joints={this_nj}, n_pose={this_np}, n_betas={this_nb}), "
                        f"expected ({n_joints}, {n_pose}, {n_betas}) from earlier input. "
                        f"All inputs must be preprocessed against the same SMAL pickle."
                    )

            # replicAnt stores cam-0 in the convention the trainer expects;
            # everything else (sleap_multiview, anything written before this
            # convention was established, merger output recursively) is treated
            # as needing canonicalisation.
            needs_canon = dtype_attr != "replicant_multiview"

            jpeg_hw = _peek_jpeg_hw(hf)

            infos.append(InputInfo(
                path=p,
                dataset_type=dtype_attr,
                num_samples=int(md["num_samples"]),
                max_views=int(md["max_views"]),
                n_joints=this_nj,
                n_pose=this_np,
                n_betas=this_nb,
                world_scale=_infer_per_input_world_scale(hf),
                target_resolution=int(md.get("target_resolution", jpeg_hw[0])),
                needs_canonicalization=needs_canon,
                decoded_jpeg_hw=jpeg_hw,
            ))
    return infos


# ---------------------------------------------------------------------------
# JPEG handling.
# ---------------------------------------------------------------------------


def _decode_jpeg(blob: np.ndarray) -> Optional[np.ndarray]:
    if blob is None or len(blob) == 0:
        return None
    img = cv2.imdecode(np.asarray(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img  # BGR uint8 (H, W, 3); None on decode failure


def _resize_image(img: np.ndarray, target_res: int) -> np.ndarray:
    if img.shape[0] == target_res and img.shape[1] == target_res:
        return img
    # Default to area interpolation when downsizing, cubic when upsizing.
    interp = cv2.INTER_AREA if (img.shape[0] > target_res or img.shape[1] > target_res) else cv2.INTER_CUBIC
    return cv2.resize(img, (target_res, target_res), interpolation=interp)


def _encode_jpeg(bgr_img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _adjust_K_for_resolution(K: np.ndarray, input_image_size_wh: Tuple[int, int], target_res: int) -> np.ndarray:
    """Rescale an OpenCV intrinsic matrix so it applies to a square
    target_res image. Input K is calibrated for input_image_size_wh."""
    W_in, H_in = float(input_image_size_wh[0]), float(input_image_size_wh[1])
    sx = float(target_res) / W_in
    sy = float(target_res) / H_in
    K_out = K.copy()
    K_out[0, 0] *= sx
    K_out[0, 2] *= sx
    K_out[1, 1] *= sy
    K_out[1, 2] *= sy
    return K_out


# ---------------------------------------------------------------------------
# HDF5 output layout.
# ---------------------------------------------------------------------------


def _create_output_layout(
    hf: h5py.File,
    num_samples: int,
    max_views: int,
    n_joints: int,
    n_pose: int,
    n_betas: int,
    target_res: int,
    chunk_size: int,
    compression: str,
    compression_level: int,
) -> Dict[str, h5py.Dataset]:
    """Allocate every dataset the SLEAPMultiViewDataset reader requires
    (plus auxiliary provenance fields). Returns a flat dict of handles
    keyed by short name."""

    md = hf.create_group("metadata")
    images_g = hf.create_group("multiview_images")
    kps_g = hf.create_group("multiview_keypoints")
    params_g = hf.create_group("parameters")
    aux_g = hf.create_group("auxiliary")

    common_kw = dict(
        compression=compression,
        compression_opts=compression_level,
    )
    # JPEG vlen datasets — one per view slot.
    vlen_uint8 = h5py.vlen_dtype(np.uint8)
    image_jpeg_datasets = []
    for v in range(max_views):
        ds = images_g.create_dataset(
            f"image_jpeg_view_{v}",
            shape=(num_samples,),
            dtype=vlen_uint8,
            chunks=(chunk_size,),
            **common_kw,
        )
        image_jpeg_datasets.append(ds)

    handles = {
        "metadata": md,
        "image_jpeg_datasets": image_jpeg_datasets,
        "view_mask": images_g.create_dataset(
            "view_mask", shape=(num_samples, max_views), dtype=np.bool_,
            chunks=(chunk_size, max_views), **common_kw),
        "keypoints_2d": kps_g.create_dataset(
            "keypoints_2d", shape=(num_samples, max_views, n_joints, 2), dtype=np.float32,
            chunks=(chunk_size, max_views, n_joints, 2), **common_kw),
        "keypoint_visibility": kps_g.create_dataset(
            "keypoint_visibility", shape=(num_samples, max_views, n_joints), dtype=np.float32,
            chunks=(chunk_size, max_views, n_joints), **common_kw),
        "camera_indices": kps_g.create_dataset(
            "camera_indices", shape=(num_samples, max_views), dtype=np.int32,
            chunks=(chunk_size, max_views), **common_kw),
        "camera_intrinsics": kps_g.create_dataset(
            "camera_intrinsics", shape=(num_samples, max_views, 3, 3), dtype=np.float32,
            chunks=(chunk_size, max_views, 3, 3), **common_kw),
        "camera_extrinsics_R": kps_g.create_dataset(
            "camera_extrinsics_R", shape=(num_samples, max_views, 3, 3), dtype=np.float32,
            chunks=(chunk_size, max_views, 3, 3), **common_kw),
        "camera_extrinsics_t": kps_g.create_dataset(
            "camera_extrinsics_t", shape=(num_samples, max_views, 3), dtype=np.float32,
            chunks=(chunk_size, max_views, 3), **common_kw),
        "image_sizes": kps_g.create_dataset(
            "image_sizes", shape=(num_samples, max_views, 2), dtype=np.int32,
            chunks=(chunk_size, max_views, 2), **common_kw),
        "keypoints_3d": kps_g.create_dataset(
            "keypoints_3d", shape=(num_samples, n_joints, 3), dtype=np.float32,
            chunks=(chunk_size, n_joints, 3), **common_kw),
        "global_rot": params_g.create_dataset(
            "global_rot", shape=(num_samples, 3), dtype=np.float32,
            chunks=(chunk_size, 3), **common_kw),
        "joint_rot": params_g.create_dataset(
            "joint_rot", shape=(num_samples, n_pose + 1, 3), dtype=np.float32,
            chunks=(chunk_size, n_pose + 1, 3), **common_kw),
        "betas": params_g.create_dataset(
            "betas", shape=(num_samples, n_betas), dtype=np.float32,
            chunks=(chunk_size, n_betas), **common_kw),
        "trans": params_g.create_dataset(
            "trans", shape=(num_samples, 3), dtype=np.float32,
            chunks=(chunk_size, 3), **common_kw),
        # Auxiliary — including new provenance fields the merger emits.
        "has_3d_data": aux_g.create_dataset(
            "has_3d_data", shape=(num_samples,), dtype=np.bool_,
            chunks=(chunk_size,), **common_kw),
        "has_ground_truth_betas": aux_g.create_dataset(
            "has_ground_truth_betas", shape=(num_samples,), dtype=np.bool_,
            chunks=(chunk_size,), **common_kw),
        "num_views": aux_g.create_dataset(
            "num_views", shape=(num_samples,), dtype=np.int32,
            chunks=(chunk_size,), **common_kw),
        "frame_idx": aux_g.create_dataset(
            "frame_idx", shape=(num_samples,), dtype=np.int32,
            chunks=(chunk_size,), **common_kw),
        "session_name": aux_g.create_dataset(
            "session_name", shape=(num_samples,), dtype=h5py.string_dtype("utf-8"),
            chunks=(chunk_size,)),
        "camera_names": aux_g.create_dataset(
            "camera_names", shape=(num_samples,), dtype=h5py.string_dtype("utf-8"),
            chunks=(chunk_size,)),
        "canonical_to_world_R": aux_g.create_dataset(
            "canonical_to_world_R", shape=(num_samples, 3, 3), dtype=np.float32,
            chunks=(chunk_size, 3, 3), **common_kw),
        "canonical_to_world_t": aux_g.create_dataset(
            "canonical_to_world_t", shape=(num_samples, 3), dtype=np.float32,
            chunks=(chunk_size, 3), **common_kw),
        "canonical_cam_id": aux_g.create_dataset(
            "canonical_cam_id", shape=(num_samples,), dtype=np.int32,
            chunks=(chunk_size,), **common_kw),
        # Provenance — new fields, the reader ignores them.
        "origin_dataset": aux_g.create_dataset(
            "origin_dataset", shape=(num_samples,), dtype=h5py.string_dtype("utf-8"),
            chunks=(chunk_size,)),
        "origin_source_file": aux_g.create_dataset(
            "origin_source_file", shape=(num_samples,), dtype=h5py.string_dtype("utf-8"),
            chunks=(chunk_size,)),
        "origin_sample_idx": aux_g.create_dataset(
            "origin_sample_idx", shape=(num_samples,), dtype=np.int32,
            chunks=(chunk_size,), **common_kw),
    }
    return handles


# ---------------------------------------------------------------------------
# Per-sample copy logic.
# ---------------------------------------------------------------------------


def _copy_one_sample(
    src: h5py.File,
    src_idx: int,
    info: InputInfo,
    handles: Dict[str, h5py.Dataset],
    out_idx: int,
    merged_max_views: int,
    jpeg_resolution: int,
    jpeg_quality: int,
) -> None:
    """Read one sample from src, transform it into the uniform output
    convention, and stream-write to the pre-allocated output handles."""

    n_joints = handles["keypoints_3d"].shape[1]
    src_max_views = info.max_views

    # 1. Read raw per-view tensors.
    view_mask_in = src["multiview_images/view_mask"][src_idx].astype(bool)
    kp2d_in = src["multiview_keypoints/keypoints_2d"][src_idx]  # already normalised [y/H, x/W]
    vis_in = src["multiview_keypoints/keypoint_visibility"][src_idx]
    K_in = src["multiview_keypoints/camera_intrinsics"][src_idx].astype(np.float64)
    R_in = src["multiview_keypoints/camera_extrinsics_R"][src_idx].astype(np.float64)
    t_in = src["multiview_keypoints/camera_extrinsics_t"][src_idx].astype(np.float64)
    image_sizes_in = src["multiview_keypoints/image_sizes"][src_idx]  # (V, 2) (W, H)
    cam_idx_in = src["multiview_keypoints/camera_indices"][src_idx]

    kp3d_in = src["multiview_keypoints/keypoints_3d"][src_idx].astype(np.float64)
    has_3d = bool(src["auxiliary/has_3d_data"][src_idx])

    # 2. Pre-scale t and kp3d by per-input world_scale so the merger emits
    # world_scale=1.0 universally. Uniform scaling is projection-invariant.
    s = float(info.world_scale)
    if s != 1.0:
        t_scaled = t_in * s
        kp3d_scaled = kp3d_in.copy()
        has_gt_3d = ~np.all(kp3d_in == 0, axis=1)
        if has_gt_3d.any():
            kp3d_scaled[has_gt_3d] = kp3d_in[has_gt_3d] * s
    else:
        t_scaled = t_in
        kp3d_scaled = kp3d_in

    # 3. Canonicalisation + alignment-with-replicAnt-storage.
    # replicAnt samples already in target convention -> skip the transform
    # (transforming would still be projection-invariant but would change the
    # stored numerical values and break round-trip tests against the source
    # replicAnt HDF5).
    if info.needs_canonicalization and bool(view_mask_in.any()):
        R_can, t_can, kp3d_can, R_0, t_0, canonical_v = canonicalize_sample(
            R_scaled := R_in, t_scaled, kp3d_scaled, view_mask_in
        )
        R_out_v, t_out_v, kp3d_out_v = align_to_pytorch3d_reader_convention(
            R_can, t_can, kp3d_can, view_mask_in
        )
        # canonical_to_world for round-trip back to the *scaled but un-canonicalised*
        # input frame. Down-stream consumers that want the raw rig-world frame
        # also need to divide by `world_scale` (recorded in the source attr).
        c2w_R = R_0.astype(np.float32)
        c2w_t = t_0.astype(np.float32)
        canonical_cam_id_out = int(canonical_v)
    else:
        # replicAnt input — already canonical-in-PyTorch3D-via-OpenCV form.
        # Pass through; preserve any stored canonical_to_world if present.
        R_out_v = R_in.copy()
        t_out_v = t_scaled.copy()
        kp3d_out_v = kp3d_scaled.copy()
        if "canonical_to_world_R" in src["auxiliary"]:
            c2w_R = src["auxiliary/canonical_to_world_R"][src_idx].astype(np.float32)
            c2w_t = src["auxiliary/canonical_to_world_t"][src_idx].astype(np.float32) * s
        else:
            c2w_R = np.eye(3, dtype=np.float32)
            c2w_t = np.zeros(3, dtype=np.float32)
        if "canonical_cam_id" in src["auxiliary"]:
            canonical_cam_id_out = int(src["auxiliary/canonical_cam_id"][src_idx])
        else:
            canonical_cam_id_out = int(np.argmax(view_mask_in))

    # 4. Padded per-view output buffers.
    kp2d_out = np.zeros((merged_max_views, n_joints, 2), dtype=np.float32)
    vis_out = np.zeros((merged_max_views, n_joints), dtype=np.float32)
    K_out = np.zeros((merged_max_views, 3, 3), dtype=np.float32)
    R_out = np.zeros((merged_max_views, 3, 3), dtype=np.float32)
    t_out = np.zeros((merged_max_views, 3), dtype=np.float32)
    sizes_out = np.zeros((merged_max_views, 2), dtype=np.int32)
    cam_idx_out = np.full((merged_max_views,), -1, dtype=np.int32)
    view_mask_out = np.zeros((merged_max_views,), dtype=bool)

    # 5. JPEG + K rescale per valid view.
    for v in range(src_max_views):
        if not view_mask_in[v]:
            continue
        # Resize image to jpeg_resolution and re-encode.
        blob = src[f"multiview_images/image_jpeg_view_{v}"][src_idx]
        img = _decode_jpeg(blob)
        if img is None:
            # Treat as missing view (defensive: shouldn't happen if view_mask=True).
            continue
        img_resized = _resize_image(img, jpeg_resolution)
        new_blob = _encode_jpeg(img_resized, jpeg_quality)

        # Rescale K from input image_sizes -> jpeg_resolution.
        W_in_v, H_in_v = int(image_sizes_in[v, 0]), int(image_sizes_in[v, 1])
        K_rescaled = _adjust_K_for_resolution(K_in[v], (W_in_v, H_in_v), jpeg_resolution)

        # Slot index: use the same slot as the source so that camera_indices
        # ordering is preserved within a sample. The cross-sample camera_indices
        # meaning is already lost on a heterogeneous merge.
        slot = v
        handles["image_jpeg_datasets"][slot][out_idx] = np.frombuffer(new_blob, dtype=np.uint8)
        kp2d_out[slot] = kp2d_in[v]            # [y/H, x/W] normalised — same numerically
        vis_out[slot] = vis_in[v]
        K_out[slot] = K_rescaled.astype(np.float32)
        R_out[slot] = R_out_v[v].astype(np.float32)
        t_out[slot] = t_out_v[v].astype(np.float32)
        sizes_out[slot] = np.array([jpeg_resolution, jpeg_resolution], dtype=np.int32)
        cam_idx_out[slot] = int(cam_idx_in[v]) if cam_idx_in[v] >= 0 else slot
        view_mask_out[slot] = True

    # 6. Shared per-sample fields. SLEAP placeholders (zeros) for trans /
    # global_rot etc. stay zero (zero is zero in any frame). replicAnt
    # real values are preserved.
    global_rot = src["parameters/global_rot"][src_idx].astype(np.float32)
    joint_rot = src["parameters/joint_rot"][src_idx].astype(np.float32)
    betas = src["parameters/betas"][src_idx].astype(np.float32)
    trans = (src["parameters/trans"][src_idx].astype(np.float64) * s).astype(np.float32)

    # 7. Stream-write to output handles.
    handles["view_mask"][out_idx] = view_mask_out
    handles["keypoints_2d"][out_idx] = kp2d_out
    handles["keypoint_visibility"][out_idx] = vis_out
    handles["camera_indices"][out_idx] = cam_idx_out
    handles["camera_intrinsics"][out_idx] = K_out
    handles["camera_extrinsics_R"][out_idx] = R_out
    handles["camera_extrinsics_t"][out_idx] = t_out
    handles["image_sizes"][out_idx] = sizes_out
    handles["keypoints_3d"][out_idx] = kp3d_out_v.astype(np.float32)

    handles["global_rot"][out_idx] = global_rot
    handles["joint_rot"][out_idx] = joint_rot
    handles["betas"][out_idx] = betas
    handles["trans"][out_idx] = trans

    handles["has_3d_data"][out_idx] = has_3d
    if "has_ground_truth_betas" in src["auxiliary"]:
        handles["has_ground_truth_betas"][out_idx] = bool(src["auxiliary/has_ground_truth_betas"][src_idx])
    else:
        handles["has_ground_truth_betas"][out_idx] = False
    handles["num_views"][out_idx] = int(view_mask_out.sum())
    if "frame_idx" in src["auxiliary"]:
        handles["frame_idx"][out_idx] = int(src["auxiliary/frame_idx"][src_idx])
    else:
        handles["frame_idx"][out_idx] = src_idx
    if "session_name" in src["auxiliary"]:
        sn = src["auxiliary/session_name"][src_idx]
        if isinstance(sn, bytes):
            sn = sn.decode("utf-8")
        handles["session_name"][out_idx] = f"{info.path.stem}::{sn}"
    else:
        handles["session_name"][out_idx] = info.path.stem
    if "camera_names" in src["auxiliary"]:
        cn = src["auxiliary/camera_names"][src_idx]
        if isinstance(cn, bytes):
            cn = cn.decode("utf-8")
        handles["camera_names"][out_idx] = cn
    else:
        handles["camera_names"][out_idx] = ""

    handles["canonical_to_world_R"][out_idx] = c2w_R
    handles["canonical_to_world_t"][out_idx] = c2w_t
    handles["canonical_cam_id"][out_idx] = canonical_cam_id_out

    handles["origin_dataset"][out_idx] = info.dataset_type
    handles["origin_source_file"][out_idx] = str(info.path.name)
    handles["origin_sample_idx"][out_idx] = src_idx


# ---------------------------------------------------------------------------
# Top-level merge.
# ---------------------------------------------------------------------------


def _apply_match_scale(infos: List["InputInfo"], input_paths: List[Path],
                       reference: Optional[Path], verbose: bool) -> Optional[float]:
    """Fold a per-input scale-match factor into each input's `world_scale`
    so all sources land at one common physical scale (median subject extent).

    `world_scale` is applied uniformly to t / kp3d / trans / canonical_to_world_t
    at copy time, and uniform world scaling is projection-invariant, so this
    only changes the magnitude of the 3D-space targets — never the images,
    2D keypoints, intrinsics, or FOV. The reference defines the target extent:
    if None, the first input is used (so the reference source is left at its
    inferred world_scale and the others are co-scaled to it).

    Returns the target extent (reader units), or None if it could not be
    measured (e.g. no 3D keypoints anywhere)."""
    # Reader-unit subject extent of each input at its current world_scale.
    extents = []
    for info, path in zip(infos, input_paths):
        with h5py.File(path, "r") as hf:
            extents.append(_median_subject_extent(hf, info.world_scale))

    if reference is not None:
        ref_resolved = Path(reference).resolve()
        match = [i for i, p in enumerate(input_paths) if p.resolve() == ref_resolved]
        if match:
            target = extents[match[0]]
        else:
            with h5py.File(reference, "r") as hf:
                target = _median_subject_extent(hf, _infer_per_input_world_scale(hf))
    else:
        target = extents[0]

    if not np.isfinite(target) or target <= 0:
        if verbose:
            print("match_scale: could not measure a reference subject extent; skipping.")
        return None

    for info, ext in zip(infos, extents):
        if np.isfinite(ext) and ext > 0:
            factor = target / ext
            info.world_scale = float(info.world_scale * factor)
            if verbose:
                print(f"  match_scale: {info.path.name} extent {ext:.4f} -> {target:.4f} "
                      f"(x{factor:.4f}); combined world_scale={info.world_scale:g}")
        elif verbose:
            print(f"  match_scale: {info.path.name} has no 3D keypoints; world_scale unchanged.")
    return float(target)


def merge(
    input_paths: List[Path],
    output_path: Path,
    jpeg_resolution: int,
    jpeg_quality: int = 95,
    chunk_size: int = 8,
    compression: str = "gzip",
    compression_level: int = 6,
    backbone_name: str = "vit_large_patch16_224",
    min_views_per_sample: int = 2,
    max_samples_per_input: Optional[int] = None,
    match_scale: bool = False,
    match_scale_reference: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, int]:
    infos = validate_inputs(input_paths)

    # Optionally cap each input for smoke testing. The infos' num_samples
    # is used both for allocation and for the per-input write loop, so a
    # single cap covers both.
    if max_samples_per_input is not None:
        for info in infos:
            info.num_samples = min(info.num_samples, int(max_samples_per_input))

    # Co-scale heterogeneous sources to one physical scale (subject extent)
    # before the copy loop, so the merged trans / mesh-scale / 3D targets do
    # not span source-dependent magnitudes. Folded into per-input world_scale.
    match_target_extent = None
    if match_scale or match_scale_reference is not None:
        if verbose:
            print("Co-scaling inputs to a common physical scale:")
        match_target_extent = _apply_match_scale(
            infos, input_paths, match_scale_reference, verbose)
        if verbose:
            print()

    n_joints = infos[0].n_joints
    n_pose = infos[0].n_pose
    n_betas = infos[0].n_betas
    merged_max_views = max(i.max_views for i in infos)
    merged_num_samples = sum(i.num_samples for i in infos)

    if verbose:
        print(f"Merging {len(infos)} inputs:")
        for i in infos:
            print(f"  - {i.path.name}: dtype={i.dataset_type}, samples={i.num_samples}, "
                  f"max_views={i.max_views}, world_scale={i.world_scale:g}, "
                  f"jpeg={i.decoded_jpeg_hw[1]}x{i.decoded_jpeg_hw[0]}, "
                  f"needs_canon={i.needs_canonicalization}")
        print(f"Output: {merged_num_samples} samples, max_views={merged_max_views}, "
              f"jpeg_resolution={jpeg_resolution}")
        print()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as out_hf:
        handles = _create_output_layout(
            out_hf, merged_num_samples, merged_max_views,
            n_joints, n_pose, n_betas, jpeg_resolution,
            chunk_size, compression, compression_level,
        )

        out_idx = 0
        per_input_written: Dict[str, int] = {}
        for info in infos:
            with h5py.File(info.path, "r") as src:
                pbar = tqdm(range(info.num_samples), desc=info.path.name, disable=not verbose)
                for src_idx in pbar:
                    _copy_one_sample(
                        src=src, src_idx=src_idx, info=info, handles=handles,
                        out_idx=out_idx, merged_max_views=merged_max_views,
                        jpeg_resolution=jpeg_resolution, jpeg_quality=jpeg_quality,
                    )
                    out_idx += 1
                per_input_written[str(info.path)] = info.num_samples

        # Metadata attrs. Schema mirrors the SLEAPMultiViewDataset reader's
        # expectations. The new value `dataset_type='merged_multiview'` is
        # not branched on by the reader.
        md = handles["metadata"]
        md.attrs["num_samples"] = int(out_idx)
        md.attrs["max_views"] = int(merged_max_views)
        md.attrs["n_joints"] = int(n_joints)
        md.attrs["n_pose"] = int(n_pose)
        md.attrs["n_betas"] = int(n_betas)
        md.attrs["target_resolution"] = int(jpeg_resolution)
        md.attrs["backbone_name"] = backbone_name
        md.attrs["jpeg_quality"] = int(jpeg_quality)
        md.attrs["dataset_type"] = "merged_multiview"
        md.attrs["is_multiview"] = True
        md.attrs["has_camera_parameters"] = True
        md.attrs["has_3d_keypoints"] = True
        md.attrs["load_3d_data"] = True
        md.attrs["world_scale"] = 1.0
        md.attrs["min_views_per_sample"] = int(min_views_per_sample)
        md.attrs["camera_extrinsics_convention"] = "opencv"
        if match_target_extent is not None:
            md.attrs["match_scale_target_extent"] = float(match_target_extent)
            md.attrs["match_scale_reference"] = (
                str(Path(match_scale_reference).resolve())
                if match_scale_reference is not None else str(input_paths[0].resolve()))
        # Per-slot canonical-camera-order is meaningless cross-source; the reader
        # has a default fallback so this is mostly cosmetic.
        md.attrs["canonical_camera_order"] = json.dumps(
            [f"slot_{v}" for v in range(merged_max_views)]
        )
        # Provenance manifest.
        md.attrs["source_files"] = json.dumps([
            {"path": str(i.path), "dataset_type": i.dataset_type,
             "num_samples": i.num_samples, "max_views": i.max_views,
             "world_scale": i.world_scale,
             "jpeg_resolution_input": [i.decoded_jpeg_hw[1], i.decoded_jpeg_hw[0]]}
            for i in infos
        ])

    if verbose:
        print(f"\nDone. Wrote {out_idx} samples to {output_path}")
    return per_input_written


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Input multi-view HDF5 paths. Order is preserved in the output sample axis.")
    p.add_argument("--output", required=True, help="Output merged HDF5 path.")
    p.add_argument("--jpeg_resolution", type=int, required=True,
                   help="Target JPEG resolution (square). All inputs are decoded, "
                        "resized to this resolution, and re-encoded; K is rescaled to match.")
    p.add_argument("--jpeg_quality", type=int, default=95)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--compression", type=str, default="gzip")
    p.add_argument("--compression_level", type=int, default=6)
    p.add_argument("--backbone_name", type=str, default="vit_large_patch16_224")
    p.add_argument("--min_views_per_sample", type=int, default=2)
    p.add_argument("--max_samples_per_input", type=int, default=None,
                   help="Cap each input to its first N samples (smoke testing only).")
    p.add_argument("--match_scale", action="store_true",
                   help="Co-scale all inputs to one physical scale (median subject "
                        "extent) before merging, so trans/mesh-scale/3D targets do not "
                        "span source-dependent magnitudes. Reference is the first input "
                        "unless --match_scale_reference is given. Projection-invariant.")
    p.add_argument("--match_scale_reference", type=str, default=None,
                   help="HDF5 whose subject scale defines the common target (e.g. the "
                        "real dataset). Implies --match_scale. May be one of --inputs or "
                        "an external file.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    input_paths = [Path(s).resolve() for s in args.inputs]
    output_path = Path(args.output).resolve()

    merge(
        input_paths=input_paths,
        output_path=output_path,
        jpeg_resolution=int(args.jpeg_resolution),
        jpeg_quality=int(args.jpeg_quality),
        chunk_size=int(args.chunk_size),
        compression=str(args.compression),
        compression_level=int(args.compression_level),
        backbone_name=str(args.backbone_name),
        min_views_per_sample=int(args.min_views_per_sample),
        max_samples_per_input=args.max_samples_per_input,
        match_scale=bool(args.match_scale or args.match_scale_reference is not None),
        match_scale_reference=Path(args.match_scale_reference).resolve()
            if args.match_scale_reference else None,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
