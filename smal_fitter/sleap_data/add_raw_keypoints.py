#!/usr/bin/env python3
"""
Add raw SLEAP 2D labels to an existing multi-view SLEAP HDF5.

Datasets built with ``--use_reprojections`` store reprojections of the
triangulated 3D as ``keypoints_2d``. This tool adds the SLEAP predictions
themselves, filtered by confidence only, for 2D-only training on labels a user
would actually have:

    multiview_keypoints/keypoints_2d_raw         (N, V, J, 2)  normalised [y, x], same crop as the images
    multiview_keypoints/keypoint_visibility_raw  (N, V, J)     confidence >= --confidence_threshold
    multiview_keypoints/keypoint_confidence_raw  (N, V, J)     SLEAP score, 0 = no detection / unmapped

New datasets built by preprocess_sleap_multiview_dataset.py already contain
them; both go through SLEAPMultiViewPreprocessor.raw_view_keypoints. No
existing dataset is modified. Two steps, so the labels can be checked before
the large file is touched:

    # 1. Build a sidecar HDF5 from the SLEAP sessions the dataset was made from
    python smal_fitter/sleap_data/add_raw_keypoints.py build DATASET.h5 SESSIONS_DIR \\
        --joint_lookup_table LOOKUP.csv --smal_file MODEL.pkl [--confidence_threshold 0.5]

    # 2. Append the sidecar to the dataset
    python smal_fitter/sleap_data/add_raw_keypoints.py append DATASET.h5 DATASET_raw_keypoints.h5

``build`` needs the same lookup table and model as the original preprocessing,
and checks that the crop rebuilt from each camera's calibration reproduces the
stored camera_intrinsics before extracting anything. Only the size-only crop
modes ('centred', 'default') are supported. ``append`` checks that the sidecar
rows match the dataset (session, frame, cameras, view mask) and reads the
written data back.
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from sleap_data_loader import SLEAPDataLoader  # noqa: E402
from smal_fitter.neuralSMIL.configs.config_utils import apply_smal_file_override  # noqa: E402
from smal_fitter.sleap_data.preprocess_sleap_multiview_dataset import (  # noqa: E402
    RAW_KEYPOINT_DATASETS,
    SLEAPMultiViewPreprocessor,
    set_raw_keypoint_attrs,
)
from smal_fitter.sleap_data.sleap_3d_loader import SLEAP3DDataLoader  # noqa: E402

# Row-alignment datasets copied into the sidecar and compared on append.
_ALIGNMENT = ("auxiliary/session_name", "auxiliary/frame_idx", "auxiliary/camera_names", "multiview_images/view_mask")
_K_TOLERANCE = 1e-3  # pixels; the stored K is float32
_APPEND_BLOCK_ROWS = 8192


def _decode(values) -> list:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def _git_commit() -> str:
    """HEAD, with "-dirty" if tracked files differ in content (line-ending-only changes don't count)."""
    try:
        repo = Path(__file__).resolve().parents[2]
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        clean = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "--ignore-cr-at-eol", "HEAD"]).returncode
        return head if clean == 0 else f"{head}-dirty"
    except Exception:
        return "unknown"


def build_sidecar(
    hdf5_path: str,
    sessions_dir: str,
    sidecar_path: str,
    joint_lookup_table: Optional[str],
    smal_file: str,
    confidence_threshold: float = 0.5,
) -> Dict[str, int]:
    """Extract raw labels for every row and view of ``hdf5_path`` into ``sidecar_path``."""
    git_commit = _git_commit()  # before the long extraction, so later edits cannot taint it
    apply_smal_file_override(smal_file)
    with h5py.File(hdf5_path, "r") as f:
        md = f["metadata"].attrs
        crop_mode = str(md["crop_mode"])
        target_resolution = int(md["target_resolution"])
        undistort = bool(md.get("undistort_images", False))
        n_joints = int(md["n_joints"])
        sessions = np.asarray(_decode(f["auxiliary/session_name"][:]), dtype=object)
        frames = f["auxiliary/frame_idx"][:].astype(np.int64)
        camera_names = [s.split(",") for s in _decode(f["auxiliary/camera_names"][:])]
        view_mask = f["multiview_images/view_mask"][:]
        K_stored = f["multiview_keypoints/camera_intrinsics"][:]
    if len(config.joint_names) != n_joints:
        raise ValueError(f"{smal_file} has {len(config.joint_names)} joints but {hdf5_path} has {n_joints}")
    for r, names in enumerate(camera_names):
        if len(names) != int(view_mask[r].sum()) or not view_mask[r, : len(names)].all():
            raise ValueError(f"Row {r}: camera_names {names} do not match view_mask {view_mask[r].tolist()}")

    pre = SLEAPMultiViewPreprocessor(
        target_resolution=target_resolution,
        crop_mode=crop_mode,
        confidence_threshold=confidence_threshold,
        undistort_images=undistort,
    )
    n_rows, max_views = view_mask.shape
    out = {
        "keypoints_2d_raw": np.zeros((n_rows, max_views, n_joints, 2), dtype=np.float32),
        "keypoint_visibility_raw": np.zeros((n_rows, max_views, n_joints), dtype=np.float32),
        "keypoint_confidence_raw": np.zeros((n_rows, max_views, n_joints), dtype=np.float32),
    }
    stats = {"views": 0, "views_without_detection": 0, "visible_keypoints": 0}

    for session in dict.fromkeys(sessions.tolist()):
        rows = np.flatnonzero(sessions == session)
        session_path = Path(sessions_dir) / session
        print(f"{session}: {len(rows)} rows")
        # The loaders log every file they open and every frame without an
        # instance; keep that out of the console.
        with contextlib.redirect_stdout(io.StringIO()):
            loader = SLEAPDataLoader(
                str(session_path), lookup_table_path=joint_lookup_table, confidence_threshold=confidence_threshold
            )
            loader_3d = SLEAP3DDataLoader(str(session_path), session_idx=0)
        if loader.smal_n_joints != n_joints:
            raise ValueError(f"Loader maps to {loader.smal_n_joints} joints, dataset has {n_joints}")

        cameras = {}
        for cam in sorted({c for r in rows for c in camera_names[r]}):
            width, height = loader.get_camera_image_size(cam)
            transform_info = pre._size_transform_info(height, width)
            intrinsic = loader_3d.get_camera_parameters(cam)["intrinsic"]
            K_adj = pre._adjust_intrinsics_for_transform(intrinsic["K"].copy().astype(np.float32), transform_info)
            slots = [(r, camera_names[r].index(cam)) for r in rows if cam in camera_names[r]]
            err = max(float(np.abs(K_stored[r, v] - K_adj).max()) for r, v in slots)
            if err > _K_TOLERANCE:
                raise ValueError(
                    f"{session}/{cam}: crop rebuilt from the calibration ({width}x{height}) does not reproduce the "
                    f"stored camera_intrinsics (max |dK| = {err:.4g}); wrong sessions_dir or calibration?"
                )
            with contextlib.redirect_stdout(io.StringIO()):
                camera_data = loader.load_camera_data(cam)
            K = intrinsic["K"] if undistort else None
            cameras[cam] = (camera_data, transform_info, K, intrinsic.get("distortion"))
            print(f"  {cam}: {len(slots)} views, {width}x{height}, stored K reproduced (max |dK| = {err:.2g})")

        with contextlib.redirect_stdout(io.StringIO()):
            for r in tqdm(rows, desc=session, leave=False):
                for v, cam in enumerate(camera_names[r]):
                    camera_data, transform_info, K, distortion = cameras[cam]
                    kp, vis, conf = pre.raw_view_keypoints(
                        loader, camera_data, int(frames[r]), transform_info, K, distortion
                    )
                    out["keypoints_2d_raw"][r, v] = kp
                    out["keypoint_visibility_raw"][r, v] = vis
                    out["keypoint_confidence_raw"][r, v] = conf
                    stats["views"] += 1
                    stats["views_without_detection"] += int(not conf.any())
                    stats["visible_keypoints"] += int(vis.sum())

    provenance = {
        "source_hdf5": os.path.abspath(hdf5_path),
        "sessions_dir": os.path.abspath(sessions_dir),
        "joint_lookup_table": os.path.abspath(joint_lookup_table) if joint_lookup_table else None,
        "smal_file": smal_file,
        "confidence_threshold": float(confidence_threshold),
        "undistorted": undistort,
        "crop_mode": crop_mode,
        "git_commit": git_commit,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **stats,
    }
    with h5py.File(sidecar_path, "w") as s, h5py.File(hdf5_path, "r") as f:
        for name, data in out.items():
            s.create_dataset(f"multiview_keypoints/{name}", data=data, compression="gzip", compression_opts=4)
        for name in _ALIGNMENT:
            s.create_dataset(name, data=f[name][:], compression="gzip", compression_opts=4)
        s.attrs["provenance"] = json.dumps(provenance)
        s.attrs["confidence_threshold"] = float(confidence_threshold)
    print(f"Sidecar written: {sidecar_path}")
    print(json.dumps(provenance, indent=2))
    return stats


def _check_alignment(f: h5py.File, s: h5py.File) -> None:
    for name in _ALIGNMENT:
        a, b = f[name][:], s[name][:]
        if a.dtype.kind == "O" or b.dtype.kind == "O":
            a, b = np.asarray(_decode(a)), np.asarray(_decode(b))
        if a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError(f"Sidecar does not match the dataset in {name}; was it built from this file?")


def append_sidecar(hdf5_path: str, sidecar_path: str, overwrite: bool = False) -> None:
    """Copy the raw-label datasets of ``sidecar_path`` into ``hdf5_path`` and read them back."""
    with h5py.File(hdf5_path, "r") as f, h5py.File(sidecar_path, "r") as s:
        _check_alignment(f, s)
        present = [n for n in RAW_KEYPOINT_DATASETS if n in f["multiview_keypoints"]]
        if present and not overwrite:
            raise ValueError(f"{hdf5_path} already has {present}; pass --overwrite to replace them")

    with h5py.File(hdf5_path, "a") as f, h5py.File(sidecar_path, "r") as s:
        group = f["multiview_keypoints"]
        for name in RAW_KEYPOINT_DATASETS:
            if name in group:
                del group[name]  # HDF5 does not reclaim the space; h5repack does
            src = s[f"multiview_keypoints/{name}"]
            # Same chunking/compression as the reprojected label it sits beside,
            # so training reads it at the same speed.
            like = group["keypoints_2d" if name == "keypoints_2d_raw" else "keypoint_visibility"]
            dst = group.create_dataset(
                name,
                shape=src.shape,
                dtype=np.float32,
                chunks=like.chunks,
                compression=like.compression,
                compression_opts=like.compression_opts,
            )
            for start in tqdm(range(0, src.shape[0], _APPEND_BLOCK_ROWS), desc=name, leave=False):
                dst[start : start + _APPEND_BLOCK_ROWS] = src[start : start + _APPEND_BLOCK_ROWS]
        set_raw_keypoint_attrs(f["metadata"], float(s.attrs["confidence_threshold"]))
        f["metadata"].attrs["raw_keypoints_provenance"] = s.attrs["provenance"]

    with h5py.File(hdf5_path, "r") as f, h5py.File(sidecar_path, "r") as s:
        for name in RAW_KEYPOINT_DATASETS:
            a, b = f[f"multiview_keypoints/{name}"], s[f"multiview_keypoints/{name}"]
            for start in range(0, a.shape[0], _APPEND_BLOCK_ROWS):
                if not np.array_equal(a[start : start + _APPEND_BLOCK_ROWS], b[start : start + _APPEND_BLOCK_ROWS]):
                    raise RuntimeError(f"Read-back of {name} differs from the sidecar at rows {start}+")
    print(f"Appended {', '.join(RAW_KEYPOINT_DATASETS)} to {hdf5_path} (read-back verified)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Extract raw labels from the SLEAP sessions into a sidecar HDF5")
    b.add_argument("hdf5", help="Multi-view HDF5 made by preprocess_sleap_multiview_dataset.py")
    b.add_argument("sessions_dir", help="Directory holding the SLEAP sessions the HDF5 was made from")
    b.add_argument("--joint_lookup_table", required=True, help="Same lookup table as the original preprocessing")
    b.add_argument("--smal_file", required=True, help="Model pkl the HDF5 was made for (sets the joint order)")
    b.add_argument("--confidence_threshold", type=float, default=0.5, help="For keypoint_visibility_raw")
    b.add_argument("--sidecar", default=None, help="Output path (default: <hdf5 stem>_raw_keypoints.h5)")

    a = sub.add_parser("append", help="Append a sidecar's raw labels to the HDF5")
    a.add_argument("hdf5")
    a.add_argument("sidecar")
    a.add_argument("--overwrite", action="store_true", help="Replace raw-label datasets already in the HDF5")

    args = parser.parse_args()
    if args.command == "build":
        if not 0.0 <= args.confidence_threshold <= 1.0:
            parser.error("--confidence_threshold must be in [0, 1]")
        sidecar = args.sidecar or str(Path(args.hdf5).with_name(Path(args.hdf5).stem + "_raw_keypoints.h5"))
        build_sidecar(
            args.hdf5, args.sessions_dir, sidecar, args.joint_lookup_table, args.smal_file, args.confidence_threshold
        )
    else:
        append_sidecar(args.hdf5, args.sidecar, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
