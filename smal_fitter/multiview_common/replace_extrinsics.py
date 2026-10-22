#!/usr/bin/env python3
"""
Replace the camera extrinsics of a multi-view SLEAP HDF5 and re-project its 2D labels.

For datasets built with ``--use_reprojections`` whose 3D points were triangulated
with different extrinsics than the calibration.toml the preprocessor read (for
example after refine_camera_params.py). The stored keypoints_3d are kept; the
stored camera_extrinsics_R / camera_extrinsics_t are replaced with those of
``--new_calibration`` and keypoints_2d / keypoint_visibility are recomputed as
the projection of keypoints_3d, with the preprocessor's reprojection rule
(visible = triangulated and inside the original image).

Modifies the HDF5 in place, so run it on a copy:

    python -m smal_fitter.multiview_common.replace_extrinsics DATASET_COPY.h5 SESSIONS_DIR \\
        --new_calibration calibration_refined_extrinsics.toml

Checks before writing: for every camera the stored R/t must equal
``--base_calibration`` and the intrinsics must be identical in both files (the
stored images were undistorted with them), and re-projecting with the base
extrinsics must reproduce the stored keypoints_2d / keypoint_visibility of every
joint that has 3D. Visible labels without stored 3D (reprojections.h5 can hold
them) cannot be re-projected; they become invisible and are counted.
Only the size-only crop modes ('centred', 'default') are supported.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smal_fitter.sleap_data.preprocess_sleap_multiview_dataset import SLEAPMultiViewPreprocessor  # noqa: E402

_ROUND_TRIP_TOL = 1e-5  # normalised units (~0.005 px at 512)


def _calibration(path: Path) -> dict:
    return {v["name"]: v for v in toml.load(path).values() if isinstance(v, dict) and "name" in v}


def _decode(values) -> list:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def _reproject(X, K, R, t, offset, scale, mask, res):
    """Preprocessor reprojection rule: kp = projection of the 3D in [y, x] / res, 0 where there is no
    3D; visible where the projection is finite and inside the original (pre-crop) image."""
    Xc = np.einsum("nvij,nkj->nvki", R, X) + t[:, :, None, :]
    proj = np.einsum("nvij,nvkj->nvki", K, Xc)
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = proj[..., :2] / proj[..., 2:3]  # 512-px (x, y)
    has3d = ~(X == 0).all(-1)[:, None, :]  # SMAL joints without 3D are stored as zeros
    finite = np.isfinite(xy).all(-1)
    original = xy / scale[:, :, None, None] + offset[:, :, None, :]  # (x, y) in the original image
    vis = mask[:, :, None] & has3d & finite & (original > 0).all(-1)
    kp = np.where((has3d & finite)[..., None], xy[..., ::-1] / res, 0.0)
    return kp.astype(np.float32), vis.astype(np.float32)


def replace_extrinsics(
    hdf5_path: str, sessions_dir: str, new_calibration: str, base_calibration: str = "calibration.toml",
    dry_run: bool = False,
) -> dict:  # fmt: skip
    with h5py.File(hdf5_path, "r") as f:
        md = f["metadata"].attrs
        res = int(md["target_resolution"])
        crop_mode = str(md["crop_mode"])
        sessions = np.asarray(_decode(f["auxiliary/session_name"][:]), dtype=object)
        cams = [s.split(",") for s in _decode(f["auxiliary/camera_names"][:])]
        mask = f["multiview_images/view_mask"][:]
        K = f["multiview_keypoints/camera_intrinsics"][:].astype(np.float64)
        R = f["multiview_keypoints/camera_extrinsics_R"][:].astype(np.float64)
        t = f["multiview_keypoints/camera_extrinsics_t"][:].astype(np.float64)
        X = f["multiview_keypoints/keypoints_3d"][:].astype(np.float64)
        kp_stored = f["multiview_keypoints/keypoints_2d"][:]
        vis_stored = f["multiview_keypoints/keypoint_visibility"][:]
    n, v = mask.shape
    cam_idx = np.full((n, v), "", dtype=object)
    for r, names in enumerate(cams):
        cam_idx[r, : len(names)] = names

    pre = SLEAPMultiViewPreprocessor(target_resolution=res, crop_mode=crop_mode)
    R_new, t_new = R.copy(), t.copy()
    offset = np.zeros((n, v, 2))  # (x, y) crop offset in original pixels
    scale = np.ones((n, v))
    for session in dict.fromkeys(sessions.tolist()):
        base = _calibration(Path(sessions_dir) / session / base_calibration)
        new = _calibration(Path(sessions_dir) / session / new_calibration)
        for cam in sorted({c for r in np.flatnonzero(sessions == session) for c in cams[r]}):
            sel = (sessions[:, None] == session) & (cam_idx == cam)
            b, nw = base[cam], new[cam]
            if not (np.allclose(b["matrix"], nw["matrix"]) and np.allclose(b["distortions"], nw["distortions"])):
                raise ValueError(f"{session}/{cam}: intrinsics differ between calibrations; the stored images "
                                 "were undistorted with the base intrinsics, so regenerate the dataset instead")  # fmt: skip
            R0 = cv2.Rodrigues(np.array(b["rotation"], float))[0]
            t0 = np.array(b["translation"], float)
            if np.abs(R[sel] - R0).max() > 1e-5 or np.abs(t[sel] - t0).max() > 1e-3:
                raise ValueError(f"{session}/{cam}: stored extrinsics are not those of {base_calibration}")
            R_new[sel] = cv2.Rodrigues(np.array(nw["rotation"], float))[0]
            t_new[sel] = np.array(nw["translation"], float)
            width, height = b["size"]
            info = pre._size_transform_info(int(height), int(width))
            offset[sel] = info["crop_offset"][::-1]
            scale[sel] = info["scale_factor"]

    kp_old, vis_old = _reproject(X, K, R, t, offset, scale, mask, res)
    valid = mask[:, :, None] & np.ones_like(vis_stored, dtype=bool)
    # Labels whose joint has no stored 3D (all zeros) cannot be re-projected. reprojections.h5 can still
    # hold a visible label there; those become invisible in the new labels and are counted.
    has3d = valid & ~(X == 0).all(-1)[:, None, :]
    without_3d = int((valid & ~has3d & (vis_stored > 0)).sum())
    kp_err = float(np.abs(kp_old - kp_stored)[has3d].max())
    vis_err = int((vis_old != vis_stored)[has3d].sum())
    if kp_err > _ROUND_TRIP_TOL or vis_err:
        raise ValueError(f"Re-projecting with {base_calibration} does not reproduce the stored labels "
                         f"(max |dkp| {kp_err:.3g}, {vis_err} visibility mismatches); not touching the file")  # fmt: skip

    kp_new, vis_new = _reproject(X, K, R_new, t_new, offset, scale, mask, res)
    both = (vis_old > 0) & (vis_new > 0)
    shift = np.linalg.norm((kp_new - kp_old) * res, axis=-1)[both]
    report = {
        "base_calibration": base_calibration,
        "new_calibration": new_calibration,
        "round_trip_max_dkp": kp_err,
        "visible_labels_without_3d_made_invisible": without_3d,
        "visibility_changed": int((vis_new != vis_stored)[valid].sum()),
        "shift_px_512_median": float(np.median(shift)),
        "shift_px_512_p99": float(np.percentile(shift, 99)),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(json.dumps(report, indent=2))
    if dry_run:
        print("Dry run: nothing written")
        return report

    with h5py.File(hdf5_path, "a") as f:
        g = f["multiview_keypoints"]
        g["camera_extrinsics_R"][...] = R_new.astype(np.float32)
        g["camera_extrinsics_t"][...] = t_new.astype(np.float32)
        g["keypoints_2d"][...] = kp_new
        g["keypoint_visibility"][...] = vis_new
        f["metadata"].attrs["extrinsics_calibration"] = new_calibration
        f["metadata"].attrs["extrinsics_replacement"] = json.dumps(report)
    with h5py.File(hdf5_path, "r") as f:
        g = f["multiview_keypoints"]
        ok = (
            np.array_equal(g["keypoints_2d"][:], kp_new)
            and np.array_equal(g["keypoint_visibility"][:], vis_new)
            and np.array_equal(g["camera_extrinsics_R"][:], R_new.astype(np.float32))
            and np.array_equal(g["camera_extrinsics_t"][:], t_new.astype(np.float32))
        )
    if not ok:
        raise RuntimeError("Read-back after writing differs")
    print(f"Replaced extrinsics and re-projected keypoints_2d in {hdf5_path} (read-back verified)")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hdf5", help="Multi-view HDF5 to modify IN PLACE (use a copy)")
    parser.add_argument("sessions_dir", help="Directory holding the SLEAP sessions (with the calibration files)")
    parser.add_argument("--new_calibration", required=True, help="Calibration file name in each session directory")
    parser.add_argument("--base_calibration", default="calibration.toml", help="Calibration the HDF5 was built with")
    parser.add_argument("--dry_run", action="store_true", help="Run the checks and report, write nothing")
    args = parser.parse_args()
    replace_extrinsics(args.hdf5, args.sessions_dir, args.new_calibration, args.base_calibration, args.dry_run)


if __name__ == "__main__":
    main()
