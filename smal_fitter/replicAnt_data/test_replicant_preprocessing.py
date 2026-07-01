#!/usr/bin/env python3
"""Round-trip verification for the multi-view replicAnt preprocessor.

For each requested sample in a preprocessed HDF5, compares the values the
production reader (`SLEAPMultiViewDataset`) exposes against a fresh call to
`load_SMIL_Unreal_multiview_sample` for the same source frame. This pins down
the conversion contract that the preprocessor relies on (OpenCV-form extrinsics
stored, PyTorch3D-form recovered at dataset-load via
`_sleap_to_pytorch3d_camera`).

Tested invariants (per sample, against tol=1e-5 unless noted):

- `y_data['cam_rot_per_view']`, `cam_trans_per_view`, `cam_fov_per_view`
  reconstructed by the dataset class match the loader's per-view PyTorch3D
  cameras.
- `keypoints_2d`, `keypoint_visibility` per slot match the loader.
- `keypoints_3d` in canonical frame matches the loader.
- Applying the stored `canonical_to_world` inverse recovers the loader's
  `keypoints_3d_world` (raw PyTorch3D-mirrored world frame).
- `parameters/trans`, `global_rot`, `betas` match the loader's
  `root_loc`, `root_rot`, `shape_betas`.

Run manually (no live CI — needs the source dataset on disk):

    python smal_fitter/replicAnt_data/test_replicant_preprocessing.py \\
        --hdf5 SMOKE_replicant_500.h5 \\
        --dataset_dir /mnt/c/replicAnt-dataset-multi-cam-mice \\
        --smal_file 3D_model_prep/SMILy_Mouse_static_joints_Falkner_conv_repose_hind_legs.pkl \\
        --sample_indices 0 100 250 499
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import h5py
import numpy as np


_REPO = Path(__file__).resolve().parents[2]


def _check_sample(
    hdf5_path: str,
    dataset_dir: str,
    sample_idx: int,
    tol: float = 1e-5,
    fov_tol_deg: float = 1e-3,
) -> bool:
    from smal_fitter.Unreal2Pytorch3D import load_SMIL_Unreal_multiview_sample
    from smal_fitter.sleap_data.sleap_multiview_dataset import SLEAPMultiViewDataset

    ds = SLEAPMultiViewDataset(hdf5_path, augment=False)
    if sample_idx >= len(ds):
        print(f"  sample_idx {sample_idx} >= dataset size {len(ds)}; skip.")
        return False

    x_data, y_data = ds[sample_idx]
    frame_idx = int(x_data["frame_idx"])
    print(f"  sample {sample_idx} = source frame {frame_idx}")

    x_mv, y_mv = load_SMIL_Unreal_multiview_sample(
        data_path=dataset_dir,
        frame_index=frame_idx,
        camera_indices=None,
        load_images=False,
        canonical_frame=True,
    )

    # The dataset drops view_mask=False slots, so its view count equals the
    # number of valid loader slots. The loader still emits ALL camera slots
    # (with `view_valid_per_view[i]=False` for the bad ones); pair the
    # dataset's views with the loader's *valid* slots in canonical order.
    n_views = int(x_data["num_active_views"])
    view_valid = list(y_mv.get("view_valid_per_view", [True] * int(x_mv["num_views"])))
    valid_loader_idx = [i for i, ok in enumerate(view_valid) if ok]
    n_valid_loader = len(valid_loader_idx)
    if n_views != n_valid_loader:
        print(
            f"  FAIL: view-count mismatch — HDF5 active={n_views}, "
            f"loader valid={n_valid_loader}/{x_mv['num_views']} "
            f"(view_valid={view_valid})"
        )
        return False
    if n_views < int(x_mv["num_views"]):
        print(f"  NOTE: {int(x_mv['num_views']) - n_views} view(s) masked out (view_valid={view_valid})")

    max_R = 0.0
    max_T = 0.0
    max_fov = 0.0
    for v_ds, v_loader in enumerate(valid_loader_idx):
        R_loader = y_mv["cam_rot_per_view"][v_loader].detach().cpu().numpy().astype(np.float32)
        T_loader = y_mv["cam_trans_per_view"][v_loader].detach().cpu().numpy().astype(np.float32)
        R_ds = y_data["cam_rot_per_view"][v_ds]
        T_ds = y_data["cam_trans_per_view"][v_ds]
        max_R = max(max_R, float(np.max(np.abs(R_loader - R_ds))))
        max_T = max(max_T, float(np.max(np.abs(T_loader - T_ds))))
        max_fov = max(
            max_fov,
            abs(float(y_mv["fov_per_view"][v_loader]) - float(y_data["cam_fov_per_view"][v_ds, 0])),
        )

    max_kp = max_vis = 0.0
    for v_ds, v_loader in enumerate(valid_loader_idx):
        max_kp = max(
            max_kp,
            float(
                np.max(
                    np.abs(
                        y_mv["keypoints_2d_per_view"][v_loader].astype(np.float32)
                        - y_data["keypoints_2d"][v_ds].astype(np.float32)
                    )
                )
            ),
        )
        max_vis = max(
            max_vis,
            float(
                np.max(
                    np.abs(
                        y_mv["keypoint_visibility_per_view"][v_loader].astype(np.float32)
                        - y_data["keypoint_visibility"][v_ds].astype(np.float32)
                    )
                )
            ),
        )

    kp3d_loader = np.asarray(y_mv["keypoints_3d"], dtype=np.float32)
    kp3d_ds = y_data["keypoints_3d"].astype(np.float32)
    max_kp3d = float(np.max(np.abs(kp3d_loader - kp3d_ds)))

    with h5py.File(hdf5_path, "r") as f:
        c2w_R = f["auxiliary/canonical_to_world_R"][sample_idx].astype(np.float32)
        c2w_t = f["auxiliary/canonical_to_world_t"][sample_idx].astype(np.float32)
        trans_h5 = f["parameters/trans"][sample_idx].astype(np.float32)
        rot_h5 = f["parameters/global_rot"][sample_idx].astype(np.float32)
        betas_h5 = f["parameters/betas"][sample_idx].astype(np.float32)
    # Canonical inverse round-trip only applies to joints with real GT.
    # Joints absent from the dataset land at the SLEAP-style (0,0,0) sentinel
    # in BOTH keypoints_3d and keypoints_3d_world — exclude them from the
    # geometric check (the canonical inverse of (0,0,0) is not (0,0,0)).
    loader_world = np.asarray(y_mv["keypoints_3d_world"], dtype=np.float32)
    has_gt_3d = ~(np.all(kp3d_ds == 0, axis=1) & np.all(loader_world == 0, axis=1))
    recovered_world = (kp3d_ds - c2w_t) @ c2w_R.T
    if has_gt_3d.any():
        max_world = float(np.max(np.abs(recovered_world[has_gt_3d] - loader_world[has_gt_3d])))
    else:
        max_world = 0.0

    max_trans = float(np.max(np.abs(np.asarray(y_mv["root_loc"], dtype=np.float32) - trans_h5)))
    max_rot = float(np.max(np.abs(np.asarray(y_mv["root_rot"], dtype=np.float32) - rot_h5)))
    max_betas = float(np.max(np.abs(np.asarray(y_mv["shape_betas"], dtype=np.float32) - betas_h5)))

    checks = {
        "R_per_view": (max_R, tol),
        "T_per_view": (max_T, tol),
        "fov_per_view": (max_fov, fov_tol_deg),
        "keypoints_2d": (max_kp, tol),
        "keypoint_vis": (max_vis, tol),
        "keypoints_3d": (max_kp3d, tol),
        "canonical_inverse": (max_world, tol),
        "parameters/trans": (max_trans, tol),
        "parameters/rot": (max_rot, tol),
        "parameters/betas": (max_betas, tol),
    }
    all_ok = True
    for name, (val, this_tol) in checks.items():
        ok = val < this_tol
        all_ok = all_ok and ok
        print(f"    {'PASS' if ok else 'FAIL'} {name:<22s}  max Δ = {val:.3e}  (tol {this_tol:.0e})")
    return all_ok


def main() -> None:
    p = argparse.ArgumentParser(description="Round-trip check for replicAnt multi-view HDF5.")
    p.add_argument("--hdf5", required=True, help="Path to the preprocessor's HDF5 output")
    p.add_argument("--dataset_dir", required=True, help="Source flat-directory replicAnt dataset")
    p.add_argument("--smal_file", required=True, help="SMAL/SMIL .pkl used at preprocess time")
    p.add_argument("--shape_family", type=int, default=None)
    p.add_argument(
        "--sample_indices", type=int, nargs="+", default=[0], help="HDF5 sample indices to verify (default: just 0)"
    )
    p.add_argument("--tol", type=float, default=1e-5)
    args = p.parse_args()

    if not os.path.isfile(args.hdf5):
        print(f"HDF5 not found: {args.hdf5}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(args.dataset_dir):
        print(f"Dataset directory not found: {args.dataset_dir}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(args.smal_file):
        print(f"SMAL pickle not found: {args.smal_file}", file=sys.stderr)
        sys.exit(2)

    # Apply override BEFORE any module touches config.dd.
    from smal_fitter.neuralSMIL.configs.config_utils import apply_smal_file_override  # noqa: E402

    apply_smal_file_override(args.smal_file, shape_family=args.shape_family)

    print(f"HDF5:         {args.hdf5}")
    print(f"Dataset dir:  {args.dataset_dir}")
    print(f"SMAL file:    {args.smal_file}")
    print(f"Tol:          {args.tol}")
    print()

    all_ok = True
    for idx in args.sample_indices:
        print(f"-- sample_idx {idx} --")
        all_ok &= _check_sample(args.hdf5, args.dataset_dir, idx, tol=args.tol)
        print()

    print("Overall:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
