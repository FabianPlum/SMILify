#!/usr/bin/env python3
"""
Iterative camera parameter refinement via alternating optimization.

Each iteration:
  1. Triangulate 3D points from 2D SLEAP using current camera params
  2. Optimize each camera's parameters to minimize reprojection error
     between the triangulated 3D and the undistorted 2D observations
  3. Evaluate improvement, check convergence

Per camera, we optimize:
    - Extrinsics: rotation (3 axis-angle) + translation (3)
    - Intrinsics: focal length (2) + principal point (2)
    Total: 10 parameters per camera

Usage:
    python refine_camera_params.py /path/to/session \
        --exclude_cameras Camera8 Camera10 Camera13 Camera15 \
        --iterations 5 --subsample_frames 5000
"""

import os
import sys
import argparse
import copy
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import cv2
import toml
from scipy.optimize import least_squares
from tqdm import tqdm

# Reuse functions from triangulation script
sys.path.insert(0, os.path.dirname(__file__))
from triangulate_3d_points import (
    load_calibration,
    load_all_2d_data,
    undistort_points,
    get_projection_matrix,
    triangulate_all,
    reprojection_analysis,
)


# ---------------------------------------------------------------------------
# Gather 3D-2D correspondences for a single camera
# ---------------------------------------------------------------------------

def gather_correspondences(
    kp_3d: np.ndarray,
    valid_3d: np.ndarray,
    coords_2d: np.ndarray,
    scores_2d: np.ndarray,
    cam: dict,
    confidence_threshold: float = 0.3,
    max_points: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gather matched 3D-2D correspondences for one camera.

    Returns:
        pts_3d: (M, 3) matched 3D points
        pts_2d: (M, 2) matched undistorted 2D points
    """
    n_frames = kp_3d.shape[0]
    n_frames_cam = coords_2d.shape[0]
    n_compare = min(n_frames, n_frames_cam)

    valid_2d_sub = coords_2d[:n_compare]
    scores_sub = scores_2d[:n_compare]
    valid_3d_sub = valid_3d[:n_compare]

    not_nan_2d = ~np.isnan(valid_2d_sub).any(axis=-1)
    not_zero_2d = (valid_2d_sub != 0).any(axis=-1)
    above_conf = np.isnan(scores_sub) | (scores_sub >= confidence_threshold)
    both_valid = valid_3d_sub & not_nan_2d & not_zero_2d & above_conf

    fi_idx, ki_idx = np.where(both_valid)
    if len(fi_idx) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2))

    pts_3d = kp_3d[fi_idx, ki_idx]       # (M, 3)
    pts_2d_raw = coords_2d[fi_idx, ki_idx]  # (M, 2)

    # Undistort using ORIGINAL distortion params (these are not optimized)
    pts_2d = undistort_points(pts_2d_raw, cam["K"], cam["dist"])

    # Subsample for efficiency
    if max_points is not None and len(pts_3d) > max_points:
        if rng is None:
            rng = np.random.default_rng(42)
        idx = rng.choice(len(pts_3d), max_points, replace=False)
        pts_3d = pts_3d[idx]
        pts_2d = pts_2d[idx]

    return pts_3d, pts_2d


# ---------------------------------------------------------------------------
# Camera parameter packing / unpacking
# ---------------------------------------------------------------------------

def pack_params(cam: dict, optimize_intrinsics: bool = True) -> np.ndarray:
    """Pack camera parameters into a flat vector."""
    rvec = cam["rvec"].ravel()  # (3,)
    t = cam["t"].ravel()        # (3,)

    if optimize_intrinsics:
        K = cam["K"]
        intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        return np.concatenate([rvec, t, intrinsics])  # 10 params
    else:
        return np.concatenate([rvec, t])  # 6 params


def unpack_params(params: np.ndarray, cam_template: dict,
                  optimize_intrinsics: bool = True) -> dict:
    """Unpack flat vector back into a camera dict."""
    cam = copy.deepcopy(cam_template)
    cam["rvec"] = params[:3]
    cam["t"] = params[3:6].reshape(3, 1)
    cam["R"], _ = cv2.Rodrigues(cam["rvec"])

    if optimize_intrinsics:
        fx, fy, cx, cy = params[6:10]
        cam["K"] = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ], dtype=np.float64)

    return cam


# ---------------------------------------------------------------------------
# Residual function for scipy.optimize.least_squares
# ---------------------------------------------------------------------------

def reprojection_residuals(
    params: np.ndarray,
    pts_3d: np.ndarray,
    pts_2d: np.ndarray,
    cam_template: dict,
    optimize_intrinsics: bool = True,
) -> np.ndarray:
    """
    Compute reprojection residuals (projected - observed).

    Returns flat array of shape (2*M,) — x and y residuals interleaved.
    """
    cam = unpack_params(params, cam_template, optimize_intrinsics)
    P = get_projection_matrix(cam)  # (3, 4)

    ones = np.ones((pts_3d.shape[0], 1), dtype=np.float64)
    pts_hom = np.hstack([pts_3d, ones])  # (M, 4)
    proj = (P @ pts_hom.T).T             # (M, 3)
    proj_2d = proj[:, :2] / proj[:, 2:3]  # (M, 2)

    return (proj_2d - pts_2d).ravel()  # (2*M,)


# ---------------------------------------------------------------------------
# Per-camera optimization
# ---------------------------------------------------------------------------

def optimize_camera(
    cam_name: str,
    cam: dict,
    pts_3d: np.ndarray,
    pts_2d: np.ndarray,
    optimize_intrinsics: bool = True,
    verbose: bool = True,
) -> Tuple[dict, dict]:
    """Optimize a single camera's parameters against 3D-2D correspondences."""
    n_pts = len(pts_3d)
    if n_pts < 20:
        if verbose:
            print(f"  {cam_name}: too few points ({n_pts}), skipping")
        return cam, {"status": "skipped", "n_points": n_pts}

    x0 = pack_params(cam, optimize_intrinsics)
    res0 = reprojection_residuals(x0, pts_3d, pts_2d, cam, optimize_intrinsics)
    err0 = np.sqrt(res0[::2]**2 + res0[1::2]**2)

    result = least_squares(
        reprojection_residuals,
        x0,
        args=(pts_3d, pts_2d, cam, optimize_intrinsics),
        method="trf",
        loss="soft_l1",
        f_scale=5.0,   # Huber transition at 5px
        max_nfev=500,
        verbose=0,
    )

    res_final = reprojection_residuals(
        result.x, pts_3d, pts_2d, cam, optimize_intrinsics
    )
    err_final = np.sqrt(res_final[::2]**2 + res_final[1::2]**2)

    refined_cam = unpack_params(result.x, cam, optimize_intrinsics)

    stats = {
        "status": "success" if result.success else "converged",
        "n_points": n_pts,
        "n_evaluations": result.nfev,
        "median_err_before": float(np.median(err0)),
        "median_err_after": float(np.median(err_final)),
        "pct_under_5px_before": float(100 * (err0 < 5).mean()),
        "pct_under_5px_after": float(100 * (err_final < 5).mean()),
        "pct_under_10px_before": float(100 * (err0 < 10).mean()),
        "pct_under_10px_after": float(100 * (err_final < 10).mean()),
    }

    if verbose:
        print(f"  {cam_name}: {n_pts:,d} pts | "
              f"median {stats['median_err_before']:.2f} -> {stats['median_err_after']:.2f} px | "
              f"<5px {stats['pct_under_5px_before']:.1f}% -> {stats['pct_under_5px_after']:.1f}% | "
              f"<10px {stats['pct_under_10px_before']:.1f}% -> {stats['pct_under_10px_after']:.1f}%")

    return refined_cam, stats


# ---------------------------------------------------------------------------
# Save refined calibration
# ---------------------------------------------------------------------------

def save_calibration_toml(cameras: Dict[str, dict], output_path: str,
                          original_cal_path: str):
    """Save refined camera parameters back to calibration.toml format."""
    original = toml.load(original_cal_path)

    for key, entry in original.items():
        if key == "metadata" or not isinstance(entry, dict):
            continue
        name = entry.get("name", key)
        if name in cameras:
            cam = cameras[name]
            entry["matrix"] = cam["K"].tolist()
            entry["rotation"] = cam["rvec"].ravel().tolist()
            entry["translation"] = cam["t"].ravel().tolist()
            # distortion is NOT optimized — kept from original

    with open(output_path, "w") as f:
        toml.dump(original, f)

    print(f"  Saved refined calibration to {output_path}")


# ---------------------------------------------------------------------------
# Compute quick reprojection stats (lighter than full reprojection_analysis)
# ---------------------------------------------------------------------------

def quick_reproj_stats(
    tracks_3d: np.ndarray,
    all_coords: Dict[str, np.ndarray],
    all_scores: Dict[str, np.ndarray],
    cameras: Dict[str, dict],
    confidence_threshold: float = 0.3,
    max_points_per_cam: int = 50000,
) -> dict:
    """
    Fast reprojection stats across all cameras (no per-joint breakdown).
    Returns dict with overall median, mean, <5px, <10px.
    """
    kp_3d = tracks_3d[:, 0]
    valid_3d = ~np.isnan(kp_3d).any(axis=-1) & (kp_3d != 0).any(axis=-1)
    rng = np.random.default_rng(42)

    all_errors = []
    for cam_name in sorted(cameras.keys()):
        if cam_name not in all_coords:
            continue
        cam = cameras[cam_name]
        pts_3d, pts_2d = gather_correspondences(
            kp_3d, valid_3d,
            all_coords[cam_name], all_scores[cam_name],
            cam, confidence_threshold,
            max_points=max_points_per_cam, rng=rng,
        )
        if len(pts_3d) == 0:
            continue

        P = get_projection_matrix(cam)
        ones = np.ones((pts_3d.shape[0], 1), dtype=np.float64)
        pts_hom = np.hstack([pts_3d, ones])
        proj = (P @ pts_hom.T).T
        proj_2d = proj[:, :2] / proj[:, 2:3]
        errors = np.linalg.norm(proj_2d - pts_2d, axis=1)
        all_errors.append(errors)

    all_errors = np.concatenate(all_errors) if all_errors else np.array([0.0])
    return {
        "median_px": float(np.median(all_errors)),
        "mean_px": float(all_errors.mean()),
        "pct_under_5px": float(100 * (all_errors < 5).mean()),
        "pct_under_10px": float(100 * (all_errors < 10).mean()),
        "n_comparisons": len(all_errors),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Iterative camera parameter refinement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("session_dir",
                        help="Path to session directory with calibration.toml")
    parser.add_argument("--output", "-o", default=None,
                        help="Output calibration.toml path "
                             "(default: <session_dir>/calibration_refined.toml)")
    parser.add_argument("--output_points3d", default=None,
                        help="Output points3d path after final re-triangulation "
                             "(default: <session_dir>/points3d_refined.h5)")
    parser.add_argument("--confidence_threshold", type=float, default=0.3)
    parser.add_argument("--min_views", type=int, default=3)
    parser.add_argument("--reproj_threshold", type=float, default=15.0)
    parser.add_argument("--max_points_per_cam", type=int, default=200000,
                        help="Max correspondences per camera for optimization (default: 200k)")
    parser.add_argument("--subsample_frames", type=int, default=5000,
                        help="Frames to use for intermediate triangulations (default: 5000)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Max refinement iterations (default: 5)")
    parser.add_argument("--convergence_threshold", type=float, default=0.05,
                        help="Stop if median reproj improvement < this (px, default: 0.05)")
    parser.add_argument("--extrinsics_only", action="store_true",
                        help="Only optimize extrinsics (R, t)")
    parser.add_argument("--exclude_cameras", type=str, nargs="+", default=None)
    parser.add_argument("--full_triangulation", action="store_true",
                        help="Run full-frame triangulation at the end (slow)")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    session_dir = args.session_dir
    if args.output is None:
        args.output = os.path.join(session_dir, "calibration_refined.toml")
    if args.output_points3d is None:
        args.output_points3d = os.path.join(session_dir, "points3d_refined.h5")

    optimize_intrinsics = not args.extrinsics_only
    verbose = not args.quiet
    exclude_set = set(args.exclude_cameras) if args.exclude_cameras else set()

    if verbose:
        print("=" * 70)
        print("ITERATIVE CAMERA PARAMETER REFINEMENT")
        print("=" * 70)
        print(f"Session:      {session_dir}")
        print(f"Output cal:   {args.output}")
        print(f"Output 3D:    {args.output_points3d}")
        print(f"Optimize:     {'extrinsics + intrinsics (10 params/cam)' if optimize_intrinsics else 'extrinsics only (6 params/cam)'}")
        print(f"Iterations:   {args.iterations}")
        print(f"Subsample:    {args.subsample_frames} frames for intermediate triangulations")
        print(f"Max pts/cam:  {args.max_points_per_cam:,d}")
        if exclude_set:
            print(f"Excluding:    {sorted(exclude_set)}")
        print()

    # ---- Load calibration ----
    cal_path = os.path.join(session_dir, "calibration.toml")
    cameras = load_calibration(cal_path)
    if exclude_set:
        cameras = {k: v for k, v in cameras.items() if k not in exclude_set}

    if verbose:
        print(f"Loaded calibration for {len(cameras)} cameras: {sorted(cameras.keys())}")

    # ---- Load 2D data ----
    if verbose:
        print("\nLoading 2D keypoint data:")
    all_coords, all_scores, node_names = load_all_2d_data(
        session_dir, cameras, verbose=verbose
    )

    n_frames_total = max(c.shape[0] for c in all_coords.values())
    n_keypoints = list(all_coords.values())[0].shape[1]

    # Subsample frame indices for intermediate iterations
    rng = np.random.default_rng(42)
    n_sub = min(args.subsample_frames, n_frames_total)
    subsample_indices = np.sort(rng.choice(n_frames_total, n_sub, replace=False))

    if verbose:
        print(f"\nTotal frames: {n_frames_total}, subsample: {n_sub} frames")

    # ---- Iterative refinement loop ----
    current_cameras = copy.deepcopy(cameras)
    prev_median = None

    for iteration in range(1, args.iterations + 1):
        if verbose:
            print(f"\n{'#'*70}")
            print(f"# ITERATION {iteration}/{args.iterations}")
            print(f"{'#'*70}")

        # Step 1: Triangulate with current camera params (on subsample)
        if verbose:
            print(f"\n  Step 1: Triangulating {n_sub} frames...")

        tracks_3d, tri_stats = triangulate_all(
            current_cameras, all_coords, all_scores,
            n_frames=n_frames_total, n_keypoints=n_keypoints,
            confidence_threshold=args.confidence_threshold,
            min_views=args.min_views,
            reproj_threshold=args.reproj_threshold,
            undistort=True, use_ransac=True,
            verbose=verbose,
            frame_indices=subsample_indices,
        )

        # Step 2: Evaluate current reprojection
        kp_3d = tracks_3d[:, 0]
        valid_3d = ~np.isnan(kp_3d).any(axis=-1) & (kp_3d != 0).any(axis=-1)

        # Build a coords/scores subset matching subsample_indices
        sub_coords = {}
        sub_scores = {}
        for cam_name in all_coords:
            c = all_coords[cam_name]
            s = all_scores[cam_name]
            # Extract only the subsampled frame rows
            valid_idx = subsample_indices[subsample_indices < c.shape[0]]
            sub_c = np.full((n_sub, n_keypoints, 2), np.nan, dtype=np.float64)
            sub_s = np.full((n_sub, n_keypoints), np.nan, dtype=np.float64)
            mask = subsample_indices < c.shape[0]
            sub_c[mask] = c[subsample_indices[mask]]
            sub_s[mask] = s[subsample_indices[mask]]
            sub_coords[cam_name] = sub_c
            sub_scores[cam_name] = sub_s

        pre_stats = quick_reproj_stats(
            tracks_3d, sub_coords, sub_scores,
            current_cameras, args.confidence_threshold,
            max_points_per_cam=args.max_points_per_cam,
        )

        if verbose:
            print(f"\n  Pre-optimization reproj: median={pre_stats['median_px']:.2f}px, "
                  f"<5px={pre_stats['pct_under_5px']:.1f}%, "
                  f"<10px={pre_stats['pct_under_10px']:.1f}%")

        # Step 3: Optimize each camera
        if verbose:
            print(f"\n  Step 2: Optimizing camera parameters...")

        cam_rng = np.random.default_rng(42 + iteration)
        for cam_name in sorted(current_cameras.keys()):
            if cam_name not in sub_coords:
                continue

            pts_3d, pts_2d = gather_correspondences(
                kp_3d, valid_3d,
                sub_coords[cam_name], sub_scores[cam_name],
                current_cameras[cam_name],
                confidence_threshold=args.confidence_threshold,
                max_points=args.max_points_per_cam,
                rng=cam_rng,
            )

            refined_cam, opt_stats = optimize_camera(
                cam_name, current_cameras[cam_name],
                pts_3d, pts_2d,
                optimize_intrinsics=optimize_intrinsics,
                verbose=verbose,
            )

            if opt_stats["status"] != "skipped":
                current_cameras[cam_name] = refined_cam

        # Step 4: Evaluate post-optimization (re-triangulate with updated params)
        if verbose:
            print(f"\n  Step 3: Re-triangulating to evaluate...")

        tracks_3d_post, _ = triangulate_all(
            current_cameras, all_coords, all_scores,
            n_frames=n_frames_total, n_keypoints=n_keypoints,
            confidence_threshold=args.confidence_threshold,
            min_views=args.min_views,
            reproj_threshold=args.reproj_threshold,
            undistort=True, use_ransac=True,
            verbose=False,
            frame_indices=subsample_indices,
        )

        # Rebuild sub_coords/sub_scores for post evaluation with updated cameras
        post_stats = quick_reproj_stats(
            tracks_3d_post, sub_coords, sub_scores,
            current_cameras, args.confidence_threshold,
            max_points_per_cam=args.max_points_per_cam,
        )

        if verbose:
            print(f"\n  Post-optimization reproj: median={post_stats['median_px']:.2f}px, "
                  f"<5px={post_stats['pct_under_5px']:.1f}%, "
                  f"<10px={post_stats['pct_under_10px']:.1f}%")
            improvement = pre_stats['median_px'] - post_stats['median_px']
            print(f"  Improvement this iteration: {improvement:+.2f}px median")

        # Convergence check
        current_median = post_stats['median_px']
        if prev_median is not None:
            delta = prev_median - current_median
            if verbose:
                print(f"  Cumulative improvement from last iteration: {delta:+.3f}px")
            if abs(delta) < args.convergence_threshold:
                if verbose:
                    print(f"\n  Converged (improvement < {args.convergence_threshold}px)")
                break
        prev_median = current_median

    # ---- Summary ----
    if verbose:
        print(f"\n{'='*70}")
        print("REFINEMENT COMPLETE")
        print(f"{'='*70}")

        print(f"\n  Parameter changes (original -> refined):")
        header = f"  {'Camera':<12} {'dR (deg)':>10} {'dt (mm)':>10}"
        if optimize_intrinsics:
            header += f" {'dfx':>10} {'dfy':>10} {'dcx':>10} {'dcy':>10}"
        print(header)
        print(f"  {'-'*len(header)}")

        for cam_name in sorted(cameras.keys()):
            orig = cameras[cam_name]
            ref = current_cameras[cam_name]

            R_diff = ref["R"] @ orig["R"].T
            angle_diff = np.degrees(np.arccos(np.clip(
                (np.trace(R_diff) - 1) / 2, -1, 1
            )))
            dt = np.linalg.norm(ref["t"] - orig["t"])

            row = f"  {cam_name:<12} {angle_diff:>9.4f}° {dt:>9.3f}"
            if optimize_intrinsics:
                dfx = ref["K"][0, 0] - orig["K"][0, 0]
                dfy = ref["K"][1, 1] - orig["K"][1, 1]
                dcx = ref["K"][0, 2] - orig["K"][0, 2]
                dcy = ref["K"][1, 2] - orig["K"][1, 2]
                row += f" {dfx:>+9.2f} {dfy:>+9.2f} {dcx:>+9.2f} {dcy:>+9.2f}"
            print(row)

    # ---- Save refined calibration ----
    save_calibration_toml(current_cameras, args.output, cal_path)

    # ---- Optional: full re-triangulation ----
    if args.full_triangulation:
        if verbose:
            print(f"\n{'#'*70}")
            print("# FINAL: Full re-triangulation with refined cameras")
            print(f"{'#'*70}")

        tracks_3d_full, final_stats = triangulate_all(
            current_cameras, all_coords, all_scores,
            n_frames=n_frames_total, n_keypoints=n_keypoints,
            confidence_threshold=args.confidence_threshold,
            min_views=args.min_views,
            reproj_threshold=args.reproj_threshold,
            undistort=True, use_ransac=True,
            verbose=verbose,
        )

        # Save
        from triangulate_3d_points import save_points3d_h5
        save_points3d_h5(tracks_3d_full, args.output_points3d)

        # Full reprojection analysis
        if verbose:
            reprojection_analysis(
                tracks_3d_full, session_dir, all_coords, all_scores,
                current_cameras, node_names,
                confidence_threshold=args.confidence_threshold,
                label="REFINED",
            )

    if verbose:
        print(f"\nDone.")
        print(f"  Refined calibration: {args.output}")
        if args.full_triangulation:
            print(f"  Refined 3D points:   {args.output_points3d}")


if __name__ == "__main__":
    main()
