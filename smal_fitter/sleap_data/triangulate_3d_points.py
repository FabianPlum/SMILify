#!/usr/bin/env python3
"""
Triangulate 3D keypoints from multi-view 2D SLEAP predictions + camera calibration.

Produces a points3d.h5 file compatible with SLEAP3DDataLoader / preprocess_sleap_multiview_dataset.py.

Usage:
    python triangulate_3d_points.py /path/to/session [--output points3d_triangulated.h5] [options]

Example:
    python triangulate_3d_points.py \
        /path/to/White_Falkner_Mouse_18_cam \
        --output points3d_full.h5 \
        --min_views 3 \
        --confidence_threshold 0.5
"""

import os
import argparse
import time
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import cv2
import toml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Calibration loading
# ---------------------------------------------------------------------------


def load_calibration(calibration_path: str) -> Dict[str, dict]:
    """
    Load camera calibration from a calibration.toml file (anipose format).

    Returns dict mapping camera name -> {K, dist, R, t, rvec, image_size}.
    """
    cal = toml.load(calibration_path)
    cameras = {}
    for key, entry in cal.items():
        if key == "metadata" or not isinstance(entry, dict):
            continue
        name = entry.get("name", key)
        size = entry["size"]  # [width, height]

        K = np.array(entry["matrix"], dtype=np.float64).reshape(3, 3)
        dist = np.array(entry.get("distortions", [0, 0, 0, 0, 0]), dtype=np.float64)
        rvec = np.array(entry["rotation"], dtype=np.float64)
        t = np.array(entry["translation"], dtype=np.float64).reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)

        cameras[name] = {
            "K": K,
            "dist": dist,
            "R": R,
            "t": t,
            "rvec": rvec,
            "image_size": (int(size[0]), int(size[1])),  # (w, h)
        }
    return cameras


def get_projection_matrix(cam: dict) -> np.ndarray:
    """Return 3×4 projection matrix P = K @ [R | t]."""
    Rt = np.hstack([cam["R"], cam["t"]])  # (3, 4)
    return cam["K"] @ Rt


# ---------------------------------------------------------------------------
# 2D keypoint loading
# ---------------------------------------------------------------------------


def load_2d_keypoints_analysis_h5(h5_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load 2D keypoints from a SLEAP analysis.h5 file.

    Returns:
        coords: (n_frames, n_keypoints, 2) float64 — (x, y) pixel coords
        scores: (n_frames, n_keypoints) float64 — confidence scores (NaN if unavailable)
        node_names: list of keypoint names
    """
    with h5py.File(h5_path, "r") as f:
        tracks = f["tracks"][:]  # (n_tracks, 2, n_keypoints, n_frames)
        node_names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["node_names"][:]]

        # Use track 0
        t0 = tracks[0]  # (2, n_keypoints, n_frames)
        x = t0[0]  # (n_keypoints, n_frames)
        y = t0[1]  # (n_keypoints, n_frames)

        coords = np.stack([x, y], axis=-1).transpose(1, 0, 2)  # (n_frames, n_keypoints, 2)

        if "point_scores" in f:
            scores = f["point_scores"][0]  # (n_keypoints, n_frames) -> take track 0
            scores = scores.T  # (n_frames, n_keypoints)
        else:
            scores = np.full((coords.shape[0], coords.shape[1]), np.nan)

    return coords, scores, node_names


def load_2d_keypoints_slp(slp_path: str, n_keypoints: int = 34) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load 2D keypoints from a SLEAP .predictions.slp file (HDF5 format).

    This is a fallback when analysis.h5 is not available or has fewer frames.

    Returns:
        coords: (n_frames, n_keypoints, 2) — NaN for missing
        scores: (n_frames, n_keypoints) — NaN for missing
    """
    with h5py.File(slp_path, "r") as f:
        frames = f["frames"][:]
        instances = f["instances"][:]
        pred_points = f["pred_points"][:]

        n_frames = len(frames)
        coords = np.full((n_frames, n_keypoints, 2), np.nan, dtype=np.float64)
        scores = np.full((n_frames, n_keypoints), np.nan, dtype=np.float64)

        for fi in range(n_frames):
            frame = frames[fi]
            inst_start = frame["instance_id_start"]
            inst_end = frame["instance_id_end"]
            if inst_end <= inst_start:
                continue

            # Pick the instance with the highest score (best detection)
            frame_instances = instances[inst_start:inst_end]
            best_idx = np.argmax(frame_instances["score"])
            inst = frame_instances[best_idx]

            pt_start = inst["point_id_start"]
            pt_end = inst["point_id_end"]
            pts = pred_points[pt_start:pt_end]

            n_pts = min(len(pts), n_keypoints)
            coords[fi, :n_pts, 0] = pts["x"][:n_pts]
            coords[fi, :n_pts, 1] = pts["y"][:n_pts]
            scores[fi, :n_pts] = pts["score"][:n_pts]

    return coords, scores


# ---------------------------------------------------------------------------
# Triangulation (vectorized for performance)
# ---------------------------------------------------------------------------


def triangulate_point_dlt(projections: List[np.ndarray], points_2d: List[np.ndarray]) -> np.ndarray:
    """
    Direct Linear Transform (DLT) triangulation from N≥2 views.

    Args:
        projections: list of (3, 4) projection matrices
        points_2d: list of (2,) arrays with (x, y) pixel coordinates

    Returns:
        (3,) world-space 3D point
    """
    n = len(projections)
    A = np.zeros((2 * n, 4), dtype=np.float64)
    for i, (P, pt) in enumerate(zip(projections, points_2d)):
        x, y = pt[0], pt[1]
        A[2 * i] = x * P[2] - P[0]
        A[2 * i + 1] = y * P[2] - P[1]

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return (X[:3] / X[3]).astype(np.float64)


def reprojection_errors_vectorized(Ps: np.ndarray, pt_3d: np.ndarray, pts_2d: np.ndarray) -> np.ndarray:
    """
    Compute reprojection errors for a 3D point across N views (vectorized).

    Args:
        Ps: (N, 3, 4) projection matrices
        pt_3d: (3,) world coordinate
        pts_2d: (N, 2) observed pixel coordinates

    Returns:
        (N,) reprojection errors in pixels
    """
    X_hom = np.append(pt_3d, 1.0)  # (4,)
    proj = Ps @ X_hom  # (N, 3)
    proj_2d = proj[:, :2] / proj[:, 2:3]  # (N, 2)
    return np.linalg.norm(proj_2d - pts_2d, axis=1)


def reprojection_error(P: np.ndarray, pt_3d: np.ndarray, pt_2d: np.ndarray) -> float:
    """Compute reprojection error (Euclidean distance in pixels)."""
    X_hom = np.append(pt_3d, 1.0)
    proj = P @ X_hom
    proj = proj[:2] / proj[2]
    return float(np.linalg.norm(proj - pt_2d))


def triangulate_point_ransac(
    Ps_arr: np.ndarray,
    pts_arr: np.ndarray,
    reproj_threshold: float = 15.0,
    min_inliers: int = 2,
    max_hypotheses: int = 50,
) -> Tuple[Optional[np.ndarray], int]:
    """
    RANSAC-based triangulation using sampled pairs (optimized).

    Args:
        Ps_arr: (N, 3, 4) projection matrices
        pts_arr: (N, 2) undistorted 2D points
        reproj_threshold: inlier threshold in pixels
        min_inliers: minimum inliers to accept a hypothesis
        max_hypotheses: max random pairs to try (caps cost for many cameras)

    Returns:
        (pt_3d, n_inliers) or (None, 0)
    """
    n = len(Ps_arr)
    if n < 2:
        return None, 0

    if n == 2:
        pt_3d = triangulate_point_dlt([Ps_arr[0], Ps_arr[1]], [pts_arr[0], pts_arr[1]])
        errors = reprojection_errors_vectorized(Ps_arr, pt_3d, pts_arr)
        n_inliers = int((errors < reproj_threshold).sum())
        if n_inliers >= min_inliers:
            return pt_3d, n_inliers
        return None, 0

    # Generate pair indices — all pairs if small, random sample if many cameras
    import itertools

    all_pairs = list(itertools.combinations(range(n), 2))
    if len(all_pairs) > max_hypotheses:
        rng = np.random.default_rng(42)
        pair_indices = [all_pairs[i] for i in rng.choice(len(all_pairs), max_hypotheses, replace=False)]
    else:
        pair_indices = all_pairs

    best_inliers = 0
    best_inlier_mask = None

    for i, j in pair_indices:
        # DLT from pair
        A = np.zeros((4, 4), dtype=np.float64)
        A[0] = pts_arr[i, 0] * Ps_arr[i, 2] - Ps_arr[i, 0]
        A[1] = pts_arr[i, 1] * Ps_arr[i, 2] - Ps_arr[i, 1]
        A[2] = pts_arr[j, 0] * Ps_arr[j, 2] - Ps_arr[j, 0]
        A[3] = pts_arr[j, 1] * Ps_arr[j, 2] - Ps_arr[j, 1]
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        pt_3d = X[:3] / X[3]

        # Vectorized inlier check
        errors = reprojection_errors_vectorized(Ps_arr, pt_3d, pts_arr)
        inlier_mask = errors < reproj_threshold
        n_inliers = int(inlier_mask.sum())

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_inlier_mask = inlier_mask

            # Early exit: if all views are inliers, can't do better
            if n_inliers == n:
                break

    # Re-triangulate using all inliers
    if best_inliers >= min_inliers and best_inlier_mask is not None:
        inlier_Ps = [Ps_arr[k] for k in range(n) if best_inlier_mask[k]]
        inlier_pts = [pts_arr[k] for k in range(n) if best_inlier_mask[k]]
        best_pt = triangulate_point_dlt(inlier_Ps, inlier_pts)
        return best_pt, best_inliers

    return None, 0


def undistort_points(points_2d: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Undistort 2D points using camera calibration.

    Args:
        points_2d: (N, 2) array of (x, y) pixel coordinates
        K: (3, 3) intrinsic matrix
        dist: distortion coefficients

    Returns:
        (N, 2) undistorted pixel coordinates
    """
    if dist is None or np.allclose(dist, 0):
        return points_2d

    pts = points_2d.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts, K, dist, P=K)
    return undist.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Reprojection analysis: reproject 3D points into each camera and compare
# with undistorted 2D SLEAP detections (fair comparison in ideal pinhole space)
# ---------------------------------------------------------------------------


def reprojection_analysis(
    tracks_3d: np.ndarray,
    session_path: str,
    all_coords: Dict[str, np.ndarray],
    all_scores: Dict[str, np.ndarray],
    cameras: Dict[str, dict],
    node_names: List[str],
    confidence_threshold: float = 0.3,
    label: str = "TRIANGULATED",
) -> dict:
    """
    Compute per-camera and per-joint reprojection errors for a set of 3D points
    against undistorted 2D SLEAP detections.

    Both sides are compared in the ideal pinhole projection space:
    - 3D → 2D via P = K[R|t] (no distortion applied to the projection)
    - Raw SLEAP 2D keypoints are undistorted via cv2.undistortPoints

    This is the fair comparison because the 3D points (whether from anipose or
    our triangulation) were computed from undistorted 2D observations.

    Args:
        tracks_3d: (n_frames, 1, n_keypoints, 3) — NaN or 0 for missing
        session_path: for logging
        all_coords: cam_name -> (n_frames, n_keypoints, 2) raw SLEAP 2D
        all_scores: cam_name -> (n_frames, n_keypoints) confidence
        cameras: calibration dict from load_calibration()
        node_names: list of joint names
        confidence_threshold: skip 2D detections below this
        label: string label for the 3D source (e.g. "REFERENCE" or "TRIANGULATED")

    Returns:
        dict with per-camera and overall error statistics
    """
    kp_3d = tracks_3d[:, 0]  # (n_frames, n_keypoints, 3)
    n_frames, n_keypoints, _ = kp_3d.shape

    # Mask for valid 3D points (non-NaN, non-zero)
    valid_3d = ~np.isnan(kp_3d).any(axis=-1) & (kp_3d != 0).any(axis=-1)

    cam_names = sorted(all_coords.keys())

    # Collect per-camera statistics
    per_cam_errors = {}  # cam -> np.ndarray of errors
    per_cam_joint_errors = {}  # cam -> joint_idx -> np.ndarray of errors
    all_errors = []  # list of np.ndarray, concatenated at end

    for cam_name in tqdm(cam_names, desc=f"Reproj analysis ({label})", leave=False):
        if cam_name not in cameras:
            continue
        cam = cameras[cam_name]
        coords_2d = all_coords[cam_name]  # (n_frames_cam, n_kp, 2)
        scores_2d = all_scores[cam_name]  # (n_frames_cam, n_kp)
        n_frames_cam = coords_2d.shape[0]
        n_compare = min(n_frames, n_frames_cam)

        # Build validity mask: valid_3d AND valid 2D (not NaN, not zero, above threshold)
        valid_2d_sub = coords_2d[:n_compare]  # (n_compare, n_kp, 2)
        scores_sub = scores_2d[:n_compare]  # (n_compare, n_kp)
        valid_3d_sub = valid_3d[:n_compare]  # (n_compare, n_kp)

        not_nan_2d = ~np.isnan(valid_2d_sub).any(axis=-1)  # (n_compare, n_kp)
        not_zero_2d = (valid_2d_sub != 0).any(axis=-1)  # (n_compare, n_kp)
        above_conf = np.isnan(scores_sub) | (scores_sub >= confidence_threshold)
        both_valid = valid_3d_sub & not_nan_2d & not_zero_2d & above_conf

        # Get indices of valid pairs
        fi_idx, ki_idx = np.where(both_valid)
        if len(fi_idx) == 0:
            per_cam_errors[cam_name] = np.array([])
            per_cam_joint_errors[cam_name] = {}
            continue

        # Gather valid 3D points and raw 2D observations
        pts_3d_valid = kp_3d[fi_idx, ki_idx]  # (M, 3)
        pts_2d_raw = coords_2d[fi_idx, ki_idx]  # (M, 2)

        # Undistort the raw 2D SLEAP keypoints into ideal pinhole space
        pts_2d_undist = undistort_points(pts_2d_raw, cam["K"], cam["dist"])  # (M, 2)

        # Project 3D → ideal 2D via P = K[R|t] (no distortion on projection side)
        P = get_projection_matrix(cam)  # (3, 4)
        ones = np.ones((pts_3d_valid.shape[0], 1), dtype=np.float64)
        pts_3d_hom = np.hstack([pts_3d_valid, ones])  # (M, 4)
        projected_hom = (P @ pts_3d_hom.T).T  # (M, 3)
        projected = projected_hom[:, :2] / projected_hom[:, 2:3]  # (M, 2)

        errors = np.linalg.norm(projected - pts_2d_undist, axis=1)  # (M,)

        per_cam_errors[cam_name] = errors
        all_errors.append(errors)

        # Per-joint breakdown for this camera
        joint_errors = {}
        for ki in range(n_keypoints):
            mask = ki_idx == ki
            if mask.any():
                joint_errors[ki] = errors[mask]
        per_cam_joint_errors[cam_name] = joint_errors

    all_errors = np.concatenate(all_errors) if all_errors else np.array([])

    # ---- Print report ----
    print(f"\n{'=' * 70}")
    print(f"REPROJECTION ANALYSIS — {label} 3D vs undistorted 2D SLEAP")
    print(f"{'=' * 70}")
    print(f"  3D source: {label} ({valid_3d.sum()} valid 3D keypoints across {n_frames} frames)")
    print("  Comparison space: undistorted (ideal pinhole) pixel coordinates")
    print(f"  Confidence threshold for 2D: {confidence_threshold}")

    if len(all_errors) == 0:
        print("  No valid comparisons!")
        return {}

    print(f"\n  Overall reprojection error ({len(all_errors)} comparisons):")
    print(f"    Mean:   {all_errors.mean():.2f} px")
    print(f"    Median: {np.median(all_errors):.2f} px")
    print(f"    Std:    {all_errors.std():.2f} px")
    print(f"    P90:    {np.percentile(all_errors, 90):.2f} px")
    print(f"    P95:    {np.percentile(all_errors, 95):.2f} px")
    print(f"    P99:    {np.percentile(all_errors, 99):.2f} px")
    print(f"    Max:    {all_errors.max():.2f} px")
    print(f"    < 5px:  {100 * (all_errors < 5).mean():.1f}%")
    print(f"    < 10px: {100 * (all_errors < 10).mean():.1f}%")
    print(f"    < 20px: {100 * (all_errors < 20).mean():.1f}%")

    # Per-camera breakdown
    print(f"\n  {'Camera':<12} {'N':>8} {'Mean':>8} {'Median':>8} {'P90':>8} {'P95':>8} {'<10px':>8}")
    print(f"  {'-' * 62}")
    for cam_name in cam_names:
        errs = per_cam_errors.get(cam_name, np.array([]))
        if len(errs) == 0:
            print(f"  {cam_name:<12} {'—':>8}")
            continue
        pct_10 = 100 * (errs < 10).mean()
        print(
            f"  {cam_name:<12} {len(errs):>8d} {errs.mean():>8.2f} "
            f"{np.median(errs):>8.2f} {np.percentile(errs, 90):>8.2f} "
            f"{np.percentile(errs, 95):>8.2f} {pct_10:>7.1f}%"
        )

    # Per-joint breakdown (aggregate across all cameras)
    print(f"\n  {'Joint':<24} {'N':>8} {'Mean':>8} {'Median':>8} {'P90':>8} {'<10px':>8}")
    print(f"  {'-' * 68}")
    for ki in range(n_keypoints):
        joint_errs = []
        for cam_name in cam_names:
            je = per_cam_joint_errors.get(cam_name, {}).get(ki, np.array([]))
            if len(je) > 0:
                joint_errs.extend(je.tolist())
        joint_errs = np.array(joint_errs)
        name = node_names[ki] if ki < len(node_names) else f"joint_{ki}"
        if len(joint_errs) == 0:
            print(f"  {name:<24} {'—':>8}")
            continue
        pct_10 = 100 * (joint_errs < 10).mean()
        print(
            f"  {name:<24} {len(joint_errs):>8d} {joint_errs.mean():>8.2f} "
            f"{np.median(joint_errs):>8.2f} {np.percentile(joint_errs, 90):>8.2f} "
            f"{pct_10:>7.1f}%"
        )

    return {
        "label": label,
        "n_comparisons": len(all_errors),
        "mean_px": float(all_errors.mean()),
        "median_px": float(np.median(all_errors)),
        "p90_px": float(np.percentile(all_errors, 90)),
        "p95_px": float(np.percentile(all_errors, 95)),
        "pct_under_10px": float(100 * (all_errors < 10).mean()),
    }


# ---------------------------------------------------------------------------
# Validation: 3D-vs-3D comparison between triangulated and reference
# ---------------------------------------------------------------------------


def validate_against_reference(
    tracks_3d: np.ndarray, reference_path: str, node_names: List[str], verbose: bool = True
) -> dict:
    """
    Compare triangulated 3D points against a reference points3d.h5 file.
    """
    with h5py.File(reference_path, "r") as f:
        ref_tracks = f["tracks"][:]  # (n_frames_ref, 1, n_kp, 3)

    n_frames_ref = ref_tracks.shape[0]
    n_frames_tri = tracks_3d.shape[0]
    n_compare = min(n_frames_ref, n_frames_tri)

    ref = ref_tracks[:n_compare, 0]  # (n_compare, n_kp, 3)
    tri = tracks_3d[:n_compare, 0]  # (n_compare, n_kp, 3)

    ref_valid = ~np.isnan(ref).any(axis=-1) & (ref != 0).any(axis=-1)
    tri_valid = ~np.isnan(tri).any(axis=-1) & (tri != 0).any(axis=-1)
    both_valid = ref_valid & tri_valid

    n_both = both_valid.sum()
    if n_both == 0:
        if verbose:
            print("No overlapping valid keypoints to compare!")
        return {"n_compared": 0}

    distances = np.linalg.norm(ref - tri, axis=-1)  # (n_compare, n_kp)
    distances_valid = distances[both_valid]

    result = {
        "n_frames_compared": n_compare,
        "n_keypoints_compared": int(n_both),
        "n_ref_valid": int(ref_valid.sum()),
        "n_tri_valid": int(tri_valid.sum()),
        "mean_error_mm": float(np.mean(distances_valid)),
        "median_error_mm": float(np.median(distances_valid)),
        "std_error_mm": float(np.std(distances_valid)),
        "p90_error_mm": float(np.percentile(distances_valid, 90)),
        "p95_error_mm": float(np.percentile(distances_valid, 95)),
        "p99_error_mm": float(np.percentile(distances_valid, 99)),
        "max_error_mm": float(np.max(distances_valid)),
        "pct_under_5mm": float(100.0 * (distances_valid < 5).sum() / len(distances_valid)),
        "pct_under_10mm": float(100.0 * (distances_valid < 10).sum() / len(distances_valid)),
        "pct_under_20mm": float(100.0 * (distances_valid < 20).sum() / len(distances_valid)),
    }

    if verbose:
        print(f"\n{'=' * 70}")
        print("3D-vs-3D VALIDATION: TRIANGULATED vs REFERENCE (anipose)")
        print(f"{'=' * 70}")
        print(f"  Frames compared: {n_compare}")
        print(f"  Keypoints compared: {n_both} (ref valid: {ref_valid.sum()}, tri valid: {tri_valid.sum()})")
        print("\n  3D Euclidean Error (mm):")
        print(f"    Mean:   {result['mean_error_mm']:.2f}")
        print(f"    Median: {result['median_error_mm']:.2f}")
        print(f"    Std:    {result['std_error_mm']:.2f}")
        print(f"    P90:    {result['p90_error_mm']:.2f}")
        print(f"    P95:    {result['p95_error_mm']:.2f}")
        print(f"    P99:    {result['p99_error_mm']:.2f}")
        print(f"    Max:    {result['max_error_mm']:.2f}")
        print("\n  Accuracy:")
        print(f"    < 5mm:  {result['pct_under_5mm']:.1f}%")
        print(f"    < 10mm: {result['pct_under_10mm']:.1f}%")
        print(f"    < 20mm: {result['pct_under_20mm']:.1f}%")

        n_kp = ref.shape[1]
        print(f"\n  {'Joint':<24} {'N':>7} {'Mean':>8} {'Median':>8} {'P90':>8} {'<5mm':>7}")
        print(f"  {'-' * 66}")
        for kp_idx in range(n_kp):
            joint_mask = both_valid[:, kp_idx]
            if joint_mask.sum() == 0:
                continue
            joint_dists = distances[:, kp_idx][joint_mask]
            name = node_names[kp_idx] if kp_idx < len(node_names) else f"joint_{kp_idx}"
            pct5 = 100 * (joint_dists < 5).mean()
            print(
                f"  {name:<24} {joint_mask.sum():>7d} {np.mean(joint_dists):>8.2f} "
                f"{np.median(joint_dists):>8.2f} {np.percentile(joint_dists, 90):>8.2f} "
                f"{pct5:>6.1f}%"
            )

    return result


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def generate_reprojections_h5(
    tracks_3d: np.ndarray,
    cameras: Dict[str, dict],
    output_path: str,
    verbose: bool = True,
) -> str:
    """
    Project triangulated 3D points back into each camera view and save as reprojections.h5.

    Uses ideal pinhole projection (P = K @ [R|t], no distortion), consistent with
    PyTorch3D FoVPerspectiveCameras used in downstream training.

    Args:
        tracks_3d: (n_frames, 1, n_keypoints, 3) — NaN for missing
        cameras: calibration dict from load_calibration()
        output_path: path to write reprojections.h5
        verbose: print progress

    Returns:
        output_path
    """
    n_frames, _, n_keypoints, _ = tracks_3d.shape
    kp_3d = tracks_3d[:, 0]  # (n_frames, n_keypoints, 3)

    # Valid mask: non-NaN and non-zero
    valid = ~np.isnan(kp_3d).any(axis=-1) & (kp_3d != 0).any(axis=-1)  # (n_frames, n_kp)

    # Build homogeneous 3D coords: (n_frames, n_keypoints, 4)
    ones = np.ones((*kp_3d.shape[:-1], 1), dtype=np.float64)
    kp_3d_safe = np.nan_to_num(kp_3d, nan=0.0)
    kp_3d_hom = np.concatenate([kp_3d_safe, ones], axis=-1)  # (n_frames, n_kp, 4)

    cam_names = sorted(cameras.keys())

    with h5py.File(output_path, "w") as f:
        for cam_name in cam_names:
            P = get_projection_matrix(cameras[cam_name])  # (3, 4)

            # Vectorized projection: (n_frames, n_kp, 4) @ (4, 3) -> (n_frames, n_kp, 3)
            projected_hom = kp_3d_hom @ P.T  # (n_frames, n_kp, 3)
            proj_2d = projected_hom[..., :2] / projected_hom[..., 2:3]  # (n_frames, n_kp, 2)

            # Set invalid points to NaN
            proj_2d[~valid] = np.nan

            # Reshape to (n_frames, 1, n_keypoints, 2) to match expected format
            proj_2d = proj_2d[:, np.newaxis, :, :]

            ds = f.create_dataset(cam_name, data=proj_2d, dtype=np.float64)
            ds.attrs["Description"] = "Shape: (n_frames, n_tracks, n_nodes, 2)."

    if verbose:
        n_valid = valid.sum()
        print(f"\nSaved reprojections: {output_path}")
        print(f"  Cameras: {len(cam_names)} ({', '.join(cam_names)})")
        print(f"  Shape per camera: ({n_frames}, 1, {n_keypoints}, 2)")
        print(
            f"  Valid projections: {n_valid} / {n_frames * n_keypoints} "
            f"({100 * n_valid / max(1, n_frames * n_keypoints):.1f}%)"
        )
        print("  Projection model: ideal pinhole (K @ [R|t], no distortion)")

    return output_path


def save_points3d_h5(tracks_3d: np.ndarray, output_path: str, frame_range: Optional[Tuple[int, int]] = None):
    """
    Save triangulated 3D points in the same format as anipose points3d.h5.

    tracks_3d: (n_frames, 1, n_keypoints, 3)
    """
    n_frames = tracks_3d.shape[0]
    if frame_range is None:
        frame_range = (0, n_frames)

    # Replace NaN with 0 to match anipose convention
    tracks_out = tracks_3d.copy()
    tracks_out = np.nan_to_num(tracks_out, nan=0.0)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("tracks", data=tracks_out, dtype=np.float64)
        f.create_dataset("frames", data=np.array(frame_range, dtype=np.int32))

    print(f"\nSaved {output_path}: tracks shape {tracks_out.shape}")


# ---------------------------------------------------------------------------
# Shared 2D data loader (used by both triangulation and analysis)
# ---------------------------------------------------------------------------


def detect_data_structure(session_path: str) -> str:
    """Detect the dataset layout: 'camera_dirs', 'session_dirs', or 'unknown'.

    Mirrors ``SLEAPDataLoader._detect_data_structure`` so the triangulation CLI
    accepts the same layouts as the rest of the SLEAP pipeline:

    - ``camera_dirs``:  ``<session>/<CameraName>/*.slp`` (one subdir per camera)
    - ``session_dirs``: ``<session>/<videoSubdir>/<prefix>_cam<X>.h5`` (one
      subdir per recorded video, flat per-camera analysis HDF5s inside)
    """
    session_path = Path(session_path)
    camera_dirs_found = 0
    session_dirs_found = 0
    for item in session_path.iterdir():
        if not item.is_dir() or item.name.startswith("."):
            continue
        if list(item.glob("*.slp")):
            camera_dirs_found += 1
        elif list(item.glob("*.h5")):
            session_dirs_found += 1
    if camera_dirs_found > 0 and session_dirs_found == 0:
        return "camera_dirs"
    if session_dirs_found > 0:
        return "session_dirs"
    return "unknown"


def _load_camera_dirs_2d_data(session_path: Path, cameras: Dict[str, dict], verbose: bool):
    """Load 2D keypoints from a ``camera_dirs`` session (one subdir per camera)."""
    all_coords, all_scores, node_names = {}, {}, None

    for cam_name in sorted(cameras.keys()):
        cam_dir = session_path / cam_name
        if not cam_dir.exists():
            if verbose:
                print(f"  Warning: Camera directory {cam_name} not found, skipping")
            continue

        h5_files = sorted(glob.glob(str(cam_dir / "*.analysis.h5")))
        slp_files = sorted(glob.glob(str(cam_dir / "*Data.mp4.predictions.slp")))
        slp_files = [f for f in slp_files if "old" not in os.path.basename(f).lower()]

        loaded = False

        for h5f in h5_files:
            try:
                coords, scores, names = load_2d_keypoints_analysis_h5(h5f)
                if node_names is None:
                    node_names = names
                all_coords[cam_name] = coords
                all_scores[cam_name] = scores
                loaded = True
                if verbose:
                    print(f"  {cam_name}: {coords.shape[0]} frames from {os.path.basename(h5f)}")
                break
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to load {h5f}: {e}")

        if not loaded and slp_files:
            for slp_f in slp_files:
                try:
                    n_kp = len(node_names) if node_names else 34
                    coords, scores = load_2d_keypoints_slp(slp_f, n_keypoints=n_kp)
                    all_coords[cam_name] = coords
                    all_scores[cam_name] = scores
                    loaded = True
                    if verbose:
                        print(f"  {cam_name}: {coords.shape[0]} frames from SLP {os.path.basename(slp_f)}")
                    break
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Failed to load SLP {slp_f}: {e}")

        if not loaded and verbose:
            print(f"  Warning: No 2D data found for {cam_name}")

    return all_coords, all_scores, node_names


def _load_session_dirs_2d_data(session_path: Path, cameras: Dict[str, dict], verbose: bool):
    """Load 2D keypoints from a ``session_dirs`` session.

    Cameras are discovered exactly like ``SLEAPDataLoader``: glob ``*_cam*.h5``
    inside every subdir and map ``stem.split('_cam')[-1]`` to a calibration
    camera name (case-insensitive, with the ``camera`` -> ``cam`` alias). The
    first file found wins per camera, so ``points3d.h5`` / ``reprojections.h5``
    (no ``_cam`` token) and stray duplicates are ignored.
    """
    calib_names = sorted(cameras.keys())
    cam_files: Dict[str, str] = {}
    for sub in sorted(session_path.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        for h5 in sorted(sub.glob("*_cam*.h5")):
            file_cam = h5.stem.split("_cam")[-1]
            for calib_name in calib_names:
                cn = calib_name.lower()
                if file_cam.lower() in (cn, cn.replace("camera", "cam")):
                    cam_files.setdefault(calib_name, str(h5))
                    break

    all_coords, all_scores, node_names = {}, {}, None
    for cam_name in calib_names:
        path = cam_files.get(cam_name)
        if path is None:
            if verbose:
                print(f"  Warning: No keypoint file found for camera {cam_name}")
            continue
        try:
            coords, scores, names = load_2d_keypoints_analysis_h5(path)
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to load {path}: {e}")
            continue
        if node_names is None:
            node_names = names
        all_coords[cam_name] = coords
        all_scores[cam_name] = scores
        if verbose:
            print(f"  {cam_name}: {coords.shape[0]} frames from {os.path.basename(path)}")

    return all_coords, all_scores, node_names


def load_all_2d_data(
    session_path: str, cameras: Dict[str, dict], verbose: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """
    Load 2D keypoint data from all cameras, auto-detecting the dataset layout
    (``camera_dirs`` or ``session_dirs``; see :func:`detect_data_structure`).

    Returns:
        all_coords: cam_name -> (n_frames, n_keypoints, 2)
        all_scores: cam_name -> (n_frames, n_keypoints)
        node_names: list of keypoint names
    """
    session_path = Path(session_path)
    structure = detect_data_structure(session_path)
    if verbose:
        print(f"  Detected data structure: {structure}")

    if structure == "session_dirs":
        all_coords, all_scores, node_names = _load_session_dirs_2d_data(session_path, cameras, verbose)
    else:
        # camera_dirs (legacy default); also the fallback for 'unknown'
        all_coords, all_scores, node_names = _load_camera_dirs_2d_data(session_path, cameras, verbose)

    if not all_coords:
        raise ValueError("No 2D keypoint data loaded from any camera")

    if node_names is None:
        node_names = [f"joint_{i}" for i in range(list(all_coords.values())[0].shape[1])]

    return all_coords, all_scores, node_names


# ---------------------------------------------------------------------------
# Core triangulation loop (reusable from other scripts)
# ---------------------------------------------------------------------------


def triangulate_all(
    cameras: Dict[str, dict],
    all_coords: Dict[str, np.ndarray],
    all_scores: Dict[str, np.ndarray],
    n_frames: int,
    n_keypoints: int,
    confidence_threshold: float = 0.3,
    min_views: int = 2,
    reproj_threshold: float = 15.0,
    undistort: bool = True,
    use_ransac: bool = True,
    verbose: bool = True,
    frame_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Triangulate 3D keypoints from multi-view 2D observations.

    Args:
        cameras: calibration dict from load_calibration()
        all_coords: cam_name -> (n_frames_cam, n_kp, 2) raw 2D coords
        all_scores: cam_name -> (n_frames_cam, n_kp) confidence scores
        n_frames: number of output frames
        n_keypoints: number of keypoints
        confidence_threshold: min SLEAP confidence for 2D keypoints
        min_views: minimum camera views to triangulate
        reproj_threshold: RANSAC inlier threshold in pixels
        undistort: whether to undistort 2D points before triangulation
        use_ransac: use RANSAC or plain DLT
        verbose: show progress
        frame_indices: optional array of frame indices to process
                       (if None, processes range(n_frames))

    Returns:
        tracks_3d: (n_out_frames, 1, n_keypoints, 3) triangulated points
        stats: dict with triangulation statistics
    """
    start = time.time()

    proj_matrices = {name: get_projection_matrix(cameras[name]) for name in cameras}

    if frame_indices is not None:
        n_out = len(frame_indices)
    else:
        n_out = n_frames
        frame_indices = np.arange(n_frames)

    tracks_3d = np.full((n_out, 1, n_keypoints, 3), np.nan, dtype=np.float64)

    stats = {
        "n_frames": n_out,
        "n_keypoints": n_keypoints,
        "n_cameras": len(all_coords),
        "total_keypoints": n_out * n_keypoints,
        "triangulated": 0,
        "failed_insufficient_views": 0,
        "failed_ransac": 0,
        "views_used_list": [],
        "reproj_error_list": [],
    }

    if verbose:
        print(f"\n  Triangulating {n_out} frames × {n_keypoints} keypoints using {len(all_coords)} cameras")
        print(f"  Confidence threshold: {confidence_threshold}")
        print(f"  Min views: {min_views}")
        print(f"  Reproj threshold: {reproj_threshold} px")
        print(f"  Undistort 2D before triangulation: {undistort}")
        print(f"  Method: {'RANSAC' if use_ransac else 'DLT'}")

    cam_name_list = sorted(all_coords.keys())

    for out_idx, frame_idx in enumerate(tqdm(frame_indices, desc="Triangulating", disable=not verbose)):
        for kp_idx in range(n_keypoints):
            view_Ps = []
            view_pts = []

            for cam_name in cam_name_list:
                coords = all_coords[cam_name]
                scores_arr = all_scores[cam_name]

                if frame_idx >= coords.shape[0]:
                    continue

                pt = coords[frame_idx, kp_idx]
                score = scores_arr[frame_idx, kp_idx]

                if np.isnan(pt).any():
                    continue
                if not np.isnan(score) and score < confidence_threshold:
                    continue
                if pt[0] == 0 and pt[1] == 0:
                    continue

                if undistort:
                    cam = cameras[cam_name]
                    pt = undistort_points(pt.reshape(1, 2), cam["K"], cam["dist"])[0]

                view_Ps.append(proj_matrices[cam_name])
                view_pts.append(pt)

            if len(view_Ps) < min_views:
                stats["failed_insufficient_views"] += 1
                continue

            Ps_arr = np.array(view_Ps)
            pts_arr = np.array(view_pts)

            if use_ransac and len(view_Ps) >= 3:
                pt_3d, n_inliers = triangulate_point_ransac(
                    Ps_arr,
                    pts_arr,
                    reproj_threshold=reproj_threshold,
                    min_inliers=min_views,
                )
                if pt_3d is None:
                    stats["failed_ransac"] += 1
                    continue
                stats["views_used_list"].append(n_inliers)
            else:
                pt_3d = triangulate_point_dlt(view_Ps, view_pts)
                stats["views_used_list"].append(len(view_Ps))

            mean_err = float(reprojection_errors_vectorized(Ps_arr, pt_3d, pts_arr).mean())
            stats["reproj_error_list"].append(mean_err)

            tracks_3d[out_idx, 0, kp_idx] = pt_3d

    elapsed = time.time() - start

    stats["pct_triangulated"] = 100.0 * stats["triangulated"] / max(1, stats["total_keypoints"])
    stats["triangulated"] = len(stats["views_used_list"])
    stats["pct_triangulated"] = 100.0 * stats["triangulated"] / max(1, stats["total_keypoints"])
    vu = stats.pop("views_used_list")
    re = stats.pop("reproj_error_list")
    stats["mean_views_used"] = float(np.mean(vu)) if vu else 0
    stats["mean_reproj_error_px"] = float(np.mean(re)) if re else 0
    stats["median_reproj_error_px"] = float(np.median(re)) if re else 0

    if verbose:
        print(f"\n  Triangulation completed in {elapsed:.1f}s")
        print(
            f"  Triangulated: {stats['triangulated']} / {stats['total_keypoints']} ({stats['pct_triangulated']:.1f}%)"
        )
        print(f"  Failed (insufficient views): {stats['failed_insufficient_views']}")
        print(f"  Failed (RANSAC):             {stats['failed_ransac']}")
        print(f"  Mean views used:  {stats['mean_views_used']:.1f}")
        print(f"  Mean reproj err:  {stats['mean_reproj_error_px']:.2f} px")
        print(f"  Median reproj err: {stats['median_reproj_error_px']:.2f} px")

    return tracks_3d, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def find_reference_points3d(session_dir: str) -> Optional[str]:
    """Locate an existing points3d.h5 to validate against.

    Searches the session root first, then one level of subdirectories (the
    ``session_dirs`` layout stores points3d.h5 inside the per-video subdir).
    """
    candidate = os.path.join(session_dir, "points3d.h5")
    if os.path.exists(candidate):
        return candidate
    for sub in sorted(Path(session_dir).iterdir()):
        if sub.is_dir() and not sub.name.startswith("."):
            sub_candidate = sub / "points3d.h5"
            if sub_candidate.exists():
                return str(sub_candidate)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Triangulate 3D keypoints from multi-view 2D SLEAP predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("session_dir", help="Path to session directory with calibration.toml and Camera* dirs")
    parser.add_argument(
        "--output", "-o", default=None, help="Output HDF5 file path (default: <session_dir>/points3d_triangulated.h5)"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Minimum SLEAP confidence for 2D keypoints (default: 0.3)",
    )
    parser.add_argument("--min_views", type=int, default=2, help="Minimum camera views to triangulate (default: 2)")
    parser.add_argument(
        "--reproj_threshold",
        type=float,
        default=15.0,
        help="Max reprojection error (px) for RANSAC inlier (default: 15.0)",
    )
    parser.add_argument("--no_undistort", action="store_true", help="Skip undistorting 2D points before triangulation")
    parser.add_argument("--no_ransac", action="store_true", help="Use plain DLT instead of RANSAC")
    parser.add_argument(
        "--validate", type=str, default=None, help="Path to reference points3d.h5 (default: auto-detect)"
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Limit to first N frames (for quick testing)")
    parser.add_argument(
        "--exclude_cameras",
        type=str,
        nargs="+",
        default=None,
        help="Camera names to exclude (e.g. Camera8 Camera10 Camera13 Camera15)",
    )
    parser.add_argument(
        "--save_reprojections", action="store_true", help="Save reprojections.h5 alongside the 3D output"
    )
    parser.add_argument(
        "--reprojections_output",
        type=str,
        default=None,
        help="Path for reprojections.h5 (default: <session_dir>/reprojections.h5)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    exclude_set = set(args.exclude_cameras) if args.exclude_cameras else set()

    session_dir = args.session_dir
    if args.output is None:
        args.output = os.path.join(session_dir, "points3d_triangulated.h5")

    # Auto-detect reference (session root or per-video subdir)
    validate_path = args.validate
    if validate_path is None:
        validate_path = find_reference_points3d(session_dir)

    verbose = not args.quiet

    if verbose:
        print("=" * 70)
        print("3D KEYPOINT TRIANGULATION FROM MULTI-VIEW SLEAP")
        print("=" * 70)
        print(f"Session:   {session_dir}")
        print(f"Output:    {args.output}")
        if validate_path:
            print(f"Reference: {validate_path}")
        if args.max_frames:
            print(f"Max frames: {args.max_frames}")
        if exclude_set:
            print(f"Excluding:  {sorted(exclude_set)}")
        print()

    # ---- Load calibration ----
    cal_path = os.path.join(session_dir, "calibration.toml")
    cameras = load_calibration(cal_path)
    if exclude_set:
        cameras = {k: v for k, v in cameras.items() if k not in exclude_set}
    cam_names = sorted(cameras.keys())
    if verbose:
        print(f"Loaded calibration for {len(cam_names)} cameras: {cam_names}")

    # ---- Load all 2D data (shared by triangulation + analysis) ----
    if verbose:
        print("\nLoading 2D keypoint data:")
    all_coords, all_scores, node_names = load_all_2d_data(session_dir, cameras, verbose=verbose)

    # ---- Determine frame count ----
    n_frames = max(c.shape[0] for c in all_coords.values())
    if args.max_frames is not None:
        n_frames = min(n_frames, args.max_frames)
    n_keypoints = list(all_coords.values())[0].shape[1]

    # ====================================================================
    # STEP 1: Reprojection analysis of REFERENCE 3D -> 2D SLEAP
    # ====================================================================
    if validate_path:
        if verbose:
            print(f"\n\n{'#' * 70}")
            print("# STEP 1: Baseline — reference (anipose) 3D reprojection into 2D")
            print(f"{'#' * 70}")

        with h5py.File(validate_path, "r") as f:
            ref_tracks = f["tracks"][:]  # (n_frames_ref, 1, n_kp, 3)

        n_frames_ref = ref_tracks.shape[0]
        if verbose:
            print(f"  Reference has {n_frames_ref} frames, {ref_tracks.shape[2]} keypoints")

        reprojection_analysis(
            ref_tracks,
            session_dir,
            all_coords,
            all_scores,
            cameras,
            node_names,
            confidence_threshold=args.confidence_threshold,
            label="REFERENCE (anipose)",
        )

    # ====================================================================
    # STEP 2: Triangulate
    # ====================================================================
    if verbose:
        print(f"\n\n{'#' * 70}")
        print("# STEP 2: Triangulating 3D points from 2D SLEAP detections")
        print(f"{'#' * 70}")

    undistort = not args.no_undistort
    use_ransac = not args.no_ransac

    conf_thr = args.confidence_threshold

    tracks_3d, stats = triangulate_all(
        cameras,
        all_coords,
        all_scores,
        n_frames=n_frames,
        n_keypoints=n_keypoints,
        confidence_threshold=conf_thr,
        min_views=args.min_views,
        reproj_threshold=args.reproj_threshold,
        undistort=undistort,
        use_ransac=use_ransac,
        verbose=verbose,
    )

    # ====================================================================
    # STEP 3: Reprojection analysis of TRIANGULATED 3D -> 2D SLEAP
    # ====================================================================
    if verbose:
        print(f"\n\n{'#' * 70}")
        print("# STEP 3: Triangulated 3D reprojection into 2D")
        print(f"{'#' * 70}")

    reprojection_analysis(
        tracks_3d,
        session_dir,
        all_coords,
        all_scores,
        cameras,
        node_names,
        confidence_threshold=conf_thr,
        label="TRIANGULATED",
    )

    # ====================================================================
    # STEP 4: 3D-vs-3D validation against reference
    # ====================================================================
    if validate_path:
        if verbose:
            print(f"\n\n{'#' * 70}")
            print("# STEP 4: 3D-vs-3D comparison (triangulated vs reference)")
            print(f"{'#' * 70}")

        validate_against_reference(tracks_3d, validate_path, node_names, verbose=verbose)

    # ====================================================================
    # Save
    # ====================================================================
    save_points3d_h5(tracks_3d, args.output)

    # ====================================================================
    # Reprojections
    # ====================================================================
    if args.save_reprojections:
        reproj_path = args.reprojections_output
        if reproj_path is None:
            reproj_path = os.path.join(session_dir, "reprojections.h5")
        generate_reprojections_h5(tracks_3d, cameras, reproj_path, verbose=verbose)

    if verbose:
        print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()
