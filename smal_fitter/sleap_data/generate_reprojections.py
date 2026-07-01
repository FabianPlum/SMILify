#!/usr/bin/env python3
"""
Generate 2D reprojections from triangulated 3D keypoints + camera calibration.

Reads a points3d.h5 file and a calibration.toml (anipose format) and writes
a reprojections.h5 file containing per-camera ideal-pinhole reprojections.

Usage:
    python generate_reprojections.py points3d_triangulated.h5 [options]

Examples:
    # Auto-detect calibration.toml in the same directory as the h5 file
    python generate_reprojections.py /path/to/session/points3d_triangulated.h5

    # Explicit calibration and output paths
    python generate_reprojections.py /path/to/session/points3d_triangulated.h5 \\
        --calibration /path/to/session/calibration.toml \\
        --output /path/to/session/reprojections.h5

    # Exclude cameras
    python generate_reprojections.py points3d.h5 \\
        --exclude_cameras Camera8 Camera10
"""

import argparse
import os
import sys

import h5py
import numpy as np

# Re-use helpers from the triangulation script
from smal_fitter.sleap_data.triangulate_3d_points import load_calibration, generate_reprojections_h5


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-camera 2D reprojections from 3D keypoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "points3d",
        help="Path to a points3d.h5 file with shape (n_frames, 1, n_keypoints, 3)",
    )
    parser.add_argument(
        "--calibration", "-c", default=None,
        help="Path to calibration.toml (default: calibration.toml next to points3d file)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output reprojections.h5 path (default: reprojections.h5 next to points3d file)",
    )
    parser.add_argument(
        "--exclude_cameras", nargs="+", default=None,
        help="Camera names to exclude (e.g. Camera8 Camera10)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    points3d_path = os.path.abspath(args.points3d)
    session_dir = os.path.dirname(points3d_path)

    # Resolve calibration path
    cal_path = args.calibration or os.path.join(session_dir, "calibration.toml")
    if not os.path.exists(cal_path):
        print(f"ERROR: calibration file not found: {cal_path}", file=sys.stderr)
        print("Pass --calibration <path> to specify its location.", file=sys.stderr)
        sys.exit(1)

    # Resolve output path
    output_path = args.output or os.path.join(session_dir, "reprojections.h5")

    if verbose:
        print("=" * 70)
        print("REPROJECTION GENERATOR")
        print("=" * 70)
        print(f"  Input 3D:    {points3d_path}")
        print(f"  Calibration: {cal_path}")
        print(f"  Output:      {output_path}")

    # Load 3D tracks
    with h5py.File(points3d_path, "r") as f:
        tracks_3d = f["tracks"][:]  # (n_frames, 1, n_keypoints, 3)

    if tracks_3d.ndim != 4 or tracks_3d.shape[1] != 1 or tracks_3d.shape[3] != 3:
        print(
            f"ERROR: expected tracks shape (n_frames, 1, n_keypoints, 3), "
            f"got {tracks_3d.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    if verbose:
        n_frames, _, n_kp, _ = tracks_3d.shape
        valid = ~np.isnan(tracks_3d[:, 0]).any(axis=-1) & (tracks_3d[:, 0] != 0).any(axis=-1)
        print(f"  Loaded: {n_frames} frames × {n_kp} keypoints "
              f"({valid.sum()} valid 3D points)")

    # Load calibration
    cameras = load_calibration(cal_path)
    if args.exclude_cameras:
        exclude_set = set(args.exclude_cameras)
        cameras = {k: v for k, v in cameras.items() if k not in exclude_set}
    if verbose:
        print(f"  Cameras: {sorted(cameras.keys())}")
        print()

    # Generate and save reprojections
    generate_reprojections_h5(tracks_3d, cameras, output_path, verbose=verbose)


if __name__ == "__main__":
    main()
