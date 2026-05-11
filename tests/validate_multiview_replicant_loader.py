"""Visual validation for multi-view replicAnt camera geometry.

For each view of one frame:
  1. Read raw 3D world keypoints from that camera's JSON.
  2. Read raw 2D keypoints from the same JSON.
  3. Read raw camera extrinsics (R, t) via parse_projection_components,
     and intrinsics (fx, fy, cx, cy) via parse_camera_intrinsics.
  4. Project the 3D keypoints through (R, t, K) and overlay the result
     on the camera image alongside the raw 2D ground truth.

If raw 2D (green) and projected 3D (red) coincide pixel-for-pixel, the
single-camera projection geometry is correct. This is the prerequisite
for any per-view multi-view storage convention.

This script intentionally **bypasses** `load_SMIL_Unreal_multiview_sample`
for now — the current draft hard-codes `config.ROOT_JOINT` (= 'b_t') which
doesn't exist in this dataset's joint set ('Mskel', 'Lumbar-Vertebrae', …).
Fixing that is part of the Phase 1 cleanup; once it's done, we'll extend
this script to also overlay the loader's output for end-to-end validation.

Usage:
    python tests/validate_multiview_replicant_loader.py \\
        --dataset_path /mnt/c/replicAnt-dataset-multi-cam-mice \\
        --frame_index 0 \\
        --output_dir TEST_plots/multiview_loader
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")  # No display in WSL.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "smal_fitter"))

from Unreal2Pytorch3D import (  # noqa: E402
    parse_camera_intrinsics,
    parse_projection_components,
)


def project_row_vector(X_world, R, t, fx, fy, cx, cy):
    """Project N world-space points through (R, t, K).

    Tries a small grid of (R | R.T) × (depth axis: x | y | z) × (sign: +/-)
    conventions and picks the one minimising pixel error against ground-truth
    midway through the script's loop. For exploration we return all variants;
    the caller scores them externally.
    """
    X_world = np.asarray(X_world, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    variants = {}
    for r_name, R_use in (("R", R), ("Rt", R.T)):
        X_cam_base = X_world @ R_use + t
        for axis_name, axis_idx in (("x", 0), ("y", 1), ("z", 2)):
            for dsign in (1.0, -1.0):
                depth = dsign * X_cam_base[:, axis_idx]
                other = [i for i in (0, 1, 2) if i != axis_idx]
                for usign in (1.0, -1.0):
                    for vsign in (1.0, -1.0):
                        u_world = usign * X_cam_base[:, other[0]]
                        v_world = vsign * X_cam_base[:, other[1]]
                        safe = np.where(np.abs(depth) < 1e-8, 1e-8, depth)
                        u = fx * u_world / safe + cx
                        v = fy * v_world / safe + cy
                        key = (
                            f"{r_name}_d={dsign:+.0f}{axis_name}"
                            f"_u={usign:+.0f}{'xyz'[other[0]]}"
                            f"_v={vsign:+.0f}{'xyz'[other[1]]}"
                        )
                        variants[key] = (u, v, depth)
    return variants


def list_camera_ids(dataset_path: Path, dataset_name: str, frame_index: int):
    pattern = re.compile(rf"^{re.escape(dataset_name)}_{frame_index:05d}_CAM(\d+)\.json$")
    ids = []
    for entry in dataset_path.iterdir():
        m = pattern.match(entry.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def load_per_camera(dataset_path: Path, dataset_name: str, frame_index: int, cam_id: int, batch_data_file):
    json_path = dataset_path / f"{dataset_name}_{frame_index:05d}_CAM{cam_id}.json"
    with open(json_path) as f:
        cam_data = json.load(f)
    pose_data = cam_data["iterationData"]["subject Data"][0]["1"]["keypoints"]
    names = list(pose_data.keys())

    kp3d = np.zeros((len(names), 3), dtype=np.float64)
    kp2d = np.zeros((len(names), 2), dtype=np.float64)
    for i, name in enumerate(names):
        p3 = pose_data[name].get("3DPos")
        p2 = pose_data[name].get("2DPos")
        if p3 is not None:
            kp3d[i] = [p3["x"], p3["y"], p3["z"]]
        if p2 is not None:
            kp2d[i] = [p2["x"], p2["y"]]

    R, t = parse_projection_components(cam_data)
    cx, cy, fx, fy = parse_camera_intrinsics(
        batch_data_file=batch_data_file, iteration_data_file=cam_data
    )

    image_path = json_path.with_suffix(".JPG")
    img = imageio.imread(image_path) if image_path.exists() else None

    return {
        "names": names,
        "kp3d_world": kp3d,
        "kp2d_pixels": kp2d,
        "R": R,
        "t": t,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "image": img,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--output_dir", default="TEST_plots/multiview_loader")
    parser.add_argument("--camera_indices", type=int, nargs="*", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_files = list(dataset_path.glob("_BatchData_*.json"))
    if not batch_files:
        sys.exit(f"No _BatchData_*.json in {dataset_path}")
    with open(batch_files[0]) as f:
        batch_data_file = json.load(f)
    dataset_name = batch_files[0].stem.replace("_BatchData_", "")

    cam_ids = args.camera_indices or list_camera_ids(dataset_path, dataset_name, args.frame_index)
    if not cam_ids:
        sys.exit(f"No camera JSONs found for frame {args.frame_index}")
    print(f"Frame {args.frame_index}: {len(cam_ids)} cameras: {cam_ids}")

    per_camera = {cid: load_per_camera(dataset_path, dataset_name, args.frame_index, cid, batch_data_file) for cid in cam_ids}

    n_views = len(cam_ids)
    n_cols = min(4, n_views)
    n_rows = (n_views + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    summary = []
    for view_idx, cid in enumerate(cam_ids):
        ax = axes_flat[view_idx]
        pc = per_camera[cid]
        img = pc["image"]
        if img is None:
            ax.set_title(f"CAM{cid} — no image")
            ax.axis("off")
            continue
        H, W = img.shape[:2]
        ax.imshow(img)

        # GT 2D from this camera's JSON, plotted as raw pixels (2DPos.x, 2DPos.y).
        gt = pc["kp2d_pixels"]
        valid_gt = ~np.all(gt == 0, axis=1)
        ax.scatter(gt[valid_gt, 0], gt[valid_gt, 1],
                   facecolors="none", edgecolors="lime", s=80, linewidths=1.5,
                   label="GT 2D (JSON 2DPos)")

        # Try every (R | R.T) × signed-depth-axis convention; pick the lowest-error one.
        variants = project_row_vector(
            pc["kp3d_world"], pc["R"], pc["t"], pc["fx"], pc["fy"], pc["cx"], pc["cy"]
        )
        best_conv, best_err, best_uv = None, float("inf"), None
        for name, (u, v, depth) in variants.items():
            in_front = depth > 0
            common = valid_gt & in_front
            if common.sum() < 4:
                continue
            err = np.sqrt((u[common] - gt[common, 0])**2 + (v[common] - gt[common, 1])**2).mean()
            if err < best_err:
                best_err = err
                best_conv = name
                best_uv = (u, v, depth)
        if best_uv is None:
            best_conv = "no-valid-conv"
            best_uv = (np.zeros(len(gt)), np.zeros(len(gt)), np.ones(len(gt)))
        u, v, depth = best_uv
        in_front = depth > 0
        ax.scatter(u[in_front], v[in_front],
                   marker="+", color="red", s=80, linewidths=1.5,
                   label=f"Projected 3D ({best_conv})")

        common = valid_gt & in_front
        if common.any():
            err = np.sqrt((u[common] - gt[common, 0])**2 + (v[common] - gt[common, 1])**2)
            mean_err = float(err.mean())
            max_err = float(err.max())
        else:
            mean_err = max_err = float("nan")
        summary.append((cid, best_conv, mean_err, max_err, int(in_front.sum()), int(valid_gt.sum())))

        ax.set_title(f"CAM{cid}  {best_conv}  mean={mean_err:.1f}px  max={max_err:.1f}px")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper right")

    for j in range(n_views, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"Frame {args.frame_index} — projection (red +) vs raw GT 2D (green circle)", fontsize=12)
    fig.tight_layout()
    out_path = output_dir / f"frame_{args.frame_index:05d}_reprojection_check.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved → {out_path}")

    print(f"\n{'CAM':>5}  {'conv':>30}  {'mean_px':>8}  {'max_px':>8}  {'in_front':>8}  {'gt_pts':>6}")
    for cid, conv, mean_err, max_err, in_front, gt_pts in summary:
        print(f"{cid:>5}  {conv:>30}  {mean_err:>8.2f}  {max_err:>8.2f}  {in_front:>8}  {gt_pts:>6}")


if __name__ == "__main__":
    main()
