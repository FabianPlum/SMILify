#!/usr/bin/env python3
"""Prototype: canonical-camera-frame transformation for a SLEAP multi-view sample.

Walks one sample from a SLEAPMultiViewDataset-format HDF5 through the
exact transformation a merger script would apply, and visually + numerically
verifies that the transformation preserves the camera <-> 3D <-> 2D
relationship.

What it does:

1. Loads one sample (--sample_idx, default first with has_3d_data and all
   max_views populated) from a SLEAP multi-view HDF5.
2. Sanity-projects the stored `keypoints_3d` through the stored `(K, R, t)`
   cameras and checks they reproduce the stored `keypoints_2d` (this is
   the "data is internally consistent" baseline; if this fails, nothing
   else can be trusted).
3. Computes the canonical-camera-frame transform with `view_mask`'s first
   True slot as the canonical camera. Formulas (column-vector OpenCV):
       R'_v   = R_v @ R_0.T
       t'_v   = t_v - R'_v @ t_0
       X'_w   = R_0 @ X_w + t_0          (so kp3d' = kp3d @ R_0.T + t_0)
   Cam 0 becomes (I, 0) by construction.
4. Reprojects the canonicalized 3D keypoints through the canonicalized
   cameras and compares to the stored 2D keypoints. Reprojection error
   must equal the baseline error to within numerical precision (the
   transform is rigid -> projection-invariant).
5. Plots:
     a) For each view: input image + stored 2D (green) + pre-transform
        reprojection (blue) + post-transform reprojection (red x)
     b) A 3D plot showing the original 3D keypoints + camera centres,
        and the canonical-frame 3D + camera centres after the transform.
     c) A summary panel with per-view max/mean reprojection errors and
        the canonical-camera identity check.

Usage (from repo root, inside WSL pytorch3d env):

    python smal_fitter/sleap_data/prototype_canonicalize_sleap_sample.py \\
        --hdf5 SMILymice_3D_6_cam_undistort.h5 \\
        --sample_idx 0 \\
        --output_png sleap_canonicalize_proto.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

# Single source of truth for the canonical-frame transform — used by the
# merger and by future SLEAP-preprocessor refactor work too.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))  # smal_fitter/ on path
from multiview_common.canonical_frame import (
    canonicalize_sample,
    cam_center_world,
    kp2d_norm_yx_to_pixel_xy,
    project_world_to_pixel,
)


# ---------------------------------------------------------------------------
# Loading helpers.
# ---------------------------------------------------------------------------


def _decode_jpeg_view(jpeg_bytes: np.ndarray) -> Optional[np.ndarray]:
    if jpeg_bytes is None or len(jpeg_bytes) == 0:
        return None
    bgr = cv2.imdecode(np.asarray(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _pick_sample_idx(hf: h5py.File, want_idx: Optional[int]) -> int:
    """If user gave a sample_idx, use it. Otherwise pick the first sample with
    has_3d_data=True AND all max_views slots populated (so the visualisation
    has every view filled in)."""
    if want_idx is not None:
        return int(want_idx)

    n = int(hf["metadata"].attrs["num_samples"])
    max_views = int(hf["metadata"].attrs["max_views"])
    has_3d = hf["auxiliary/has_3d_data"][:]
    view_mask = hf["multiview_images/view_mask"][:]  # (N, max_views)

    for i in range(n):
        if bool(has_3d[i]) and int(view_mask[i].sum()) == max_views:
            return i
    # Fall back: first has_3d_data sample even if some views are masked.
    for i in range(n):
        if bool(has_3d[i]):
            return i
    raise RuntimeError("No sample with has_3d_data=True in this HDF5")


def _load_sample(hdf5_path: Path, sample_idx: int):
    """Read one sample's per-view fields and 3D keypoints."""
    with h5py.File(hdf5_path, "r") as hf:
        if sample_idx < 0:
            sample_idx = _pick_sample_idx(hf, None)
        max_views = int(hf["metadata"].attrs["max_views"])
        n_joints = int(hf["metadata"].attrs["n_joints"])

        view_mask = hf["multiview_images/view_mask"][sample_idx].astype(bool)
        kp2d = hf["multiview_keypoints/keypoints_2d"][sample_idx].astype(np.float64)  # (V, J, 2) [y/H, x/W]
        kp_vis = hf["multiview_keypoints/keypoint_visibility"][sample_idx].astype(np.float64)  # (V, J)
        K = hf["multiview_keypoints/camera_intrinsics"][sample_idx].astype(np.float64)  # (V, 3, 3)
        R = hf["multiview_keypoints/camera_extrinsics_R"][sample_idx].astype(np.float64)  # (V, 3, 3)
        t = hf["multiview_keypoints/camera_extrinsics_t"][sample_idx].astype(np.float64)  # (V, 3)
        image_sizes = hf["multiview_keypoints/image_sizes"][sample_idx].astype(np.int64)  # (V, 2)  (W, H)
        kp3d = hf["multiview_keypoints/keypoints_3d"][sample_idx].astype(np.float64)  # (J, 3)
        has_3d = bool(hf["auxiliary/has_3d_data"][sample_idx])

        images = []
        for v in range(max_views):
            try:
                blob = hf[f"multiview_images/image_jpeg_view_{v}"][sample_idx]
            except KeyError:
                blob = None
            images.append(_decode_jpeg_view(blob))

        # Effective world_scale: the SLEAP preprocessor doesn't write it; the
        # reader heuristic kicks in when ||t|| > 50 and multiplies by 1e-3. We
        # apply the same heuristic for the prototype so the reprojection check
        # matches the reader's working frame.
        world_scale_attr = hf["metadata"].attrs.get("world_scale", None)
        if world_scale_attr is None:
            t_norms = np.linalg.norm(t[view_mask], axis=1)
            if t_norms.size and float(t_norms.max()) > 50.0:
                world_scale = 1.0e-3
            else:
                world_scale = 1.0
        else:
            world_scale = float(world_scale_attr)

    return {
        "sample_idx": sample_idx,
        "n_joints": n_joints,
        "max_views": max_views,
        "view_mask": view_mask,
        "kp2d_norm_yx": kp2d,
        "kp_vis": kp_vis,
        "K": K,
        "R": R,
        "t": t,
        "image_sizes": image_sizes,  # (V, 2) (W, H)
        "kp3d": kp3d,
        "has_3d": has_3d,
        "images": images,
        "world_scale": world_scale,
    }


# ---------------------------------------------------------------------------
# Reprojection comparison.
# ---------------------------------------------------------------------------


# `_kp2d_norm_yx_to_pixel_xy` previously lived here as a local helper. Promoted
# to `multiview_common.canonical_frame.kp2d_norm_yx_to_pixel_xy` so the merger,
# viewer, and this prototype share one definition.
_kp2d_norm_yx_to_pixel_xy = kp2d_norm_yx_to_pixel_xy  # back-compat alias for callers below


def per_view_reproj_error(
    sample: dict, R: np.ndarray, t: np.ndarray, kp3d: np.ndarray
) -> dict:
    """Project kp3d through (K, R, t) for each valid view, compare to stored
    2D. Returns per-view max/mean error in pixels (over visible AND has_gt_3d
    joints only).

    Note: no world_scale multiplier here. The reader scales both `t` and
    `kp3d` uniformly at load time, which is a no-op for projection accuracy
    (the z divides out). Projecting at raw units keeps the check honest.
    """
    view_mask = sample["view_mask"]
    kp_vis = sample["kp_vis"]
    image_sizes = sample["image_sizes"]
    K_all = sample["K"]
    kp2d_norm = sample["kp2d_norm_yx"]

    has_gt_3d = ~np.all(kp3d == 0, axis=1)

    per_view = {}
    for v in range(len(R)):
        if not view_mask[v]:
            continue
        W, H = int(image_sizes[v, 0]), int(image_sizes[v, 1])
        gt_pix = _kp2d_norm_yx_to_pixel_xy(kp2d_norm[v], W, H)  # (J, 2)
        pred_pix = project_world_to_pixel(kp3d, R[v], t[v], K_all[v])  # (J, 2)
        mask = has_gt_3d & (kp_vis[v] > 0)
        if not mask.any():
            per_view[v] = {"max": np.nan, "mean": np.nan, "n": 0}
            continue
        err = np.linalg.norm(pred_pix[mask] - gt_pix[mask], axis=1)
        per_view[v] = {
            "max": float(np.nanmax(err)),
            "mean": float(np.nanmean(err)),
            "n": int(mask.sum()),
        }
    return per_view


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------


# cam_center_world is imported from multiview_common.canonical_frame.
_cam_center = cam_center_world


def make_plot(
    sample: dict,
    R_orig: np.ndarray,
    t_orig: np.ndarray,
    kp3d_orig: np.ndarray,
    R_can: np.ndarray,
    t_can: np.ndarray,
    kp3d_can: np.ndarray,
    canonical_v: int,
    err_pre: dict,
    err_post: dict,
    out_png: Path,
) -> None:
    view_mask = sample["view_mask"]
    image_sizes = sample["image_sizes"]
    K = sample["K"]
    kp2d_norm = sample["kp2d_norm_yx"]
    kp_vis = sample["kp_vis"]
    images = sample["images"]
    valid_v = [int(v) for v in np.where(view_mask)[0]]
    V = len(valid_v)

    fig = plt.figure(figsize=(4 * V, 14))
    gs = fig.add_gridspec(3, max(V, 2), height_ratios=[1.2, 1.0, 0.8])

    # Row 1: per-view image overlays.
    # Note: SLEAP HDF5 stores the resized 224x224 JPEG but the calibration K +
    # 2D keypoints are in the original 1280x1024 frame. We stretch the image
    # over the calibration extent so the keypoint overlays land correctly.
    for col, v in enumerate(valid_v):
        ax = fig.add_subplot(gs[0, col])
        W, H = int(image_sizes[v, 0]), int(image_sizes[v, 1])
        img = images[v]
        if img is None:
            ax.set_facecolor("gray")
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
        else:
            ax.imshow(img, extent=(0, W, H, 0))
        gt_pix = _kp2d_norm_yx_to_pixel_xy(kp2d_norm[v], W, H)
        pre_pix = project_world_to_pixel(kp3d_orig, R_orig[v], t_orig[v], K[v])
        post_pix = project_world_to_pixel(kp3d_can, R_can[v], t_can[v], K[v])
        vis = (kp_vis[v] > 0) & ~np.all(kp3d_orig == 0, axis=1)
        ax.scatter(gt_pix[vis, 0], gt_pix[vis, 1], s=22, marker="o",
                   facecolors="none", edgecolors="lime", linewidths=1.5, label="stored 2D")
        ax.scatter(pre_pix[vis, 0], pre_pix[vis, 1], s=10, marker="s",
                   c="cornflowerblue", label="proj (original)")
        ax.scatter(post_pix[vis, 0], post_pix[vis, 1], s=24, marker="x",
                   c="red", linewidths=1.2, label="proj (canonical)")
        title = f"view {v} ({'canonical' if v == canonical_v else 'other'})\n"
        title += f"pre max={err_pre[v]['max']:.3g}px  post max={err_post[v]['max']:.3g}px"
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.legend(loc="lower right", fontsize=7)

    # Row 2: 3D views (left = original world, right = canonical frame).
    has_gt_3d = ~np.all(kp3d_orig == 0, axis=1)

    ax3d_a = fig.add_subplot(gs[1, : max(V // 2, 1)], projection="3d")
    ax3d_a.set_title("Original world frame", fontsize=10)
    ax3d_a.scatter(kp3d_orig[has_gt_3d, 0],
                   kp3d_orig[has_gt_3d, 1],
                   kp3d_orig[has_gt_3d, 2],
                   c="black", s=8, label="kp3d")
    for v in valid_v:
        c = _cam_center(R_orig[v], t_orig[v])
        col = "red" if v == canonical_v else "gray"
        ax3d_a.scatter(c[0], c[1], c[2], c=col, s=30, marker="^")
        ax3d_a.text(c[0], c[1], c[2], f" cam{v}", fontsize=7, color=col)
    ax3d_a.set_xlabel("X"); ax3d_a.set_ylabel("Y"); ax3d_a.set_zlabel("Z")

    ax3d_b = fig.add_subplot(gs[1, max(V // 2, 1):], projection="3d")
    ax3d_b.set_title(f"Canonical frame (cam {canonical_v} at origin)", fontsize=10)
    ax3d_b.scatter(kp3d_can[has_gt_3d, 0],
                   kp3d_can[has_gt_3d, 1],
                   kp3d_can[has_gt_3d, 2],
                   c="black", s=8)
    for v in valid_v:
        c = _cam_center(R_can[v], t_can[v])
        col = "red" if v == canonical_v else "gray"
        ax3d_b.scatter(c[0], c[1], c[2], c=col, s=30, marker="^")
        ax3d_b.text(c[0], c[1], c[2], f" cam{v}", fontsize=7, color=col)
    ax3d_b.set_xlabel("X"); ax3d_b.set_ylabel("Y"); ax3d_b.set_zlabel("Z")

    # Row 3: text summary.
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")
    lines = [
        f"sample_idx = {sample['sample_idx']}",
        f"max_views (HDF5)   = {sample['max_views']}",
        f"valid views        = {len(valid_v)} -> {valid_v}",
        f"canonical view     = {canonical_v}",
        f"keypoints with GT 3D = {int(has_gt_3d.sum())} / {len(kp3d_orig)}",
        "",
        "Cam canonical identity check (post-transform):",
        f"  ||R_can[{canonical_v}] - I||_max = {np.max(np.abs(R_can[canonical_v] - np.eye(3))):.3e}",
        f"  ||t_can[{canonical_v}]||_max     = {np.max(np.abs(t_can[canonical_v])):.3e}",
        "",
        "Per-view reprojection error (pixels):",
    ]
    for v in valid_v:
        a, b = err_pre[v], err_post[v]
        lines.append(
            f"  cam {v:>2}: pre max={a['max']:8.4g}, post max={b['max']:8.4g} "
            f"| pre mean={a['mean']:8.4g}, post mean={b['mean']:8.4g} (n={a['n']})"
        )
    pre_max_all = np.nanmax([err_pre[v]["max"] for v in valid_v])
    post_max_all = np.nanmax([err_post[v]["max"] for v in valid_v])
    lines.append("")
    lines.append(f"  ALL views: pre max = {pre_max_all:.4g} px, post max = {post_max_all:.4g} px")
    lines.append(f"  delta (post - pre, max) = {post_max_all - pre_max_all:+.4g} px")
    ax_txt.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=9, va="top")

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5", required=True, help="SLEAP multi-view HDF5 path")
    p.add_argument("--sample_idx", type=int, default=-1,
                   help="Sample to use (default: first with has_3d_data=True and all views populated)")
    p.add_argument("--output_png", type=str, default="sleap_canonicalize_proto.png")
    p.add_argument("--no_world_scale", action="store_true",
                   help="Skip the SLEAP reader's mm->m world_scale heuristic (debugging)")
    args = p.parse_args()

    hdf5_path = Path(args.hdf5)
    if not hdf5_path.is_file():
        print(f"HDF5 not found: {hdf5_path}", file=sys.stderr)
        sys.exit(2)

    sample = _load_sample(hdf5_path, args.sample_idx)
    if args.no_world_scale:
        sample["world_scale"] = 1.0

    print(f"Loaded sample {sample['sample_idx']} from {hdf5_path.name}")
    print(f"  max_views = {sample['max_views']}, valid = {int(sample['view_mask'].sum())}")
    print(f"  has_3d_data = {sample['has_3d']}")

    R_orig = sample["R"]
    t_orig = sample["t"]
    kp3d_orig = sample["kp3d"]

    # Baseline reprojection (no canonical transform). Operates on raw units —
    # world_scale is a uniform train-time rescale, so it doesn't affect
    # projection accuracy and we omit it here for honesty.
    err_pre = per_view_reproj_error(sample, R_orig, t_orig, kp3d_orig)

    # Canonical-frame transform.
    R_can, t_can, kp3d_can, R_0, t_0, canonical_v = canonicalize_sample(
        R_orig, t_orig, kp3d_orig, sample["view_mask"]
    )

    err_post = per_view_reproj_error(sample, R_can, t_can, kp3d_can)

    # Identity sanity.
    R_id_err = float(np.max(np.abs(R_can[canonical_v] - np.eye(3))))
    t_id_err = float(np.max(np.abs(t_can[canonical_v])))
    print(f"Canonical view = {canonical_v}")
    print(f"  ||R'_{canonical_v} - I||_max = {R_id_err:.3e}")
    print(f"  ||t'_{canonical_v}||_max     = {t_id_err:.3e}")

    print("Per-view reprojection error (pixels, on visible+has_3d joints):")
    valid_v = [int(v) for v in np.where(sample["view_mask"])[0]]
    for v in valid_v:
        a, b = err_pre[v], err_post[v]
        print(f"  cam {v:>2}: pre max={a['max']:8.4g} mean={a['mean']:8.4g} | "
              f"post max={b['max']:8.4g} mean={b['mean']:8.4g} (n={a['n']})")
    pre_max_all = float(np.nanmax([err_pre[v]["max"] for v in valid_v]))
    post_max_all = float(np.nanmax([err_post[v]["max"] for v in valid_v]))
    print(f"  ALL views: pre max={pre_max_all:.4g}px, post max={post_max_all:.4g}px, "
          f"delta={post_max_all - pre_max_all:+.4g}px")

    make_plot(sample, R_orig, t_orig, kp3d_orig, R_can, t_can, kp3d_can,
              canonical_v, err_pre, err_post, Path(args.output_png))
    print(f"Plot written to {args.output_png}")


if __name__ == "__main__":
    main()
