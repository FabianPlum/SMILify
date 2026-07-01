#!/usr/bin/env python3
"""Diagnostic plots for the per-view depth-occlusion visibility refinement.

Loads one frame of a multi-camera replicAnt dataset twice (id-mask only vs
id-mask + depth-buffer self-occlusion) and writes per-view PNGs comparing
which keypoints survive each stage. Use this to tune `depth_tolerance_cm` /
`depth_max_cm` or to sanity-check that the surface depth lines up with the
projected keypoints.

For each camera the script produces a 1x4 panel:
  1. Original RGB + ALL in-frame keypoints (cyan).
  2. After ID-mask culling: green = kept, red X = culled.
  3. After ID + depth culling: green = kept, red X = culled by ID,
     orange X = culled by depth (passed ID but blocked by surface).
  4. Same as 3 with the depth pass (R-channel grayscale) blended at 50%.

Usage (from repo root):
    python smal_fitter/replicAnt_data/visualize_multiview_depth_occlusion.py \\
        <data_path> \\
        --smal_file 3D_model_prep/SMILy_Mouse_static_joints_Falkner_conv_repose_hind_legs.pkl \\
        --frame 100 \\
        --out_dir multiview_depth_occlusion_viz

Note: the SMAL/SMIL file must match the dataset's skeleton, otherwise the
J_names mapping in the loader produces all-zero keypoints and every panel
will be empty. The default `config.SMAL_FILE` is the ant model — pass
`--smal_file` pointing at the mouse pkl for the multi-cam mice dataset.
"""
import argparse
import os
import sys
from pathlib import Path

# Standard sys.path pattern used by sibling scripts in this directory so
# `import config` and `from neuralSMIL...` resolve when the script is run
# directly (rather than as a module).

import imageio.v2 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from smal_fitter.neuralSMIL.configs.config_utils import apply_smal_file_override
from smal_fitter.Unreal2Pytorch3D import load_SMIL_Unreal_multiview_sample


def _plot_panel(
    ax,
    image,
    depth_gray,
    px_col,
    px_row,
    in_frame,
    vis_id,
    vis_dep,
    joint_names,
    panel,
    show_legend,
    label_depth_culled,
):
    """One of the four panels for a single view.

    panel = 0 -> raw image + all in-frame keypoints (cyan)
    panel = 1 -> raw image + id-mask kept/culled
    panel = 2 -> raw image + id+depth kept/culled (id-culled red, depth-culled orange)
    panel = 3 -> depth-overlay image + id+depth kept/culled
    """
    if panel == 3:
        depth_rgb = np.stack([depth_gray] * 3, axis=-1)
        blended = (image.astype(np.float32) / 255.0) * 0.5 + depth_rgb * 0.5
        ax.imshow(np.clip(blended, 0.0, 1.0))
    else:
        ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])

    if panel == 0:
        ax.scatter(
            px_col[in_frame],
            px_row[in_frame],
            s=60,
            c="cyan",
            edgecolors="black",
            linewidths=0.7,
        )
        for j in np.where(in_frame)[0]:
            ax.annotate(
                joint_names[j],
                (px_col[j], px_row[j]),
                fontsize=6,
                color="white",
                xytext=(3, -3),
                textcoords="offset points",
            )
        return

    kept = (vis_dep == 1.0) if panel >= 2 else (vis_id == 1.0)
    culled_id = (vis_id == 0.0) & in_frame
    culled_depth_only = (vis_id == 1.0) & (vis_dep == 0.0)

    ax.scatter(
        px_col[kept],
        px_row[kept],
        s=80,
        c="lime",
        edgecolors="black",
        linewidths=0.7,
        label="kept",
    )
    ax.scatter(
        px_col[culled_id],
        px_row[culled_id],
        s=70,
        c="red",
        marker="x",
        linewidths=2.0,
        label="culled (ID)",
    )
    if panel >= 2:
        ax.scatter(
            px_col[culled_depth_only],
            px_row[culled_depth_only],
            s=90,
            c="orange",
            marker="x",
            linewidths=2.2,
            label="culled (depth)",
        )
        if label_depth_culled:
            for j in np.where(culled_depth_only)[0]:
                ax.annotate(
                    joint_names[j],
                    (px_col[j], px_row[j]),
                    fontsize=6,
                    color="orange",
                    xytext=(4, -4),
                    textcoords="offset points",
                )
    if show_legend:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


def _view_geometry(x_payload, y_payload, view_idx):
    """Pull image, depth grayscale, and pixel-space keypoints for one view."""
    image = x_payload["image_data"][view_idx]
    H, W = image.shape[:2]
    kp2d = y_payload["keypoints_2d_per_view"][view_idx]

    # The loader stores 2D keypoints in an axis-swapped normalised form
    # (norm_x = 2DPos.y / H, norm_y = 2DPos.x / W). matplotlib's imshow
    # uses (col, row) as (x, y), so col = norm_y*W, row = norm_x*H.
    px_col = kp2d[:, 1] * W
    px_row = kp2d[:, 0] * H
    # AND in-dataset so model-only joints (left at the [0, 0] sentinel by
    # the loader) don't get drawn at the image corner.
    in_dataset = y_payload["keypoint_in_dataset_per_view"][view_idx]
    in_frame = in_dataset & (px_col >= 0) & (px_col < W) & (px_row >= 0) & (px_row < H)

    depth_path = Path(x_payload["depth_paths"][view_idx])
    if depth_path.exists():
        depth_gray = iio.imread(str(depth_path))[..., 0].astype(np.float32) / 255.0
    else:
        depth_gray = np.zeros((H, W), dtype=np.float32)

    return image, depth_gray, px_col, px_row, in_frame


def visualize_frame(
    data_path: str,
    frame_index: int,
    out_dir: Path,
    depth_max_cm: float,
    depth_tolerance_cm: float,
    depth_neighborhood: int,
    write_composite: bool,
):
    """Generate the per-view (and optional composite) PNGs for one frame."""
    x_id, y_id = load_SMIL_Unreal_multiview_sample(
        data_path,
        frame_index,
        depth_occlusion_check=False,
        load_images=True,
    )
    x_d, y_d = load_SMIL_Unreal_multiview_sample(
        data_path,
        frame_index,
        depth_occlusion_check=True,
        depth_max_cm=depth_max_cm,
        depth_tolerance_cm=depth_tolerance_cm,
        depth_neighborhood=depth_neighborhood,
        load_images=True,
    )

    V = x_d["num_views"]
    joint_names = y_d["joint_names"]
    J = len(joint_names)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_titles = ("all kp", "after ID", "after ID+depth", "ID+depth, depth overlay")
    totals = {"id": 0, "depth": 0, "depth_only_culled": 0}

    # Per-view full-resolution PNGs.
    print(f"frame {frame_index}: writing {V} per-view PNGs to {out_dir}/")
    for v in range(V):
        cam_id = x_d["camera_ids"][v]
        image, depth_gray, px_col, px_row, in_frame = _view_geometry(x_d, y_d, v)
        vis_id = y_id["keypoint_visibility_per_view"][v]
        vis_dep = y_d["keypoint_visibility_per_view"][v]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
        for col, base_title in enumerate(panel_titles):
            ax = axes[col]
            counts = {
                0: f"({int(in_frame.sum())}/{J} in frame)",
                1: f"({int(vis_id.sum())} kept)",
                2: f"({int(vis_dep.sum())} kept)",
                3: "",
            }[col]
            ax.set_title(f"{base_title} {counts}".strip(), fontsize=11)
            _plot_panel(
                ax,
                image,
                depth_gray,
                px_col,
                px_row,
                in_frame,
                vis_id,
                vis_dep,
                joint_names,
                panel=col,
                show_legend=(col > 0),
                label_depth_culled=(col >= 2),
            )
        fig.suptitle(
            f"frame {frame_index} cam{cam_id}  "
            f"id={int(vis_id.sum())} -> id+depth={int(vis_dep.sum())}",
            fontsize=12,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pv_out = out_dir / f"frame_{frame_index:05d}_cam{cam_id:02d}.png"
        plt.savefig(pv_out, dpi=110, bbox_inches="tight")
        plt.close(fig)

        totals["id"] += int(vis_id.sum())
        totals["depth"] += int(vis_dep.sum())
        totals["depth_only_culled"] += int(((vis_id == 1.0) & (vis_dep == 0.0)).sum())

    # Optional composite (12 rows x 4 cols on one page).
    if write_composite:
        fig, axes = plt.subplots(V, 4, figsize=(16, 4 * V), squeeze=False)
        for v in range(V):
            cam_id = x_d["camera_ids"][v]
            image, depth_gray, px_col, px_row, in_frame = _view_geometry(x_d, y_d, v)
            vis_id = y_id["keypoint_visibility_per_view"][v]
            vis_dep = y_d["keypoint_visibility_per_view"][v]
            for col in range(4):
                ax = axes[v, col]
                ax.set_title(f"cam{cam_id}: {panel_titles[col]}", fontsize=8)
                _plot_panel(
                    ax,
                    image,
                    depth_gray,
                    px_col,
                    px_row,
                    in_frame,
                    vis_id,
                    vis_dep,
                    joint_names,
                    panel=col,
                    show_legend=(v == 0 and col > 0),
                    label_depth_culled=False,  # too cluttered at this zoom
                )
        fig.suptitle(
            f"Frame {frame_index} - id-mask vs id+depth "
            f"(totals across {V} views: id={totals['id']}  "
            f"id+depth={totals['depth']}  "
            f"newly culled by depth={totals['depth_only_culled']})",
            fontsize=11,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.995])
        comp_out = out_dir / f"frame_{frame_index:05d}_composite.png"
        plt.savefig(comp_out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  composite -> {comp_out.name}")

    # Summary table.
    print()
    print(f"  {'cam':>4} {'id_vis':>7} {'depth_vis':>10} {'culled_by_depth':>16}")
    for v in range(V):
        vid = int(y_id["keypoint_visibility_per_view"][v].sum())
        vd = int(y_d["keypoint_visibility_per_view"][v].sum())
        print(f"  {x_d['camera_ids'][v]:>4} {vid:>7} {vd:>10} {vid - vd:>16}")
    print(
        f"  totals: id={totals['id']}  id+depth={totals['depth']}  "
        f"culled_by_depth={totals['depth_only_culled']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Per-view depth-occlusion visibility diagnostic for the multi-cam "
            "replicAnt loader (load_SMIL_Unreal_multiview_sample)."
        ),
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the flat-directory multi-camera replicAnt dataset.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=100,
        help="Frame index to visualize (default: 100).",
    )
    parser.add_argument(
        "--smal_file",
        type=str,
        default=None,
        help=(
            "Path to the SMAL/SMIL .pkl file matching the dataset's skeleton. "
            "If omitted, the global config.SMAL_FILE is used. For the multi-cam "
            "mice dataset pass "
            "3D_model_prep/SMILy_Mouse_static_joints_Falkner_conv_repose_hind_legs.pkl."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="multiview_depth_occlusion_viz",
        help="Output directory for the PNGs (default: %(default)s).",
    )
    parser.add_argument(
        "--depth_max_cm",
        type=float,
        default=1000.0,
        help="Encoded range of the depth pass in cm (default: %(default)s).",
    )
    parser.add_argument(
        "--depth_tolerance_cm",
        type=float,
        default=5.0,
        help=(
            "Margin added to surface depth before declaring a joint occluded. "
            "Default 5.0 cm covers one depth-LSB (~3.92 cm @ 1000 cm range) "
            "plus ~1 cm interior offset."
        ),
    )
    parser.add_argument(
        "--depth_neighborhood",
        type=int,
        default=1,
        help="Half-window in pixels for surface min-depth sample (default: 1 = 3x3).",
    )
    parser.add_argument(
        "--no_composite",
        action="store_true",
        help="Skip the 12-row composite PNG (faster, less disk).",
    )
    args = parser.parse_args()

    if args.smal_file is not None:
        apply_smal_file_override(args.smal_file)
    print(
        f"SMAL file: {config.SMAL_FILE}  "
        f"({len(config.dd['J_names'])} joints)"
    )

    visualize_frame(
        data_path=args.data_path,
        frame_index=args.frame,
        out_dir=Path(args.out_dir),
        depth_max_cm=args.depth_max_cm,
        depth_tolerance_cm=args.depth_tolerance_cm,
        depth_neighborhood=args.depth_neighborhood,
        write_composite=not args.no_composite,
    )


if __name__ == "__main__":
    main()
