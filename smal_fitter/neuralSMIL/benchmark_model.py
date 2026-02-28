#!/usr/bin/env python3
"""
Benchmark a SMIL model checkpoint (single-view or multi-view) on an HDF5 dataset.

The model type is auto-detected from the checkpoint state dict:
  - If ``view_embeddings.weight`` is present → multi-view
  - Otherwise → single-view

Outputs:
  - PCK@5 (pixel threshold on original image size)
  - PCK curve over multiple thresholds
  - MPJPE in mm (after converting back to original world scale) [multi-view only, when 3D GT available]
  - Dataset stats and HDF5 key inventory
  - Plots and a text report in a dedicated output directory
"""

# Set matplotlib backend BEFORE any other imports
import matplotlib
matplotlib.use("Agg")

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

# Add parent directories to path (mirrors train_multiview_regressor.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from configs import apply_smal_file_override

# Multi-view imports (always available)
from sleap_data.sleap_multiview_dataset import SLEAPMultiViewDataset, multiview_collate_fn
from multiview_smil_regressor import create_multiview_regressor
from train_multiview_regressor import MultiViewTrainingConfig, load_checkpoint, set_random_seeds

# Single-view imports
from smil_image_regressor import SMILImageRegressor
from smil_datasets import UnifiedSMILDataset
from training_config import TrainingConfig
from train_smil_regressor import custom_collate_fn, set_random_seeds as sv_set_random_seeds


def _detect_model_type(checkpoint: dict) -> str:
    """Detect whether a checkpoint is from a multi-view or single-view model.

    Multi-view checkpoints contain ``view_embeddings.weight`` in their state
    dict; single-view checkpoints do not.
    """
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if "view_embeddings.weight" in state_dict:
        return "multiview"
    return "singleview"


def _safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.replace(" ", "_")


def _format_value(val) -> str:
    if isinstance(val, (list, tuple)):
        return ", ".join(str(v) for v in val)
    return str(val)


def _collect_hdf5_inventory(hdf5_path: str) -> List[str]:
    lines = []
    with h5py.File(hdf5_path, "r") as f:
        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                lines.append(f"DATASET: {name} | shape={obj.shape} | dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                lines.append(f"GROUP:   {name}")
                if obj.attrs:
                    for k, v in obj.attrs.items():
                        lines.append(f"  ATTR {name}.{k}: {_format_value(v)}")
        f.visititems(_visit)
    return lines


def _collect_dataset_summary(dataset: SLEAPMultiViewDataset) -> List[str]:
    info = dataset.get_dataset_info()
    lines = ["DATASET SUMMARY:"]
    for key in sorted(info.keys()):
        lines.append(f"  {key}: {_format_value(info[key])}")
    lines.append(f"  has_camera_parameters: {dataset.has_camera_parameters}")
    lines.append(f"  has_3d_keypoints: {dataset.has_3d_keypoints}")
    lines.append(f"  world_scale: {dataset.world_scale}")
    return lines


def _get_original_image_size(
    y_data: Dict,
    view_idx: int,
    default_resolution: int,
    override_size: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    if override_size is not None:
        return override_size
    # Stored as (width, height)
    if y_data.get("image_sizes") is not None:
        sz = np.array(y_data["image_sizes"][view_idx]).reshape(-1)
        if len(sz) >= 2:
            return int(sz[1]), int(sz[0])  # return (H, W)
    return int(default_resolution), int(default_resolution)


def _log_keypoint_rescaling_info(
    log_fn,
    dataset: SLEAPMultiViewDataset,
    override_size: Optional[Tuple[int, int]],
):
    if override_size is not None:
        log_fn(f"2D keypoint scaling: using override size {override_size[1]}x{override_size[0]}")
        return
    try:
        _, y0 = dataset[0]
        if y0.get("image_sizes") is not None and len(y0["image_sizes"]) > 0:
            sizes = np.array(y0["image_sizes"], dtype=np.int32)
            widths = sizes[:, 0]
            heights = sizes[:, 1]
            log_fn(
                "2D keypoint scaling: using per-view image_sizes from dataset "
                f"(W range {widths.min()}-{widths.max()}, H range {heights.min()}-{heights.max()})"
            )
        else:
            log_fn(
                "2D keypoint scaling: image_sizes missing, using target_resolution "
                f"{dataset.target_resolution}x{dataset.target_resolution}"
            )
    except Exception as e:
        log_fn(
            "2D keypoint scaling: failed to read image_sizes, using target_resolution "
            f"{dataset.target_resolution}x{dataset.target_resolution} (reason: {e})"
        )


def _collect_aspect_ratio_tensor(y_data_batch: List[Dict], view_idx: int, device: torch.device) -> Optional[torch.Tensor]:
    aspects = []
    has_any = False
    for yd in y_data_batch:
        aspect_arr = yd.get("cam_aspect_per_view")
        if aspect_arr is not None and view_idx < len(aspect_arr):
            aspect_val = float(np.array(aspect_arr[view_idx]).reshape(-1)[0])
            aspects.append(aspect_val)
            has_any = True
        else:
            aspects.append(1.0)
    if not has_any:
        return None
    return torch.tensor(aspects, device=device, dtype=torch.float32).unsqueeze(1)


def _compute_pck_errors(
    model,
    predicted_params: Dict[str, torch.Tensor],
    y_data_batch: List[Dict],
    default_resolution: int,
    override_size: Optional[Tuple[int, int]],
    device: torch.device,
) -> List[float]:
    errors_px = []
    num_views = len(predicted_params.get("fov_per_view", []))
    for v in range(num_views):
        fov_v = predicted_params["fov_per_view"][v]
        cam_rot_v = predicted_params["cam_rot_per_view"][v]
        cam_trans_v = predicted_params["cam_trans_per_view"][v]
        aspect_v = _collect_aspect_ratio_tensor(y_data_batch, v, device=device)

        rendered = model._render_keypoints_with_camera(
            predicted_params, fov_v, cam_rot_v, cam_trans_v, aspect_ratio=aspect_v
        )  # (B, J, 2) normalized

        rendered_np = rendered.detach().cpu().numpy()

        for b_idx, y_data in enumerate(y_data_batch):
            view_valid = y_data.get("view_valid")
            if view_valid is not None and v < len(view_valid) and not bool(view_valid[v]):
                continue
            kp_2d = y_data.get("keypoints_2d")
            kp_vis = y_data.get("keypoint_visibility")
            if kp_2d is None or kp_vis is None or v >= kp_2d.shape[0]:
                continue
            gt = kp_2d[v]
            vis = kp_vis[v]
            if gt is None or vis is None:
                continue

            H, W = _get_original_image_size(
                y_data, v, default_resolution=default_resolution, override_size=override_size
            )

            pred = rendered_np[b_idx]
            # Convert normalized [y, x] -> pixels using original size
            pred_y = pred[:, 0] * H
            pred_x = pred[:, 1] * W
            gt_y = gt[:, 0] * H
            gt_x = gt[:, 1] * W

            gt_zero_mask = (np.abs(gt_y) < 1e-6) & (np.abs(gt_x) < 1e-6)
            valid = np.isfinite(gt_y) & np.isfinite(gt_x) & (vis > 0.5) & (~gt_zero_mask)
            if not np.any(valid):
                continue

            dy = pred_y[valid] - gt_y[valid]
            dx = pred_x[valid] - gt_x[valid]
            dist = np.sqrt(dy * dy + dx * dx)
            errors_px.extend(dist.tolist())

    return errors_px


def _compute_mpjpe_mm(
    model,
    predicted_params: Dict[str, torch.Tensor],
    y_data_batch: List[Dict],
    world_scale: float,
) -> Tuple[List[float], int]:
    errors_mm = []
    valid_samples = 0
    pred_joints = model._predict_canonical_joints_3d(predicted_params)  # (B, J, 3)
    pred_joints_np = pred_joints.detach().cpu().numpy()

    for b_idx, y_data in enumerate(y_data_batch):
        if not bool(y_data.get("has_3d_data", False)):
            continue
        gt = y_data.get("keypoints_3d")
        if gt is None:
            continue
        gt = np.array(gt, dtype=np.float32)
        pred = pred_joints_np[b_idx]

        J = min(gt.shape[0], pred.shape[0])
        if J == 0:
            continue
        # Apply training-style masking: exclude zero joints and non-finite values
        gt_slice = gt[:J]
        pred_slice = pred[:J]
        joint_norms = np.linalg.norm(gt_slice, axis=1)
        finite_mask = np.isfinite(gt_slice).all(axis=1)
        valid_joint_mask = (joint_norms > 1e-6) & finite_mask
        if not np.any(valid_joint_mask):
            continue

        diff = pred_slice[valid_joint_mask] - gt_slice[valid_joint_mask]
        dist = np.linalg.norm(diff, axis=1)  # in scaled world units

        # Convert back to original world scale (typically mm)
        scale = 1.0 / float(world_scale) if float(world_scale) != 0.0 else 1.0
        dist_mm = dist * scale
        errors_mm.extend(dist_mm.tolist())
        valid_samples += 1

    return errors_mm, valid_samples


def _assign_percentile_bins(errors_mm: np.ndarray, thresholds: List[float]) -> np.ndarray:
    """
    Assign each error to a percentile bin index based on sorted thresholds.
    thresholds should be increasing, e.g. [P50, P75, P90, P95, P99].
    Returns bin indices in [0, len(thresholds)].
    """
    bins = np.zeros_like(errors_mm, dtype=np.int32)
    for i, t in enumerate(thresholds):
        bins = np.where(errors_mm > t, i + 1, bins)
    return bins


def _plot_3d_keypoints_by_percentile(
    samples: List[Dict],
    percentile_thresholds: List[float],
    output_dir: str,
):
    # Define colors for bins: <=P50, <=P75, <=P90, <=P95, <=P99, >P99
    bin_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
    bin_labels = ["<=P50", "<=P75", "<=P90", "<=P95", "<=P99", ">P99"]

    for idx, sample in enumerate(samples):
        gt = sample["gt"]
        pred = sample["pred"]
        err = sample["errors_mm"]
        bins = _assign_percentile_bins(err, percentile_thresholds)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        for b in range(len(bin_colors)):
            mask = bins == b
            if not np.any(mask):
                continue
            # GT points (circles)
            ax.scatter(
                gt[mask, 0], gt[mask, 1], gt[mask, 2],
                color=bin_colors[b], s=30, marker="o", label=f"{bin_labels[b]} GT", alpha=0.9
            )
            # Pred points (crosses)
            ax.scatter(
                pred[mask, 0], pred[mask, 1], pred[mask, 2],
                color=bin_colors[b], s=45, marker="x", label=f"{bin_labels[b]} Pred", alpha=0.9
            )
            # Connect GT -> Pred
            for i in np.where(mask)[0]:
                ax.plot(
                    [gt[i, 0], pred[i, 0]],
                    [gt[i, 1], pred[i, 1]],
                    [gt[i, 2], pred[i, 2]],
                    color=bin_colors[b],
                    linewidth=1.0,
                    alpha=0.6,
                )

        ax.set_title(f"GT vs Pred Keypoints (Sample {idx})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Compact legend on the right
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.25, 0.5), fontsize=8)

        plot_path = os.path.join(output_dir, f"sample_{idx:02d}_3d_keypoints_percentiles.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Single-view helpers
# ---------------------------------------------------------------------------

def _create_singleview_model(
    checkpoint: dict,
    device: torch.device,
    smal_file_override: Optional[str],
    shape_family_override: Optional[int],
    log_fn=print,
) -> Tuple[SMILImageRegressor, dict]:
    """Create and load a single-view SMILImageRegressor from *checkpoint*.

    Returns ``(model, resolved_config)`` where *resolved_config* is the flat
    dict produced by merging checkpoint config with ``TrainingConfig`` defaults.
    """
    ckpt_config = checkpoint.get("config", {})
    training_config_fallback = TrainingConfig.get_all_config()
    fallback_model = training_config_fallback["model_config"].copy()
    fallback_params = training_config_fallback["training_params"]

    if ckpt_config:
        model_config = {**fallback_model, **ckpt_config.get("model_config", {})}
        rotation_representation = (
            (ckpt_config.get("training_params") or {}).get("rotation_representation")
            or fallback_params.get("rotation_representation", "6d")
        )
        scale_trans_mode = ckpt_config.get("scale_trans_mode") or TrainingConfig.get_scale_trans_mode()
        shape_family = ckpt_config.get("shape_family", config.SHAPE_FAMILY)
    else:
        model_config = fallback_model
        rotation_representation = fallback_params["rotation_representation"]
        scale_trans_mode = TrainingConfig.get_scale_trans_mode()
        shape_family = config.SHAPE_FAMILY

    # CLI overrides
    smal_file = smal_file_override or ckpt_config.get("smal_file")
    if shape_family_override is not None:
        shape_family = shape_family_override

    if not smal_file or not os.path.exists(smal_file):
        print(
            f"ERROR: Cannot resolve SMAL model file.\n"
            f"  From checkpoint config: {ckpt_config.get('smal_file', '(not stored)')}\n"
            f"  From --smal-file arg:   {smal_file_override or '(not provided)'}\n"
            f"  Resolved path:          {smal_file or '(none)'}",
            file=sys.stderr,
        )
        sys.exit(1)
    apply_smal_file_override(smal_file, shape_family=shape_family)

    backbone_name = model_config["backbone_name"]
    input_resolution = 224 if backbone_name.startswith("vit") else 512

    log_fn(f"Singleview model config:")
    log_fn(f"  backbone: {backbone_name}")
    log_fn(f"  head_type: {model_config.get('head_type', 'mlp')}")
    log_fn(f"  rotation_representation: {rotation_representation}")
    log_fn(f"  scale_trans_mode: {scale_trans_mode}")
    log_fn(f"  shape_family: {shape_family}")
    log_fn(f"  input_resolution: {input_resolution}")

    # CRITICAL: Placeholder must be 512x512 regardless of backbone. The renderer
    # (SMALFitter) derives its image_size from data_batch.shape, and
    # _compute_rendered_outputs normalises projected joints by a hardcoded 512.
    # Training always uses 512x512 placeholder data (create_placeholder_data_batch),
    # so we must match that here to keep the rendering coordinate system consistent.
    placeholder_data = torch.zeros((1, 3, 512, 512))
    model = SMILImageRegressor(
        device=device,
        data_batch=placeholder_data,
        batch_size=1,
        shape_family=shape_family,
        use_unity_prior=model_config.get("use_unity_prior", False),
        rgb_only=model_config.get("rgb_only", True),
        freeze_backbone=model_config.get("freeze_backbone", True),
        hidden_dim=model_config.get("hidden_dim", 1024),
        use_ue_scaling=True,
        rotation_representation=rotation_representation,
        input_resolution=input_resolution,
        backbone_name=backbone_name,
        head_type=model_config.get("head_type", "mlp"),
        transformer_config=model_config.get("transformer_config", {}),
        scale_trans_mode=scale_trans_mode,
    ).to(device)

    # Load weights (filter out SMAL optimization params, same as inference script)
    state_dict = checkpoint["model_state_dict"]
    smal_optimization_params = [
        "global_rotation", "joint_rotations", "trans", "log_beta_scales",
        "betas_trans", "betas", "fov", "target_joints", "target_visibility",
    ]
    nn_state_dict = {
        k: v for k, v in state_dict.items()
        if not any(k == p or k.startswith(p + ".") for p in smal_optimization_params)
    }
    missing, unexpected = model.load_state_dict(nn_state_dict, strict=False)
    if missing:
        log_fn(f"  Missing keys (will use init): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        log_fn(f"  Unexpected keys (ignored):    {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.eval()
    log_fn(f"Loaded singleview model ({sum(p.numel() for p in model.parameters()):,} params)")

    # Build a flat resolved config dict for the benchmark loop
    resolved = {
        "model_config": model_config,
        "rotation_representation": rotation_representation,
        "scale_trans_mode": scale_trans_mode,
        "shape_family": shape_family,
        "backbone_name": backbone_name,
        "input_resolution": input_resolution,
        "batch_size": ckpt_config.get("training_params", {}).get("batch_size",
                      fallback_params.get("batch_size", 4)),
        "seed": ckpt_config.get("training_params", {}).get("seed",
                fallback_params.get("seed", 0)),
        "train_ratio": training_config_fallback.get("split_config", {}).get("train_size",
                       1.0 - training_config_fallback.get("split_config", {}).get("val_size", 0.1)
                             - training_config_fallback.get("split_config", {}).get("test_size", 0.1)),
        "val_ratio": training_config_fallback.get("split_config", {}).get("val_size", 0.1),
    }
    return model, resolved


def _compute_pck_errors_singleview(
    model: SMILImageRegressor,
    x_data_batch: list,
    y_data_batch: list,
    default_resolution: int,
    override_size: Optional[Tuple[int, int]],
) -> List[float]:
    """Compute per-joint 2D pixel errors for a single-view batch.

    Uses the model's own ``predict_from_batch`` and ``_compute_rendered_outputs``
    to obtain normalised predicted joint positions, then compares against the
    ground-truth ``keypoints_2d`` and ``keypoint_visibility`` stored in
    *y_data_batch*.
    """
    result = model.predict_from_batch(x_data_batch, y_data_batch)
    if result[0] is None:
        return []

    predicted_params, _, _ = result

    # Render predicted 2D joints (normalised [0, 1] in [y, x] order)
    rendered_joints, _, _ = model._compute_rendered_outputs(
        predicted_params, compute_joints=True, compute_silhouette=False, compute_joints_3d=False,
    )
    if rendered_joints is None:
        return []

    rendered_np = rendered_joints.detach().cpu().numpy()  # (B, J, 2)

    errors_px: List[float] = []
    for b_idx, y_data in enumerate(y_data_batch):
        gt = y_data.get("keypoints_2d")
        vis = y_data.get("keypoint_visibility")
        if gt is None or vis is None:
            continue

        gt = np.asarray(gt, dtype=np.float32)
        vis = np.asarray(vis, dtype=np.float32)
        pred = rendered_np[b_idx]

        J = min(gt.shape[0], pred.shape[0])
        if J == 0:
            continue

        gt = gt[:J]
        pred = pred[:J]
        vis = vis[:J]

        if override_size is not None:
            H, W = override_size
        else:
            H = W = default_resolution

        # Both are normalised [0, 1]; scale to pixel space
        pred_y = pred[:, 0] * H
        pred_x = pred[:, 1] * W
        gt_y = gt[:, 0] * H
        gt_x = gt[:, 1] * W

        gt_zero_mask = (np.abs(gt_y) < 1e-6) & (np.abs(gt_x) < 1e-6)
        valid = np.isfinite(gt_y) & np.isfinite(gt_x) & (vis > 0.5) & (~gt_zero_mask)
        if not np.any(valid):
            continue

        dy = pred_y[valid] - gt_y[valid]
        dx = pred_x[valid] - gt_x[valid]
        dist = np.sqrt(dy * dy + dx * dx)
        errors_px.extend(dist.tolist())

    return errors_px


def _run_singleview_benchmark(
    args,
    checkpoint: dict,
    device: torch.device,
    output_dir: str,
    log_fn,
    override_size: Optional[Tuple[int, int]],
):
    """Full benchmark loop for a single-view checkpoint."""
    model, sv_config = _create_singleview_model(
        checkpoint, device,
        smal_file_override=args.smal_file,
        shape_family_override=args.shape_family,
        log_fn=log_fn,
    )

    # Override batch size / workers from CLI
    batch_size = args.batch_size if args.batch_size is not None else sv_config["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else 4

    sv_set_random_seeds(sv_config["seed"])

    # Dataset
    log_fn("\nHDF5 INVENTORY:")
    for line in _collect_hdf5_inventory(args.dataset_path):
        log_fn(line)

    backbone_name = sv_config["backbone_name"]
    rotation_representation = sv_config["rotation_representation"]
    dataset = UnifiedSMILDataset.from_path(
        args.dataset_path,
        rotation_representation=rotation_representation,
        backbone_name=backbone_name,
    )
    target_resolution = dataset.get_target_resolution()
    log_fn(f"\nDataset size: {len(dataset)}")
    log_fn(f"Target resolution: {target_resolution}x{target_resolution}")

    _log_keypoint_rescaling_info_sv(log_fn, target_resolution, override_size)

    # Split (mirror training script)
    total_size = len(dataset)
    train_ratio = sv_config["train_ratio"]
    val_ratio = sv_config["val_ratio"]
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(sv_config["seed"]),
    )
    log_fn(f"\nDataset split sizes:")
    log_fn(f"  Train: {len(train_set)}")
    log_fn(f"  Val:   {len(val_set)}")
    log_fn(f"  Test:  {len(test_set)}")

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    pixel_scale = target_resolution
    if override_size is not None:
        pixel_scale = max(override_size)
    log_fn(f"\nPCK pixel scale: {pixel_scale}px (resolution used for error → pixel conversion)")

    # Benchmark loop
    all_2d_errors_px: List[float] = []
    with torch.no_grad():
        for batch_idx, (x_data_batch, y_data_batch) in enumerate(test_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            batch_errors = _compute_pck_errors_singleview(
                model, x_data_batch, y_data_batch,
                default_resolution=target_resolution,
                override_size=override_size,
            )
            all_2d_errors_px.extend(batch_errors)

    # Compute PCK metrics
    errors_px = np.array(all_2d_errors_px, dtype=np.float32)
    pck_thresholds = np.array([1, 2, 5, 10, 20, 30, 40, 50], dtype=np.float32)
    if errors_px.size > 0:
        pck_values = [(errors_px <= t).mean() for t in pck_thresholds]
        pck_at_5 = float((errors_px <= 5.0).mean())
        mean_2d_error = float(np.mean(errors_px))
        median_2d_error = float(np.median(errors_px))
    else:
        pck_values = [0.0 for _ in pck_thresholds]
        pck_at_5 = 0.0
        mean_2d_error = 0.0
        median_2d_error = 0.0

    log_fn("\n==== BENCHMARK RESULTS (TEST SPLIT) ====")
    log_fn(f"PCK@5px: {pck_at_5:.4f}")
    log_fn(f"Mean 2D error (px): {mean_2d_error:.4f}")
    log_fn(f"Median 2D error (px): {median_2d_error:.4f}")
    log_fn(f"2D joint errors count: {errors_px.size}")

    log_fn("\nPCK curve:")
    for t, v in zip(pck_thresholds, pck_values):
        log_fn(f"  PCK@{int(t)}px: {v:.4f}")

    # Plots
    _save_pck_plot(pck_thresholds, pck_values, output_dir)
    _save_error_histogram(errors_px, output_dir)

    # Save raw errors
    np.save(os.path.join(output_dir, "errors_2d_px.npy"), errors_px)

    log_fn(f"\nSaved outputs to: {output_dir}")
    log_fn(f"  PCK plot: {os.path.join(output_dir, 'pck_curve.png')}")
    log_fn(f"  Error histogram: {os.path.join(output_dir, 'error_histogram.png')}")

    return errors_px


def _log_keypoint_rescaling_info_sv(log_fn, target_resolution: int, override_size: Optional[Tuple[int, int]]):
    if override_size is not None:
        log_fn(f"2D keypoint scaling: using override size {override_size[1]}x{override_size[0]}")
    else:
        log_fn(
            f"2D keypoint scaling: using target_resolution "
            f"{target_resolution}x{target_resolution}"
        )


# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------

def _save_pck_plot(pck_thresholds, pck_values, output_dir: str):
    pck_plot_path = os.path.join(output_dir, "pck_curve.png")
    plt.figure(figsize=(8, 5))
    plt.plot(pck_thresholds, pck_values, marker="o")
    plt.title("PCK vs Pixel Threshold")
    plt.xlabel("Threshold (px)")
    plt.ylabel("PCK")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(pck_plot_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_error_histogram(errors_px: np.ndarray, output_dir: str):
    hist_plot_path = os.path.join(output_dir, "error_histogram.png")
    plt.figure(figsize=(8, 5))
    if errors_px.size > 0:
        max_err = max(50.0, float(np.max(errors_px)))
        bins = np.logspace(np.log10(max(0.1, float(errors_px[errors_px > 0].min()))), np.log10(max_err), 50)
        plt.hist(errors_px, bins=bins, color="#4C72B0", alpha=0.8)
    plt.xscale("log")
    plt.title("2D Keypoint Error Histogram (px)")
    plt.xlabel("Error (px)")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(hist_plot_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SMIL Model (auto-detects single-view vs multi-view)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=None, help="Override DataLoader workers")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0 or cpu")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit number of test batches")
    parser.add_argument("--orig_width", type=int, default=None, help="Override original image width (pixels)")
    parser.add_argument("--orig_height", type=int, default=None, help="Override original image height (pixels)")
    parser.add_argument("--smal-file", type=str, default=None,
                       help="Path to SMAL/SMIL model pickle. Overrides checkpoint value. "
                            "Required if checkpoint does not contain smal_file.")
    parser.add_argument("--shape-family", type=int, default=None,
                       help="Shape family index (overrides checkpoint value)")
    # Multi-view only options
    parser.add_argument("--num_views_to_use", type=int, default=None, help="Override num views per sample (multi-view only)")
    parser.add_argument("--no_random_view_sampling", action="store_true", help="Disable random view sampling (multi-view only)")
    args = parser.parse_args()

    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint and detect model type
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_type = _detect_model_type(checkpoint)

    # Output directory (includes model type for clarity)
    ckpt_stem = _safe_stem(args.checkpoint)
    dataset_stem = _safe_stem(args.dataset_path)
    output_dir = os.path.join(
        os.getcwd(), f"benchmark_{model_type}_{ckpt_stem}_on_{dataset_stem}"
    )
    os.makedirs(output_dir, exist_ok=True)

    log_lines = []
    def log(msg: str):
        print(msg)
        log_lines.append(str(msg))

    log("=" * 60)
    log(f"SMILify Benchmark ({model_type})")
    log("=" * 60)
    log(f"Model type: {model_type}")
    log(f"Checkpoint: {args.checkpoint}")
    log(f"Dataset: {args.dataset_path}")
    log(f"Output dir: {output_dir}")
    log(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    log(f"Device: {device}")

    # Parse override size (shared)
    override_size = None
    if args.orig_width is not None or args.orig_height is not None:
        if args.orig_width is None or args.orig_height is None:
            raise ValueError("Both --orig_width and --orig_height must be provided when overriding size.")
        override_size = (int(args.orig_height), int(args.orig_width))
        log(f"Override original image size: {args.orig_width}x{args.orig_height}")

    # ---------------------------------------------------------------
    # Dispatch to model-specific benchmark
    # ---------------------------------------------------------------
    if model_type == "singleview":
        _run_singleview_benchmark(args, checkpoint, device, output_dir, log, override_size)
    else:
        _run_multiview_benchmark(args, checkpoint, device, output_dir, log, override_size)

    # Write report to txt
    report_path = os.path.join(output_dir, "benchmark_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(log_lines))
        f.write("\n")

    log(f"\nReport saved to: {report_path}")


def _run_multiview_benchmark(
    args,
    checkpoint: dict,
    device: torch.device,
    output_dir: str,
    log_fn,
    override_size: Optional[Tuple[int, int]],
):
    """Full benchmark loop for a multi-view checkpoint (original behaviour)."""
    config_from_ckpt = checkpoint.get("config", {})
    if not config_from_ckpt:
        config_from_ckpt = MultiViewTrainingConfig.get_config()

    # Resolve SMAL model: CLI arg > checkpoint config > abort
    smal_file = args.smal_file or config_from_ckpt.get("smal_file")
    shape_family = args.shape_family if args.shape_family is not None else config_from_ckpt.get("shape_family")
    if not smal_file or not os.path.exists(smal_file):
        print(
            f"ERROR: Cannot resolve SMAL model file.\n"
            f"  From checkpoint config: {config_from_ckpt.get('smal_file', '(not stored)')}\n"
            f"  From --smal-file arg:   {args.smal_file or '(not provided)'}\n"
            f"  Resolved path:          {smal_file or '(none)'}\n\n"
            f"Provide a valid path via --smal-file, e.g.:\n"
            f"  python benchmark_multiview_model.py --checkpoint {args.checkpoint} "
            f"--dataset_path {args.dataset_path} --smal-file path/to/model.pkl",
            file=sys.stderr,
        )
        sys.exit(1)
    apply_smal_file_override(smal_file, shape_family=shape_family)

    config_from_ckpt["dataset_path"] = args.dataset_path
    if args.batch_size is not None:
        config_from_ckpt["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config_from_ckpt["num_workers"] = args.num_workers
    if args.num_views_to_use is not None:
        config_from_ckpt["num_views_to_use"] = args.num_views_to_use
    random_view_sampling = not args.no_random_view_sampling

    # Reproducibility
    set_random_seeds(int(config_from_ckpt.get("seed", 0)))

    # Dataset inventory and summary
    log_fn("\nHDF5 INVENTORY:")
    for line in _collect_hdf5_inventory(args.dataset_path):
        log_fn(line)

    dataset = SLEAPMultiViewDataset(
        hdf5_path=args.dataset_path,
        rotation_representation=config_from_ckpt["rotation_representation"],
        num_views_to_use=config_from_ckpt.get("num_views_to_use"),
        random_view_sampling=random_view_sampling,
    )

    log_fn("\nDATASET SUMMARY:")
    for line in _collect_dataset_summary(dataset):
        log_fn(line)

    # Get dataset max_views and canonical_camera_order
    dataset_max_views = dataset.get_max_views_in_dataset()
    dataset_canonical_camera_order = dataset.get_canonical_camera_order()
    log_fn(f"\nDataset max_views: {dataset_max_views}")
    log_fn(f"Dataset canonical camera order: {dataset_canonical_camera_order}")

    # CRITICAL: Infer max_views and canonical_camera_order from checkpoint
    log_fn(f"\nInferring model architecture from checkpoint...")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if 'view_embeddings.weight' in state_dict:
        max_views = state_dict['view_embeddings.weight'].shape[0]
        log_fn(f"Inferred max_views={max_views} from checkpoint view_embeddings.weight shape")
    else:
        max_views = config_from_ckpt.get("max_views", dataset_max_views)
        log_fn(f"Using max_views={max_views} from checkpoint config or dataset")

    canonical_camera_order = config_from_ckpt.get("canonical_camera_order", None)
    if canonical_camera_order is None:
        canonical_camera_order = dataset_canonical_camera_order
        if len(canonical_camera_order) != max_views:
            canonical_camera_order = [f"Camera{i}" for i in range(max_views)]
            log_fn(f"Created placeholder canonical camera order (indices 0-{max_views-1})")
    else:
        log_fn(f"Loaded canonical camera order from checkpoint: {canonical_camera_order}")

    log_fn(f"Model architecture: max_views={max_views}, canonical_camera_order has {len(canonical_camera_order)} cameras")
    if max_views > dataset_max_views:
        log_fn(f"Note: Model supports {max_views} views, dataset has up to {dataset_max_views} views")
        log_fn(f"      Model will handle samples with fewer views via view_mask")
    elif max_views < dataset_max_views:
        log_fn(f"WARNING: Model supports {max_views} views but dataset has up to {dataset_max_views} views")
        log_fn(f"         Samples with >{max_views} views will be truncated")

    log_fn(f"\nLoaded data resolution (target): {dataset.target_resolution}x{dataset.target_resolution}")
    log_fn(f"Original world scale: {dataset.world_scale}")
    if dataset.world_scale != 0.0:
        log_fn(f"World scale conversion factor to original units: {1.0 / dataset.world_scale:.6f}")

    _log_keypoint_rescaling_info(log_fn, dataset, override_size)

    # Data splits (mirror train_multiview_regressor.py)
    total_size = len(dataset)
    train_size = int(total_size * config_from_ckpt["train_ratio"])
    val_size = int(total_size * config_from_ckpt["val_ratio"])
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config_from_ckpt["seed"])
    )

    log_fn("\nDataset split sizes:")
    log_fn(f"  Train: {len(train_set)}")
    log_fn(f"  Val: {len(val_set)}")
    log_fn(f"  Test: {len(test_set)}")

    test_loader = DataLoader(
        test_set,
        batch_size=config_from_ckpt["batch_size"],
        shuffle=False,
        num_workers=config_from_ckpt.get("num_workers", 4),
        pin_memory=config_from_ckpt.get("pin_memory", True),
        collate_fn=multiview_collate_fn
    )

    # Create model (mirror training script)
    backbone_name = config_from_ckpt["backbone_name"]
    if backbone_name.startswith("vit"):
        input_resolution = 224
    else:
        input_resolution = 512
    log_fn(f"\nUsing input resolution: {input_resolution}x{input_resolution} (backbone: {backbone_name})")

    allow_mesh_scaling = config_from_ckpt.get("allow_mesh_scaling", False)
    mesh_scale_init = config_from_ckpt.get("mesh_scale_init", 1.0)
    use_gt_camera_init = config_from_ckpt.get("use_gt_camera_init", False)
    if allow_mesh_scaling:
        log_fn(f"Mesh scaling enabled with init={mesh_scale_init}")
    if use_gt_camera_init:
        log_fn(f"GT camera initialization enabled - model predicts deltas from GT camera params")

    model = create_multiview_regressor(
        device=device,
        batch_size=config_from_ckpt["batch_size"],
        shape_family=config_from_ckpt.get("shape_family", config.SHAPE_FAMILY),
        use_unity_prior=config_from_ckpt.get("use_unity_prior", False),
        max_views=max_views,
        canonical_camera_order=canonical_camera_order,
        cross_attention_layers=config_from_ckpt["cross_attention_layers"],
        cross_attention_heads=config_from_ckpt["cross_attention_heads"],
        cross_attention_dropout=config_from_ckpt["cross_attention_dropout"],
        backbone_name=backbone_name,
        freeze_backbone=config_from_ckpt["freeze_backbone"],
        head_type=config_from_ckpt["head_type"],
        hidden_dim=config_from_ckpt["hidden_dim"],
        rotation_representation=config_from_ckpt["rotation_representation"],
        scale_trans_mode=config_from_ckpt["scale_trans_mode"],
        use_ue_scaling=config_from_ckpt.get("use_ue_scaling", False),
        input_resolution=input_resolution,
        allow_mesh_scaling=allow_mesh_scaling,
        mesh_scale_init=mesh_scale_init,
        use_gt_camera_init=use_gt_camera_init
    )
    model = model.to(device)

    _ = load_checkpoint(args.checkpoint, model, optimizer=None, scheduler=None, device=device)
    model.eval()

    # Benchmark loop
    all_2d_errors_px = []
    all_3d_errors_mm = []
    samples_with_3d = 0
    samples_3d_for_plot = []

    with torch.no_grad():
        for batch_idx, (x_data_batch, y_data_batch) in enumerate(test_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            predicted_params, _, _ = model.predict_from_multiview_batch(
                x_data_batch, y_data_batch
            )
            pred_joints_np = model._predict_canonical_joints_3d(predicted_params).detach().cpu().numpy()

            batch_errors_px = _compute_pck_errors(
                model=model,
                predicted_params=predicted_params,
                y_data_batch=y_data_batch,
                default_resolution=dataset.target_resolution,
                override_size=override_size,
                device=device,
            )
            all_2d_errors_px.extend(batch_errors_px)

            batch_errors_mm, batch_samples_with_3d = _compute_mpjpe_mm(
                model=model,
                predicted_params=predicted_params,
                y_data_batch=y_data_batch,
                world_scale=dataset.world_scale,
            )
            all_3d_errors_mm.extend(batch_errors_mm)
            samples_with_3d += batch_samples_with_3d

            if len(samples_3d_for_plot) < 5:
                for b_idx, y_data in enumerate(y_data_batch):
                    if len(samples_3d_for_plot) >= 5:
                        break
                    if not bool(y_data.get("has_3d_data", False)):
                        continue
                    gt = y_data.get("keypoints_3d")
                    if gt is None:
                        continue
                    gt = np.array(gt, dtype=np.float32)
                    pred = pred_joints_np[b_idx].astype(np.float32)
                    J = min(gt.shape[0], pred.shape[0])
                    if J == 0:
                        continue
                    gt_slice = gt[:J]
                    pred_slice = pred[:J]
                    joint_norms = np.linalg.norm(gt_slice, axis=1)
                    finite_mask = np.isfinite(gt_slice).all(axis=1)
                    valid_joint_mask = (joint_norms > 1e-6) & finite_mask
                    if not np.any(valid_joint_mask):
                        continue

                    diff = pred_slice[valid_joint_mask] - gt_slice[valid_joint_mask]
                    dist = np.linalg.norm(diff, axis=1)
                    scale = 1.0 / float(dataset.world_scale) if float(dataset.world_scale) != 0.0 else 1.0
                    dist_mm = dist * scale
                    samples_3d_for_plot.append({
                        "gt": gt_slice[valid_joint_mask],
                        "pred": pred_slice[valid_joint_mask],
                        "errors_mm": dist_mm,
                    })

    # Compute PCK metrics
    errors_px = np.array(all_2d_errors_px, dtype=np.float32)
    pck_thresholds = np.array([1, 2, 5, 10, 20, 30, 40, 50], dtype=np.float32)
    if errors_px.size > 0:
        pck_values = [(errors_px <= t).mean() for t in pck_thresholds]
        pck_at_5 = float((errors_px <= 5.0).mean())
        mean_2d_error = float(np.mean(errors_px))
        median_2d_error = float(np.median(errors_px))
    else:
        pck_values = [0.0 for _ in pck_thresholds]
        pck_at_5 = 0.0
        mean_2d_error = 0.0
        median_2d_error = 0.0

    # Compute MPJPE
    errors_mm = np.array(all_3d_errors_mm, dtype=np.float32)
    if errors_mm.size > 0:
        mpjpe_mm = float(np.mean(errors_mm))
        median_mpjpe_mm = float(np.median(errors_mm))
    else:
        mpjpe_mm = 0.0
        median_mpjpe_mm = 0.0

    log_fn("\n==== BENCHMARK RESULTS (TEST SPLIT) ====")
    log_fn(f"PCK@5px: {pck_at_5:.4f}")
    log_fn(f"Mean 2D error (px): {mean_2d_error:.4f}")
    log_fn(f"Median 2D error (px): {median_2d_error:.4f}")
    log_fn(f"MPJPE (mm): {mpjpe_mm:.4f}")
    log_fn(f"Median MPJPE (mm): {median_mpjpe_mm:.4f}")
    if errors_mm.size > 0:
        percentiles = [50, 75, 90, 95, 99]
        pct_values = np.percentile(errors_mm, percentiles).tolist()
        log_fn("MPJPE percentiles (mm):")
        for p, v in zip(percentiles, pct_values):
            log_fn(f"  P{p}: {v:.4f}")
    log_fn(f"3D samples with GT: {samples_with_3d}")
    log_fn(f"2D joint errors count: {errors_px.size}")
    log_fn(f"3D joint errors count: {errors_mm.size}")

    log_fn("\nPCK curve:")
    for t, v in zip(pck_thresholds, pck_values):
        log_fn(f"  PCK@{int(t)}px: {v:.4f}")

    # 3D percentile plots
    if errors_mm.size > 0 and samples_3d_for_plot:
        percentile_thresholds = np.percentile(errors_mm, [50, 75, 90, 95, 99]).tolist()
        _plot_3d_keypoints_by_percentile(
            samples=samples_3d_for_plot,
            percentile_thresholds=percentile_thresholds,
            output_dir=output_dir,
        )

    # Shared plots
    _save_pck_plot(pck_thresholds, pck_values, output_dir)
    _save_error_histogram(errors_px, output_dir)

    # MPJPE histogram (multi-view only)
    mpjpe_hist_path = os.path.join(output_dir, "mpjpe_histogram.png")
    plt.figure(figsize=(8, 5))
    if errors_mm.size > 0:
        max_err_mm = max(200.0, float(np.max(errors_mm)))
        bins_mm = np.logspace(np.log10(max(0.1, float(errors_mm[errors_mm > 0].min()))), np.log10(max_err_mm), 50)
        plt.hist(errors_mm, bins=bins_mm, color="#55A868", alpha=0.8)
    plt.xscale("log")
    plt.title("3D Joint Error Histogram (mm)")
    plt.xlabel("Error (mm)")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(mpjpe_hist_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Save raw error arrays
    np.save(os.path.join(output_dir, "errors_2d_px.npy"), errors_px)
    np.save(os.path.join(output_dir, "errors_3d_mm.npy"), errors_mm)

    log_fn(f"\nSaved outputs to: {output_dir}")
    log_fn(f"  PCK plot: {os.path.join(output_dir, 'pck_curve.png')}")
    log_fn(f"  Error histogram: {os.path.join(output_dir, 'error_histogram.png')}")
    log_fn(f"  MPJPE histogram: {mpjpe_hist_path}")
    if errors_mm.size > 0 and samples_3d_for_plot:
        log_fn(f"  3D percentile plots: {os.path.join(output_dir, 'sample_00_3d_keypoints_percentiles.png')} (and next 4)")


if __name__ == "__main__":
    main()
