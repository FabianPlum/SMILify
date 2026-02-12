#!/usr/bin/env python3
"""
Benchmark a Multi-View SMIL model checkpoint on a multi-view SLEAP dataset.

Outputs:
  - PCK@5 (pixel threshold on original image size)
  - PCK curve over multiple thresholds
  - MPJPE in mm (after converting back to original world scale)
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
from sleap_data.sleap_multiview_dataset import SLEAPMultiViewDataset, multiview_collate_fn
from multiview_smil_regressor import create_multiview_regressor
from train_multiview_regressor import MultiViewTrainingConfig, load_checkpoint, set_random_seeds


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


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Multi-View SMIL Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to multi-view HDF5 dataset")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=None, help="Override DataLoader workers")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0 or cpu")
    parser.add_argument("--num_views_to_use", type=int, default=None, help="Override num views per sample")
    parser.add_argument("--no_random_view_sampling", action="store_true", help="Disable random view sampling")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit number of test batches")
    parser.add_argument("--orig_width", type=int, default=None, help="Override original image width (pixels)")
    parser.add_argument("--orig_height", type=int, default=None, help="Override original image height (pixels)")
    args = parser.parse_args()

    # Output directory
    ckpt_stem = _safe_stem(args.checkpoint)
    dataset_stem = _safe_stem(args.dataset_path)
    output_dir = os.path.join(
        os.getcwd(), f"benchmark_multiview_model_{ckpt_stem}_on_{dataset_stem}"
    )
    os.makedirs(output_dir, exist_ok=True)

    log_lines = []
    def log(msg: str):
        print(msg)
        log_lines.append(str(msg))

    log("=" * 60)
    log("SMILify Multi-View Benchmark")
    log("=" * 60)
    log(f"Checkpoint: {args.checkpoint}")
    log(f"Dataset: {args.dataset_path}")
    log(f"Output dir: {output_dir}")
    log(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")

    # Device setup (mirrors training behavior)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # Load checkpoint (use TrainingConfig logic from training script)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config_from_ckpt = checkpoint.get("config", {})
    if not config_from_ckpt:
        config_from_ckpt = MultiViewTrainingConfig.get_config()
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
    log("\nHDF5 INVENTORY:")
    for line in _collect_hdf5_inventory(args.dataset_path):
        log(line)

    dataset = SLEAPMultiViewDataset(
        hdf5_path=args.dataset_path,
        rotation_representation=config_from_ckpt["rotation_representation"],
        num_views_to_use=config_from_ckpt.get("num_views_to_use"),
        random_view_sampling=random_view_sampling,
    )

    log("\nDATASET SUMMARY:")
    for line in _collect_dataset_summary(dataset):
        log(line)

    # Get dataset max_views and canonical_camera_order
    dataset_max_views = dataset.get_max_views_in_dataset()
    dataset_canonical_camera_order = dataset.get_canonical_camera_order()
    log(f"\nDataset max_views: {dataset_max_views}")
    log(f"Dataset canonical camera order: {dataset_canonical_camera_order}")

    # CRITICAL: Infer max_views and canonical_camera_order from checkpoint
    # The model architecture must match the checkpoint, not the dataset.
    # The model can still handle samples with fewer views than max_views via view_mask.
    log(f"\nInferring model architecture from checkpoint...")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Infer max_views from checkpoint state dict
    if 'view_embeddings.weight' in state_dict:
        max_views = state_dict['view_embeddings.weight'].shape[0]
        log(f"Inferred max_views={max_views} from checkpoint view_embeddings.weight shape")
    else:
        # Fall back to config or dataset
        max_views = config_from_ckpt.get("max_views", dataset_max_views)
        log(f"Using max_views={max_views} from checkpoint config or dataset")

    # Get canonical_camera_order from checkpoint
    canonical_camera_order = config_from_ckpt.get("canonical_camera_order", None)
    if canonical_camera_order is None:
        # Fall back to dataset or create placeholder
        canonical_camera_order = dataset_canonical_camera_order
        if len(canonical_camera_order) != max_views:
            # Create placeholder if lengths don't match
            canonical_camera_order = [f"Camera{i}" for i in range(max_views)]
            log(f"Created placeholder canonical camera order (indices 0-{max_views-1})")
    else:
        log(f"Loaded canonical camera order from checkpoint: {canonical_camera_order}")

    log(f"Model architecture: max_views={max_views}, canonical_camera_order has {len(canonical_camera_order)} cameras")
    if max_views > dataset_max_views:
        log(f"Note: Model supports {max_views} views, dataset has up to {dataset_max_views} views")
        log(f"      Model will handle samples with fewer views via view_mask")
    elif max_views < dataset_max_views:
        log(f"WARNING: Model supports {max_views} views but dataset has up to {dataset_max_views} views")
        log(f"         Samples with >{max_views} views will be truncated")

    log(f"\nLoaded data resolution (target): {dataset.target_resolution}x{dataset.target_resolution}")
    log(f"Original world scale: {dataset.world_scale}")
    if dataset.world_scale != 0.0:
        log(f"World scale conversion factor to original units: {1.0 / dataset.world_scale:.6f}")
    override_size = None
    if args.orig_width is not None or args.orig_height is not None:
        if args.orig_width is None or args.orig_height is None:
            raise ValueError("Both --orig_width and --orig_height must be provided when overriding size.")
        override_size = (int(args.orig_height), int(args.orig_width))
        log(f"Override original image size: {args.orig_width}x{args.orig_height}")

    _log_keypoint_rescaling_info(log, dataset, override_size)

    # Data splits (mirror train_multiview_regressor.py)
    total_size = len(dataset)
    train_size = int(total_size * config_from_ckpt["train_ratio"])
    val_size = int(total_size * config_from_ckpt["val_ratio"])
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config_from_ckpt["seed"])
    )

    log("\nDataset split sizes:")
    log(f"  Train: {len(train_set)}")
    log(f"  Val: {len(val_set)}")
    log(f"  Test: {len(test_set)}")

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
    log(f"\nUsing input resolution: {input_resolution}x{input_resolution} (backbone: {backbone_name})")

    allow_mesh_scaling = config_from_ckpt.get("allow_mesh_scaling", False)
    mesh_scale_init = config_from_ckpt.get("mesh_scale_init", 1.0)
    use_gt_camera_init = config_from_ckpt.get("use_gt_camera_init", False)
    if allow_mesh_scaling:
        log(f"Mesh scaling enabled with init={mesh_scale_init}")
    if use_gt_camera_init:
        log(f"GT camera initialization enabled - model predicts deltas from GT camera params")

    # Create model with architecture from checkpoint (not dataset)
    # CRITICAL: max_views and canonical_camera_order come from checkpoint to ensure
    # model architecture matches the trained checkpoint. The model can still handle
    # samples with fewer views than max_views via view_mask.
    model = create_multiview_regressor(
        device=device,
        batch_size=config_from_ckpt["batch_size"],
        shape_family=config_from_ckpt.get("shape_family", config.SHAPE_FAMILY),
        use_unity_prior=config_from_ckpt.get("use_unity_prior", False),
        max_views=max_views,  # From checkpoint, not dataset
        canonical_camera_order=canonical_camera_order,  # From checkpoint, not dataset
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

    # Load model weights (exactly as training script)
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

            # PCK errors
            batch_errors_px = _compute_pck_errors(
                model=model,
                predicted_params=predicted_params,
                y_data_batch=y_data_batch,
                default_resolution=dataset.target_resolution,
                override_size=override_size,
                device=device,
            )
            all_2d_errors_px.extend(batch_errors_px)

            # MPJPE errors
            batch_errors_mm, batch_samples_with_3d = _compute_mpjpe_mm(
                model=model,
                predicted_params=predicted_params,
                y_data_batch=y_data_batch,
                world_scale=dataset.world_scale,
            )
            all_3d_errors_mm.extend(batch_errors_mm)
            samples_with_3d += batch_samples_with_3d

            # Collect first 5 samples with 3D data for percentile-colored plotting
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

    log("\n==== BENCHMARK RESULTS (TEST SPLIT) ====")
    log(f"PCK@5px: {pck_at_5:.4f}")
    log(f"Mean 2D error (px): {mean_2d_error:.4f}")
    log(f"Median 2D error (px): {median_2d_error:.4f}")
    log(f"MPJPE (mm): {mpjpe_mm:.4f}")
    log(f"Median MPJPE (mm): {median_mpjpe_mm:.4f}")
    if errors_mm.size > 0:
        percentiles = [50, 75, 90, 95, 99]
        pct_values = np.percentile(errors_mm, percentiles).tolist()
        log("MPJPE percentiles (mm):")
        for p, v in zip(percentiles, pct_values):
            log(f"  P{p}: {v:.4f}")
    log(f"3D samples with GT: {samples_with_3d}")
    log(f"2D joint errors count: {errors_px.size}")
    log(f"3D joint errors count: {errors_mm.size}")

    log("\nPCK curve:")
    for t, v in zip(pck_thresholds, pck_values):
        log(f"  PCK@{int(t)}px: {v:.4f}")

    # Plot first 5 3D keypoint sets colored by percentile bins
    if errors_mm.size > 0 and samples_3d_for_plot:
        percentile_thresholds = np.percentile(errors_mm, [50, 75, 90, 95, 99]).tolist()
        _plot_3d_keypoints_by_percentile(
            samples=samples_3d_for_plot,
            percentile_thresholds=percentile_thresholds,
            output_dir=output_dir,
        )

    # Save plots
    pck_plot_path = os.path.join(output_dir, "pck_curve.png")
    plt.figure(figsize=(8, 5))
    plt.plot(pck_thresholds, pck_values, marker="o")
    plt.title("PCK vs Pixel Threshold (Original Image Size)")
    plt.xlabel("Threshold (px)")
    plt.ylabel("PCK")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(pck_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    hist_plot_path = os.path.join(output_dir, "error_histogram.png")
    plt.figure(figsize=(8, 5))
    if errors_px.size > 0:
        max_err = max(50.0, float(np.max(errors_px)))
        bins = np.arange(0, max_err + 1, 1.0)
        plt.hist(errors_px, bins=bins, color="#4C72B0", alpha=0.8)
    plt.title("2D Keypoint Error Histogram (px)")
    plt.xlabel("Error (px)")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(hist_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    mpjpe_hist_path = os.path.join(output_dir, "mpjpe_histogram.png")
    plt.figure(figsize=(8, 5))
    if errors_mm.size > 0:
        max_err_mm = max(200.0, float(np.max(errors_mm)))
        bins_mm = np.arange(0, max_err_mm + 1, 1.0)
        plt.hist(errors_mm, bins=bins_mm, color="#55A868", alpha=0.8)
    plt.title("3D Joint Error Histogram (mm)")
    plt.xlabel("Error (mm)")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(mpjpe_hist_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Save raw error arrays for later comparison
    np.save(os.path.join(output_dir, "errors_2d_px.npy"), errors_px)
    np.save(os.path.join(output_dir, "errors_3d_mm.npy"), errors_mm)

    # Write report to txt
    report_path = os.path.join(output_dir, "benchmark_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(log_lines))
        f.write("\n")

    log(f"\nSaved outputs to: {output_dir}")
    log(f"  Report: {report_path}")
    log(f"  PCK plot: {pck_plot_path}")
    log(f"  Error histogram: {hist_plot_path}")
    log(f"  MPJPE histogram: {mpjpe_hist_path}")
    if errors_mm.size > 0 and samples_3d_for_plot:
        log(f"  3D percentile plots: {os.path.join(output_dir, 'sample_00_3d_keypoints_percentiles.png')} (and next 4)")


if __name__ == "__main__":
    main()
