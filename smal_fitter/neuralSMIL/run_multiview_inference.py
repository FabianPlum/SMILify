#!/usr/bin/env python3
"""
Simple multi-view inference + visualization using a preprocessed SLEAP dataset.

This script mirrors the visualization logic used during training in
`train_multiview_regressor.py`, but runs over all samples in a preprocessed
multi-view HDF5 dataset and writes two videos in the current working directory:

  - "<DATASET>_multiview_inference.mp4" (multi-view grid visualization)
  - "<DATASET>_smultiview_first_camera_render.mp4" (single-view render for view 0)
"""

# ===== CRITICAL: Force IPv4 BEFORE any other imports =====
# This prevents "Address family not supported by protocol" (errno: 97) errors
# on HPC systems that don't have full IPv6 support
import socket
_original_getaddrinfo = socket.getaddrinfo

def _getaddrinfo_ipv4_only(*args, **kwargs):
    """Force getaddrinfo to return only IPv4 results."""
    responses = _original_getaddrinfo(*args, **kwargs)
    # Filter to only IPv4 (AF_INET) results
    ipv4_responses = [r for r in responses if r[0] == socket.AF_INET]
    # If we have IPv4 results, use them; otherwise fall back to original
    return ipv4_responses if ipv4_responses else responses

socket.getaddrinfo = _getaddrinfo_ipv4_only
# ===== End IPv4 forcing =====

import argparse
import os
import re
import tempfile
import pickle
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import timedelta

import numpy as np
import cv2
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from multiview_smil_regressor import create_multiview_regressor, MultiViewSMILImageRegressor
from smil_image_regressor import rotation_6d_to_axis_angle
from smal_fitter import SMALFitter
from sleap_data.sleap_multiview_dataset import SLEAPMultiViewDataset
import config


DEFAULT_CHECKPOINTS = [
    "multiview_checkpoints/best_model.pth",
    "multiview_checkpoints/final_model.pth",
]


def is_torchrun_launched():
    """
    Check if the script was launched via torchrun/torch.distributed.launch.
    
    When launched via torchrun, environment variables RANK, LOCAL_RANK, and WORLD_SIZE
    are set automatically.
    
    Returns:
        bool: True if launched via torchrun, False otherwise
    """
    return all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])


def setup_ddp(rank: int, world_size: int, port: str = '12345', local_rank: int = None):
    """
    Initialize DDP environment with robust IPv4-only TCP store.
    
    Args:
        rank: Current process rank (global rank across all nodes)
        world_size: Total number of processes
        port: Master port for communication (default: 12345, ignored if MASTER_PORT env var is set)
        local_rank: Local rank within the node (for GPU assignment). If None, uses rank.
    """
    # Get master address and port from environment
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = int(os.environ.get('MASTER_PORT', port or '12345'))
    
    # Validate that master_addr is an IPv4 address (not a hostname)
    ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if not re.match(ipv4_pattern, master_addr):
        print(f"WARNING: MASTER_ADDR '{master_addr}' is not an IPv4 address!")
        print(f"  Attempting to resolve to IPv4...")
        try:
            # Force IPv4 resolution
            result = socket.getaddrinfo(master_addr, master_port, socket.AF_INET, socket.SOCK_STREAM)
            if result:
                master_addr = result[0][4][0]
                print(f"  Resolved to: {master_addr}")
            else:
                print(f"  ERROR: Could not resolve {master_addr} to IPv4!")
        except Exception as e:
            print(f"  ERROR resolving hostname: {e}")
    
    # Use local_rank for GPU assignment (important for multi-node setups)
    # Do this BEFORE init_process_group so NCCL binds to the correct GPU
    gpu_rank = local_rank if local_rank is not None else rank
    
    # Debug: show available CUDA devices
    if rank == 0:
        print(f"CUDA devices available: {torch.cuda.device_count()}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    torch.cuda.set_device(gpu_rank)
    
    # Initialize process group if not already initialized
    if not dist.is_initialized():
        print(f"[Rank {rank}] Initializing distributed: WORLD_SIZE={world_size}, "
              f"LOCAL_RANK/GPU={gpu_rank}, MASTER={master_addr}:{master_port}")
        
        try:
            # Create explicit TCPStore with IPv4 address to avoid IPv6 issues
            # This bypasses the default hostname resolution that can return IPv6
            is_master = (rank == 0)
            
            # Create TCP store with explicit timeout
            store = dist.TCPStore(
                host_name=master_addr,
                port=master_port,
                world_size=world_size,
                is_master=is_master,
                timeout=timedelta(seconds=1800),
                use_libuv=False  # Disable libuv to avoid potential IPv6 issues
            )
            
            # Initialize process group with explicit store (bypasses env:// which can use IPv6)
            dist.init_process_group(
                backend="nccl",
                store=store,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=1800)
            )
            print(f"[Rank {rank}] Successfully initialized NCCL process group")
            
        except Exception as e:
            print(f"Error initializing process group with NCCL + TCPStore: {e}")
            print(f"  MASTER_ADDR: {master_addr}")
            print(f"  MASTER_PORT: {master_port}")
            print(f"  RANK: {rank}, WORLD_SIZE: {world_size}")
            print(f"  LOCAL_RANK: {local_rank}, GPU_RANK: {gpu_rank}")
            
            # Try gloo backend as fallback with explicit store
            print("Attempting fallback to gloo backend with TCPStore...")
            try:
                is_master = (rank == 0)
                store = dist.TCPStore(
                    host_name=master_addr,
                    port=master_port + 1,  # Use different port for gloo
                    world_size=world_size,
                    is_master=is_master,
                    timeout=timedelta(seconds=1800),
                    use_libuv=False
                )
                dist.init_process_group(
                    backend="gloo",
                    store=store,
                    rank=rank,
                    world_size=world_size,
                    timeout=timedelta(seconds=1800)
                )
                print(f"[Rank {rank}] Successfully initialized with gloo backend!")
            except Exception as e2:
                print(f"Gloo fallback also failed: {e2}")
                raise e  # Re-raise original NCCL error


def cleanup_ddp():
    """Clean up DDP environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _find_default_checkpoint() -> Path:
    for rel_path in DEFAULT_CHECKPOINTS:
        path = Path(rel_path)
        if path.exists():
            return path
    return Path(DEFAULT_CHECKPOINTS[0])


def load_multiview_model_from_checkpoint(checkpoint_path: Path,
                                          device: str,
                                          max_views: int = None,
                                          canonical_camera_order: List[str] = None) -> MultiViewSMILImageRegressor:
    """
    Load a trained MultiViewSMILImageRegressor model from checkpoint.
    
    CRITICAL: max_views and canonical_camera_order are inferred from the checkpoint,
    not from the dataset. The model architecture (view_embeddings, camera_heads) is
    determined by max_views used during training. The model can still handle samples
    with fewer views than max_views through the view_mask mechanism.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: PyTorch device
        max_views: Optional max_views (if None, inferred from checkpoint)
        canonical_camera_order: Optional canonical camera order (if None, loaded from checkpoint)
    
    Returns:
        MultiViewSMILImageRegressor model
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    ckpt_config = checkpoint.get("config", {})
    
    # Get state dict for inferring model structure
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    backbone_name = ckpt_config.get("backbone_name", "vit_large_patch16_224")
    head_type = ckpt_config.get("head_type", "transformer_decoder")
    hidden_dim = ckpt_config.get("hidden_dim", 512)
    rotation_representation = ckpt_config.get("rotation_representation", "6d")
    scale_trans_mode = ckpt_config.get("scale_trans_mode", "separate")
    freeze_backbone = ckpt_config.get("freeze_backbone", True)
    use_unity_prior = ckpt_config.get("use_unity_prior", False)
    use_ue_scaling = ckpt_config.get("use_ue_scaling", False)
    allow_mesh_scaling = ckpt_config.get("allow_mesh_scaling", False)
    mesh_scale_init = ckpt_config.get("mesh_scale_init", 1.0)

    cross_attention_layers = ckpt_config.get("cross_attention_layers", 2)
    cross_attention_heads = ckpt_config.get("cross_attention_heads", 8)
    cross_attention_dropout = ckpt_config.get("cross_attention_dropout", 0.1)
    transformer_config = ckpt_config.get("transformer_config", {})
    use_gt_camera_init = ckpt_config.get("use_gt_camera_init", False)

    # CRITICAL: Infer max_views from checkpoint state dict to ensure model architecture matches
    # The view_embeddings.weight shape determines the number of camera positions
    if max_views is None:
        if 'view_embeddings.weight' in state_dict:
            max_views = state_dict['view_embeddings.weight'].shape[0]
            print(f"Inferred max_views={max_views} from checkpoint view_embeddings.weight shape")
        else:
            # Fall back to config or default
            max_views = ckpt_config.get("max_views", 4)
            print(f"Using max_views={max_views} from checkpoint config or default")
    
    # Get canonical_camera_order from checkpoint if not provided
    if canonical_camera_order is None:
        canonical_camera_order = ckpt_config.get("canonical_camera_order", None)
        if canonical_camera_order is None:
            # Create placeholder list - indices are what matter, not names
            canonical_camera_order = [f"Camera{i}" for i in range(max_views)]
            print(f"Created placeholder canonical camera order (indices 0-{max_views-1})")
        else:
            print(f"Loaded canonical camera order from checkpoint: {canonical_camera_order}")
    
    print(f"Model architecture: max_views={max_views}, canonical_camera_order has {len(canonical_camera_order)} cameras")
    print(f"Note: Model can handle samples with fewer views than max_views via view_mask")

    input_resolution = 224 if backbone_name.startswith("vit") else 512

    model = create_multiview_regressor(
        device=device,
        batch_size=1,
        shape_family=config.SHAPE_FAMILY,
        use_unity_prior=use_unity_prior,
        max_views=max_views,
        canonical_camera_order=canonical_camera_order,
        cross_attention_layers=cross_attention_layers,
        cross_attention_heads=cross_attention_heads,
        cross_attention_dropout=cross_attention_dropout,
        backbone_name=backbone_name,
        freeze_backbone=freeze_backbone,
        head_type=head_type,
        hidden_dim=hidden_dim,
        rotation_representation=rotation_representation,
        scale_trans_mode=scale_trans_mode,
        use_ue_scaling=use_ue_scaling,
        input_resolution=input_resolution,
        transformer_config=transformer_config,
        allow_mesh_scaling=allow_mesh_scaling,
        mesh_scale_init=mesh_scale_init,
        use_gt_camera_init=use_gt_camera_init,
    ).to(device)
    
    if use_gt_camera_init:
        print(f"  Note: Model trained with GT camera initialization - will use GT camera params as base for delta predictions")

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    smal_optimization_params = [
        "global_rotation", "joint_rotations", "trans", "log_beta_scales",
        "betas_trans", "betas", "fov", "target_joints", "target_visibility"
    ]
    nn_state_dict = {
        k: v for k, v in state_dict.items()
        if not any(k == param or k.startswith(param + ".") for param in smal_optimization_params)
    }
    model.load_state_dict(nn_state_dict, strict=False)
    model.eval()
    return model


def create_multiview_visualization(model: MultiViewSMILImageRegressor,
                                   x_data: dict,
                                   y_data: dict,
                                   device: str,
                                   disable_scaling: bool = False,
                                   disable_translation: bool = False) -> Optional[np.ndarray]:
    """
    Create multi-view grid visualization matching training visualization exactly.
    
    CRITICAL: This function must match the training visualization in train_multiview_regressor.py
    exactly, especially for PCA transformation and parameter application order.
    
    This function handles variable numbers of views per sample (same as training):
    - Grid dimensions are calculated dynamically based on actual num_views
    - view_mask is created with shape (1, num_views) where num_views is the actual number of views
    - The model's forward_multiview adapts to the actual number of views in the sample
    
    Args:
        model: MultiViewSMILImageRegressor model
        x_data: Input data dictionary
        y_data: Target data dictionary
        device: PyTorch device
        disable_scaling: If True, zero out log_beta_scales for visualization (testing/debugging)
        disable_translation: If True, zero out betas_trans for visualization (testing/debugging)
    """
    images = x_data.get("images", [])
    num_views = len(images)
    if num_views == 0:
        return None

    # Get camera indices - may have fewer entries than max_views
    cam_indices = x_data.get("camera_indices", list(range(num_views)))
    if isinstance(cam_indices, np.ndarray):
        cam_indices = cam_indices.tolist()
    
    # Ensure cam_indices matches num_views (handle cases where dataset has fewer views)
    if len(cam_indices) != num_views:
        if len(cam_indices) > num_views:
            cam_indices = cam_indices[:num_views]
        else:
            # Pad with sequential indices if needed
            cam_indices = list(cam_indices) + list(range(len(cam_indices), num_views))

    images_per_view = []
    for img in images:
        img_tensor = model.preprocess_image(img).to(device)  # (1, 3, H, W)
        images_per_view.append(img_tensor.squeeze(0))  # (3, H, W)

    images_tensors = [img.unsqueeze(0) for img in images_per_view]
    camera_indices_tensor = torch.tensor([cam_indices], device=device)
    # Create view_mask with shape (1, num_views) where num_views is the actual number of views
    # The model adapts to this number - it doesn't require max_views views
    view_mask = torch.ones(1, num_views, dtype=torch.bool, device=device)

    # Pass y_data as target_data for GT camera initialization (if model trained with use_gt_camera_init=True)
    # This ensures the model uses GT camera params as base values when predicting camera deltas
    with torch.no_grad():
        predicted_params = model.forward_multiview(images_tensors, camera_indices_tensor, view_mask, target_data=[y_data])
    
    # NOTE: We do NOT modify predicted_params here. Instead, pass disable flags to
    # create_rendered_view_with_keypoints() which will handle them by creating a local copy.

    # Calculate grid dimensions dynamically based on actual num_views (same as training)
    # This allows samples with fewer views than max_views to be visualized correctly
    img_size = 224
    margin = 5
    grid_width = num_views * img_size + (num_views + 1) * margin
    grid_height = 2 * img_size + 3 * margin

    canvas = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 40
    target_keypoints = y_data.get("keypoints_2d", None)
    target_visibility = y_data.get("keypoint_visibility", None)

    for v in range(num_views):
        x_offset = margin + v * (img_size + margin)
        input_img = images[v]
        if isinstance(input_img, np.ndarray):
            if input_img.shape[0] != img_size or input_img.shape[1] != img_size:
                from PIL import Image
                pil_img = Image.fromarray((input_img * 255).astype(np.uint8) if input_img.max() <= 1 else input_img.astype(np.uint8))
                pil_img = pil_img.resize((img_size, img_size), Image.BILINEAR)
                input_img = np.array(pil_img)
            if input_img.max() <= 1.0:
                input_img = (input_img * 255).astype(np.uint8)
            else:
                input_img = input_img.astype(np.uint8)
            if len(input_img.shape) == 2:
                input_img = np.stack([input_img] * 3, axis=-1)
            elif input_img.shape[-1] == 4:
                input_img = input_img[:, :, :3]
            canvas[margin:margin + img_size, x_offset:x_offset + img_size] = input_img

        try:
            # Extract aspect ratio for this view if available
            aspect_ratio = None
            try:
                if y_data.get("cam_aspect_per_view") is not None:
                    aspect_ratio = float(np.array(y_data["cam_aspect_per_view"][v]).reshape(-1)[0])
            except Exception:
                aspect_ratio = None
            
            rendered_img = create_rendered_view_with_keypoints(
                model, predicted_params, v,
                target_keypoints, target_visibility,
                device, img_size, aspect_ratio=aspect_ratio,
                disable_scaling=disable_scaling,
                disable_translation=disable_translation
            )
            canvas[2 * margin + img_size:2 * margin + 2 * img_size,
                   x_offset:x_offset + img_size] = rendered_img
        except Exception as e:
            print(f"Warning: Could not render view {v}: {e}")
            placeholder = np.ones((img_size, img_size, 3), dtype=np.uint8) * 128
            canvas[2 * margin + img_size:2 * margin + 2 * img_size,
                   x_offset:x_offset + img_size] = placeholder

    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
        draw.text((5, margin + img_size // 2 - 6), "Input", fill=(255, 255, 255), font=font)
        draw.text((5, 2 * margin + img_size + img_size // 2 - 6), "Pred", fill=(255, 255, 255), font=font)
        for v in range(num_views):
            x_pos = margin + v * (img_size + margin) + img_size // 2 - 10
            cam_name = x_data.get("camera_names", [f"V{v}"])[v] if v < len(x_data.get("camera_names", [])) else f"V{v}"
            draw.text((x_pos, 2), str(cam_name)[:8], fill=(255, 255, 255), font=font)
        canvas = np.array(pil_canvas)
    except Exception:
        pass

    return canvas


def create_rendered_view_with_keypoints(model: MultiViewSMILImageRegressor,
                                        predicted_params: dict,
                                        view_idx: int,
                                        target_keypoints: np.ndarray,
                                        target_visibility: np.ndarray,
                                        device: str,
                                        img_size: int,
                                        aspect_ratio: Optional[float] = None,
                                        disable_scaling: bool = False,
                                        disable_translation: bool = False) -> np.ndarray:
    """
    Create a rendered view with keypoint overlays matching training visualization exactly.
    
    CRITICAL: This function must match the training visualization in train_multiview_regressor.py
    exactly, especially for PCA transformation and parameter application order.
    
    Args:
        model: MultiViewSMILImageRegressor model
        predicted_params: Dictionary of predicted parameters (NOT modified by this function)
        view_idx: Which view to render
        target_keypoints: Ground truth keypoints
        target_visibility: Ground truth visibility
        device: PyTorch device
        img_size: Output image size
        aspect_ratio: Optional camera aspect ratio for correct projection
        disable_scaling: If True, zero out log_beta_scales for visualization (testing/debugging)
        disable_translation: If True, zero out betas_trans for visualization (testing/debugging)
    """
    # Create a modified copy of predicted_params for visualization if flags are set
    # This ensures we don't modify the original params dict
    vis_params = predicted_params
    if disable_scaling or disable_translation:
        vis_params = predicted_params.copy()
        if disable_scaling and "log_beta_scales" in vis_params:
            vis_params["log_beta_scales"] = torch.zeros_like(vis_params["log_beta_scales"])
        if disable_translation and "betas_trans" in vis_params:
            vis_params["betas_trans"] = torch.zeros_like(vis_params["betas_trans"])
    
    fov = vis_params["fov_per_view"][view_idx]
    cam_rot = vis_params["cam_rot_per_view"][view_idx]
    cam_trans = vis_params["cam_trans_per_view"][view_idx]

    pred_kps = None
    try:
        with torch.no_grad():
            # Convert aspect_ratio to tensor if provided
            aspect_tensor = None
            if aspect_ratio is not None:
                aspect_tensor = torch.tensor([aspect_ratio], dtype=torch.float32, device=device)
            
            rendered_joints = model._render_keypoints_with_camera(
                vis_params, fov, cam_rot, cam_trans, aspect_ratio=aspect_tensor
            )
        pred_kps = rendered_joints[0].detach().cpu().numpy()
        pred_kps = pred_kps * img_size
    except Exception as e:
        print(f"Keypoint rendering failed: {e}")

    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 60
    for i in range(img_size):
        img[i, :, 0] = min(255, 60 + view_idx * 30)

    gt_kps = None
    gt_vis = None
    if target_keypoints is not None:
        if len(target_keypoints.shape) == 3:
            if view_idx < target_keypoints.shape[0]:
                gt_kps = target_keypoints[view_idx] * img_size
                if target_visibility is not None and view_idx < target_visibility.shape[0]:
                    gt_vis = target_visibility[view_idx]
        elif len(target_keypoints.shape) == 2 and view_idx == 0:
            gt_kps = target_keypoints * img_size
            gt_vis = target_visibility

    from PIL import Image, ImageDraw
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    if gt_kps is not None:
        for j, (y, x) in enumerate(gt_kps):
            if gt_vis is None or gt_vis[j] > 0.5:
                x, y = float(x), float(y)
                if 0 <= x < img_size and 0 <= y < img_size:
                    draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline="green", width=2)

    if pred_kps is not None:
        for j, (y, x) in enumerate(pred_kps):
            x, y = float(x), float(y)
            if 0 <= x < img_size and 0 <= y < img_size:
                draw.line([x - 4, y, x + 4, y], fill="red", width=2)
                draw.line([x, y - 4, x, y + 4], fill="red", width=2)

    return np.array(pil_img)


class _InMemoryImageExporter:
    def __init__(self):
        self.image = None

    def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces, img_idx=0):
        self.image = collage_np


def render_singleview_collage(model: MultiViewSMILImageRegressor,
                              x_data: dict,
                              y_data: dict,
                              device: str,
                              view_idx: int = 0,
                              disable_scaling: bool = False,
                              disable_translation: bool = False) -> Optional[np.ndarray]:
    """
    Render single-view mesh visualization matching training visualization exactly.
    
    CRITICAL: This function must match the training visualization in train_multiview_regressor.py
    exactly, especially for PCA transformation and parameter application order.
    
    This function handles variable numbers of views per sample (same as training):
    - Works correctly even if the sample has fewer views than max_views used during training
    - view_mask is created with shape (1, num_views) where num_views is the actual number of views
    
    Operation order (must match training):
    1. Create SMALFitter
    2. Set propagate_scaling to match model
    3. Set body parameters (global_rot, joint_rot, betas, trans, fov)
    4. Set scales/trans (with PCA transformation if in separate mode with PCA)
    5. Set camera parameters (R, T, fov, aspect_ratio)
    6. Generate visualization
    
    Args:
        model: MultiViewSMILImageRegressor model
        x_data: Input data dictionary
        y_data: Target data dictionary
        device: PyTorch device
        view_idx: Which view to render (must be < num_views for this sample)
        disable_scaling: If True, skip applying log_beta_scales (testing/debugging)
        disable_translation: If True, skip applying betas_trans (testing/debugging)
    """
    images = x_data.get("images", [])
    num_views = len(images)
    if num_views == 0 or view_idx >= num_views:
        return None

    # Get camera indices - may have fewer entries than max_views
    cam_indices = x_data.get("camera_indices", list(range(num_views)))
    if isinstance(cam_indices, np.ndarray):
        cam_indices = cam_indices.tolist()
    
    # Ensure cam_indices matches num_views (handle cases where dataset has fewer views)
    if len(cam_indices) != num_views:
        if len(cam_indices) > num_views:
            cam_indices = cam_indices[:num_views]
        else:
            # Pad with sequential indices if needed
            cam_indices = list(cam_indices) + list(range(len(cam_indices), num_views))

    images_per_view = []
    for img in images:
        img_tensor = model.preprocess_image(img).to(device)
        images_per_view.append(img_tensor.squeeze(0))

    images_tensors = [img.unsqueeze(0) for img in images_per_view]
    camera_indices_tensor = torch.tensor([cam_indices], device=device)
    # Create view_mask with shape (1, num_views) where num_views is the actual number of views
    # The model adapts to this number - it doesn't require max_views views
    view_mask = torch.ones(1, num_views, dtype=torch.bool, device=device)

    # Pass y_data as target_data for GT camera initialization (if model trained with use_gt_camera_init=True)
    # This ensures the model uses GT camera params as base values when predicting camera deltas
    with torch.no_grad():
        predicted_params = model.forward_multiview(images_tensors, camera_indices_tensor, view_mask, target_data=[y_data])

    fov_per_view = predicted_params.get("fov_per_view", None)
    cam_rot_per_view = predicted_params.get("cam_rot_per_view", None)
    cam_trans_per_view = predicted_params.get("cam_trans_per_view", None)

    target_size = int(getattr(model.renderer, "image_size", 224))

    original_image = images[view_idx]
    from PIL import Image
    pil_img = Image.fromarray((original_image * 255).astype(np.uint8))
    pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
    resized_image = np.array(pil_img).astype(np.float32) / 255.0

    resized_image = np.clip(resized_image, 0.0, 1.0)
    resized_image_bgr = resized_image[:, :, [2, 1, 0]]
    rgb = torch.from_numpy(resized_image_bgr).permute(2, 0, 1).unsqueeze(0).float()

    keypoints_2d = y_data.get("keypoints_2d", None)
    visibility = y_data.get("keypoint_visibility", None)

    view_keypoints = None
    view_visibility = None
    if keypoints_2d is not None:
        if len(keypoints_2d.shape) == 3:
            view_keypoints = keypoints_2d[view_idx] if view_idx < keypoints_2d.shape[0] else None
            if visibility is not None and view_idx < visibility.shape[0]:
                view_visibility = visibility[view_idx]
        else:
            view_keypoints = keypoints_2d if view_idx == 0 else None
            view_visibility = visibility if view_idx == 0 else None

    sil = torch.zeros(1, 1, target_size, target_size)
    if view_keypoints is not None and view_visibility is not None:
        pixel_coords = view_keypoints.copy()
        pixel_coords[:, 0] = pixel_coords[:, 0] * target_size
        pixel_coords[:, 1] = pixel_coords[:, 1] * target_size
        num_joints = len(view_keypoints)
        joints = torch.tensor(pixel_coords.reshape(1, num_joints, 2), dtype=torch.float32)
        vis = torch.tensor(view_visibility.reshape(1, num_joints), dtype=torch.float32)
        temp_batch = (rgb, sil, joints, vis)
        rgb_only = False
    else:
        temp_batch = rgb
        rgb_only = True

    temp_fitter = SMALFitter(
        device=device,
        data_batch=temp_batch,
        batch_size=1,
        shape_family=config.SHAPE_FAMILY,
        use_unity_prior=False,
        rgb_only=rgb_only,
    )
    
    # CRITICAL: Match propagate_scaling to the training model's setting.
    # The model learns scales with propagate_scaling=True (set in SMILImageRegressor.__init__),
    # so visualization must also use propagate_scaling=True for consistent geometry.
    temp_fitter.propagate_scaling = model.propagate_scaling

    if view_keypoints is not None and view_visibility is not None:
        pixel_coords = view_keypoints.copy()
        pixel_coords[:, 0] = pixel_coords[:, 0] * target_size
        pixel_coords[:, 1] = pixel_coords[:, 1] * target_size
        temp_fitter.target_joints = torch.tensor(pixel_coords, dtype=torch.float32, device=device).unsqueeze(0)
        temp_fitter.target_visibility = torch.tensor(view_visibility, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        n_joints = temp_fitter.joint_rotations.shape[1] + 1
        temp_fitter.target_joints = torch.zeros((1, n_joints, 2), device=device)
        temp_fitter.target_visibility = torch.zeros((1, n_joints), device=device)

    if model.rotation_representation == "6d":
        global_rot_aa = rotation_6d_to_axis_angle(predicted_params["global_rot"][0:1].detach())
        joint_rot_aa = rotation_6d_to_axis_angle(predicted_params["joint_rot"][0:1].detach())
    else:
        global_rot_aa = predicted_params["global_rot"][0:1].detach()
        joint_rot_aa = predicted_params["joint_rot"][0:1].detach()

    temp_fitter.global_rotation.data = global_rot_aa.to(device)
    temp_fitter.joint_rotations.data = joint_rot_aa.to(device)
    temp_fitter.betas.data = predicted_params["betas"][0].detach().to(device)
    temp_fitter.trans.data = predicted_params["trans"][0:1].detach().to(device)

    if fov_per_view is not None and view_idx < len(fov_per_view):
        fov_val = fov_per_view[view_idx][0, 0].detach().to(device)
        temp_fitter.fov.data = fov_val.unsqueeze(0)
    elif "fov" in predicted_params:
        temp_fitter.fov.data = predicted_params["fov"][0:1].detach().to(device)

    # Set scale and translation parameters if available
    #
    # IMPORTANT: Scales ARE applied during training loss computation in `_predict_canonical_joints_3d`
    # and `_render_keypoints_with_camera` for 'separate' mode. The model learns scales implicitly
    # through 2D/3D keypoint supervision. We must apply them here to match training behavior.
    #
    # CRITICAL: This section must match training's render_singleview_for_sample() exactly:
    # - Same PCA transformation logic
    # - Same shape handling: (1, N_BETAS) input → (1, n_joints, 3) output
    # - Same order: transform PCA → apply scales → apply trans (if not disabled)
    if "log_beta_scales" in predicted_params and "betas_trans" in predicted_params:
        if model.scale_trans_mode == "ignore":
            # 'ignore' mode: scales/translations are not used
            pass
        elif model.scale_trans_mode == "separate":
            # 'separate' mode: check if using PCA or per-joint values
            from training_config import TrainingConfig
            scale_trans_config = TrainingConfig.get_scale_trans_config()
            use_pca_transformation = scale_trans_config.get('separate', {}).get('use_pca_transformation', True)
            
            scales = predicted_params["log_beta_scales"][0:1].detach()
            trans = predicted_params["betas_trans"][0:1].detach()
            
            if use_pca_transformation:
                # PCA weights - convert to per-joint values for SMALFitter
                # This matches training exactly: scales is (1, N_BETAS), result is (1, n_joints, 3)
                try:
                    scales, trans = model._transform_separate_pca_weights_to_joint_values(scales, trans)
                except Exception as e:
                    print(f"Warning: Failed to convert PCA limb scales for visualization: {e}")
                    # Fall back to not applying scales
                    scales = None
                    trans = None
            # else: Already per-joint values (batch_size, n_joints, 3) - use directly
            
            # Apply scales and translations independently based on disable flags
            if not disable_scaling and scales is not None:
                temp_fitter.log_beta_scales.data = scales.to(device)
            if not disable_translation and trans is not None:
                temp_fitter.betas_trans.data = trans.to(device)
        else:
            # 'entangled_with_betas' mode: values are already per-joint
            if not disable_scaling:
                temp_fitter.log_beta_scales.data = predicted_params["log_beta_scales"][0:1].detach().to(device)
            if not disable_translation:
                temp_fitter.betas_trans.data = predicted_params["betas_trans"][0:1].detach().to(device)

    if cam_rot_per_view is not None and cam_trans_per_view is not None and view_idx < len(cam_rot_per_view):
        cam_rot = cam_rot_per_view[view_idx][0:1].detach().to(device)
        cam_trans = cam_trans_per_view[view_idx][0:1].detach().to(device)
        if fov_per_view is not None and view_idx < len(fov_per_view):
            view_fov_val = fov_per_view[view_idx][0, 0].detach().to(device)
            view_fov = view_fov_val.unsqueeze(0)
            temp_fitter.fov.data = view_fov
        else:
            view_fov = temp_fitter.fov.data

        aspect = None
        try:
            if y_data.get("cam_aspect_per_view") is not None:
                aspect = float(np.array(y_data["cam_aspect_per_view"][view_idx]).reshape(-1)[0])
        except Exception:
            aspect = None

        temp_fitter.renderer.set_camera_parameters(
            R=cam_rot, T=cam_trans, fov=view_fov, aspect_ratio=aspect
        )

    exporter = _InMemoryImageExporter()
    vis_mesh_scale = None
    if model.allow_mesh_scaling and "mesh_scale" in predicted_params:
        vis_mesh_scale = predicted_params["mesh_scale"][0:1].detach()
    temp_fitter.generate_visualization(
        exporter,
        apply_UE_transform=model.use_ue_scaling,  # MUST match model setting for consistency with 3D keypoints!
        img_idx=view_idx,
        mesh_scale=vis_mesh_scale,
    )
    return exporter.image


def _pad_or_resize(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_size
    if frame.shape[1] == target_w and frame.shape[0] == target_h:
        return frame
    if frame.shape[1] > target_w or frame.shape[0] > target_h:
        return cv2.resize(frame, (target_w, target_h))
    padded = np.ones((target_h, target_w, 3), dtype=np.uint8) * 40
    h = min(target_h, frame.shape[0])
    w = min(target_w, frame.shape[1])
    padded[:h, :w] = frame[:h, :w]
    return padded


def process_dataset_portion(
    dataset: SLEAPMultiViewDataset,
    model: MultiViewSMILImageRegressor,
    device: str,
    rank: int,
    world_size: int,
    sampler: Optional[DistributedSampler],
    grid_width: int,
    grid_height: int,
    singleview_size: Tuple[int, int],
    max_frames: Optional[int] = None,
    disable_scaling: bool = False,
    disable_translation: bool = False,
    view_indices: List[int] = None,
) -> Tuple[List[np.ndarray], Dict[int, List[np.ndarray]], List[int], Dict[int, List[int]]]:
    """
    Process a portion of the dataset on a single GPU.
    
    Args:
        dataset: The multi-view dataset
        model: The trained model
        device: Device string
        rank: Process rank
        world_size: Total number of processes
        sampler: Optional distributed sampler
        grid_width: Width of multiview grid
        grid_height: Height of multiview grid
        singleview_size: Size of singleview frames
        max_frames: Maximum total frames to process across all ranks (None = all)
        disable_scaling: If True, skip applying part scaling for comparison/debugging
        disable_translation: If True, skip applying part translation for comparison/debugging
        view_indices: List of camera view indices to render for singleview output (default: [0])
    
    Returns:
        Tuple of (multiview_frames, singleview_frames_per_view, mv_frame_indices, sv_frame_indices_per_view)
        - multiview_frames: List of multiview grid frames
        - singleview_frames_per_view: Dict mapping view_idx -> list of frames
        - mv_frame_indices: List of dataset indices for multiview frames
        - sv_frame_indices_per_view: Dict mapping view_idx -> list of dataset indices
    """
    if view_indices is None:
        view_indices = [0]
    
    multiview_frames = []
    mv_frame_indices = []
    
    # Initialize per-view storage
    singleview_frames_per_view: Dict[int, List[np.ndarray]] = {v: [] for v in view_indices}
    sv_frame_indices_per_view: Dict[int, List[int]] = {v: [] for v in view_indices}
    
    # Set sampler epoch for distributed processing
    if sampler is not None:
        sampler.set_epoch(0)
    
    # Get indices for this rank
    # DistributedSampler divides the dataset into chunks based on rank
    if sampler is not None:
        # Compute which indices belong to this rank
        # DistributedSampler logic: each rank gets dataset[rank::world_size]
        indices = list(range(rank, len(dataset), world_size))
    else:
        # Single GPU: process all samples
        indices = list(range(len(dataset)))
    
    # Apply max_frames limit (distributed across ranks)
    if max_frames is not None:
        # Each rank processes max_frames / world_size frames
        frames_per_rank = max(1, max_frames // world_size)
        indices = indices[:frames_per_rank]
        if rank == 0:
            print(f"Limiting to {max_frames} total frames ({frames_per_rank} per rank)")
    
    model.eval()
    
    for local_idx, global_idx in enumerate(indices):
        try:
            x_data, y_data = dataset[global_idx]
            num_available_views = len(x_data.get("images", []))
            
            # Skip samples with no views
            if num_available_views == 0:
                if rank == 0 and local_idx == 0:
                    print(f"Warning: Sample {global_idx} has no views, skipping")
                continue
            
            # Process multiview visualization
            # The visualization function handles variable numbers of views correctly:
            # - Grid dimensions are calculated based on actual num_views
            # - view_mask is created with shape (1, num_views) where num_views is actual
            # - Model adapts to the actual number of views
            mv_frame = create_multiview_visualization(
                model, x_data, y_data, device,
                disable_scaling=disable_scaling,
                disable_translation=disable_translation
            )
            if mv_frame is not None:
                # Pad or resize to consistent size for video output
                # Samples with fewer views will be padded to max_views grid size
                mv_frame = _pad_or_resize(mv_frame, (grid_width, grid_height))
                mv_bgr = cv2.cvtColor(mv_frame, cv2.COLOR_RGB2BGR)
                multiview_frames.append(mv_bgr)
                mv_frame_indices.append(global_idx)
            
            # Process singleview visualization for each requested view
            for view_idx in view_indices:
                if view_idx >= num_available_views:
                    # Skip if this view doesn't exist in the sample
                    # This handles cases where samples have fewer views than requested
                    continue
                sv_frame = render_singleview_collage(
                    model, x_data, y_data, device,
                    view_idx=view_idx,
                    disable_scaling=disable_scaling,
                    disable_translation=disable_translation
                )
                if sv_frame is not None:
                    sv_frame = _pad_or_resize(sv_frame, singleview_size)
                    sv_bgr = cv2.cvtColor(sv_frame, cv2.COLOR_RGB2BGR)
                    singleview_frames_per_view[view_idx].append(sv_bgr)
                    sv_frame_indices_per_view[view_idx].append(global_idx)
            
            if (local_idx + 1) % 100 == 0:
                print(f"[Rank {rank}] Processed {local_idx + 1}/{len(indices)} samples")
                
        except Exception as e:
            print(f"[Rank {rank}] Error processing sample {global_idx}: {e}")
            continue
    
    total_sv_frames = sum(len(frames) for frames in singleview_frames_per_view.values())
    print(f"[Rank {rank}] Completed processing {len(multiview_frames)} multiview frames, {total_sv_frames} singleview frames across {len(view_indices)} views")
    return multiview_frames, singleview_frames_per_view, mv_frame_indices, sv_frame_indices_per_view


def write_frames_to_temp_storage(
    multiview_frames: List[np.ndarray],
    singleview_frames_per_view: Dict[int, List[np.ndarray]],
    mv_frame_indices: List[int],
    sv_frame_indices_per_view: Dict[int, List[int]],
    temp_dir: Path,
    rank: int,
) -> Path:
    """
    Write frames to temporary storage on disk to avoid memory issues with all_gather.
    
    Each rank writes its frames to individual image files and a manifest pickle file.
    
    Args:
        multiview_frames: List of multiview grid frames
        singleview_frames_per_view: Dict mapping view_idx -> list of singleview frames
        mv_frame_indices: Original dataset indices for multiview frames
        sv_frame_indices_per_view: Dict mapping view_idx -> list of dataset indices
        temp_dir: Directory to store temporary files
        rank: Current process rank
        
    Returns:
        Path to rank directory containing manifests
    """
    rank_dir = temp_dir / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    
    mv_manifest = []
    
    # Write multiview frames
    for i, (frame, idx) in enumerate(zip(multiview_frames, mv_frame_indices)):
        frame_path = rank_dir / f"mv_{i:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        mv_manifest.append((idx, str(frame_path)))
    
    # Write singleview frames per view
    sv_manifests_per_view = {}
    for view_idx, frames in singleview_frames_per_view.items():
        indices = sv_frame_indices_per_view[view_idx]
        sv_manifest = []
        for i, (frame, idx) in enumerate(zip(frames, indices)):
            frame_path = rank_dir / f"sv_view{view_idx}_{i:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            sv_manifest.append((idx, str(frame_path)))
        sv_manifests_per_view[view_idx] = sv_manifest
    
    # Write manifests
    mv_manifest_path = rank_dir / "mv_manifest.pkl"
    sv_manifest_path = rank_dir / "sv_manifests_per_view.pkl"
    
    with open(mv_manifest_path, 'wb') as f:
        pickle.dump(mv_manifest, f)
    with open(sv_manifest_path, 'wb') as f:
        pickle.dump(sv_manifests_per_view, f)
    
    total_sv = sum(len(m) for m in sv_manifests_per_view.values())
    print(f"[Rank {rank}] Wrote {len(mv_manifest)} multiview frames and {total_sv} singleview frames to {rank_dir}")
    
    return rank_dir


def merge_frames_and_write_videos(
    temp_dir: Path,
    world_size: int,
    multiview_out: Path,
    singleview_out_base: Path,
    fps: int,
    grid_width: int,
    grid_height: int,
    singleview_size: Tuple[int, int],
    view_indices: List[int],
):
    """
    Merge frames from all ranks and write final videos (called only by rank 0).
    
    Reads manifests from all ranks, sorts frames by original index, and writes videos.
    Creates separate singleview videos for each view index.
    """
    print(f"\nMerging frames from {world_size} ranks...")
    
    # Collect all manifests
    all_mv_entries = []
    all_sv_entries_per_view: Dict[int, List] = {v: [] for v in view_indices}
    
    for rank_idx in range(world_size):
        rank_dir = temp_dir / f"rank_{rank_idx}"
        mv_manifest_path = rank_dir / "mv_manifest.pkl"
        sv_manifest_path = rank_dir / "sv_manifests_per_view.pkl"
        
        if mv_manifest_path.exists():
            with open(mv_manifest_path, 'rb') as f:
                all_mv_entries.extend(pickle.load(f))
        
        if sv_manifest_path.exists():
            with open(sv_manifest_path, 'rb') as f:
                sv_manifests = pickle.load(f)
                for view_idx, entries in sv_manifests.items():
                    if view_idx in all_sv_entries_per_view:
                        all_sv_entries_per_view[view_idx].extend(entries)
    
    # Sort by original dataset index
    all_mv_entries.sort(key=lambda x: x[0])
    for view_idx in all_sv_entries_per_view:
        all_sv_entries_per_view[view_idx].sort(key=lambda x: x[0])
    
    print(f"Writing {len(all_mv_entries)} multiview frames...")
    
    # Write multiview video
    if len(all_mv_entries) > 0:
        multiview_writer = cv2.VideoWriter(
            str(multiview_out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (grid_width, grid_height),
        )
        for _, frame_path in all_mv_entries:
            frame = cv2.imread(frame_path)
            if frame is not None:
                multiview_writer.write(frame)
        multiview_writer.release()
        print(f"Wrote {multiview_out}")
    
    # Write singleview video for each view
    for view_idx in view_indices:
        sv_entries = all_sv_entries_per_view.get(view_idx, [])
        if len(sv_entries) > 0:
            # Create output path with view index
            if len(view_indices) == 1:
                # Single view: use original naming
                singleview_out = singleview_out_base
            else:
                # Multiple views: add view index to filename
                stem = singleview_out_base.stem
                singleview_out = singleview_out_base.parent / f"{stem}_view{view_idx}.mp4"
            
            singleview_writer = cv2.VideoWriter(
                str(singleview_out),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                singleview_size,
            )
            for _, frame_path in sv_entries:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    singleview_writer.write(frame)
            singleview_writer.release()
            print(f"Wrote {singleview_out} ({len(sv_entries)} frames)")


def main_inference(
    args,
    rank: int = 0,
    world_size: int = 1,
    device_override: Optional[str] = None,
):
    """
    Main inference function that can run in single-GPU or multi-GPU mode.
    
    Args:
        args: Parsed command line arguments
        rank: Process rank (0 for single-GPU)
        world_size: Total number of processes (1 for single-GPU)
        device_override: Optional device string override
    """
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Set device
    if device_override:
        device = device_override
    else:
        if world_size > 1:
            # Multi-GPU: use rank-specific device
            device = f"cuda:{rank}"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Parse view indices from comma-separated string
    view_indices = [int(x.strip()) for x in args.view_indices.split(",")]
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("MULTI-VIEW INFERENCE")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Dataset: {dataset_path}")
        if args.max_frames is not None:
            print(f"Max frames: {args.max_frames} (testing mode)")
        if args.disable_scaling:
            print(f"Part scaling: DISABLED (comparison mode)")
        if args.disable_translation:
            print(f"Part translation: DISABLED (comparison mode)")
        print(f"View indices for singleview: {view_indices}")
        print(f"{'='*60}\n")
    
    checkpoint_path = _find_default_checkpoint()
    
    # Load dataset
    # Use num_views_to_use=None to use all available views per sample (same as training)
    # This allows the dataset to have variable numbers of views per sample
    dataset = SLEAPMultiViewDataset(
        hdf5_path=str(dataset_path),
        rotation_representation="6d",
        num_views_to_use=None,  # Use all available views (handles variable view counts)
        random_view_sampling=True,
    )
    
    dataset_max_views = dataset.get_max_views_in_dataset()
    dataset_canonical_camera_order = dataset.get_canonical_camera_order()
    
    if rank == 0:
        print(f"Dataset size: {len(dataset)}")
        print(f"Max views in dataset: {dataset_max_views}")
        print(f"Dataset canonical camera order: {dataset_canonical_camera_order}")
        print(f"Note: Samples may have fewer views than dataset max_views\n")
    
    # Load model
    # CRITICAL: max_views and canonical_camera_order are inferred from the checkpoint,
    # not from the dataset. The model architecture must match what was used during training.
    # The model can still handle samples with fewer views than max_views via view_mask.
    model = load_multiview_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        max_views=None,  # Infer from checkpoint
        canonical_camera_order=None,  # Load from checkpoint
    )
    
    # Get model's max_views (from checkpoint architecture)
    model_max_views = model.max_views
    
    if rank == 0:
        print(f"Model max_views (from checkpoint): {model_max_views}")
        print(f"Dataset max_views: {dataset_max_views}")
        if model_max_views > dataset_max_views:
            print(f"Note: Model supports {model_max_views} views, dataset has up to {dataset_max_views} views")
            print(f"      Model will handle samples with fewer views via view_mask\n")
        elif model_max_views < dataset_max_views:
            print(f"WARNING: Model supports {model_max_views} views but dataset has up to {dataset_max_views} views")
            print(f"         Samples with >{model_max_views} views will be truncated\n")
    
    # Create distributed sampler if multi-GPU
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Calculate frame dimensions
    # Note: Each sample may have fewer views than model_max_views. The visualization
    # functions calculate grid dimensions dynamically based on actual num_views per sample.
    # We use model_max_views here as the maximum grid size for consistent video output.
    # Samples with fewer views will be padded to this size by _pad_or_resize().
    img_size = 224
    margin = 5
    grid_width = model_max_views * img_size + (model_max_views + 1) * margin
    grid_height = 2 * img_size + 3 * margin
    
    # Determine singleview frame size (use first sample to get actual size)
    singleview_size = (224, 224)  # Default
    if len(dataset) > 0:
        try:
            test_frame = render_singleview_collage(
                model, dataset[0][0], dataset[0][1], device, view_idx=0
            )
            if test_frame is not None:
                singleview_size = (test_frame.shape[1], test_frame.shape[0])
        except Exception:
            pass  # Use default
    
    # Process dataset portion on this rank
    multiview_frames, singleview_frames_per_view, mv_frame_indices, sv_frame_indices_per_view = process_dataset_portion(
        dataset=dataset,
        model=model,
        device=device,
        rank=rank,
        world_size=world_size,
        sampler=sampler,
        grid_width=grid_width,
        grid_height=grid_height,
        singleview_size=singleview_size,
        max_frames=args.max_frames,
        disable_scaling=args.disable_scaling,
        disable_translation=args.disable_translation,
        view_indices=view_indices,
    )
    
    # Output paths
    dataset_name = dataset_path.stem
    multiview_out = Path(f"{dataset_name}_multiview_inference.mp4")
    singleview_out_base = Path(f"{dataset_name}_singleview_inference.mp4")
    
    # Use file-based gathering to avoid OOM with all_gather_object on large frame lists
    if world_size > 1:
        # Create a shared temporary directory for frame storage
        # Use cwd or dataset parent dir which should be on shared filesystem
        temp_base = Path.cwd() / f".inference_temp_{dataset_name}"
        
        # Only rank 0 creates the directory
        if rank == 0:
            if temp_base.exists():
                shutil.rmtree(temp_base)
            temp_base.mkdir(parents=True, exist_ok=True)
        
        # Synchronize to ensure temp directory exists before other ranks write
        dist.barrier()
        
        # Each rank writes its frames to temporary storage
        write_frames_to_temp_storage(
            multiview_frames=multiview_frames,
            singleview_frames_per_view=singleview_frames_per_view,
            mv_frame_indices=mv_frame_indices,
            sv_frame_indices_per_view=sv_frame_indices_per_view,
            temp_dir=temp_base,
            rank=rank,
        )
        
        # Free memory immediately after writing to disk
        del multiview_frames, singleview_frames_per_view, mv_frame_indices, sv_frame_indices_per_view
        torch.cuda.empty_cache()
        
        # Wait for all ranks to finish writing
        dist.barrier()
        
        # Only rank 0 merges and writes the final videos
        if rank == 0:
            merge_frames_and_write_videos(
                temp_dir=temp_base,
                world_size=world_size,
                multiview_out=multiview_out,
                singleview_out_base=singleview_out_base,
                fps=args.fps,
                grid_width=grid_width,
                grid_height=grid_height,
                singleview_size=singleview_size,
                view_indices=view_indices,
            )
            
            # Clean up temporary files
            print(f"Cleaning up temporary directory: {temp_base}")
            shutil.rmtree(temp_base)
        
        # Final barrier to ensure cleanup is done before exit
        dist.barrier()
    else:
        # Single GPU: write directly (no temp storage needed)
        if len(multiview_frames) > 0:
            multiview_writer = cv2.VideoWriter(
                str(multiview_out),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (grid_width, grid_height),
            )
            for frame in multiview_frames:
                multiview_writer.write(frame)
            multiview_writer.release()
            print(f"Wrote {multiview_out}")
        
        # Write singleview videos for each requested view
        for view_idx in view_indices:
            frames = singleview_frames_per_view.get(view_idx, [])
            if len(frames) > 0:
                # Create output path with view index
                if len(view_indices) == 1:
                    singleview_out = singleview_out_base
                else:
                    stem = singleview_out_base.stem
                    singleview_out = singleview_out_base.parent / f"{stem}_view{view_idx}.mp4"
                
                singleview_writer = cv2.VideoWriter(
                    str(singleview_out),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    singleview_size,
                )
                for frame in frames:
                    singleview_writer.write(frame)
                singleview_writer.release()
                print(f"Wrote {singleview_out} ({len(frames)} frames)")


def ddp_main_inference(rank: int, world_size: int, args, master_port: str):
    """
    DDP wrapper around main_inference function.
    
    Supports two launch modes:
    1. mp.spawn (single-node): rank is passed by spawn, local_rank == rank
    2. torchrun/SLURM (multi-node): environment variables are auto-detected and used
    """
    # Check if running under torchrun/SLURM - if so, use environment variables
    if is_torchrun_launched():
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu_rank = local_rank
    else:
        # mp.spawn mode (single-node) - local_rank == rank
        gpu_rank = rank
    
    setup_ddp(rank, world_size, master_port, local_rank=gpu_rank)
    
    # Set device override for this rank
    device_override = f"cuda:{gpu_rank}"
    
    try:
        main_inference(args, rank=rank, world_size=world_size, device_override=device_override)
    finally:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(
        description="Run simple multi-view inference on a preprocessed dataset"
    )
    parser.add_argument("--dataset", required=True, type=str, help="Path to preprocessed SLEAP HDF5 dataset")
    parser.add_argument("--fps", type=int, default=60, help="Output video FPS (default: 60)")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process (default: all frames). Useful for quick testing.")
    parser.add_argument("--disable_scaling", action="store_true", help="Disable part scaling (log_beta_scales) for comparison/debugging")
    parser.add_argument("--disable_translation", action="store_true", default=True, help="Disable part translation (betas_trans) for comparison/debugging")
    parser.add_argument("--view_indices", type=str, default="0", help="Comma-separated list of camera view indices to render for singleview output (default: '0'). E.g., '0,4,11' renders views 0, 4, and 11.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (default: 1, ignored when using torchrun)")
    parser.add_argument("--master-port", type=str, default=None, help="Master port for distributed processing (default: from MASTER_PORT env var or 12355)")
    args = parser.parse_args()
    
    # Get master port from args or environment variable
    master_port = args.master_port or os.environ.get('MASTER_PORT', '12355')
    
    # Check if launched via torchrun/torch.distributed.launch (HPC environment)
    if is_torchrun_launched():
        # Launched via torchrun - processes are already spawned by the launcher
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # Only print from rank 0 to avoid duplicate output
        if rank == 0:
            local_rank = int(os.environ['LOCAL_RANK'])
            print(f"Detected torchrun/HPC launch environment:")
            print(f"  Global rank: {rank}")
            print(f"  Local rank (GPU): {local_rank}")
            print(f"  World size: {world_size}")
            print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
            print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")
        
        # Call ddp_main_inference directly
        ddp_main_inference(rank, world_size, args, master_port)
    
    elif args.num_gpus > 1:
        # Manual multi-GPU launch using mp.spawn
        if not torch.cuda.is_available():
            print("ERROR: Multi-GPU processing requested but CUDA is not available!")
            exit(1)
        available_gpus = torch.cuda.device_count()
        if args.num_gpus > available_gpus:
            print(f"ERROR: Requested {args.num_gpus} GPUs but only {available_gpus} available!")
            exit(1)
        
        print(f"Launching multi-GPU inference on {args.num_gpus} GPUs (using mp.spawn)...")
        print(f"Master port: {master_port}")
        
        # Launch multi-GPU processing using spawn
        mp.spawn(
            ddp_main_inference,
            args=(args.num_gpus, args, master_port),
            nprocs=args.num_gpus,
            join=True
        )
    else:
        # Single GPU processing (existing path)
        print("Launching single-GPU inference...")
        main_inference(args, rank=0, world_size=1, device_override=None)


if __name__ == "__main__":
    main()
