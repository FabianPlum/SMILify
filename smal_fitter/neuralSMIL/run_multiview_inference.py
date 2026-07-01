#!/usr/bin/env python3
"""
Simple multi-view inference + visualization using a preprocessed SLEAP dataset.

This script mirrors the visualization logic used during training in
`train_multiview_regressor.py`, but runs over all samples in a preprocessed
multi-view HDF5 dataset and writes two videos in the current working directory:

  - "<DATASET>_multiview_inference.avi" (multi-view grid visualization, AVI+MJPG
    so wide grids exceed MPEG-4's 8192px width cap)
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
import sys
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
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from smal_fitter.neuralSMIL.multiview_smil_regressor import create_multiview_regressor, MultiViewSMILImageRegressor
from smal_fitter.neuralSMIL.smil_image_regressor import rotation_6d_to_axis_angle
from smal_fitter.fitter import SMALFitter
from smal_fitter.sleap_data.sleap_multiview_dataset import SLEAPMultiViewDataset
from smal_fitter.neuralSMIL.configs import apply_smal_file_override
import config
from smal_fitter.neuralSMIL.animation_export import build_recorder_from_config, build_multiview_cameras
from smal_fitter.neuralSMIL.multiview_visualization import (
    compute_multiview_grid_layout,
    create_multiview_visualization,
)


DEFAULT_CHECKPOINTS = [
    "multiview_checkpoints/best_model.pth",
    "multiview_checkpoints/final_model.pth",
]


class PredictionSmoother:
    """Temporal smoother that applies moving average over predicted parameters.

    Maintains a ring buffer of the last ``window_size`` predictions and returns
    their element-wise mean.  Metadata keys (non-tensor values and lists whose
    length varies across frames) are passed through from the latest frame.
    """

    _PER_VIEW_KEYS = {"fov_per_view", "cam_rot_per_view", "cam_trans_per_view"}
    _METADATA_KEYS = {"num_views", "view_mask", "camera_indices"}

    def __init__(self, window_size: int):
        self.window_size = window_size
        self._buffer: List[Dict[str, Any]] = []

    def __call__(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add *params* to the buffer and return the smoothed result."""
        if self.window_size <= 0:
            return params

        self._buffer.append(params)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

        if len(self._buffer) == 1:
            return params

        smoothed: Dict[str, Any] = {}
        for key in params:
            if key in self._METADATA_KEYS:
                smoothed[key] = params[key]
            elif key in self._PER_VIEW_KEYS:
                num_views = len(params[key])
                smoothed_list = []
                for v in range(num_views):
                    tensors = [
                        buf[key][v]
                        for buf in self._buffer
                        if key in buf and v < len(buf[key])
                    ]
                    if tensors:
                        smoothed_list.append(torch.stack(tensors).mean(dim=0))
                    else:
                        smoothed_list.append(params[key][v])
                smoothed[key] = smoothed_list
            elif isinstance(params[key], torch.Tensor):
                tensors = [buf[key] for buf in self._buffer if key in buf]
                smoothed[key] = torch.stack(tensors).mean(dim=0)
            else:
                smoothed[key] = params[key]

        return smoothed


def _params_to_cpu(params: Dict[str, Any]) -> Dict[str, Any]:
    """Detach and move all tensors in a predicted_params dict to CPU."""
    out: Dict[str, Any] = {}
    for key, val in params.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.detach().cpu()
        elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
            out[key] = [t.detach().cpu() for t in val]
        else:
            out[key] = val
    return out


def _params_to_device(params: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Move all tensors in a predicted_params dict to *device*."""
    out: Dict[str, Any] = {}
    for key, val in params.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.to(device)
        elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
            out[key] = [t.to(device) for t in val]
        else:
            out[key] = val
    return out


def run_forward_multiview(model: MultiViewSMILImageRegressor,
                          x_data: dict,
                          y_data: dict,
                          device: str) -> Optional[dict]:
    """Run a single forward pass and return predicted_params (or None if no views)."""
    images = x_data.get("images", [])
    num_views = len(images)
    if num_views == 0:
        return None

    cam_indices = x_data.get("camera_indices", list(range(num_views)))
    if isinstance(cam_indices, np.ndarray):
        cam_indices = cam_indices.tolist()
    if len(cam_indices) != num_views:
        if len(cam_indices) > num_views:
            cam_indices = cam_indices[:num_views]
        else:
            cam_indices = list(cam_indices) + list(range(len(cam_indices), num_views))

    images_per_view = []
    for img in images:
        img_tensor = model.preprocess_image(img).to(device)
        images_per_view.append(img_tensor.squeeze(0))

    images_tensors = [img.unsqueeze(0) for img in images_per_view]
    camera_indices_tensor = torch.tensor([cam_indices], device=device)
    view_mask = torch.ones(1, num_views, dtype=torch.bool, device=device)

    with torch.no_grad():
        predicted_params = model.forward_multiview(
            images_tensors, camera_indices_tensor, view_mask, target_data=[y_data]
        )

    return predicted_params


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

    from smal_fitter.neuralSMIL.backbone_factory import BackboneFactory
    input_resolution = BackboneFactory.get_default_input_resolution(backbone_name)

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


class _InMemoryImageExporter:
    def __init__(self):
        self.image = None

    def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces, img_idx=0, epoch=None):
        self.image = collage_np


def render_singleview_collage(model: MultiViewSMILImageRegressor,
                              x_data: dict,
                              y_data: dict,
                              device: str,
                              view_idx: int = 0,
                              disable_scaling: bool = False,
                              disable_translation: bool = False,
                              predicted_params: Optional[dict] = None,
                              render_resolution: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Render single-view mesh visualization matching training visualization exactly.

    CRITICAL: This function must match the training visualization in train_multiview_regressor.py
    exactly, especially for PCA transformation and parameter application order.

    Args:
        model: MultiViewSMILImageRegressor model
        x_data: Input data dictionary
        y_data: Target data dictionary
        device: PyTorch device
        view_idx: Which view to render (must be < num_views for this sample)
        disable_scaling: If True, skip applying log_beta_scales (testing/debugging)
        disable_translation: If True, skip applying betas_trans (testing/debugging)
        predicted_params: Pre-computed predicted parameters. If None, runs forward pass internally.
        render_resolution: If set, render the mesh + composite the footage at this square
            pixel resolution instead of the model's native ``renderer.image_size`` (224).
            Footage is interpolated up via PIL bilinear when this exceeds the native footage
            resolution (512). Does not affect model inference.
    """
    images = x_data.get("images", [])
    num_views = len(images)
    if num_views == 0 or view_idx >= num_views:
        return None

    # Run forward pass if predicted_params not provided
    if predicted_params is None:
        predicted_params = run_forward_multiview(model, x_data, y_data, device)
        if predicted_params is None:
            return None

    fov_per_view = predicted_params.get("fov_per_view", None)
    cam_rot_per_view = predicted_params.get("cam_rot_per_view", None)
    cam_trans_per_view = predicted_params.get("cam_trans_per_view", None)

    target_size = int(render_resolution) if render_resolution else int(getattr(model.renderer, "image_size", 224))

    original_image = images[view_idx]
    from PIL import Image
    pil_img = Image.fromarray((original_image * 255).astype(np.uint8))
    pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
    resized_image = np.array(pil_img).astype(np.float32) / 255.0

    resized_image = np.clip(resized_image, 0.0, 1.0)
    rgb = torch.from_numpy(resized_image).permute(2, 0, 1).unsqueeze(0).float()

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
            from smal_fitter.neuralSMIL.training_config import TrainingConfig
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


def _compute_rank_indices(
    dataset: SLEAPMultiViewDataset,
    rank: int,
    world_size: int,
    sampler: Optional[DistributedSampler],
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> List[int]:
    """Return the global dataset indices in ``[start_idx, end_idx)`` assigned to *rank*.

    With ``world_size > 1`` the range is striped across ranks (rank 0 takes
    ``start_idx``, ``start_idx + world_size``, ...), matching the original
    distributed behavior over the full dataset.
    """
    if end_idx is None:
        end_idx = len(dataset)
    end_idx = min(end_idx, len(dataset))
    start_idx = max(0, start_idx)

    if sampler is not None:
        sampler.set_epoch(0)
        indices = list(range(start_idx + rank, end_idx, world_size))
    else:
        indices = list(range(start_idx, end_idx))

    return indices


def _compute_subclip_ranges(
    dataset_size: int,
    max_frames: Optional[int],
    num_subclips: int,
    rank: int = 0,
) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` index ranges (end exclusive) for each subclip.

    With ``num_subclips > 1`` the dataset is divided into evenly-spaced slots
    of size ``dataset_size // num_subclips``; each subclip starts at slot ``i``
    and runs for ``max_frames`` frames. Falls back to a single full-dataset
    clip when subclips can't fit (``--max_frames`` not set or per-slot space
    smaller than ``max_frames``).
    """
    if num_subclips <= 1:
        end = min(max_frames, dataset_size) if max_frames is not None else dataset_size
        return [(0, end)]

    if max_frames is None:
        if rank == 0:
            print(f"WARNING: --generate_num_subclips={num_subclips} requires --max_frames; "
                  f"falling back to a single full-dataset clip.")
        return [(0, dataset_size)]

    slot_size = dataset_size // num_subclips
    if slot_size < max_frames:
        if rank == 0:
            print(f"WARNING: dataset has {dataset_size} frames; {num_subclips} subclips of "
                  f"{max_frames} frames each don't fit (only {slot_size} frames per slot). "
                  f"Falling back to a single full-dataset clip.")
        return [(0, dataset_size)]

    ranges: List[Tuple[int, int]] = []
    for i in range(num_subclips):
        start = i * dataset_size // num_subclips
        end = min(start + max_frames, dataset_size)
        ranges.append((start, end))
    return ranges


def _export_animation(
    raw_predictions: List[Tuple[int, dict]],
    rank: int,
    world_size: int,
    dataset: SLEAPMultiViewDataset,
    model: MultiViewSMILImageRegressor,
    checkpoint_path: Path,
    dataset_path: Path,
    export_path: str,
    fps: float,
) -> None:
    """Gather predictions to rank 0 and write an AMASS-style .npz + .json clip.

    The export captures the *raw*, pre-smoothing network predictions so downstream
    consumers (Blender addon, etc.) can apply their own smoothing if desired.
    """
    if world_size > 1:
        # Reuse the existing temp-dir gather machinery dedicated to this export.
        # Derive the temp dir name from the export path so it stays unique when
        # multiple subclips are exported in the same run.
        export_temp_base = Path.cwd() / f".animation_export_temp_{Path(export_path).name}"
        if rank == 0:
            if export_temp_base.exists():
                shutil.rmtree(export_temp_base)
            export_temp_base.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        write_predictions_to_temp(raw_predictions, export_temp_base, rank)
        dist.barrier()

        if rank == 0:
            all_predictions = load_all_predictions_from_temp(export_temp_base, world_size)
        else:
            all_predictions = None

        dist.barrier()
        if rank == 0:
            shutil.rmtree(export_temp_base, ignore_errors=True)
    else:
        all_predictions = sorted(raw_predictions, key=lambda x: x[0])

    if rank != 0 or not all_predictions:
        return

    recorder = build_recorder_from_config(
        output_path=export_path,
        rotation_representation=getattr(model, "rotation_representation", "6d"),
        fps=fps,
        source_checkpoint=str(checkpoint_path),
        source_input=str(dataset_path),
        model_id=getattr(model, "model_id", None),
    )

    cameras = build_multiview_cameras(all_predictions, dataset.get_canonical_camera_order())
    if cameras:
        recorder.set_cameras(cameras)

    for _, params in all_predictions:
        recorder.record(params)

    written = recorder.write()
    print(f"Animation export written: {written['npz']} + {written['json']} "
          f"({recorder.num_frames()} frames)")


def run_inference_phase(
    dataset: SLEAPMultiViewDataset,
    model: MultiViewSMILImageRegressor,
    device: str,
    indices: List[int],
    rank: int,
) -> List[Tuple[int, dict]]:
    """Run forward passes on the assigned indices and return raw predictions on CPU.

    Returns:
        List of ``(global_idx, predicted_params_cpu)`` tuples.
    """
    model.eval()
    raw_predictions: List[Tuple[int, dict]] = []

    iterator = tqdm(indices, desc="Running inference", disable=(rank != 0))
    for global_idx in iterator:
        try:
            x_data, y_data = dataset[global_idx]
            if len(x_data.get("images", [])) == 0:
                continue
            predicted_params = run_forward_multiview(model, x_data, y_data, device)
            if predicted_params is None:
                continue
            raw_predictions.append((global_idx, _params_to_cpu(predicted_params)))
        except Exception as e:
            print(f"[Rank {rank}] Error in inference for sample {global_idx}: {e}")
            continue

    print(f"[Rank {rank}] Inference complete: {len(raw_predictions)} predictions")
    return raw_predictions


def run_render_phase(
    dataset: SLEAPMultiViewDataset,
    model: MultiViewSMILImageRegressor,
    device: str,
    smoothed_params: Dict[int, dict],
    indices: List[int],
    rank: int,
    grid_width: int,
    grid_height: int,
    singleview_size: Tuple[int, int],
    disable_scaling: bool = False,
    disable_translation: bool = False,
    view_indices: Optional[List[int]] = None,
    total_view_slots: Optional[int] = None,
    render_resolution: Optional[int] = None,
) -> Tuple[List[np.ndarray], Dict[int, List[np.ndarray]], List[int], Dict[int, List[int]]]:
    """Render visualizations for the assigned indices using pre-computed smoothed params.

    Returns the same tuple as the old ``process_dataset_portion``.
    """
    if view_indices is None:
        view_indices = [0]

    multiview_frames: List[np.ndarray] = []
    mv_frame_indices: List[int] = []
    singleview_frames_per_view: Dict[int, List[np.ndarray]] = {v: [] for v in view_indices}
    sv_frame_indices_per_view: Dict[int, List[int]] = {v: [] for v in view_indices}

    iterator = tqdm(indices, desc="Rendering visualizations", disable=(rank != 0))
    for global_idx in iterator:
        if global_idx not in smoothed_params:
            continue
        try:
            x_data, y_data = dataset[global_idx]
            num_available_views = len(x_data.get("images", []))
            if num_available_views == 0:
                continue

            params = _params_to_device(smoothed_params[global_idx], device)

            mv_frame = create_multiview_visualization(
                model, x_data, y_data, device,
                disable_scaling=disable_scaling,
                disable_translation=disable_translation,
                predicted_params=params,
                total_view_slots=total_view_slots,
            )
            if mv_frame is not None:
                mv_frame = _pad_or_resize(mv_frame, (grid_width, grid_height))
                multiview_frames.append(cv2.cvtColor(mv_frame, cv2.COLOR_RGB2BGR))
                mv_frame_indices.append(global_idx)

            for view_idx in view_indices:
                if view_idx >= num_available_views:
                    continue
                sv_frame = render_singleview_collage(
                    model, x_data, y_data, device,
                    view_idx=view_idx,
                    disable_scaling=disable_scaling,
                    disable_translation=disable_translation,
                    predicted_params=params,
                    render_resolution=render_resolution,
                )
                if sv_frame is not None:
                    sv_frame = _pad_or_resize(sv_frame, singleview_size)
                    singleview_frames_per_view[view_idx].append(
                        cv2.cvtColor(sv_frame, cv2.COLOR_RGB2BGR)
                    )
                    sv_frame_indices_per_view[view_idx].append(global_idx)

        except Exception as e:
            print(f"[Rank {rank}] Error rendering sample {global_idx}: {e}")
            continue

    total_sv = sum(len(f) for f in singleview_frames_per_view.values())
    print(f"[Rank {rank}] Rendering complete: {len(multiview_frames)} multiview, {total_sv} singleview frames")
    return multiview_frames, singleview_frames_per_view, mv_frame_indices, sv_frame_indices_per_view


def write_predictions_to_temp(
    raw_predictions: List[Tuple[int, dict]],
    temp_dir: Path,
    rank: int,
) -> None:
    """Pickle raw predictions for this rank to shared temp storage."""
    rank_dir = temp_dir / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    pred_path = rank_dir / "predictions.pkl"
    with open(pred_path, "wb") as f:
        pickle.dump(raw_predictions, f)
    print(f"[Rank {rank}] Wrote {len(raw_predictions)} predictions to {pred_path}")


def load_all_predictions_from_temp(
    temp_dir: Path,
    world_size: int,
) -> List[Tuple[int, dict]]:
    """Load and merge prediction pickles from all ranks, sorted by global index."""
    all_predictions: List[Tuple[int, dict]] = []
    for rank_idx in range(world_size):
        pred_path = temp_dir / f"rank_{rank_idx}" / "predictions.pkl"
        if pred_path.exists():
            with open(pred_path, "rb") as f:
                all_predictions.extend(pickle.load(f))
    all_predictions.sort(key=lambda x: x[0])
    print(f"Loaded {len(all_predictions)} total predictions from {world_size} ranks")
    return all_predictions


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
    
    # Write multiview video. MPEG-4 caps frame dimensions at 8192 px, so wide
    # multi-camera grids fall over silently with mp4v. AVI+MJPG has no such
    # cap and stays well-supported everywhere we play these back.
    if len(all_mv_entries) > 0:
        multiview_writer = cv2.VideoWriter(
            str(multiview_out),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (grid_width, grid_height),
        )
        if not multiview_writer.isOpened():
            raise RuntimeError(
                f"Failed to open multiview VideoWriter for {multiview_out} "
                f"at {grid_width}x{grid_height}"
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
    
    # Apply SMAL model override if provided (similar to training script)
    # This must be done before loading the dataset/model to ensure config.dd, config.N_POSE, etc. are correct
    if args.smal_file:
        if rank == 0:
            print(f"Applying SMAL file override: {args.smal_file}")
        shape_family = args.shape_family if args.shape_family is not None else config.SHAPE_FAMILY
        apply_smal_file_override(args.smal_file, shape_family=shape_family)
        if rank == 0:
            print(f"  Shape family: {config.SHAPE_FAMILY}")
            print(f"  N_POSE: {config.N_POSE}")
            print(f"  N_BETAS: {config.N_BETAS}")
    
    # Parse view indices from comma-separated string
    view_indices = [int(x.strip()) for x in args.view_indices.split(",")]
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("MULTI-VIEW INFERENCE")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Dataset: {dataset_path}")
        print(f"Checkpoint: {args.checkpoint if args.checkpoint else '(auto-detect)'}")
        if args.max_frames is not None:
            print(f"Max frames: {args.max_frames} (testing mode)")
        if args.generate_num_subclips > 1:
            print(f"Subclips: {args.generate_num_subclips} (per-clip length: {args.max_frames})")
        if args.disable_scaling:
            print(f"Part scaling: DISABLED (comparison mode)")
        if args.disable_translation:
            print(f"Part translation: DISABLED (comparison mode)")
        print(f"View indices for singleview: {view_indices}")
        if args.smoothing_window > 0:
            print(f"Temporal smoothing: {args.smoothing_window} frames")
        print(f"{'='*60}\n")
    
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else _find_default_checkpoint()
    
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

    # Optional higher-resolution single-view visualization. When set, the single-view
    # mesh collage is rendered (and footage interpolated up) at this square resolution.
    # The multi-view grid is left at the native renderer resolution on purpose.
    render_resolution = getattr(args, "render_resolution", None)
    if render_resolution is not None:
        if render_resolution <= 0:
            raise ValueError(
                f"--render_resolution must be a positive integer, got {render_resolution}"
            )
        if rank == 0:
            native_footage = 512  # decoded JPEG crops stored in the HDF5 are 512x512
            if render_resolution < native_footage:
                print(f"WARNING: --render_resolution={render_resolution} is below the native "
                      f"footage resolution ({native_footage}); rendering below native footage "
                      f"wastes available detail.")
            if render_resolution > 4096:
                print(f"WARNING: --render_resolution={render_resolution} is very large; "
                      f"render time and single-view file size grow ~quadratically and "
                      f"memory use may be high.")
            print(f"Single-view render resolution: {render_resolution} "
                  f"(native renderer image_size={int(model.renderer.image_size)})")

    # Calculate frame dimensions — derive from the model's renderer resolution.
    # Layout (single-row vs wrapped 6-per-row block layout for >12 views) is
    # decided once from model.max_views so every frame in the output video has
    # identical dimensions, even when individual samples have fewer views.
    # NOTE: the multi-view grid intentionally stays at the native renderer
    # resolution; only the single-view collage honors --render_resolution.
    img_size = int(model.renderer.image_size)
    mv_layout = compute_multiview_grid_layout(model_max_views, img_size)
    grid_width = mv_layout['grid_width']
    grid_height = mv_layout['grid_height']
    if rank == 0:
        print(f"Multiview grid layout: {mv_layout['num_blocks']} block(s) × "
              f"{mv_layout['cols']} col(s) → {grid_width}×{grid_height}")

    # Determine singleview frame size (use first sample to get actual size).
    # Fallback default reflects the effective single-view resolution so a failed
    # test render still yields sane dimensions.
    sv_img_size = render_resolution if render_resolution else img_size
    singleview_size = (sv_img_size, sv_img_size)  # Default, overridden below by test render
    if len(dataset) > 0:
        try:
            test_frame = render_singleview_collage(
                model, dataset[0][0], dataset[0][1], device, view_idx=0,
                render_resolution=render_resolution,
            )
            if test_frame is not None:
                singleview_size = (test_frame.shape[1], test_frame.shape[0])
        except Exception:
            pass  # Use default

    # Determine subclip ranges (single full-dataset clip by default).
    subclip_ranges = _compute_subclip_ranges(
        dataset_size=len(dataset),
        max_frames=args.max_frames,
        num_subclips=args.generate_num_subclips,
        rank=rank,
    )
    multi_subclip = len(subclip_ranges) > 1

    dataset_name = dataset_path.stem
    smoothing_window = args.smoothing_window
    export_animation_path = getattr(args, "export_animation", None)

    for clip_idx, (start_idx, end_idx) in enumerate(subclip_ranges):
        if multi_subclip and rank == 0:
            print(f"\n{'#'*60}")
            print(f"# SUBCLIP {clip_idx + 1}/{len(subclip_ranges)}: "
                  f"frames [{start_idx}, {end_idx}) ({end_idx - start_idx} frames)")
            print(f"{'#'*60}")

        range_suffix = f"_frames{start_idx:06d}-{end_idx:06d}" if multi_subclip else ""

        # Compute which dataset indices this rank is responsible for in this subclip
        assigned_indices = _compute_rank_indices(
            dataset, rank, world_size, sampler,
            start_idx=start_idx, end_idx=end_idx,
        )

        # ── Phase 1: Inference (all ranks in parallel) ──────────────────────
        if rank == 0:
            print("\n── Phase 1: Running inference ──")
        raw_predictions = run_inference_phase(dataset, model, device, assigned_indices, rank)

        # Free GPU memory after inference — rendering will reload params to device as needed
        torch.cuda.empty_cache()

        # ── Phase 1b: Optional animation export (raw, pre-smoothing) ────────
        if export_animation_path:
            clip_export_path = f"{export_animation_path}{range_suffix}"
            _export_animation(
                raw_predictions=raw_predictions,
                rank=rank,
                world_size=world_size,
                dataset=dataset,
                model=model,
                checkpoint_path=checkpoint_path,
                dataset_path=dataset_path,
                export_path=clip_export_path,
                fps=float(args.fps),
            )

        # ── Phase 2: Gather + smooth predictions ────────────────────────────
        temp_base = Path.cwd() / f".inference_temp_{dataset_name}{range_suffix}"

        if world_size > 1 and smoothing_window > 0:
            # Multi-GPU with smoothing: gather all predictions so every rank
            # can build the full temporally-ordered sequence for correct smoothing.
            if rank == 0:
                print(f"\n── Phase 2: Gathering predictions across {world_size} ranks for smoothing (window={smoothing_window}) ──")
                if temp_base.exists():
                    shutil.rmtree(temp_base)
                temp_base.mkdir(parents=True, exist_ok=True)
            dist.barrier()

            write_predictions_to_temp(raw_predictions, temp_base, rank)
            dist.barrier()

            all_predictions = load_all_predictions_from_temp(temp_base, world_size)

            # Apply smoothing over the full sorted sequence
            smoother = PredictionSmoother(smoothing_window)
            smoothed_params: Dict[int, dict] = {}
            iterator = tqdm(all_predictions, desc="Applying temporal smoothing", disable=(rank != 0))
            for global_idx, params in iterator:
                smoothed_params[global_idx] = smoother(params)

            # Clean up prediction temp files (keep temp_base for frame storage later)
            for rank_idx in range(world_size):
                pred_path = temp_base / f"rank_{rank_idx}" / "predictions.pkl"
                if pred_path.exists():
                    pred_path.unlink()

            del raw_predictions, all_predictions

        elif smoothing_window > 0:
            # Single GPU with smoothing
            if rank == 0:
                print(f"\n── Phase 2: Applying temporal smoothing (window={smoothing_window}) ──")
            raw_predictions.sort(key=lambda x: x[0])
            smoother = PredictionSmoother(smoothing_window)
            smoothed_params = {}
            for global_idx, params in tqdm(raw_predictions, desc="Applying temporal smoothing", disable=(rank != 0)):
                smoothed_params[global_idx] = smoother(params)
            del raw_predictions

        else:
            # No smoothing: use raw predictions directly
            if rank == 0:
                print("\n── Phase 2: No smoothing (window=0) ──")
            smoothed_params = {idx: params for idx, params in raw_predictions}
            del raw_predictions

        # ── Phase 3: Render visualizations (all ranks in parallel) ──────────
        if rank == 0:
            print("\n── Phase 3: Rendering visualizations ──")
        multiview_frames, singleview_frames_per_view, mv_frame_indices, sv_frame_indices_per_view = run_render_phase(
            dataset=dataset,
            model=model,
            device=device,
            smoothed_params=smoothed_params,
            indices=assigned_indices,
            rank=rank,
            grid_width=grid_width,
            grid_height=grid_height,
            singleview_size=singleview_size,
            disable_scaling=args.disable_scaling,
            disable_translation=args.disable_translation,
            view_indices=view_indices,
            total_view_slots=model_max_views,
            render_resolution=render_resolution,
        )
        del smoothed_params

        # ── Phase 4: Write output videos ────────────────────────────────────
        if rank == 0:
            print("\n── Phase 4: Writing output videos ──")
        multiview_out = Path(f"{dataset_name}{range_suffix}_multiview_inference.avi")
        singleview_out_base = Path(f"{dataset_name}{range_suffix}_singleview_inference.mp4")

        if world_size > 1:
            if rank == 0:
                if not temp_base.exists():
                    temp_base.mkdir(parents=True, exist_ok=True)
            dist.barrier()

            write_frames_to_temp_storage(
                multiview_frames=multiview_frames,
                singleview_frames_per_view=singleview_frames_per_view,
                mv_frame_indices=mv_frame_indices,
                sv_frame_indices_per_view=sv_frame_indices_per_view,
                temp_dir=temp_base,
                rank=rank,
            )
            del multiview_frames, singleview_frames_per_view, mv_frame_indices, sv_frame_indices_per_view
            torch.cuda.empty_cache()

            dist.barrier()

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
                print(f"Cleaning up temporary directory: {temp_base}")
                shutil.rmtree(temp_base)

            dist.barrier()
        else:
            # Single GPU: write directly. See note in merge_frames_and_write_videos
            # for why multiview uses AVI+MJPG instead of MP4.
            if len(multiview_frames) > 0:
                multiview_writer = cv2.VideoWriter(
                    str(multiview_out),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    args.fps,
                    (grid_width, grid_height),
                )
                if not multiview_writer.isOpened():
                    raise RuntimeError(
                        f"Failed to open multiview VideoWriter for {multiview_out} "
                        f"at {grid_width}x{grid_height}"
                    )
                for frame in multiview_frames:
                    multiview_writer.write(frame)
                multiview_writer.release()
                print(f"Wrote {multiview_out}")

            for view_idx in view_indices:
                frames = singleview_frames_per_view.get(view_idx, [])
                if len(frames) > 0:
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
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process (default: all frames). Useful for quick testing. With --generate_num_subclips > 1, this is the length of each subclip.")
    parser.add_argument("--generate_num_subclips", type=int, default=1,
                        help="Generate N subclips evenly spaced across the dataset, each --max_frames "
                             "long. Subclip i starts at index i * len(dataset) / N. Each output "
                             "video and exported animation file is suffixed with the frame range. "
                             "Falls back to a single full-dataset clip if subclips don't fit. "
                             "Default: 1 (single clip).")
    parser.add_argument("--disable_scaling", action="store_true", help="Disable part scaling (log_beta_scales) for comparison/debugging")
    parser.add_argument("--disable_translation", action=argparse.BooleanOptionalAction, default=True, help="Disable part translation (betas_trans) for comparison/debugging. Pass --no-disable_translation to keep translation enabled.")
    parser.add_argument("--view_indices", type=str, default="0", help="Comma-separated list of camera view indices to render for singleview output (default: '0'). E.g., '0,4,11' renders views 0, 4, and 11.")
    parser.add_argument("--smoothing_window", type=int, default=0, help="Number of frames to average predictions over for temporal smoothing (default: 0, disabled)")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (default: 1, ignored when using torchrun)")
    parser.add_argument("--master-port", type=str, default=None, help="Master port for distributed processing (default: from MASTER_PORT env var or 12355)")
    parser.add_argument("--smal_file", type=str, default=None, help="Path to SMAL model file to override config.py SMAL_FILE (optional)")
    parser.add_argument("--shape_family", type=int, default=None, help="Shape family to use with --smal_file (optional, defaults to config.SHAPE_FAMILY)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth) to use for inference (default: auto-detected from multiview_checkpoints/)")
    parser.add_argument("--export_animation", type=str, default=None,
                        help="Optional output path stem for SMIL animation export. "
                             "Writes <stem>.npz + <stem>.json with raw (pre-smoothing) parameters "
                             "and per-view cameras. Gathered to rank 0 in multi-GPU runs. "
                             "NOTE: any string is accepted as-is (e.g. \"True\" writes True.npz) — "
                             "no validation is performed, so pass a real path/filename stem.")
    parser.add_argument("--render_resolution", type=int, default=None,
                        help="Square pixel resolution for the single-view mesh visualization. "
                             "The mesh is rendered and the background footage interpolated up to "
                             "match (native footage is 512). Default: None = renderer's native "
                             "image_size (224). Does NOT affect model inference / backbone input, "
                             "and does NOT change the multi-view grid. Note: render time and "
                             "single-view file size scale ~quadratically with this value.")
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
