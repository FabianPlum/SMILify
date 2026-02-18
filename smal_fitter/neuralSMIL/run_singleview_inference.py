#!/usr/bin/env python3
"""
SMIL Image Regressor Inference Script

This script loads a trained SMILImageRegressor model from a checkpoint and runs inference
on images or videos. It generates visualizations using the existing SMIL visualization
functions and saves results to an output folder.

Usage:
    # For images
    python run_inference.py --checkpoint path/to/checkpoint.pth --input-folder path/to/images --output-folder path/to/output
    
    # For video
    python run_inference.py --checkpoint path/to/checkpoint.pth --input-video path/to/video.mp4 --output-folder path/to/output

Features:
    - Loads trained model from checkpoint
    - Processes images or video files
    - Supports center-crop preprocessing (matching training)
    - Generates SMIL model visualizations
    - Saves predicted parameters and visualizations
    - For videos: generates output video and per-frame results
    - Handles different image/video sizes and formats
"""

import os
import sys
import argparse
import glob
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import h5py
import json
import pickle as pkl

import torch
import torch.nn as nn
import numpy as np
import cv2
import imageio
from tqdm import tqdm

# Set matplotlib backend BEFORE any other imports to prevent tkinter issues
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Add the parent directories to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smil_image_regressor import SMILImageRegressor, rotation_6d_to_axis_angle
from training_config import TrainingConfig
from smal_fitter import SMALFitter
from Unreal2Pytorch3D import return_placeholder_data
import config
from sleap_data_loader import SLEAPDataLoader
import importlib.util
import importlib


def _import_sleap_preprocessor():
    try:
        module = importlib.import_module("smal_fitter.sleap_data.preprocess_sleap_dataset")
        return module.SLEAPDatasetPreprocessor
    except ModuleNotFoundError:
        module_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "sleap_data", "preprocess_sleap_dataset.py")
        )
        spec = importlib.util.spec_from_file_location("preprocess_sleap_dataset", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not locate preprocess_sleap_dataset at {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SLEAPDatasetPreprocessor


SLEAPDatasetPreprocessor = _import_sleap_preprocessor()


class SLEAPCroppingHelper:
    """Leverage preprocessing logic from preprocess_sleap_dataset for bbox cropping."""

    def __init__(self, project_path: str, crop_mode: str, target_resolution: int,
                 backbone_name: str, use_reprojections: bool = True):
        self.project_path = Path(project_path)
        self.crop_mode = crop_mode
        self.preprocessor = SLEAPDatasetPreprocessor(
            joint_lookup_table_path=None,
            shape_betas_table_path=None,
            target_resolution=target_resolution,
            backbone_name=backbone_name,
            crop_mode=crop_mode,
            use_reprojections=use_reprojections
        )

        session_paths = self.preprocessor.discover_sleap_sessions(project_path)
        if not session_paths:
            raise ValueError(f"No SLEAP sessions found in {project_path}")

        self.session_map: Dict[str, Path] = {Path(s).name: Path(s) for s in session_paths}
        self.single_session_name = next(iter(self.session_map)) if len(self.session_map) == 1 else None

        self.session_loaders: Dict[str, SLEAPDataLoader] = {}
        self.camera_data_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.reprojection_cache: Dict[str, Tuple[Optional[h5py.File], Dict[str, h5py.File]]] = {}
        self._warned_missing_camera: Set[str] = set()
        self._warned_missing_frame: Set[Tuple[str, int]] = set()

    def close(self):
        """Close any open reprojection file handles."""
        for root_handle, sub_handles in self.reprojection_cache.values():
            try:
                if root_handle is not None:
                    root_handle.close()
            except Exception:
                pass
            for handle in sub_handles.values():
                try:
                    handle.close()
                except Exception:
                    pass

    def list_cameras(self) -> List[str]:
        cameras = set()
        for session_name in self.session_map:
            loader = self._get_loader(session_name)
            cameras.update(loader.camera_views)
        return sorted(cameras)

    def _resolve_session(self, media_path: str) -> Tuple[str, Path]:
        for session_name, session_path in self.session_map.items():
            if session_name in media_path:
                return session_name, session_path
        if self.single_session_name is not None:
            session_path = self.session_map[self.single_session_name]
            return self.single_session_name, session_path
        raise ValueError(f"Could not infer SLEAP session for path: {media_path}")

    def _get_loader(self, session_name: str) -> SLEAPDataLoader:
        if session_name not in self.session_loaders:
            session_path = self.session_map[session_name]
            self.session_loaders[session_name] = SLEAPDataLoader(
                project_path=str(session_path),
                lookup_table_path=self.preprocessor.joint_lookup_table_path,
                shape_betas_path=self.preprocessor.shape_betas_table_path
            )
        return self.session_loaders[session_name]

    def _get_reprojection_handles(self, session_name: str) -> Tuple[Optional[h5py.File], Dict[str, h5py.File]]:
        if session_name in self.reprojection_cache:
            return self.reprojection_cache[session_name]

        session_path = self.session_map[session_name]
        root_handle = None
        root_reproj = session_path / 'reprojections.h5'
        if root_reproj.exists():
            try:
                root_handle = h5py.File(str(root_reproj), 'r')
            except Exception as exc:
                print(f"Warning: Failed to open reprojections file {root_reproj}: {exc}")
                root_handle = None

        sub_handles = self.preprocessor._find_all_reprojection_files(str(session_path)) if self.preprocessor.use_reprojections else {}
        self.reprojection_cache[session_name] = (root_handle, sub_handles)
        return self.reprojection_cache[session_name]

    def _get_camera_data(self, session_name: str, camera_name: str, loader: SLEAPDataLoader) -> Dict[str, Any]:
        key = (session_name, camera_name)
        if key not in self.camera_data_cache:
            self.camera_data_cache[key] = loader.load_camera_data(camera_name)
        return self.camera_data_cache[key]

    def _infer_camera(self, media_path: str, loader: SLEAPDataLoader, explicit_camera: Optional[str]) -> Optional[str]:
        if explicit_camera:
            return explicit_camera

        path_lower = str(media_path).lower()
        stem_lower = Path(media_path).stem.lower()
        parent_names = {p.name.lower() for p in Path(media_path).parents}
        for camera in loader.camera_views:
            cam_lower = camera.lower()
            if f"_cam{cam_lower}" in path_lower or f"-cam{cam_lower}" in path_lower:
                return camera
            if cam_lower in stem_lower.split('_'):
                return camera
            # Also check directory names in the path (e.g., .../Camera2/...)
            if cam_lower in parent_names:
                return camera
        if len(loader.camera_views) == 1:
            return loader.camera_views[0]
        return None

    def _get_reprojection_handle_for_camera(self, session_name: str, camera_name: str) -> Optional[h5py.File]:
        root_handle, sub_handles = self._get_reprojection_handles(session_name)
        if root_handle is not None:
            return root_handle
        session_path = self.session_map[session_name]
        camera_subdir = self.preprocessor._find_camera_subdir(str(session_path), camera_name)
        if camera_subdir and camera_subdir in sub_handles:
            return sub_handles[camera_subdir]
        return None

    def _extract_keypoints(self, loader: SLEAPDataLoader, camera_data: Dict[str, Any],
                           camera_name: str, frame_idx: int, session_name: str) -> Optional[np.ndarray]:
        reproj_handle = None
        if self.preprocessor.use_reprojections:
            reproj_handle = self._get_reprojection_handle_for_camera(session_name, camera_name)
        try:
            keypoints_2d, visibility = self.preprocessor._extract_2d_keypoints_for_frame(
                loader=loader,
                camera_data=camera_data,
                camera_name=camera_name,
                frame_idx=frame_idx,
                reproj_handle=reproj_handle
            )
        except Exception as exc:
            key = (camera_name, frame_idx)
            if key not in self._warned_missing_frame:
                print(f"Warning: Failed to extract keypoints for camera '{camera_name}', frame {frame_idx}: {exc}")
                self._warned_missing_frame.add(key)
            return None

        keypoints_2d = self.preprocessor._sanitize_array(keypoints_2d, default_value=np.nan)
        if keypoints_2d is None:
            return None

        keypoints = np.asarray(keypoints_2d, dtype=np.float32)
        vis_mask = None
        if visibility is not None:
            visibility = np.asarray(visibility)
            if visibility.dtype == bool:
                vis_mask = visibility
            else:
                vis_mask = visibility > 0.5
        if vis_mask is not None and vis_mask.shape[0] == keypoints.shape[0]:
            keypoints = keypoints[vis_mask]

        if keypoints.size == 0:
            return None

        valid_mask = np.isfinite(keypoints).all(axis=1)
        keypoints = keypoints[valid_mask]
        keypoints = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
        return keypoints if keypoints.size > 0 else None

    def preprocess_image(self, image: np.ndarray, media_path: str, frame_idx: int,
                         explicit_camera: Optional[str] = None) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        session_name, _ = self._resolve_session(media_path)
        loader = self._get_loader(session_name)
        camera_name = self._infer_camera(media_path, loader, explicit_camera)
        if camera_name is None:
            if media_path not in self._warned_missing_camera:
                print(f"Warning: Could not infer SLEAP camera for '{media_path}'")
                self._warned_missing_camera.add(media_path)
            return None

        camera_data = self._get_camera_data(session_name, camera_name, loader)
        keypoints = self._extract_keypoints(loader, camera_data, camera_name, frame_idx, session_name)
        if keypoints is None or len(keypoints) == 0:
            return None

        if image.max() <= 1.0:
            work_image = (image * 255.0).astype(np.uint8)
        else:
            work_image = image.astype(np.uint8)

        processed_image, transform_info = self.preprocessor._preprocess_image(work_image, keypoints)
        return processed_image, transform_info


class InferenceImageExporter:
    """Enhanced image exporter for inference results."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the image exporter.
        
        Args:
            output_dir: Directory to save visualization images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export(self, collage_np: np.ndarray, batch_id: int, global_id: int, 
               img_parameters: Dict[str, Any], vertices: torch.Tensor, faces: np.ndarray, 
               img_idx: int = 0, image_name: str = "image"):
        """
        Export visualization image and parameters.
        
        Args:
            collage_np: Visualization collage as numpy array
            batch_id: Batch ID
            global_id: Global ID  
            img_parameters: Dictionary of SMIL parameters
            vertices: Model vertices
            faces: Model faces
            img_idx: Image index
            image_name: Base name for the image
        """
        # Save visualization image
        vis_filename = f"{image_name}_visualization.png"
        vis_path = os.path.join(self.output_dir, vis_filename)
        imageio.imsave(vis_path, collage_np)
        
        # Save parameters as JSON (for human readability)
        params_filename = f"{image_name}_parameters.json"
        params_path = os.path.join(self.output_dir, params_filename)
        
        # Convert numpy arrays and tensors to lists for JSON serialization
        json_parameters = {}
        for key, value in img_parameters.items():
            if isinstance(value, np.ndarray):
                json_parameters[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                json_parameters[key] = value.detach().cpu().numpy().tolist()
            elif hasattr(value, 'numpy'):  # Handle other tensor-like objects
                json_parameters[key] = value.numpy().tolist()
            else:
                json_parameters[key] = value
        
        with open(params_path, 'w') as f:
            json.dump(json_parameters, f, indent=2)
        
        # Save parameters as pickle (for exact reproduction)
        pkl_filename = f"{image_name}_parameters.pkl"
        pkl_path = os.path.join(self.output_dir, pkl_filename)
        with open(pkl_path, 'wb') as f:
            pkl.dump(img_parameters, f)
        
        print(f"Saved results for {image_name}:")
        print(f"  Visualization: {vis_path}")
        print(f"  Parameters (JSON): {params_path}")
        print(f"  Parameters (PKL): {pkl_path}")


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> Tuple[SMILImageRegressor, Dict[str, Any]]:
    """
    Load a trained SMILImageRegressor model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: PyTorch device ('cuda' or 'cpu')
        
    Returns:
        Tuple of (loaded_model, model_config)
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("Checkpoint loaded successfully")

        # Prefer config from checkpoint so inference matches training; fall back to training_config if missing
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
            config_source = "checkpoint (fallback: training_config for missing keys)"
        else:
            model_config = fallback_model
            rotation_representation = fallback_params["rotation_representation"]
            scale_trans_mode = TrainingConfig.get_scale_trans_mode()
            shape_family = config.SHAPE_FAMILY
            config_source = "training_config.py (no config in checkpoint)"

        print(f"Configuration from {config_source}:")
        print(f"  backbone_name: {model_config['backbone_name']}")
        print(f"  head_type: {model_config.get('head_type', 'mlp')}")
        print(f"  rotation_representation: {rotation_representation}")
        print(f"  scale_trans_mode: {scale_trans_mode}")
        print(f"  shape_family: {shape_family}")

        # If the checkpoint specifies a SMAL/SMIL model file, re-derive
        # config.dd, N_POSE, N_BETAS, joint_names, etc. from that file.
        if ckpt_config and ckpt_config.get("smal_file"):
            from configs import apply_smal_file_override
            apply_smal_file_override(
                ckpt_config["smal_file"],
                shape_family=shape_family,
            )
        
        # Verify this matches the checkpoint by checking state dict keys
        state_dict = checkpoint['model_state_dict']
        
        # Check for transformer head
        has_transformer_head = any('transformer_head' in key for key in state_dict.keys())
        
        # Infer backbone type from feature dimensions in the checkpoint
        # Note: Backbone weights are NOT saved (frozen pretrained weights are re-downloaded on load)
        # So we detect backbone type from the input dimension of the regression head
        inferred_backbone = None
        if has_transformer_head:
            # Check transformer head input dimension from token_embedding weight
            token_emb_key = 'transformer_head.token_embedding.weight'
            if token_emb_key in state_dict:
                feature_dim = state_dict[token_emb_key].shape[0]
                if feature_dim == 1024:
                    inferred_backbone = 'vit_large'
                elif feature_dim == 768:
                    inferred_backbone = 'vit_base'
                elif feature_dim == 2048:
                    inferred_backbone = 'resnet'
                print(f"Inferred backbone from checkpoint feature dim ({feature_dim}): {inferred_backbone}")
        
        # Validate config matches inferred backbone
        config_backbone = model_config['backbone_name']
        if inferred_backbone:
            config_is_vit_large = 'vit_large' in config_backbone
            config_is_vit_base = 'vit_base' in config_backbone
            config_is_resnet = config_backbone.startswith('resnet')
            
            if inferred_backbone == 'vit_large' and not config_is_vit_large:
                print(f"WARNING: Checkpoint was trained with ViT-Large but config specifies {config_backbone}")
            elif inferred_backbone == 'vit_base' and not config_is_vit_base:
                print(f"WARNING: Checkpoint was trained with ViT-Base but config specifies {config_backbone}")
            elif inferred_backbone == 'resnet' and not config_is_resnet:
                print(f"WARNING: Checkpoint was trained with ResNet but config specifies {config_backbone}")
            else:
                print(f"Backbone configuration matches checkpoint: {config_backbone}")
        
        if model_config['head_type'] == 'transformer_decoder' and not has_transformer_head:
            print("WARNING: Config specifies transformer_decoder but checkpoint doesn't contain transformer_head")
        
        print(f"Checkpoint verification:")
        print(f"  Inferred backbone: {inferred_backbone or 'unknown'}")
        print(f"  Contains transformer_head: {has_transformer_head}")
        
        print(f"Model configuration:")
        for key, value in model_config.items():
            if key != 'transformer_config':
                print(f"  {key}: {value}")
        
        print(f"Using rotation representation: {rotation_representation}")
        
        # For inference, always use batch_size=1
        # The checkpoint may have been saved with a different batch size during training,
        # but for inference we process one image at a time
        batch_size = 1
        print(f"Using batch size: {batch_size} (inference mode)")
        
        # Create placeholder data for model initialization
        placeholder_data = torch.zeros((batch_size, 3, 512, 512))
        
        # Determine input resolution based on backbone
        if model_config['backbone_name'].startswith('vit'):
            input_resolution = 224
        else:
            input_resolution = 512
        
        print(f"Creating model with input resolution: {input_resolution}")
        
        # Initialize model with detected configuration
        model = SMILImageRegressor(
            device=device,
            data_batch=placeholder_data,
            batch_size=batch_size,
            shape_family=shape_family,
            use_unity_prior=model_config.get('use_unity_prior', False),
            rgb_only=model_config.get('rgb_only', True),
            freeze_backbone=model_config.get('freeze_backbone', True),
            hidden_dim=model_config.get('hidden_dim', 1024),
            use_ue_scaling=True,  # Default for replicAnt data
            rotation_representation=rotation_representation,
            input_resolution=input_resolution,
            backbone_name=model_config['backbone_name'],
            head_type=model_config.get('head_type', 'mlp'),
            transformer_config=model_config.get('transformer_config', {}),
            scale_trans_mode=scale_trans_mode  # Critical for correct output dimensions
        ).to(device)
        
        # Load model state, handling batch size differences
        # For inference, we need the neural network weights, but skip SMAL optimization parameters
        state_dict = checkpoint['model_state_dict']
        
        # Filter out SMAL optimization parameters that have batch size dependencies
        # These are specific to the SMALFitter optimization process, not the neural network
        smal_optimization_params = [
            'global_rotation', 'joint_rotations', 'trans', 'log_beta_scales', 
            'betas_trans', 'betas', 'fov', 'target_joints', 'target_visibility'
        ]
        
        # Keep all neural network parameters (backbone, transformer_head, fc layers, etc.)
        nn_state_dict = {}
        skipped_params = []
        
        for k, v in state_dict.items():
            # Skip SMAL optimization parameters that are specific to the optimization process
            if any(k == param or k.startswith(param + '.') for param in smal_optimization_params):
                skipped_params.append(k)
            else:
                nn_state_dict[k] = v
        
        print(f"Loading {len(nn_state_dict)} neural network parameters")
        print(f"Skipping {len(skipped_params)} SMAL optimization parameters: {skipped_params[:5]}{'...' if len(skipped_params) > 5 else ''}")
        
        # Load the neural network weights
        missing_keys, unexpected_keys = model.load_state_dict(nn_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys (will use random initialization): {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''}")
        model.eval()
        
        print("Model loaded and set to evaluation mode")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Head type: {model.head_type}")
        print(f"  Backbone: {model.backbone_name}")
        print(f"  Input resolution: {input_resolution}")
        
        return model, model_config
        
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")


def find_image_files(input_folder: str, supported_extensions: List[str] = None) -> List[str]:
    """
    Find all image files in the input folder.
    
    Args:
        input_folder: Path to folder containing images
        supported_extensions: List of supported file extensions
        
    Returns:
        List of image file paths
    """
    if supported_extensions is None:
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF']
    
    image_files = []
    input_path = Path(input_folder)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_folder}")
    
    # Find all image files
    for ext in supported_extensions:
        pattern = f"*{ext}"
        image_files.extend(input_path.glob(pattern))
    
    # Convert to strings and sort
    image_files = [str(f) for f in image_files]
    image_files.sort()
    
    print(f"Found {len(image_files)} image files in {input_folder}")
    
    return image_files


def preprocess_frame(image: np.ndarray, target_resolution: int, crop_mode: str = 'centred') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Preprocess a frame using centred or default resize.
    
    Args:
        image: Input image (H, W, C) in range [0, 255] or [0, 1]
        target_resolution: Target resolution for model input
        crop_mode: 'centred' or 'default'
        
    Returns:
        Tuple of (preprocessed_image, transform_info)
    """
    if crop_mode == 'bbox_crop':
        raise ValueError("bbox_crop should be handled via SLEAPCroppingHelper; preprocess_frame only supports 'centred' or 'default'.")

    # Ensure image is in [0, 255] range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    original_h, original_w = image.shape[:2]
    transform_info = {
        'original_size': (original_h, original_w),
        'crop_offset': (0, 0),
        'crop_size': (original_h, original_w),
        'scale_factor': 1.0,
        'mode': crop_mode
    }
    
    if crop_mode == 'centred':
        crop_size = min(original_h, original_w)
        y_offset = (original_h - crop_size) // 2
        x_offset = (original_w - crop_size) // 2
        image = image[y_offset:y_offset + crop_size, x_offset:x_offset + crop_size]
        transform_info['crop_offset'] = (y_offset, x_offset)
        transform_info['crop_size'] = (crop_size, crop_size)
        scale_factor = target_resolution / crop_size
        transform_info['scale_factor'] = scale_factor
        image = cv2.resize(image, (target_resolution, target_resolution))
    
    else:
        scale_y = target_resolution / original_h
        scale_x = target_resolution / original_w
        transform_info['scale_factor'] = (scale_y, scale_x)
        image = cv2.resize(image, (target_resolution, target_resolution))
    
    image = image.astype(np.float32) / 255.0
    return image, transform_info


def load_and_preprocess_image(image_path: str, model: SMILImageRegressor, crop_mode: str = 'centred',
                              keypoints_2d: Optional[np.ndarray] = None) -> Tuple[np.ndarray, torch.Tensor, Dict[str, Any]]:
    """
    Load and preprocess an image for inference.
    
    Args:
        image_path: Path to the image file
        model: SMILImageRegressor model for preprocessing
        crop_mode: Cropping mode ('centred' or 'default')
        
    Returns:
        Tuple of (original_image_array, preprocessed_tensor, transform_info)
    """
    try:
        # Load image
        image_data = imageio.v2.imread(image_path)
        
        # Keep original for visualization
        original_image = image_data.copy()
        
        # Preprocess with proper cropping
        target_resolution = model.input_resolution
        preprocessed_image, transform_info = preprocess_frame(
            image_data, target_resolution, crop_mode, keypoints_2d=keypoints_2d
        )
        
        # Convert to tensor (C, H, W) format
        preprocessed_tensor = torch.from_numpy(preprocessed_image).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        
        return original_image, preprocessed_tensor, transform_info
        
    except Exception as e:
        raise RuntimeError(f"Failed to load/preprocess image {image_path}: {e}")


def run_inference_on_image(model: SMILImageRegressor, image_tensor: torch.Tensor, 
                          device: str) -> Dict[str, torch.Tensor]:
    """
    Run inference on a preprocessed image tensor.
    
    Args:
        model: SMILImageRegressor model
        image_tensor: Preprocessed image tensor (1, C, H, W)
        device: PyTorch device
        
    Returns:
        Dictionary of predicted SMIL parameters
    """
    try:
        with torch.no_grad():
            # Move to device
            image_tensor = image_tensor.to(device)
            
            # Get batch size from tensor
            batch_size = image_tensor.shape[0]
            
            # Run inference through the model's forward pass
            # The model's forward() method handles batches correctly
            predicted_params = model.forward(image_tensor)
            
            # Move results back to CPU for visualization
            cpu_params = {}
            for key, value in predicted_params.items():
                if isinstance(value, torch.Tensor):
                    cpu_params[key] = value.cpu()
                else:
                    cpu_params[key] = value
            
            return cpu_params
            
    except Exception as e:
        import traceback
        print("\n" + "="*60)
        print("INFERENCE ERROR DEBUG INFO")
        print("="*60)
        print(f"Input tensor shape: {image_tensor.shape}")
        print(f"Device: {device}")
        print(f"Model batch_size attribute: {model.batch_size}")
        print(f"Model head_type: {model.head_type}")
        print(f"Model rotation_representation: {model.rotation_representation}")
        print("="*60)
        traceback.print_exc()
        print("="*60)
        raise RuntimeError(f"Inference failed: {e}")


def render_model_only(model: SMILImageRegressor, predicted_params: Dict[str, torch.Tensor],
                      device: str, render_size: int) -> np.ndarray:
    """
    Render only the predicted 3D model without any background.
    
    Args:
        model: SMILImageRegressor model
        predicted_params: Dictionary of predicted SMIL parameters
        device: PyTorch device
        render_size: Target render resolution
        
    Returns:
        Rendered model image (render_size, render_size, 3) in RGB, range [0, 255]
    """
    try:
        # Convert rotations to axis-angle if they're in 6D representation
        if model.rotation_representation == '6d':
            global_rot_aa = rotation_6d_to_axis_angle(predicted_params['global_rot'])
            joint_rot_aa = rotation_6d_to_axis_angle(predicted_params['joint_rot'])
        else:
            global_rot_aa = predicted_params['global_rot']
            joint_rot_aa = predicted_params['joint_rot']
        
        # Create a blank RGB tensor for rendering
        rgb_tensor = torch.zeros((1, 3, render_size, render_size), device=device)
        
        # Create temporary SMALFitter for rendering
        temp_fitter = SMALFitter(
            device=device,
            data_batch=rgb_tensor,
            batch_size=1,
            shape_family=config.SHAPE_FAMILY,
            use_unity_prior=False,
            rgb_only=True
        )
        
        # CRITICAL: Match propagate_scaling to the training model's setting.
        # The model learns scales with propagate_scaling=True (set in SMILImageRegressor.__init__),
        # so visualization must also use propagate_scaling=True for consistent geometry.
        temp_fitter.propagate_scaling = model.propagate_scaling
        
        # Set the predicted parameters
        temp_fitter.global_rotation.data = global_rot_aa.to(device)
        temp_fitter.joint_rotations.data = joint_rot_aa.to(device)
        temp_fitter.betas.data = predicted_params['betas'].to(device)
        temp_fitter.trans.data = predicted_params['trans'].to(device)
        temp_fitter.fov.data = predicted_params['fov'].to(device)
        
        # Set joint scales and translations if available
        if 'log_beta_scales' in predicted_params:
            temp_fitter.log_beta_scales.data = predicted_params['log_beta_scales'].to(device)
        if 'betas_trans' in predicted_params:
            temp_fitter.betas_trans.data = predicted_params['betas_trans'].to(device)
        
        # Set camera parameters
        if 'cam_rot' in predicted_params and 'cam_trans' in predicted_params:
            temp_fitter.renderer.set_camera_parameters(
                R=predicted_params['cam_rot'].to(device),
                T=predicted_params['cam_trans'].to(device),
                fov=predicted_params['fov'].to(device)
            )
        
        # Render the model
        with torch.no_grad():
            # Get vertices and joints from SMAL model
            verts, joints, Rs, v_shaped = temp_fitter.smal_model(
                temp_fitter.betas,
                torch.cat([
                    temp_fitter.global_rotation.unsqueeze(1),
                    temp_fitter.joint_rotations
                ], dim=1),
                betas_logscale=temp_fitter.log_beta_scales,
                betas_trans=temp_fitter.betas_trans,
                propagate_scaling=temp_fitter.propagate_scaling
            )
            
            # Apply UE scaling transformation (10x scale)
            verts = (verts - joints[:, 0, :].unsqueeze(1)) * 10 + temp_fitter.trans.unsqueeze(1)
            joints = (joints - joints[:, 0, :].unsqueeze(1)) * 10 + temp_fitter.trans.unsqueeze(1)
            
            # Get canonical model joints
            canonical_joints = joints[:, config.CANONICAL_MODEL_JOINTS]
            
            # Prepare faces
            faces_batch = temp_fitter.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1)
            
            # Render with texture
            rendered_silhouettes, rendered_joints, rendered_image = temp_fitter.renderer(
                verts, canonical_joints, faces_batch, render_texture=True
            )
        
        # Convert rendered image to numpy (already in (B, C, H, W) format)
        rendered_np = rendered_image[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        rendered_np = np.clip(rendered_np, 0, 1)
        
        # Convert to [0, 255] range
        return (rendered_np * 255).astype(np.uint8)
        
    except Exception as e:
        print(f"Warning: Failed to render model: {e}")
        # Return black image on error
        return np.zeros((render_size, render_size, 3), dtype=np.uint8)


def render_prediction_on_frame(model: SMILImageRegressor, predicted_params: Dict[str, torch.Tensor],
                               original_frame: np.ndarray, device: str,
                               transform_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Render the predicted 3D model onto the original frame.
    
    Args:
        model: SMILImageRegressor model
        predicted_params: Dictionary of predicted SMIL parameters
        original_frame: Original frame (H, W, 3) in RGB, range [0, 255]
        device: PyTorch device
        
    Returns:
        Rendered frame with 3D model overlay (H, W, 3) in RGB, range [0, 255]
    """
    try:
        # Convert rotations to axis-angle if they're in 6D representation
        if model.rotation_representation == '6d':
            global_rot_aa = rotation_6d_to_axis_angle(predicted_params['global_rot'])
            joint_rot_aa = rotation_6d_to_axis_angle(predicted_params['joint_rot'])
        else:
            global_rot_aa = predicted_params['global_rot']
            joint_rot_aa = predicted_params['joint_rot']
        
        # Get frame dimensions
        frame_h, frame_w = original_frame.shape[:2]
        
        # Resize to model's expected input size for rendering
        if model.backbone_name.startswith('vit'):
            render_size = 224
        else:
            render_size = 512
        
        # Prepare image for rendering
        if original_frame.max() > 1.0:
            rgb_image = original_frame.astype(np.float32) / 255.0
        else:
            rgb_image = original_frame.astype(np.float32)
        
        # Resize to render size
        rgb_resized = cv2.resize(rgb_image, (render_size, render_size))
        
        # Convert to tensor format expected by SMALFitter
        rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Create temporary SMALFitter for rendering
        temp_fitter = SMALFitter(
            device=device,
            data_batch=rgb_tensor,
            batch_size=1,
            shape_family=config.SHAPE_FAMILY,
            use_unity_prior=False,
            rgb_only=True
        )
        
        # CRITICAL: Match propagate_scaling to the training model's setting.
        # The model learns scales with propagate_scaling=True (set in SMILImageRegressor.__init__),
        # so visualization must also use propagate_scaling=True for consistent geometry.
        temp_fitter.propagate_scaling = model.propagate_scaling
        
        # Set the predicted parameters
        temp_fitter.global_rotation.data = global_rot_aa.to(device)
        temp_fitter.joint_rotations.data = joint_rot_aa.to(device)
        temp_fitter.betas.data = predicted_params['betas'].to(device)
        temp_fitter.trans.data = predicted_params['trans'].to(device)
        temp_fitter.fov.data = predicted_params['fov'].to(device)
        
        # Set joint scales and translations if available
        if 'log_beta_scales' in predicted_params:
            temp_fitter.log_beta_scales.data = predicted_params['log_beta_scales'].to(device)
        if 'betas_trans' in predicted_params:
            temp_fitter.betas_trans.data = predicted_params['betas_trans'].to(device)
        
        # Set camera parameters
        if 'cam_rot' in predicted_params and 'cam_trans' in predicted_params:
            temp_fitter.renderer.set_camera_parameters(
                R=predicted_params['cam_rot'].to(device),
                T=predicted_params['cam_trans'].to(device),
                fov=predicted_params['fov'].to(device)
            )
        
        # Render the model
        with torch.no_grad():
            # Get vertices and joints from SMAL model
            verts, joints, Rs, v_shaped = temp_fitter.smal_model(
                temp_fitter.betas,
                torch.cat([
                    temp_fitter.global_rotation.unsqueeze(1),
                    temp_fitter.joint_rotations
                ], dim=1),
                betas_logscale=temp_fitter.log_beta_scales,
                betas_trans=temp_fitter.betas_trans,
                propagate_scaling=temp_fitter.propagate_scaling
            )
            
            # Apply UE scaling transformation (10x scale)
            verts = (verts - joints[:, 0, :].unsqueeze(1)) * 10 + temp_fitter.trans.unsqueeze(1)
            joints = (joints - joints[:, 0, :].unsqueeze(1)) * 10 + temp_fitter.trans.unsqueeze(1)
            
            # Get canonical model joints
            canonical_joints = joints[:, config.CANONICAL_MODEL_JOINTS]
            
            # Prepare faces
            faces_batch = temp_fitter.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1)
            
            # Render with texture
            rendered_silhouettes, rendered_joints, rendered_image = temp_fitter.renderer(
                verts, canonical_joints, faces_batch, render_texture=True
            )
        
        # Convert rendered image to numpy (already in (B, C, H, W) format)
        rendered_np = rendered_image[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        rendered_np = np.clip(rendered_np, 0, 1)
        
        alpha = 0.6  # Transparency of the overlay
        overlay_base = (original_frame.astype(np.float32) / 255.0)

        placed_overlay = overlay_base.copy()

        if transform_info is not None and transform_info.get('mode') in ('centred', 'bbox_crop'):
            crop_height, crop_width = transform_info.get('crop_size', (frame_h, frame_w))
            y_offset, x_offset = transform_info.get('crop_offset', (0, 0))

            crop_height = int(round(crop_height))
            crop_width = int(round(crop_width))
            y_offset = int(round(y_offset))
            x_offset = int(round(x_offset))

            if crop_height <= 0 or crop_width <= 0:
                rendered_resized = cv2.resize(rendered_np, (frame_w, frame_h))
                blended = alpha * rendered_resized + (1 - alpha) * overlay_base
                return (blended * 255).astype(np.uint8)

            overlay_y_end = min(y_offset + crop_height, frame_h)
            overlay_x_end = min(x_offset + crop_width, frame_w)

            if overlay_y_end <= y_offset or overlay_x_end <= x_offset:
                rendered_resized = cv2.resize(rendered_np, (frame_w, frame_h))
                blended = alpha * rendered_resized + (1 - alpha) * overlay_base
                return (blended * 255).astype(np.uint8)

            target_height = overlay_y_end - y_offset
            target_width = overlay_x_end - x_offset

            rendered_resized = cv2.resize(rendered_np, (target_width, target_height))

            base_region = overlay_base[y_offset:overlay_y_end, x_offset:overlay_x_end]
            blended_region = alpha * rendered_resized + (1 - alpha) * base_region
            placed_overlay[y_offset:overlay_y_end, x_offset:overlay_x_end] = blended_region
            return (placed_overlay * 255).astype(np.uint8)
        else:
            rendered_resized = cv2.resize(rendered_np, (frame_w, frame_h))
            blended = alpha * rendered_resized + (1 - alpha) * overlay_base
            return (blended * 255).astype(np.uint8)
        
    except Exception as e:
        print(f"Warning: Failed to render prediction: {e}")
        # Return original frame on error
        return original_frame


def generate_visualization(model: SMILImageRegressor, predicted_params: Dict[str, torch.Tensor],
                          original_image: np.ndarray, image_exporter: InferenceImageExporter,
                          image_name: str, device: str) -> None:
    """
    Generate visualization using the SMIL model and predicted parameters.
    
    Args:
        model: SMILImageRegressor model
        predicted_params: Dictionary of predicted SMIL parameters
        original_image: Original input image
        image_exporter: Image exporter for saving results
        image_name: Base name for the image
        device: PyTorch device
    """
    try:
        # Convert rotations to axis-angle if they're in 6D representation
        if model.rotation_representation == '6d':
            global_rot_aa = rotation_6d_to_axis_angle(predicted_params['global_rot'])
            joint_rot_aa = rotation_6d_to_axis_angle(predicted_params['joint_rot'])
        else:
            global_rot_aa = predicted_params['global_rot']
            joint_rot_aa = predicted_params['joint_rot']
        
        # Create a simplified SMALFitter for visualization
        # Use the original image as RGB input
        if original_image.max() > 1.0:
            rgb_image = original_image.astype(np.float32) / 255.0
        else:
            rgb_image = original_image.astype(np.float32)
        
        # Resize to model's expected input size
        if model.backbone_name.startswith('vit'):
            target_size = (224, 224)
        else:
            target_size = (512, 512)
        
        if rgb_image.shape[:2] != target_size:
            rgb_image = cv2.resize(rgb_image, target_size)
        
        # Convert to tensor format expected by SMALFitter
        rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Create temporary SMALFitter for visualization
        temp_fitter = SMALFitter(
            device=device,
            data_batch=rgb_tensor,
            batch_size=1,
            shape_family=config.SHAPE_FAMILY,
            use_unity_prior=False,
            rgb_only=True
        )
        
        # CRITICAL: Match propagate_scaling to the training model's setting.
        # The model learns scales with propagate_scaling=True (set in SMILImageRegressor.__init__),
        # so visualization must also use propagate_scaling=True for consistent geometry.
        temp_fitter.propagate_scaling = model.propagate_scaling
        
        # Set the predicted parameters (ensure they're on the right device)
        temp_fitter.global_rotation.data = global_rot_aa.to(device)
        temp_fitter.joint_rotations.data = joint_rot_aa.to(device)
        temp_fitter.betas.data = predicted_params['betas'].to(device)
        temp_fitter.trans.data = predicted_params['trans'].to(device)
        temp_fitter.fov.data = predicted_params['fov'].to(device)
        
        # Set joint scales and translations if available
        if 'log_beta_scales' in predicted_params:
            temp_fitter.log_beta_scales.data = predicted_params['log_beta_scales'].to(device)
        if 'betas_trans' in predicted_params:
            temp_fitter.betas_trans.data = predicted_params['betas_trans'].to(device)
        
        # Set camera parameters using predicted values
        if 'cam_rot' in predicted_params and 'cam_trans' in predicted_params:
            temp_fitter.renderer.set_camera_parameters(
                R=predicted_params['cam_rot'].to(device),
                T=predicted_params['cam_trans'].to(device),
                fov=predicted_params['fov'].to(device)
            )
        
        # Set dummy target joints and visibility for visualization
        temp_fitter.target_joints = torch.zeros((1, config.N_POSE, 2), device=device)
        temp_fitter.target_visibility = torch.ones((1, config.N_POSE), device=device)
        
        # Generate visualization with custom image exporter wrapper
        class NamedImageExporter:
            def __init__(self, base_exporter, image_name):
                self.base_exporter = base_exporter
                self.image_name = image_name
            
            def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces, img_idx=0):
                # Call the base exporter with the specific image name
                self.base_exporter.export(
                    collage_np, batch_id, global_id, img_parameters, vertices, faces, 
                    img_idx=img_idx, image_name=self.image_name
                )
        
        named_exporter = NamedImageExporter(image_exporter, image_name)
        temp_fitter.generate_visualization(named_exporter, apply_UE_transform=model.use_ue_scaling, img_idx=0)
        
        print(f"Generated visualization for {image_name}")
        
    except Exception as e:
        print(f"Warning: Failed to generate visualization for {image_name}: {e}")
        # Save just the parameters without visualization
        img_parameters = {k: v.cpu().data.numpy() if isinstance(v, torch.Tensor) else v 
                         for k, v in predicted_params.items()}
        
        # Create a simple visualization showing the original image
        simple_vis = original_image.copy()
        if simple_vis.max() > 1.0:
            simple_vis = (simple_vis).astype(np.uint8)
        else:
            simple_vis = (simple_vis * 255).astype(np.uint8)
        
        image_exporter.export(
            simple_vis, 0, 0, img_parameters, 
            torch.zeros(1, 1000, 3), np.zeros((1000, 3), dtype=int), 
            img_idx=0, image_name=image_name
        )


def process_images_batch(model: SMILImageRegressor, image_files: List[str],
                        output_folder: str, device: str, crop_mode: str = 'centred',
                        batch_size: int = 1, sleap_helper: Optional[SLEAPCroppingHelper] = None,
                        sleap_camera: Optional[str] = None) -> None:
    """
    Process a batch of images for inference.
    
    Args:
        model: SMILImageRegressor model
        image_files: List of image file paths
        output_folder: Output folder for results
        device: PyTorch device
        crop_mode: Cropping mode ('centred', 'default', or 'bbox_crop')
        batch_size: Batch size for processing (currently only supports 1)
        sleap_helper: Optional helper for bbox_crop using SLEAP keypoints
        sleap_camera: Optional camera override for bbox_crop
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Create image exporter
    image_exporter = InferenceImageExporter(output_folder)
    
    print(f"Processing {len(image_files)} images...")
    print(f"Crop mode: {crop_mode}")
    
    # Process images with progress bar
    for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Get image name for output files
            image_name = Path(image_path).stem
            
            print(f"\nProcessing image {i+1}/{len(image_files)}: {image_name}")
            
            if crop_mode == 'bbox_crop' and sleap_helper is not None:
                original_image = imageio.v2.imread(image_path)
                if original_image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")
                preprocess_result = sleap_helper.preprocess_image(
                    original_image, image_path, frame_idx=i, explicit_camera=sleap_camera
                )
                if preprocess_result is None:
                    print("Warning: bbox_crop requested but no keypoints available; falling back to centred crop")
                    preprocessed_image, transform_info = preprocess_frame(
                        original_image, model.input_resolution, crop_mode='centred'
                    )
                else:
                    preprocessed_image, transform_info = preprocess_result
                preprocessed_tensor = torch.from_numpy(preprocessed_image).permute(2, 0, 1).unsqueeze(0)
            else:
                original_image, preprocessed_tensor, transform_info = load_and_preprocess_image(
                    image_path, model, crop_mode
                )
            
            # Run inference
            predicted_params = run_inference_on_image(model, preprocessed_tensor, device)
            
            # Generate visualization with unique image name
            generate_visualization(
                model, predicted_params, original_image, 
                image_exporter, image_name, device
            )
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    print(f"\nProcessing complete! Results saved to: {output_folder}")


def smooth_camera_parameters(predicted_params: Dict[str, torch.Tensor], 
                            camera_buffer: Dict[str, List], 
                            window_size: int) -> Dict[str, torch.Tensor]:
    """
    Apply moving average smoothing to camera parameters.
    
    Args:
        predicted_params: Dictionary of predicted parameters
        camera_buffer: Buffer storing recent camera parameters
        window_size: Size of the moving average window
        
    Returns:
        Smoothed parameters dictionary
    """
    smoothed_params = predicted_params.copy()
    
    # Add current predictions to buffers
    camera_buffer['cam_rot'].append(predicted_params['cam_rot'].clone())
    camera_buffer['cam_trans'].append(predicted_params['cam_trans'].clone())
    camera_buffer['fov'].append(predicted_params['fov'].clone())
    
    # Keep only the last window_size frames
    if len(camera_buffer['cam_rot']) > window_size:
        camera_buffer['cam_rot'].pop(0)
        camera_buffer['cam_trans'].pop(0)
        camera_buffer['fov'].pop(0)
    
    # Compute moving average
    if len(camera_buffer['cam_rot']) > 0:
        smoothed_params['cam_rot'] = torch.stack(camera_buffer['cam_rot']).mean(dim=0)
        smoothed_params['cam_trans'] = torch.stack(camera_buffer['cam_trans']).mean(dim=0)
        smoothed_params['fov'] = torch.stack(camera_buffer['fov']).mean(dim=0)
    
    return smoothed_params


def process_video(model: SMILImageRegressor, video_path: str, output_folder: str,
                 device: str, crop_mode: str = 'centred', fps: Optional[int] = None,
                 save_frames: bool = False, max_frames: int = -1,
                 camera_smoothing_window: int = 10,
                 sleap_helper: Optional[SLEAPCroppingHelper] = None,
                 sleap_camera: Optional[str] = None,
                 video_export_mode: str = 'overlay') -> None:
    """
    Process a video file for inference.
    
    Args:
        model: SMILImageRegressor model
        video_path: Path to input video file
        output_folder: Output folder for results
        device: PyTorch device
        crop_mode: Cropping mode ('centred', 'default', or 'bbox_crop')
        fps: Output video FPS (None = same as input)
        save_frames: Whether to save individual frame results
        max_frames: Maximum number of frames to process (-1 for all frames)
        camera_smoothing_window: Number of frames for moving average of camera parameters (default: 10)
        sleap_helper: Optional helper for bbox_crop using SLEAP keypoints
        sleap_camera: Optional camera override when using bbox_crop
        video_export_mode: Export mode ('overlay' or 'side_by_side')
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_fps = fps if fps is not None else input_fps
    
    # Determine how many frames to process
    if max_frames > 0:
        frames_to_process = min(max_frames, total_frames)
    else:
        frames_to_process = total_frames
    
    print(f"Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  Frames to process: {frames_to_process}")
    print(f"  Input FPS: {input_fps}")
    print(f"  Output FPS: {output_fps}")
    print(f"  Resolution: {frame_width}x{frame_height}")
    print(f"  Crop mode: {crop_mode}")
    print(f"  Camera smoothing window: {camera_smoothing_window} frames")
    print(f"  Video export mode: {video_export_mode}")
    
    # Determine output video dimensions based on export mode
    if video_export_mode == 'side_by_side':
        # Determine render size based on model backbone
        if model.backbone_name.startswith('vit'):
            render_size = 224
        else:
            render_size = 512
        
        # For side-by-side: input video will be rescaled to match render_size height
        # Output width will be 2 * render_size
        output_height = render_size
        output_width = render_size * 2
    else:
        # For overlay mode: keep original video dimensions
        output_height = frame_height
        output_width = frame_width
    
    # Create video writer for output
    output_video_path = os.path.join(output_folder, Path(video_path).stem + "_inference.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (output_width, output_height))
    
    # Initialize moving average buffers for camera parameters
    camera_buffer = {
        'cam_rot': [],
        'cam_trans': [],
        'fov': []
    }
    
    # Optionally create frame exporter
    if save_frames:
        frames_folder = os.path.join(output_folder, "frames")
        os.makedirs(frames_folder, exist_ok=True)
        frame_exporter = InferenceImageExporter(frames_folder)
    else:
        frame_exporter = None
    
    # Process frames
    frame_idx = 0
    
    try:
        pbar = tqdm(total=frames_to_process, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Stop if we've reached max_frames limit
            if max_frames > 0 and frame_idx >= max_frames:
                break
            
            try:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Preprocess frame
                target_resolution = model.input_resolution
                if crop_mode == 'bbox_crop' and sleap_helper is not None:
                    preprocess_result = sleap_helper.preprocess_image(
                        frame_rgb, video_path, frame_idx=frame_idx, explicit_camera=sleap_camera
                    )
                    if preprocess_result is None:
                        print(f"Warning: Missing SLEAP keypoints for frame {frame_idx}; "
                              "falling back to centred crop for this frame")
                        preprocessed_image, transform_info = preprocess_frame(
                            frame_rgb, target_resolution, crop_mode='centred'
                        )
                    else:
                        preprocessed_image, transform_info = preprocess_result
                else:
                    preprocessed_image, transform_info = preprocess_frame(
                        frame_rgb, target_resolution, crop_mode
                    )
                
                # Convert to tensor
                preprocessed_tensor = torch.from_numpy(preprocessed_image).permute(2, 0, 1).unsqueeze(0)
                
                # Run inference
                predicted_params = run_inference_on_image(model, preprocessed_tensor, device)
                
                # Apply camera parameter smoothing
                if camera_smoothing_window > 0:
                    smoothed_params = smooth_camera_parameters(
                        predicted_params, camera_buffer, camera_smoothing_window
                    )
                else:
                    smoothed_params = predicted_params
                
                # Process frame based on export mode
                if video_export_mode == 'side_by_side':
                    # Render model only
                    rendered_model = render_model_only(model, smoothed_params, device, render_size)
                    
                    # Resize input frame to match render_size
                    input_resized = cv2.resize(frame_rgb, (render_size, render_size))
                    
                    # Create side-by-side visualization
                    side_by_side = np.hstack([input_resized, rendered_model])
                    
                    # Convert RGB to BGR for OpenCV
                    output_frame_bgr = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)
                else:
                    # Render prediction onto frame (using smoothed camera parameters)
                    rendered_frame = render_prediction_on_frame(
                        model, smoothed_params, frame_rgb, device, transform_info=transform_info
                    )
                    
                    # Convert RGB back to BGR for OpenCV
                    output_frame_bgr = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)
                
                # Write output frame to video
                out.write(output_frame_bgr)
                
                # Optionally save frame results
                if save_frames and frame_idx % 10 == 0:  # Save every 10th frame
                    frame_name = f"frame_{frame_idx:06d}"
                    try:
                        generate_visualization(
                            model, predicted_params, frame_rgb,
                            frame_exporter, frame_name, device
                        )
                    except Exception as e:
                        print(f"Warning: Failed to save frame {frame_idx}: {e}")
                
                frame_idx += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"Warning: Failed to process frame {frame_idx}: {e}")
                # Write original frame on error
                out.write(frame)
                frame_idx += 1
                pbar.update(1)
                continue
        
        pbar.close()
        
    finally:
        cap.release()
        out.release()
    
    print(f"\nVideo processing complete!")
    print(f"  Output video: {output_video_path}")
    print(f"  Processed {frame_idx} frames")
    if save_frames:
        print(f"  Frame results: {frames_folder}")


def main():
    """Main function for the inference script."""
    parser = argparse.ArgumentParser(
        description='Run SMIL inference on images or video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process images
  python run_singleview_inference.py --checkpoint checkpoints/best_model.pth --input_folder test_images --output_folder results
  python run_singleview_inference.py -c model.pth -i images/ -o output/ --crop_mode centred

  # Process video
  python run_singleview_inference.py --checkpoint model.pth --input_video video.mp4 --output_folder results
  python run_singleview_inference.py -c model.pth -v video.mp4 -o output/ --save_frames --fps 30

  # With different preprocessing
  python run_singleview_inference.py -c model.pth -i images/ -o output/ --crop_mode default

Supported image formats: jpg, jpeg, png, bmp, tiff, tif (case-insensitive)
Supported video formats: mp4, avi, mov, mkv (anything supported by OpenCV)
        """
    )
    
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                       help='Path to the trained model checkpoint (.pth file)')
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input_folder', type=str,
                            help='Path to folder containing input images')
    input_group.add_argument('-v', '--input_video', type=str,
                            help='Path to input video file')
    
    parser.add_argument('-o', '--output_folder', type=str, required=True,
                       help='Path to folder for saving results')
    
    # Preprocessing options
    parser.add_argument('--crop_mode', type=str, default='centred',
                       choices=['centred', 'default', 'bbox_crop'],
                       help='Image preprocessing mode: centred=center crop (preserves aspect ratio), '
                            'default=direct resize (may distort), bbox_crop=SLEAP-driven bounding box crop. '
                            'Should match training preprocessing. (default: centred)')
    
    # Processing options
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference (default: auto), using cuda if available')
    
    # Video-specific options
    parser.add_argument('--fps', type=int, default=None,
                       help='Output video FPS (default: same as input)')
    parser.add_argument('--save_frames', action='store_true',
                       help='Save individual frame results when processing video')
    parser.add_argument('--max_frames', type=int, default=-1,
                       help='Maximum number of frames to process from video (default: -1 for all frames)')
    parser.add_argument('--camera_smoothing', type=int, default=0,
                       help='Moving average window size for camera parameter smoothing (default: 0, set to 0 to disable)')
    parser.add_argument('--video_export_mode', type=str, default='overlay',
                       choices=['overlay', 'side_by_side'],
                       help='Video export mode: overlay=blend model onto input (default), '
                            'side_by_side=display input and rendered model side by side at same resolution')
    
    # SLEAP-specific options
    parser.add_argument('--sleap_project', type=str, default=None,
                       help='Path to SLEAP project directory (required for bbox_crop)')
    parser.add_argument('--sleap_camera', type=str, default=None,
                       help='Optional camera name override when using bbox_crop with SLEAP data')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SMIL Image Regressor - Inference Script")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    if args.input_folder:
        print(f"Input folder: {args.input_folder}")
    if args.input_video:
        print(f"Input video: {args.input_video}")
    print(f"Output folder: {args.output_folder}")
    print(f"Crop mode: {args.crop_mode}")
    if args.sleap_project:
        print(f"SLEAP project: {args.sleap_project}")
        if args.sleap_camera:
            print(f"SLEAP camera override: {args.sleap_camera}")
    if args.input_video:
        print(f"Save frames: {args.save_frames}")
        if args.fps:
            print(f"Output FPS: {args.fps}")
        if args.max_frames > 0:
            print(f"Max frames: {args.max_frames}")
        else:
            print(f"Max frames: All frames")
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Device: {device}")
    
    if args.crop_mode == 'bbox_crop' and not args.sleap_project:
        print("Error: bbox_crop mode requires --sleap-project to supply keypoints.")
        return 1
    
    sleap_helper = None
    try:
        # Load model from checkpoint
        print("\n" + "="*40)
        print("Loading model...")
        model, model_config = load_model_from_checkpoint(args.checkpoint, device)
        
        if args.crop_mode == 'bbox_crop':
            sleap_helper = SLEAPCroppingHelper(
                project_path=args.sleap_project,
                crop_mode=args.crop_mode,
                target_resolution=model.input_resolution,
                backbone_name=model.backbone_name
            )
            print(f"SLEAP project loaded from {args.sleap_project}")
            print(f"Available SLEAP cameras: {sleap_helper.list_cameras()}")
        
        # Process based on input type
        if args.input_folder:
            # Find image files
            print("\n" + "="*40)
            print("Finding images...")
            image_files = find_image_files(args.input_folder)
            
            if len(image_files) == 0:
                print("No image files found in the input folder!")
                return 1
            
            # Process images
            print("\n" + "="*40)
            print("Running inference on images...")
            process_images_batch(
                model, image_files, args.output_folder,
                device, args.crop_mode, args.batch_size,
                sleap_helper=sleap_helper,
                sleap_camera=args.sleap_camera
            )
            
        elif args.input_video:
            # Process video
            print("\n" + "="*40)
            print("Running inference on video...")
            process_video(
                model, args.input_video, args.output_folder,
                device, args.crop_mode, args.fps, args.save_frames, args.max_frames,
                args.camera_smoothing,
                sleap_helper=sleap_helper,
                sleap_camera=args.sleap_camera,
                video_export_mode=args.video_export_mode
            )
        
        print("\n" + "="*60)
        print("Inference completed successfully!")
        print(f"Results saved to: {args.output_folder}")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nInference interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if sleap_helper is not None:
            sleap_helper.close()


if __name__ == "__main__":
    exit(main())
