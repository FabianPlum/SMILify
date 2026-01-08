#!/usr/bin/env python3
"""
Multi-View SMIL Image Regressor Inference Script

This script loads a trained MultiViewSMILImageRegressor model from a checkpoint and runs 
inference on synchronized multi-view videos from a SLEAP project. It generates visualizations 
showing input images and rendered meshes for each camera view.

Usage:
    python run_inference_multiview.py --checkpoint path/to/multiview_checkpoint.pth \
        --sleap-project path/to/sleap/sessions --output-folder path/to/output

Features:
    - Discovers SLEAP sessions and synchronized multi-view frames
    - Processes synchronized views through the multi-view model
    - Generates grid visualization:
        - Top row: Input images from each camera
        - Bottom row: Rendered mesh with predicted/GT keypoints
    - Outputs video with all views synchronized
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import cv2
import h5py
import imageio
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# Set matplotlib backend BEFORE any other imports
import matplotlib
matplotlib.use('Agg')

# Add parent directories to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multiview_smil_regressor import MultiViewSMILImageRegressor, create_multiview_regressor
from smil_image_regressor import rotation_6d_to_axis_angle
from training_config import TrainingConfig
from smal_fitter import SMALFitter
import config
from sleap_data_loader import SLEAPDataLoader


class SLEAPMultiViewInferenceHelper:
    """
    Helper class to load and process multi-view data from SLEAP projects for inference.
    
    Similar to SLEAPMultiViewPreprocessor but optimized for real-time inference
    rather than preprocessing to HDF5.
    """
    
    def __init__(self, 
                 project_path: str,
                 target_resolution: int = 224,
                 crop_mode: str = 'bbox_crop',
                 confidence_threshold: float = 0.5,
                 min_views_per_sample: int = 2,
                 joint_lookup_table_path: Optional[str] = None,
                 shape_betas_table_path: Optional[str] = None):
        """
        Initialize the multi-view inference helper.
        
        Args:
            project_path: Path to SLEAP project (containing sessions)
            target_resolution: Target image resolution
            crop_mode: Image cropping mode
            confidence_threshold: Minimum keypoint confidence
            min_views_per_sample: Minimum views required per frame
            joint_lookup_table_path: Path to joint lookup table
            shape_betas_table_path: Path to shape betas table
        """
        self.project_path = Path(project_path)
        self.target_resolution = target_resolution
        self.crop_mode = crop_mode
        self.confidence_threshold = confidence_threshold
        self.min_views_per_sample = min_views_per_sample
        self.joint_lookup_table_path = joint_lookup_table_path
        self.shape_betas_table_path = shape_betas_table_path
        
        # Initialize caches FIRST (before any method that uses them)
        self.loader_cache: Dict[str, SLEAPDataLoader] = {}
        self.camera_data_cache: Dict[Tuple[str, str], Dict] = {}
        self.video_cap_cache: Dict[Tuple[str, str], cv2.VideoCapture] = {}
        
        # Discover sessions
        self.sessions = self._discover_sessions()
        if not self.sessions:
            raise ValueError(f"No SLEAP sessions found in {project_path}")
        
        print(f"Found {len(self.sessions)} SLEAP sessions")
        
        # Establish canonical camera order (caches must be initialized first)
        self.canonical_camera_order = self._establish_canonical_camera_order()
        print(f"Canonical camera order: {self.canonical_camera_order}")
    
    def _discover_sessions(self) -> List[Path]:
        """Discover all SLEAP session directories."""
        sessions = []
        
        for item in self.project_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if self._is_sleap_session(item):
                    sessions.append(item)
        
        sessions.sort(key=lambda x: x.name)
        return sessions
    
    def _is_sleap_session(self, session_path: Path) -> bool:
        """Check if a directory looks like a SLEAP session."""
        # Check for session subdirectories (session_dirs structure)
        session_subdirs = [d for d in session_path.iterdir() if d.is_dir()]
        if session_subdirs:
            for subdir in session_subdirs:
                h5_files = list(subdir.glob('*.h5'))
                if h5_files:
                    return True
        
        # Check for camera directories (camera_dirs structure)
        for cam_dir in session_path.iterdir():
            if cam_dir.is_dir():
                slp_files = list(cam_dir.glob('*.slp'))
                if slp_files:
                    return True
        
        # Check for direct SLEAP files
        sleap_indicators = ['calibration.toml', 'points3d.h5']
        for indicator in sleap_indicators:
            if (session_path / indicator).exists():
                return True
        
        return False
    
    def _establish_canonical_camera_order(self) -> List[str]:
        """Establish canonical camera ordering from all sessions."""
        all_cameras: Set[str] = set()
        
        for session_path in self.sessions:
            try:
                loader = self._get_loader(session_path)
                all_cameras.update(loader.camera_views)
            except Exception as e:
                print(f"Warning: Failed to load cameras from {session_path}: {e}")
        
        return sorted(list(all_cameras))
    
    def _get_loader(self, session_path: Path) -> SLEAPDataLoader:
        """Get or create a SLEAPDataLoader for a session."""
        key = str(session_path)
        if key not in self.loader_cache:
            self.loader_cache[key] = SLEAPDataLoader(
                project_path=str(session_path),
                lookup_table_path=self.joint_lookup_table_path,
                shape_betas_path=self.shape_betas_table_path,
                confidence_threshold=self.confidence_threshold
            )
        return self.loader_cache[key]
    
    def _get_camera_data(self, session_path: Path, camera_name: str) -> Dict:
        """Get or cache camera data."""
        key = (str(session_path), camera_name)
        if key not in self.camera_data_cache:
            loader = self._get_loader(session_path)
            self.camera_data_cache[key] = loader.load_camera_data(camera_name)
        return self.camera_data_cache[key]
    
    def _get_video_capture(self, session_path: Path, camera_name: str, 
                           verbose: bool = False) -> Optional[cv2.VideoCapture]:
        """Get or cache video capture."""
        key = (str(session_path), camera_name)
        if key not in self.video_cap_cache:
            loader = self._get_loader(session_path)
            video_file = self._find_video_file(loader, camera_name)
            if video_file:
                if verbose:
                    print(f"  Loading video for camera '{camera_name}': {video_file}")
                cap = cv2.VideoCapture(str(video_file))
                if cap.isOpened():
                    self.video_cap_cache[key] = cap
                else:
                    print(f"  Warning: Failed to open video file: {video_file}")
                    return None
            else:
                if verbose:
                    print(f"  Warning: No video file found for camera '{camera_name}'")
                return None
        return self.video_cap_cache[key]
    
    def _find_video_file(self, loader: SLEAPDataLoader, camera_name: str) -> Optional[Path]:
        """Find video file for a camera."""
        if loader.data_structure_type == 'camera_dirs':
            camera_dir = loader.project_path / camera_name
            
            try:
                h5_candidates = list(camera_dir.glob("*.analysis.h5"))
                if not h5_candidates:
                    h5_candidates = list(camera_dir.glob("*.predictions.h5"))
                
                if h5_candidates:
                    h5_file = h5_candidates[0]
                    with h5py.File(h5_file, 'r') as f:
                        if 'video_path' in f:
                            raw_path = f['video_path'][()]
                            if isinstance(raw_path, bytes):
                                video_path_str = raw_path.decode('utf-8')
                            else:
                                video_path_str = str(raw_path)
                            video_filename = Path(video_path_str).name
                            candidate_path = camera_dir / video_filename
                            if candidate_path.exists():
                                return candidate_path
            except Exception:
                pass
            
            video_files = list(camera_dir.glob("*.mp4"))
            return video_files[0] if video_files else None
            
        elif loader.data_structure_type == 'session_dirs':
            if loader.session_name:
                session_dir = loader.project_path / loader.session_name
                camera_h5_files = list(session_dir.glob(f"*_cam{camera_name}.h5"))
                if camera_h5_files:
                    camera_h5_file = camera_h5_files[0]
                    try:
                        with h5py.File(camera_h5_file, 'r') as f:
                            if 'video_path' in f:
                                raw_path = f['video_path'][()]
                                if isinstance(raw_path, bytes):
                                    video_path_str = raw_path.decode('utf-8')
                                else:
                                    video_path_str = str(raw_path)
                                video_filename = Path(video_path_str).name
                                candidate_path = camera_h5_file.parent / video_filename
                                if candidate_path.exists():
                                    return candidate_path
                    except Exception:
                        pass
            
            video_files = list(loader.project_path.glob(f"*_cam{camera_name}.mp4"))
            return video_files[0] if video_files else None
        
        return None
    
    def discover_multiview_frames(self, session_path: Path, 
                                   max_frames: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Discover all frames that have multi-view data in a session.
        
        Returns:
            List of frame info dicts with keys: frame_idx, available_cameras
        """
        loader = self._get_loader(session_path)
        
        # Collect annotated frame indices per camera
        camera_frames: Dict[str, Set[int]] = {}
        
        for camera_name in loader.camera_views:
            try:
                camera_data = self._get_camera_data(session_path, camera_name)
                annotated = self._get_annotated_frames(camera_data, loader.data_structure_type)
                camera_frames[camera_name] = set(annotated)
            except Exception as e:
                print(f"Warning: Failed to get frames for camera {camera_name}: {e}")
                camera_frames[camera_name] = set()
        
        # Find frames that appear in multiple cameras
        all_frames: Set[int] = set()
        for frames in camera_frames.values():
            all_frames.update(frames)
        
        multiview_frames = []
        for frame_idx in sorted(all_frames):
            available_cameras = [cam for cam, frames in camera_frames.items() 
                               if frame_idx in frames]
            
            if len(available_cameras) >= self.min_views_per_sample:
                multiview_frames.append({
                    'frame_idx': frame_idx,
                    'available_cameras': available_cameras,
                    'num_views': len(available_cameras)
                })
        
        if max_frames is not None:
            multiview_frames = multiview_frames[:max_frames]
        
        return multiview_frames
    
    def _get_annotated_frames(self, camera_data: Dict[str, Any], 
                              data_structure_type: str) -> List[int]:
        """Get list of frame indices that have annotation data."""
        annotated_frames = []
        
        if data_structure_type == 'camera_dirs':
            if 'instances' in camera_data:
                instances = camera_data['instances']
                if len(instances) > 0:
                    frame_ids = np.unique(instances['frame_id'])
                    annotated_frames = sorted(frame_ids.tolist())
                    
        elif data_structure_type == 'session_dirs':
            if 'tracks' in camera_data:
                tracks = camera_data['tracks']
                if len(tracks.shape) >= 4:
                    num_frames = tracks.shape[3]
                    for frame_idx in range(num_frames):
                        frame_tracks = tracks[:, :, :, frame_idx]
                        if np.any(frame_tracks != 0):
                            annotated_frames.append(frame_idx)
        
        return annotated_frames
    
    def get_video_info(self, session_path: Path, verbose: bool = True) -> Dict[str, Any]:
        """Get video information for a session."""
        loader = self._get_loader(session_path)
        
        info = {
            'session_name': session_path.name,
            'cameras': loader.camera_views,
            'fps': None,
            'total_frames': 0,
            'video_files': {}
        }
        
        # Load video captures for all cameras and record their paths
        for camera_name in loader.camera_views:
            video_file = self._find_video_file(loader, camera_name)
            if video_file:
                info['video_files'][camera_name] = str(video_file)
            cap = self._get_video_capture(session_path, camera_name, verbose=verbose)
            if cap is not None and info['fps'] is None:
                info['fps'] = cap.get(cv2.CAP_PROP_FPS)
                info['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                info['frame_width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info['frame_height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return info
    
    def load_multiview_frame(self, session_path: Path, 
                              frame_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load a single multi-view frame (all camera views for one time instant).
        
        Args:
            session_path: Path to SLEAP session
            frame_info: Frame info dict with frame_idx and available_cameras
            
        Returns:
            Dict with images, keypoints, visibility for each view
        """
        frame_idx = frame_info['frame_idx']
        available_cameras = frame_info['available_cameras']
        loader = self._get_loader(session_path)
        
        view_images = []
        view_keypoints = []
        view_visibility = []
        view_camera_names = []
        view_camera_indices = []
        view_original_images = []  # For visualization
        
        for camera_name in available_cameras:
            cap = self._get_video_capture(session_path, camera_name)
            if cap is None:
                continue
            
            camera_data = self._get_camera_data(session_path, camera_name)
            
            try:
                # Get canonical camera index
                if camera_name in self.canonical_camera_order:
                    cam_idx = self.canonical_camera_order.index(camera_name)
                else:
                    cam_idx = len(self.canonical_camera_order)
                
                # Read video frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                view_original_images.append(frame_rgb.copy())
                
                # Extract keypoints (for bbox cropping)
                keypoints_2d, visibility = loader.extract_2d_keypoints(camera_data, frame_idx)
                
                # Preprocess image
                processed_image, transform_info = self._preprocess_image(frame_rgb, keypoints_2d)
                
                # Adjust keypoints for visualization
                adjusted_keypoints = self._adjust_keypoints_for_transform(keypoints_2d, transform_info)
                
                # Map to SMAL format
                preprocessed_size = (self.target_resolution, self.target_resolution)
                smal_keypoints, smal_visibility = loader.map_keypoints_to_smal_model(
                    adjusted_keypoints, visibility, preprocessed_size
                )
                
                # Sanitize
                smal_keypoints = np.nan_to_num(smal_keypoints, nan=0.0, posinf=0.0, neginf=0.0)
                smal_visibility = np.nan_to_num(smal_visibility, nan=0.0, posinf=0.0, neginf=0.0)
                
                view_images.append(processed_image.astype(np.float32))
                view_keypoints.append(smal_keypoints.astype(np.float32))
                view_visibility.append(smal_visibility.astype(np.float32))
                view_camera_names.append(camera_name)
                view_camera_indices.append(cam_idx)
                
            except Exception as e:
                print(f"Warning: Failed to load camera {camera_name} frame {frame_idx}: {e}")
                continue
        
        if len(view_images) < self.min_views_per_sample:
            return None
        
        return {
            'images': view_images,  # List of (H, W, 3) float32 in [0, 1]
            'original_images': view_original_images,  # List of (H, W, 3) uint8
            'keypoints_2d': np.stack(view_keypoints, axis=0),  # (num_views, N_joints, 2)
            'keypoint_visibility': np.stack(view_visibility, axis=0),  # (num_views, N_joints)
            'camera_names': view_camera_names,
            'camera_indices': np.array(view_camera_indices, dtype=np.int32),
            'num_views': len(view_images),
            'frame_idx': frame_idx
        }
    
    def _preprocess_image(self, image: np.ndarray, 
                          keypoints_2d: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess image for inference."""
        original_h, original_w = image.shape[:2]
        transform_info = {
            'original_size': (original_h, original_w),
            'crop_offset': (0, 0),
            'crop_size': (original_h, original_w),
            'scale_factor': 1.0,
            'mode': self.crop_mode
        }
        
        if self.crop_mode == 'centred':
            crop_size = min(original_h, original_w)
            y_offset = (original_h - crop_size) // 2
            x_offset = (original_w - crop_size) // 2
            image = image[y_offset:y_offset + crop_size, x_offset:x_offset + crop_size]
            transform_info['crop_offset'] = (y_offset, x_offset)
            transform_info['crop_size'] = (crop_size, crop_size)
            scale_factor = self.target_resolution / crop_size
            transform_info['scale_factor'] = scale_factor
            image = cv2.resize(image, (self.target_resolution, self.target_resolution))
            
        elif self.crop_mode == 'bbox_crop':
            if keypoints_2d is None or len(keypoints_2d) == 0:
                scale_y = self.target_resolution / original_h
                scale_x = self.target_resolution / original_w
                transform_info['scale_factor'] = (scale_y, scale_x)
                image = cv2.resize(image, (self.target_resolution, self.target_resolution))
            else:
                valid_kpts = keypoints_2d[~np.isnan(keypoints_2d).any(axis=1)]
                valid_kpts = valid_kpts[(valid_kpts[:, 0] > 0) & (valid_kpts[:, 1] > 0)]
                
                if len(valid_kpts) == 0:
                    scale_y = self.target_resolution / original_h
                    scale_x = self.target_resolution / original_w
                    transform_info['scale_factor'] = (scale_y, scale_x)
                    image = cv2.resize(image, (self.target_resolution, self.target_resolution))
                else:
                    x_min, y_min = valid_kpts.min(axis=0)
                    x_max, y_max = valid_kpts.max(axis=0)
                    center_x = (x_min + x_max) / 2.0
                    center_y = (y_min + y_max) / 2.0
                    width = x_max - x_min
                    height = y_max - y_min
                    bbox_size = max(width, height) * 1.05
                    
                    half_size = bbox_size / 2.0
                    x_start = center_x - half_size
                    y_start = center_y - half_size
                    x_end = center_x + half_size
                    y_end = center_y + half_size
                    
                    # Clamp to image bounds
                    if x_start < 0:
                        x_end = min(original_w, x_end - x_start)
                        x_start = 0
                    if x_end > original_w:
                        x_start = max(0, x_start - (x_end - original_w))
                        x_end = original_w
                    if y_start < 0:
                        y_end = min(original_h, y_end - y_start)
                        y_start = 0
                    if y_end > original_h:
                        y_start = max(0, y_start - (y_end - original_h))
                        y_end = original_h
                    
                    actual_width = x_end - x_start
                    actual_height = y_end - y_start
                    
                    image = image[int(y_start):int(y_end), int(x_start):int(x_end)]
                    transform_info['crop_offset'] = (int(y_start), int(x_start))
                    transform_info['crop_size'] = (int(actual_height), int(actual_width))
                    scale_factor = self.target_resolution / max(actual_height, actual_width)
                    transform_info['scale_factor'] = scale_factor
                    image = cv2.resize(image, (self.target_resolution, self.target_resolution))
        else:
            scale_y = self.target_resolution / original_h
            scale_x = self.target_resolution / original_w
            transform_info['scale_factor'] = (scale_y, scale_x)
            image = cv2.resize(image, (self.target_resolution, self.target_resolution))
        
        image = image.astype(np.float32) / 255.0
        return image, transform_info
    
    def _adjust_keypoints_for_transform(self, keypoints_2d: np.ndarray,
                                         transform_info: Dict[str, Any]) -> np.ndarray:
        """Adjust 2D keypoints based on image preprocessing transformations."""
        adjusted_keypoints = keypoints_2d.copy()
        
        if transform_info['mode'] in ['centred', 'bbox_crop']:
            y_offset, x_offset = transform_info['crop_offset']
            adjusted_keypoints[:, 0] -= x_offset
            adjusted_keypoints[:, 1] -= y_offset
            scale_factor = transform_info['scale_factor']
            if isinstance(scale_factor, tuple):
                adjusted_keypoints[:, 0] *= scale_factor[1]
                adjusted_keypoints[:, 1] *= scale_factor[0]
            else:
                adjusted_keypoints *= scale_factor
        else:
            scale_y, scale_x = transform_info['scale_factor']
            adjusted_keypoints[:, 0] *= scale_x
            adjusted_keypoints[:, 1] *= scale_y
        
        return adjusted_keypoints
    
    def close(self):
        """Release all video captures."""
        for cap in self.video_cap_cache.values():
            try:
                cap.release()
            except Exception:
                pass


def load_multiview_model_from_checkpoint(checkpoint_path: str, 
                                          device: str) -> Tuple[MultiViewSMILImageRegressor, Dict[str, Any]]:
    """
    Load a trained MultiViewSMILImageRegressor model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: PyTorch device
        
    Returns:
        Tuple of (model, model_config)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading multi-view checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration from checkpoint
    ckpt_config = checkpoint.get('config', {})
    
    # Get state dict for inferring model structure
    state_dict = checkpoint['model_state_dict']
    
    # Get model configuration
    backbone_name = ckpt_config.get('backbone_name', 'vit_large_patch16_224')
    head_type = ckpt_config.get('head_type', 'transformer_decoder')
    hidden_dim = ckpt_config.get('hidden_dim', 512)
    rotation_representation = ckpt_config.get('rotation_representation', '6d')
    scale_trans_mode = ckpt_config.get('scale_trans_mode', 'separate')
    freeze_backbone = ckpt_config.get('freeze_backbone', True)
    use_unity_prior = ckpt_config.get('use_unity_prior', False)
    use_ue_scaling = ckpt_config.get('use_ue_scaling', False)
    
    # Multi-view specific config
    # CRITICAL: Infer max_views from state dict to ensure camera head count matches
    # view_embeddings.weight shape determines the number of camera positions
    max_views = None
    num_camera_heads = None
    
    if 'view_embeddings.weight' in state_dict:
        max_views = state_dict['view_embeddings.weight'].shape[0]
        print(f"Inferred max_views from view_embeddings.weight: {max_views}")
    
    # Also check camera_heads count
    camera_head_keys = [k for k in state_dict.keys() if k.startswith('camera_heads.')]
    if camera_head_keys:
        # Extract indices from keys like 'camera_heads.0.fc1.weight'
        indices = set()
        for k in camera_head_keys:
            parts = k.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                indices.add(int(parts[1]))
        num_camera_heads = max(indices) + 1 if indices else None
        print(f"Inferred num_camera_heads from state dict: {num_camera_heads}")
    
    # Use the detected values, or fall back to config/defaults
    if max_views is None:
        max_views = ckpt_config.get('max_views', 4)
    
    # Ensure max_views matches camera heads
    if num_camera_heads is not None and num_camera_heads != max_views:
        print(f"WARNING: view_embeddings says {max_views} views but found {num_camera_heads} camera heads")
        print(f"Using {num_camera_heads} to match checkpoint camera heads")
        max_views = num_camera_heads
    
    cross_attention_layers = ckpt_config.get('cross_attention_layers', 2)
    cross_attention_heads = ckpt_config.get('cross_attention_heads', 8)
    cross_attention_dropout = ckpt_config.get('cross_attention_dropout', 0.1)
    transformer_config = ckpt_config.get('transformer_config', {})
    
    # Get canonical camera order from checkpoint or create placeholder
    # The actual camera names don't matter for inference - what matters is the index mapping
    canonical_camera_order = ckpt_config.get('canonical_camera_order', None)
    if canonical_camera_order is None:
        # Create placeholder list - indices are what matter, not names
        canonical_camera_order = [f"Camera{i}" for i in range(max_views)]
        print(f"Created placeholder canonical camera order (indices 0-{max_views-1})")
    else:
        print(f"Loaded canonical camera order from checkpoint: {canonical_camera_order}")
    
    print(f"Model configuration:")
    print(f"  Backbone: {backbone_name}")
    print(f"  Head type: {head_type}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Max views: {max_views}")
    print(f"  Num canonical cameras (camera heads): {len(canonical_camera_order)}")
    print(f"  Cross-attention layers: {cross_attention_layers}")
    print(f"  Cross-attention heads: {cross_attention_heads}")
    print(f"  Rotation representation: {rotation_representation}")
    print(f"  Scale/trans mode: {scale_trans_mode}")
    print(f"  Freeze backbone: {freeze_backbone}")
    print(f"  Use unity prior: {use_unity_prior}")
    print(f"  Use UE scaling: {use_ue_scaling}")
    
    # Determine input resolution
    if backbone_name.startswith('vit'):
        input_resolution = 224
    else:
        input_resolution = 512
    
    # Create model
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
        transformer_config=transformer_config
    )
    
    model = model.to(device)
    
    # Filter out SMAL optimization parameters from state_dict (already loaded above)
    smal_optimization_params = [
        'global_rotation', 'joint_rotations', 'trans', 'log_beta_scales',
        'betas_trans', 'betas', 'fov', 'target_joints', 'target_visibility'
    ]
    
    nn_state_dict = {}
    skipped_keys = []
    for k, v in state_dict.items():
        if not any(k == param or k.startswith(param + '.') for param in smal_optimization_params):
            nn_state_dict[k] = v
        else:
            skipped_keys.append(k)
    
    print(f"\nState dict analysis:")
    print(f"  Total keys in checkpoint: {len(state_dict)}")
    print(f"  Keys to load: {len(nn_state_dict)}")
    print(f"  Skipped SMAL params: {len(skipped_keys)}")
    
    # Print some key model components to verify they exist
    key_components = ['view_embeddings', 'view_fusion', 'body_aggregator', 'camera_heads', 
                      'transformer_head', 'backbone', 'fc1', 'regressor']
    print(f"\n  Key components in checkpoint state dict:")
    for comp in key_components:
        matching = [k for k in nn_state_dict.keys() if comp in k]
        if matching:
            print(f"    {comp}: {len(matching)} parameters")
            # Print first few
            for k in matching[:3]:
                print(f"      - {k}: {nn_state_dict[k].shape}")
    
    missing_keys, unexpected_keys = model.load_state_dict(nn_state_dict, strict=False)
    
    if missing_keys:
        print(f"\n  Missing keys (using defaults): {len(missing_keys)} parameters")
        # Print first 10 missing keys
        for k in missing_keys[:10]:
            print(f"    - {k}")
        if len(missing_keys) > 10:
            print(f"    ... and {len(missing_keys) - 10} more")
    
    if unexpected_keys:
        print(f"\n  Unexpected keys (ignored): {len(unexpected_keys)} parameters")
        for k in unexpected_keys[:10]:
            print(f"    - {k}")
        if len(unexpected_keys) > 10:
            print(f"    ... and {len(unexpected_keys) - 10} more")
    
    model.eval()
    print("Model loaded and set to evaluation mode")
    
    return model, ckpt_config


def run_multiview_inference(model: MultiViewSMILImageRegressor,
                            frame_data: Dict[str, Any],
                            device: str,
                            debug: bool = False) -> Dict[str, torch.Tensor]:
    """
    Run inference on a multi-view frame.
    
    Args:
        model: MultiViewSMILImageRegressor
        frame_data: Dict containing images, camera_indices, etc.
        device: PyTorch device
        debug: Print debug information
        
    Returns:
        Dict of predicted parameters
    """
    with torch.no_grad():
        # Prepare images
        images_per_view = []
        for img in frame_data['images']:
            # Convert to tensor (C, H, W)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            images_per_view.append(img_tensor)
        
        # Prepare camera indices
        camera_indices = torch.tensor([frame_data['camera_indices']], device=device)
        
        # Prepare view mask
        num_views = frame_data['num_views']
        view_mask = torch.ones(1, num_views, dtype=torch.bool, device=device)
        
        if debug:
            print(f"\n[DEBUG] Frame {frame_data['frame_idx']}:")
            print(f"  Camera names: {frame_data['camera_names']}")
            print(f"  Camera indices: {frame_data['camera_indices']}")
            print(f"  Num views: {num_views}")
        
        # Forward pass
        predicted_params = model.forward_multiview(images_per_view, camera_indices, view_mask)
        
        if debug:
            print(f"\n[DEBUG] Predicted parameters:")
            print(f"  global_rot shape: {predicted_params['global_rot'].shape}")
            print(f"  global_rot values: {predicted_params['global_rot'][0, :6].cpu().numpy()}")
            print(f"  joint_rot shape: {predicted_params['joint_rot'].shape}")
            print(f"  betas shape: {predicted_params['betas'].shape}")
            print(f"  betas values (first 5): {predicted_params['betas'][0, :5].cpu().numpy()}")
            print(f"  trans shape: {predicted_params['trans'].shape}")
            print(f"  trans values: {predicted_params['trans'][0].cpu().numpy()}")
            
            # Print per-view camera params
            fov_per_view = predicted_params.get('fov_per_view', [])
            cam_rot_per_view = predicted_params.get('cam_rot_per_view', [])
            cam_trans_per_view = predicted_params.get('cam_trans_per_view', [])
            print(f"  num per-view camera params: {len(fov_per_view)}")
            for v in range(min(3, len(fov_per_view))):
                print(f"    View {v}: fov={fov_per_view[v][0, 0].item():.2f}, "
                      f"cam_trans={cam_trans_per_view[v][0].cpu().numpy()}")
        
        return predicted_params


def create_multiview_visualization_grid(model: MultiViewSMILImageRegressor,
                                         frame_data: Dict[str, Any],
                                         predicted_params: Dict[str, torch.Tensor],
                                         device: str,
                                         img_size: int = 224,
                                         render_mesh: bool = True) -> np.ndarray:
    """
    Create a grid visualization for multi-view inference.
    
    Layout:
    ┌────────────┬────────────┬────────────┬─────┐
    │  Input V0  │  Input V1  │  Input V2  │ ... │  <- Input images with GT keypoints
    ├────────────┼────────────┼────────────┼─────┤
    │  Pred V0   │  Pred V1   │  Pred V2   │ ... │  <- Predicted keypoints overlay
    ├────────────┼────────────┼────────────┼─────┤
    │  Mesh V0   │  Mesh V1   │  Mesh V2   │ ... │  <- Rendered 3D meshes (if render_mesh)
    └────────────┴────────────┴────────────┴─────┘
    
    Args:
        model: MultiViewSMILImageRegressor
        frame_data: Input frame data
        predicted_params: Predicted parameters from model
        device: PyTorch device
        img_size: Size of each image in the grid
        render_mesh: Whether to render 3D meshes in a third row
        
    Returns:
        Numpy array of visualization (H, W, 3) uint8
    """
    num_views = frame_data['num_views']
    margin = 5
    
    # Grid dimensions - 3 rows if render_mesh, 2 rows otherwise
    num_rows = 3 if render_mesh else 2
    grid_width = num_views * img_size + (num_views + 1) * margin
    grid_height = num_rows * img_size + (num_rows + 1) * margin
    
    # Create canvas with dark background
    canvas = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 40
    
    # Get GT keypoints
    target_keypoints = frame_data['keypoints_2d']  # (num_views, n_joints, 2)
    target_visibility = frame_data['keypoint_visibility']  # (num_views, n_joints)
    
    # Get per-view camera params for rendering predicted keypoints
    fov_per_view = predicted_params.get('fov_per_view', [])
    cam_rot_per_view = predicted_params.get('cam_rot_per_view', [])
    cam_trans_per_view = predicted_params.get('cam_trans_per_view', [])
    
    for v in range(num_views):
        x_offset = margin + v * (img_size + margin)
        
        # === ROW 1: Input images with GT keypoints ===
        input_img = frame_data['images'][v].copy()
        
        # Ensure correct size
        if input_img.shape[0] != img_size or input_img.shape[1] != img_size:
            input_img = cv2.resize(input_img, (img_size, img_size))
        
        # Convert to uint8
        if input_img.max() <= 1.0:
            input_img_uint8 = (input_img * 255).astype(np.uint8)
        else:
            input_img_uint8 = input_img.astype(np.uint8)
        
        # Ensure RGB
        if len(input_img_uint8.shape) == 2:
            input_img_uint8 = np.stack([input_img_uint8] * 3, axis=-1)
        elif input_img_uint8.shape[-1] == 4:
            input_img_uint8 = input_img_uint8[:, :, :3]
        
        # Draw GT keypoints on input image (green circles)
        pil_input = Image.fromarray(input_img_uint8)
        draw_input = ImageDraw.Draw(pil_input)
        gt_kps = target_keypoints[v] * img_size  # Scale to pixel coords
        gt_vis = target_visibility[v]
        for j, (y, x) in enumerate(gt_kps):
            if gt_vis[j] > 0.5:
                x, y = float(x), float(y)
                if 0 <= x < img_size and 0 <= y < img_size:
                    draw_input.ellipse([x - 3, y - 3, x + 3, y + 3], outline='green', width=2)
        
        canvas[margin:margin + img_size, x_offset:x_offset + img_size] = np.array(pil_input)
        
        # === ROW 2: Predicted keypoints visualization ===
        # Use copy of input image as background
        pred_img = input_img_uint8.copy()
        pil_pred = Image.fromarray(pred_img)
        draw_pred = ImageDraw.Draw(pil_pred)
        
        # Draw GT keypoints (green circles) for reference
        for j, (y, x) in enumerate(gt_kps):
            if gt_vis[j] > 0.5:
                x, y = float(x), float(y)
                if 0 <= x < img_size and 0 <= y < img_size:
                    draw_pred.ellipse([x - 3, y - 3, x + 3, y + 3], outline='green', width=2)
        
        # Render predicted 2D keypoints (red crosses)
        if v < len(fov_per_view):
            try:
                fov = fov_per_view[v]
                cam_rot = cam_rot_per_view[v]
                cam_trans = cam_trans_per_view[v]
                
                with torch.no_grad():
                    rendered_joints = model._render_keypoints_with_camera(
                        predicted_params, fov, cam_rot, cam_trans
                    )
                    pred_kps = rendered_joints[0].detach().cpu().numpy()
                    pred_kps = pred_kps * img_size  # Scale to pixel coords
                
                for j, (y, x) in enumerate(pred_kps):
                    x, y = float(x), float(y)
                    if 0 <= x < img_size and 0 <= y < img_size:
                        # Red cross for predictions
                        draw_pred.line([x - 4, y, x + 4, y], fill='red', width=2)
                        draw_pred.line([x, y - 4, x, y + 4], fill='red', width=2)
            except Exception as e:
                print(f"Warning: Failed to render predicted keypoints for view {v}: {e}")
        
        row2_y_start = 2 * margin + img_size
        canvas[row2_y_start:row2_y_start + img_size,
               x_offset:x_offset + img_size] = np.array(pil_pred)
        
        # === ROW 3: Rendered 3D meshes ===
        if render_mesh and v < len(fov_per_view):
            try:
                fov = fov_per_view[v]
                cam_rot = cam_rot_per_view[v]
                cam_trans = cam_trans_per_view[v]
                
                # Render mesh for this view
                mesh_img = render_mesh_for_view(
                    model, predicted_params, fov, cam_rot, cam_trans, device, img_size
                )
                
                row3_y_start = 3 * margin + 2 * img_size
                canvas[row3_y_start:row3_y_start + img_size,
                       x_offset:x_offset + img_size] = mesh_img
            except Exception as e:
                print(f"Warning: Failed to render mesh for view {v}: {e}")
                # Fill with placeholder
                row3_y_start = 3 * margin + 2 * img_size
                placeholder = np.ones((img_size, img_size, 3), dtype=np.uint8) * 60
                canvas[row3_y_start:row3_y_start + img_size,
                       x_offset:x_offset + img_size] = placeholder
    
    # Add labels
    try:
        pil_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_canvas)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Row labels
        draw.text((5, margin + img_size // 2 - 6), "Input", fill=(255, 255, 255), font=font)
        draw.text((5, 2 * margin + img_size + img_size // 2 - 6), "KP", fill=(255, 255, 255), font=font)
        if render_mesh:
            draw.text((5, 3 * margin + 2 * img_size + img_size // 2 - 6), "Mesh", fill=(255, 255, 255), font=font)
        
        # View labels with camera index
        for v in range(num_views):
            x_pos = margin + v * (img_size + margin) + img_size // 2 - 30
            cam_name = frame_data['camera_names'][v]
            cam_idx = frame_data['camera_indices'][v]
            draw.text((x_pos, 2), f"{cam_name}[{cam_idx}]", fill=(255, 255, 255), font=font)
        
        # Frame index
        frame_idx = frame_data['frame_idx']
        draw.text((grid_width - 80, 2), f"Frame: {frame_idx}", fill=(255, 255, 0), font=font)
        
        # Legend
        draw.text((5, grid_height - 15), "○ GT  + Pred", fill=(255, 255, 255), font=font)
        
        canvas = np.array(pil_canvas)
    except Exception as e:
        print(f"Warning: Failed to add labels: {e}")
    
    return canvas


def create_rendered_view_with_keypoints(model: MultiViewSMILImageRegressor,
                                         predicted_params: Dict[str, torch.Tensor],
                                         view_idx: int,
                                         target_keypoints: np.ndarray,
                                         target_visibility: np.ndarray,
                                         device: str,
                                         img_size: int,
                                         render_mesh: bool = True) -> np.ndarray:
    """
    Create a rendered view with mesh and keypoint overlays.
    
    Args:
        model: MultiViewSMILImageRegressor
        predicted_params: Predicted parameters from forward_multiview
        view_idx: Which view to render
        target_keypoints: GT keypoints for this view (n_joints, 2)
        target_visibility: GT visibility for this view (n_joints,)
        device: PyTorch device
        img_size: Output image size
        render_mesh: Whether to render the 3D mesh
        
    Returns:
        Rendered image with mesh and keypoints (img_size, img_size, 3) uint8
    """
    # Get per-view camera parameters
    fov_per_view = predicted_params.get('fov_per_view', None)
    cam_rot_per_view = predicted_params.get('cam_rot_per_view', None)
    cam_trans_per_view = predicted_params.get('cam_trans_per_view', None)
    
    # Get camera params for this view
    if fov_per_view is not None and view_idx < len(fov_per_view):
        fov = fov_per_view[view_idx]
        cam_rot = cam_rot_per_view[view_idx]
        cam_trans = cam_trans_per_view[view_idx]
    else:
        # Fallback
        fov = predicted_params.get('fov', torch.tensor([[30.0]], device=device))
        cam_rot = torch.eye(3, device=device).unsqueeze(0)
        cam_trans = torch.zeros(1, 3, device=device)
    
    # Create base image - render mesh if requested
    if render_mesh:
        try:
            img = render_mesh_for_view(
                model, predicted_params, fov, cam_rot, cam_trans, device, img_size
            )
        except Exception as e:
            print(f"Mesh rendering failed for view {view_idx}: {e}")
            img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 60
            for i in range(img_size):
                img[i, :, 0] = min(255, 60 + view_idx * 30)
    else:
        # Simple background without mesh
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 60
        for i in range(img_size):
            img[i, :, 0] = min(255, 60 + view_idx * 30)
    
    # Render 2D keypoints
    pred_kps = None
    try:
        with torch.no_grad():
            rendered_joints = model._render_keypoints_with_camera(
                predicted_params, fov, cam_rot, cam_trans
            )
            pred_kps = rendered_joints[0].detach().cpu().numpy()
            pred_kps = pred_kps * img_size
    except Exception as e:
        print(f"Keypoint rendering failed for view {view_idx}: {e}")
    
    # Scale GT keypoints to image size
    gt_kps = target_keypoints * img_size
    gt_vis = target_visibility
    
    # Draw on image
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Draw GT keypoints (green circles)
    for j, (y, x) in enumerate(gt_kps):
        if gt_vis[j] > 0.5:
            x, y = float(x), float(y)
            if 0 <= x < img_size and 0 <= y < img_size:
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline='green', width=2)
    
    # Draw predicted keypoints (red crosses)
    if pred_kps is not None:
        for j, (y, x) in enumerate(pred_kps):
            x, y = float(x), float(y)
            if 0 <= x < img_size and 0 <= y < img_size:
                draw.line([x - 4, y, x + 4, y], fill='red', width=2)
                draw.line([x, y - 4, x, y + 4], fill='red', width=2)
    
    # Add legend
    try:
        font = ImageFont.load_default()
        draw.text((5, img_size - 25), "○ GT", fill='green', font=font)
        draw.text((5, img_size - 12), "+ Pred", fill='red', font=font)
    except:
        pass
    
    return np.array(pil_img)


def render_mesh_for_view(model: MultiViewSMILImageRegressor,
                          predicted_params: Dict[str, torch.Tensor],
                          fov: torch.Tensor,
                          cam_rot: torch.Tensor,
                          cam_trans: torch.Tensor,
                          device: str,
                          img_size: int) -> np.ndarray:
    """
    Render the 3D mesh for a specific view using SMALFitter.
    
    Args:
        model: MultiViewSMILImageRegressor
        predicted_params: Predicted body parameters
        fov: Field of view for this view
        cam_rot: Camera rotation matrix for this view (1, 3, 3)
        cam_trans: Camera translation for this view (1, 3)
        device: PyTorch device
        img_size: Output image size
        
    Returns:
        Rendered mesh image (img_size, img_size, 3) uint8
    """
    with torch.no_grad():
        # Create a blank RGB tensor for the SMALFitter
        rgb_tensor = torch.zeros((1, 3, img_size, img_size), device=device)
        
        # Create temporary SMALFitter for rendering
        temp_fitter = SMALFitter(
            device=device,
            data_batch=rgb_tensor,
            batch_size=1,
            shape_family=config.SHAPE_FAMILY,
            use_unity_prior=False,
            rgb_only=True
        )
        
        # Convert 6D rotation to axis-angle if needed
        if model.rotation_representation == '6d':
            global_rot_aa = rotation_6d_to_axis_angle(predicted_params['global_rot'][0:1].detach())
            joint_rot_aa = rotation_6d_to_axis_angle(predicted_params['joint_rot'][0:1].detach())
        else:
            global_rot_aa = predicted_params['global_rot'][0:1].detach()
            joint_rot_aa = predicted_params['joint_rot'][0:1].detach()
        
        # Set body parameters
        temp_fitter.global_rotation.data = global_rot_aa.to(device)
        temp_fitter.joint_rotations.data = joint_rot_aa.to(device)
        temp_fitter.betas.data = predicted_params['betas'][0].detach().to(device)
        temp_fitter.trans.data = predicted_params['trans'][0:1].detach().to(device)
        
        # Set FOV
        if fov.dim() == 2:
            fov_val = fov[0, 0].detach().to(device)
        else:
            fov_val = fov[0].detach().to(device)
        temp_fitter.fov.data = fov_val.unsqueeze(0)
        
        # Set scale and translation parameters if available
        if 'log_beta_scales' in predicted_params and 'betas_trans' in predicted_params:
            if model.scale_trans_mode in ['separate', 'ignore']:
                try:
                    scale_weights = predicted_params['log_beta_scales'][0:1].detach()
                    trans_weights = predicted_params['betas_trans'][0:1].detach()
                    log_beta_scales_joint, betas_trans_joint = model._transform_separate_pca_weights_to_joint_values(
                        scale_weights, trans_weights
                    )
                    temp_fitter.log_beta_scales.data = log_beta_scales_joint.to(device)
                    temp_fitter.betas_trans.data = betas_trans_joint.to(device)
                except Exception:
                    pass
            else:
                temp_fitter.log_beta_scales.data = predicted_params['log_beta_scales'][0:1].detach().to(device)
                temp_fitter.betas_trans.data = predicted_params['betas_trans'][0:1].detach().to(device)
        
        # Set camera parameters
        temp_fitter.renderer.set_camera_parameters(
            R=cam_rot.detach().to(device),
            T=cam_trans.detach().to(device),
            fov=temp_fitter.fov.data
        )
        
        # Get mesh vertices
        verts, joints, Rs, v_shaped = temp_fitter.smal_model(
            temp_fitter.betas.expand(1, -1),
            torch.cat([
                temp_fitter.global_rotation.unsqueeze(1),
                temp_fitter.joint_rotations
            ], dim=1),
            betas_logscale=temp_fitter.log_beta_scales,
            betas_trans=temp_fitter.betas_trans,
            propagate_scaling=temp_fitter.propagate_scaling
        )
        
        # Apply translation
        verts = verts + temp_fitter.trans.unsqueeze(1)
        joints = joints + temp_fitter.trans.unsqueeze(1)
        
        # Get canonical joints
        canonical_joints = joints[:, config.CANONICAL_MODEL_JOINTS]
        
        # Prepare faces
        faces_batch = temp_fitter.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1)
        
        # Render with texture
        rendered_silhouettes, rendered_joints, rendered_image = temp_fitter.renderer(
            verts, canonical_joints, faces_batch, render_texture=True
        )
        
        # Convert to numpy
        rendered_np = rendered_image[0].permute(1, 2, 0).cpu().numpy()
        rendered_np = np.clip(rendered_np, 0, 1)
        
        return (rendered_np * 255).astype(np.uint8)


def process_multiview_session(model: MultiViewSMILImageRegressor,
                               helper: SLEAPMultiViewInferenceHelper,
                               session_path: Path,
                               output_folder: str,
                               device: str,
                               max_frames: Optional[int] = None,
                               output_fps: int = 10,
                               img_size: int = 224,
                               render_mesh: bool = True) -> str:
    """
    Process a single SLEAP session and generate output video.
    
    Args:
        model: MultiViewSMILImageRegressor
        helper: SLEAPMultiViewInferenceHelper
        session_path: Path to SLEAP session
        output_folder: Output directory
        device: PyTorch device
        max_frames: Maximum frames to process
        output_fps: Output video FPS
        img_size: Image size for visualization
        render_mesh: Whether to render the 3D mesh
        
    Returns:
        Path to output video
    """
    # Get video info
    video_info = helper.get_video_info(session_path, verbose=True)
    print(f"\nProcessing session: {video_info['session_name']}")
    print(f"  Cameras: {video_info['cameras']}")
    print(f"  Canonical camera order: {helper.canonical_camera_order}")
    print(f"  Input FPS: {video_info['fps']}")
    print(f"  Total frames: {video_info['total_frames']}")
    print(f"  Video files per camera:")
    for cam, vfile in video_info.get('video_files', {}).items():
        print(f"    {cam}: {vfile}")
    
    # Discover multi-view frames
    multiview_frames = helper.discover_multiview_frames(session_path, max_frames)
    print(f"  Multi-view frames: {len(multiview_frames)}")
    
    if len(multiview_frames) == 0:
        print("  No multi-view frames found, skipping session")
        return None
    
    # Calculate output video dimensions
    sample_frame = helper.load_multiview_frame(session_path, multiview_frames[0])
    if sample_frame is None:
        print("  Failed to load sample frame, skipping session")
        return None
    
    num_views = sample_frame['num_views']
    margin = 5
    grid_width = num_views * img_size + (num_views + 1) * margin
    # Calculate grid height based on number of rows (3 if render_mesh, 2 otherwise)
    num_rows = 3 if render_mesh else 2
    grid_height = num_rows * img_size + (num_rows + 1) * margin
    
    # Create output video writer
    output_video_path = os.path.join(output_folder, f"{session_path.name}_multiview_inference.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (grid_width, grid_height))
    
    # Verify video writer is initialized correctly
    if not out.isOpened():
        print(f"ERROR: Failed to initialize video writer for {output_video_path}")
        print(f"  Dimensions: {grid_width}x{grid_height}, FPS: {output_fps}")
        return None
    
    # Process frames
    frames_processed = 0
    for frame_info in tqdm(multiview_frames, desc=f"Processing {session_path.name}"):
        try:
            # Load frame data
            frame_data = helper.load_multiview_frame(session_path, frame_info)
            if frame_data is None:
                continue
            
            # Run inference (debug output for first frame)
            debug_first = (frames_processed == 0)
            predicted_params = run_multiview_inference(model, frame_data, device, debug=debug_first)
            
            # Create visualization
            vis_frame = create_multiview_visualization_grid(
                model, frame_data, predicted_params, device, img_size, render_mesh
            )
            
            # Verify frame dimensions match expected size
            if vis_frame.shape[0] != grid_height or vis_frame.shape[1] != grid_width:
                print(f"Warning: Frame dimensions mismatch! Expected {grid_width}x{grid_height}, got {vis_frame.shape[1]}x{vis_frame.shape[0]}")
                # Resize to match expected dimensions
                vis_frame = cv2.resize(vis_frame, (grid_width, grid_height))
            
            # Convert RGB to BGR for OpenCV
            vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            
            # Verify frame is valid before writing
            if vis_frame_bgr is None or vis_frame_bgr.size == 0:
                print(f"Warning: Invalid frame generated for frame {frame_info['frame_idx']}, skipping")
                continue
            
            out.write(vis_frame_bgr)
            
            frames_processed += 1
            
        except Exception as e:
            print(f"Warning: Failed to process frame {frame_info['frame_idx']}: {e}")
            continue
    
    out.release()
    print(f"  Processed {frames_processed} frames")
    print(f"  Output video: {output_video_path}")
    
    return output_video_path


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-view SMIL inference on SLEAP project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_inference_multiview.py --checkpoint multiview_checkpoints/best_model.pth \\
      --sleap-project /path/to/sleap/sessions --output-folder output/

  # With frame limit
  python run_inference_multiview.py -c model.pth -s /path/to/sleap -o output/ --max-frames 100

  # Custom settings
  python run_inference_multiview.py -c model.pth -s /path/to/sleap -o output/ \\
      --fps 30 --crop-mode centred --img-size 256
        """
    )
    
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                       help='Path to trained multi-view model checkpoint')
    parser.add_argument('-s', '--sleap_project', type=str, required=True,
                       help='Path to SLEAP project directory containing sessions')
    parser.add_argument('-o', '--output_folder', type=str, required=True,
                       help='Output folder for videos and results')
    
    # Processing options
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum frames to process per session (default: all)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Output video FPS (default: 10)')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size for visualization grid (default: 224)')
    parser.add_argument('--crop_mode', type=str, default='bbox_crop',
                       choices=['centred', 'default', 'bbox_crop'],
                       help='Image preprocessing mode (default: bbox_crop)')
    parser.add_argument('--min_views', type=int, default=2,
                       help='Minimum views required per frame (default: 2)')
    
    # Optional lookup tables
    parser.add_argument('--joint_lookup_table', type=str, default=None,
                       help='Path to joint lookup table CSV')
    parser.add_argument('--shape_betas_table', type=str, default=None,
                       help='Path to shape betas table CSV')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device for inference (default: auto)')
    
    # Rendering options
    parser.add_argument('--no_mesh', action='store_true',
                       help='Skip mesh rendering (faster, shows only keypoints)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MULTI-VIEW SMIL IMAGE REGRESSOR - INFERENCE")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"SLEAP project: {args.sleap_project}")
    print(f"Output folder: {args.output_folder}")
    print(f"Crop mode: {args.crop_mode}")
    print(f"Min views: {args.min_views}")
    print(f"Render mesh: {not args.no_mesh}")
    if args.max_frames:
        print(f"Max frames: {args.max_frames}")
    print("=" * 60)
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, using CPU")
        device = 'cpu'
    
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    try:
        # Load model first to get configuration
        print("\nLoading model...")
        model, model_config = load_multiview_model_from_checkpoint(args.checkpoint, device)
        
        # Determine target resolution from model
        if model.backbone_name.startswith('vit'):
            target_resolution = 224
        else:
            target_resolution = 512
        
        # Initialize SLEAP helper
        print("\nInitializing SLEAP data loader...")
        helper = SLEAPMultiViewInferenceHelper(
            project_path=args.sleap_project,
            target_resolution=target_resolution,
            crop_mode=args.crop_mode,
            min_views_per_sample=args.min_views,
            joint_lookup_table_path=args.joint_lookup_table,
            shape_betas_table_path=args.shape_betas_table
        )
        
        # Verify camera counts match
        print(f"\nCamera configuration:")
        print(f"  Model num_canonical_cameras: {model.num_canonical_cameras}")
        print(f"  SLEAP canonical camera order: {helper.canonical_camera_order}")
        print(f"  SLEAP camera count: {len(helper.canonical_camera_order)}")
        
        if len(helper.canonical_camera_order) > model.num_canonical_cameras:
            print(f"  WARNING: SLEAP has more cameras ({len(helper.canonical_camera_order)}) "
                  f"than model supports ({model.num_canonical_cameras})")
            print(f"  Cameras with index >= {model.num_canonical_cameras} will use modulo mapping")
        
        # Process each session
        render_mesh = not args.no_mesh
        output_videos = []
        for session_path in helper.sessions:
            output_video = process_multiview_session(
                model=model,
                helper=helper,
                session_path=session_path,
                output_folder=args.output_folder,
                device=device,
                max_frames=args.max_frames,
                output_fps=args.fps,
                img_size=args.img_size,
                render_mesh=render_mesh
            )
            if output_video:
                output_videos.append(output_video)
        
        # Clean up
        helper.close()
        
        print("\n" + "=" * 60)
        print("INFERENCE COMPLETE")
        print("=" * 60)
        print(f"Sessions processed: {len(output_videos)}")
        print(f"Output videos:")
        for video in output_videos:
            print(f"  - {video}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nInference interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
