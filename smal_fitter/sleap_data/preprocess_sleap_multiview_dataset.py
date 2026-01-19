#!/usr/bin/env python3
"""
Multi-View SLEAP Dataset Preprocessing Script

This script preprocesses multi-view SLEAP pose estimation datasets into optimized HDF5 format
compatible with the SMILify multi-view training pipeline. Each sample contains all synchronized
camera views for a single time instant.

Usage:
    python preprocess_sleap_multiview_dataset.py sessions_dir output.h5 [options]

Example:
    python preprocess_sleap_multiview_dataset.py /path/to/sleap/sessions multiview_sleap.h5 \\
        --joint_lookup_table /path/to/joint_lookup.csv \\
        --shape_betas_table /path/to/shape_betas.csv \\
        --max_views 4
"""

import os
import sys
import argparse
import time
import h5py
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import json

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import SLEAPDataLoader with fallback
try:
    from sleap_data_loader import SLEAPDataLoader
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from sleap_data_loader import SLEAPDataLoader

# Import SLEAP3DDataLoader for 3D keypoints and camera parameters
try:
    from sleap_3d_loader import SLEAP3DDataLoader
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from smal_fitter.sleap_data.sleap_3d_loader import SLEAP3DDataLoader

import config


class SLEAPMultiViewPreprocessor:
    """
    Preprocesses multi-view SLEAP datasets into optimized HDF5 format.
    
    Key differences from single-view preprocessor:
    1. Groups samples by (session, frame_idx) to ensure time synchronization
    2. Stores all camera views together per sample
    3. Maintains canonical camera ordering for consistent training
    4. Handles variable number of views per sample (no padding)
    """
    
    def __init__(self, 
                 joint_lookup_table_path: Optional[str] = None,
                 shape_betas_table_path: Optional[str] = None,
                 target_resolution: int = 224,
                 backbone_name: str = 'vit_large_patch16_224',
                 jpeg_quality: int = 95,
                 chunk_size: int = 8,
                 compression: str = 'gzip',
                 compression_level: int = 6,
                 max_frames_per_session: Optional[int] = None,
                 crop_mode: str = 'default',
                 use_reprojections: bool = False,
                 confidence_threshold: float = 0.5,
                 min_views_per_sample: int = 2,
                 load_3d_data: bool = True,
                 debug: bool = False):
        """
        Initialize the multi-view SLEAP dataset preprocessor.
        
        Args:
            joint_lookup_table_path: Path to CSV lookup table for joint name mapping
            shape_betas_table_path: Path to CSV lookup table for ground truth shape betas
            target_resolution: Target image resolution for preprocessing
            backbone_name: Backbone network name for resolution selection
            jpeg_quality: JPEG compression quality (1-100)
            chunk_size: HDF5 chunk size (should match training batch size)
            compression: HDF5 compression algorithm
            compression_level: Compression level (1-9)
            max_frames_per_session: Maximum frames to process per session (None for all)
            crop_mode: Image cropping mode ('default', 'centred', 'bbox_crop'). Default: 'default'
            use_reprojections: If True, use reprojected 2D coordinates
            confidence_threshold: Minimum confidence for keypoints
            min_views_per_sample: Minimum views required per sample (skip samples with fewer)
            load_3d_data: If True, attempt to load 3D keypoints and camera parameters from SLEAP 3D data
            debug: If True, print detailed information about filtered outlier keypoints (default: False)
        """
        self.joint_lookup_table_path = joint_lookup_table_path
        self.shape_betas_table_path = shape_betas_table_path
        self.target_resolution = target_resolution
        self.backbone_name = backbone_name
        self.jpeg_quality = jpeg_quality
        self.chunk_size = chunk_size
        self.compression = compression
        self.compression_level = compression_level
        self.max_frames_per_session = max_frames_per_session
        self.crop_mode = crop_mode
        self.use_reprojections = use_reprojections
        self.confidence_threshold = confidence_threshold
        self.min_views_per_sample = min_views_per_sample
        self.load_3d_data = load_3d_data
        self.debug = debug
        
        # Validate crop_mode
        if self.crop_mode not in ['default', 'centred', 'bbox_crop']:
            raise ValueError(f"crop_mode must be 'default', 'centred', or 'bbox_crop', got: {self.crop_mode}")
        
        # Statistics tracking
        self.stats = {
            'total_sessions': 0,
            'total_multiview_samples': 0,
            'processed_samples': 0,
            'skipped_insufficient_views': 0,
            'failed_samples': 0,
            'sessions_processed': 0,
            'sessions_failed': 0,
            'views_per_sample_histogram': defaultdict(int),
            'sessions_with_3d_data': 0,
            'samples_with_3d_data': 0,
            'total_outlier_keypoints_filtered': 0,
            'samples_with_outliers_filtered': 0
        }
        
        # Canonical camera ordering (will be determined from first session)
        self.canonical_camera_order: List[str] = []
    
    def discover_sleap_sessions(self, sessions_dir: str) -> List[str]:
        """
        Discover all SLEAP session directories.
        
        Args:
            sessions_dir: Path to directory containing SLEAP sessions
            
        Returns:
            List of session directory paths
        """
        sessions_dir = Path(sessions_dir)
        sessions = []
        
        for item in sessions_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if self._is_sleap_session(item):
                    sessions.append(str(item))
        
        sessions.sort()
        return sessions
    
    def _is_sleap_session(self, session_path: Path) -> bool:
        """Check if a directory looks like a SLEAP session."""
        # Check for common SLEAP files
        sleap_indicators = ['calibration.toml', 'points3d.h5']
        
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
        for indicator in sleap_indicators:
            if (session_path / indicator).exists():
                return True
        
        return False
    
    def _establish_canonical_camera_order(self, sessions: List[str]) -> List[str]:
        """
        Establish canonical camera ordering from all sessions.
        
        Args:
            sessions: List of session paths
            
        Returns:
            List of camera names in canonical order
        """
        all_cameras: Set[str] = set()
        
        for session_path in sessions:
            try:
                loader = SLEAPDataLoader(
                    project_path=session_path,
                    lookup_table_path=self.joint_lookup_table_path,
                    shape_betas_path=self.shape_betas_table_path,
                    confidence_threshold=self.confidence_threshold
                )
                all_cameras.update(loader.camera_views)
            except Exception as e:
                print(f"Warning: Failed to load cameras from {session_path}: {e}")
        
        # Sort for consistent ordering
        canonical_order = sorted(list(all_cameras))
        print(f"Established canonical camera order: {canonical_order}")
        return canonical_order
    
    def _discover_multiview_frames(self, loader: SLEAPDataLoader, 
                                   session_path: str) -> List[Dict[str, Any]]:
        """
        Discover all frames that have multi-view data.
        
        Args:
            loader: SLEAPDataLoader instance
            session_path: Path to session
            
        Returns:
            List of frame info dicts with keys: frame_idx, available_cameras
        """
        # Collect all annotated frame indices per camera
        camera_frames: Dict[str, Set[int]] = {}
        
        for camera_name in loader.camera_views:
            try:
                camera_data = loader.load_camera_data(camera_name)
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
        
        # Apply frame limit if specified
        if self.max_frames_per_session is not None:
            multiview_frames = multiview_frames[:self.max_frames_per_session]
        
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
    
    def process_single_session(self, session_path: str) -> List[Dict[str, Any]]:
        """
        Process a single SLEAP session and extract multi-view samples.
        
        Args:
            session_path: Path to SLEAP session directory
            
        Returns:
            List of processed multi-view sample dictionaries
        """
        try:
            loader = SLEAPDataLoader(
                project_path=session_path,
                lookup_table_path=self.joint_lookup_table_path,
                shape_betas_path=self.shape_betas_table_path,
                confidence_threshold=self.confidence_threshold
            )
            
            # Try to load 3D data if enabled
            loader_3d = None
            has_3d_data = False
            if self.load_3d_data:
                try:
                    loader_3d = SLEAP3DDataLoader(session_path, session_idx=0)
                    has_3d_data = True
                    self.stats['sessions_with_3d_data'] += 1
                    print(f"Loaded 3D data for {Path(session_path).name}: {loader_3d.n_frames} frames, {loader_3d.n_keypoints} keypoints")
                except Exception as e:
                    print(f"Warning: Could not load 3D data for {session_path}: {e}")
                    loader_3d = None
                    has_3d_data = False
            
            # Discover multi-view frames
            multiview_frames = self._discover_multiview_frames(loader, session_path)
            
            if len(multiview_frames) == 0:
                print(f"No multi-view frames found in {session_path}")
                return []
            
            print(f"Found {len(multiview_frames)} multi-view frames in {Path(session_path).name}")
            
            # Pre-load camera data for all cameras
            camera_data_cache: Dict[str, Dict] = {}
            video_cap_cache: Dict[str, cv2.VideoCapture] = {}
            # Track current frame position per camera for optimized sequential reading
            video_frame_positions: Dict[str, int] = {}
            
            for camera_name in loader.camera_views:
                try:
                    camera_data_cache[camera_name] = loader.load_camera_data(camera_name)
                    video_file = self._find_video_file(loader, camera_name)
                    if video_file:
                        cap = cv2.VideoCapture(str(video_file))
                        if cap.isOpened():
                            video_cap_cache[camera_name] = cap
                            video_frame_positions[camera_name] = 0  # Initialize at frame 0
                except Exception as e:
                    print(f"Warning: Failed to load camera {camera_name}: {e}")
            
            # Process each multi-view frame
            samples = []
            ground_truth_betas = loader.get_ground_truth_shape_betas()
            
            for frame_info in tqdm(multiview_frames, 
                                   desc=f"Processing {Path(session_path).name}",
                                   leave=False):
                try:
                    sample = self._process_multiview_frame(
                        loader=loader,
                        frame_info=frame_info,
                        session_path=session_path,
                        camera_data_cache=camera_data_cache,
                        video_cap_cache=video_cap_cache,
                        video_frame_positions=video_frame_positions,
                        ground_truth_betas=ground_truth_betas,
                        loader_3d=loader_3d,
                        has_3d_data=has_3d_data
                    )
                    if sample is not None:
                        samples.append(sample)
                        self.stats['views_per_sample_histogram'][sample['num_views']] += 1
                        if has_3d_data and sample.get('has_3d_data', False):
                            self.stats['samples_with_3d_data'] += 1
                except Exception as e:
                    print(f"Warning: Failed to process frame {frame_info['frame_idx']}: {e}")
                    self.stats['failed_samples'] += 1
            
            # Close video captures
            for cap in video_cap_cache.values():
                cap.release()
            
            self.stats['sessions_processed'] += 1
            self.stats['processed_samples'] += len(samples)
            return samples
            
        except Exception as e:
            print(f"Error processing session {session_path}: {e}")
            self.stats['sessions_failed'] += 1
            return []
    
    def _process_multiview_frame(self, loader: SLEAPDataLoader,
                                  frame_info: Dict[str, Any],
                                  session_path: str,
                                  camera_data_cache: Dict[str, Dict],
                                  video_cap_cache: Dict[str, cv2.VideoCapture],
                                  video_frame_positions: Dict[str, int],
                                  ground_truth_betas: Optional[np.ndarray],
                                  loader_3d: Optional['SLEAP3DDataLoader'] = None,
                                  has_3d_data: bool = False) -> Optional[Dict[str, Any]]:
        """
        Process a single multi-view frame (all cameras for one time instant).
        
        Returns:
            Dict containing images and keypoints for all views, or None if failed
        """
        frame_idx = frame_info['frame_idx']
        available_cameras = frame_info['available_cameras']
        
        # Load 3D keypoints if available
        keypoints_3d = None
        outlier_joint_mask = None
        if has_3d_data and loader_3d is not None:
            try:
                if frame_idx < loader_3d.n_frames:
                    keypoints_3d_raw = loader_3d.get_3d_keypoints(frame_idx)  # (N_sleap, 3)
                    # IMPORTANT: Map 3D keypoints to the SMAL joint order used by training.
                    # 2D keypoints are already mapped via `loader.map_keypoints_to_smal_model(...)`,
                    # so 3D must match the same joint indexing/count (n_joints) for 3D supervision.
                    keypoints_3d = self._map_3d_keypoints_to_smal(loader, keypoints_3d_raw)
                    
                    # Filter outlier 3D keypoints (keypoints far from median)
                    if keypoints_3d is not None:
                        # Get joint names from loader if available
                        joint_names = getattr(loader, 'smal_joint_names', None)
                        if joint_names is None or len(joint_names) == 0:
                            # Fallback: try to get from config (already imported at module level)
                            try:
                                if hasattr(config, 'joint_names'):
                                    joint_names = config.joint_names
                            except Exception:
                                joint_names = None
                        
                        keypoints_3d, outlier_joint_mask = self._filter_outlier_3d_keypoints(
                            keypoints_3d, 
                            joint_names=joint_names
                        )
                        if outlier_joint_mask is not None and outlier_joint_mask.any():
                            num_outliers = outlier_joint_mask.sum()
                            self.stats['total_outlier_keypoints_filtered'] += num_outliers
                            self.stats['samples_with_outliers_filtered'] += 1
                            if self.debug:
                                print(f"  Filtered {num_outliers} outlier 3D keypoint(s) in frame {frame_idx}")
            except Exception as e:
                print(f"Warning: Failed to load 3D keypoints for frame {frame_idx}: {e}")
                keypoints_3d = None
                outlier_joint_mask = None
        
        # Collect data for each view
        view_images = []
        view_keypoints = []
        view_visibility = []
        view_camera_names = []
        view_camera_indices = []
        view_camera_intrinsics = []  # K matrices
        view_camera_extrinsics_R = []  # Rotation matrices
        view_camera_extrinsics_t = []  # Translation vectors
        view_image_sizes = []  # Original image sizes (width, height)
        
        for camera_name in available_cameras:
            if camera_name not in camera_data_cache:
                continue
            if camera_name not in video_cap_cache:
                continue
            
            camera_data = camera_data_cache[camera_name]
            cap = video_cap_cache[camera_name]
            
            try:
                # Get canonical index for this camera
                if camera_name in self.canonical_camera_order:
                    cam_idx = self.canonical_camera_order.index(camera_name)
                else:
                    cam_idx = len(self.canonical_camera_order)  # Unknown camera
                
                # Read video frame with optimized sequential reading
                # Since frames are processed in sorted order, we can read forward sequentially
                # which is much faster than seeking
                current_pos = video_frame_positions.get(camera_name, -1)
                
                if current_pos == frame_idx:
                    # Already at the right position, just read
                    ret, frame = cap.read()
                elif current_pos < frame_idx:
                    # Target frame is ahead - use optimal strategy based on gap size
                    gap = frame_idx - current_pos
                    if gap <= 10:
                        # Small gap: read forward sequentially (very fast, no seeks)
                        for _ in range(gap - 1):
                            cap.read()  # Skip intermediate frames
                        ret, frame = cap.read()  # Read target frame
                    else:
                        # Large gap: seek is faster than reading many frames
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                else:
                    # Target frame is behind - must seek backward (slow, but rare if sorted)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Update position after successful read
                video_frame_positions[camera_name] = frame_idx + 1
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract keypoints
                keypoints_2d, visibility = loader.extract_2d_keypoints(camera_data, frame_idx)
                if len(keypoints_2d) == 0:
                    continue
                
                # Get image size for mapping
                image_size = loader.get_camera_image_size(camera_name)
                
                # Preprocess image
                processed_image, transform_info = self._preprocess_image(frame_rgb, keypoints_2d)
                
                # Adjust keypoints
                adjusted_keypoints = self._adjust_keypoints_for_transform(keypoints_2d, transform_info)
                
                # Map to SMAL format
                preprocessed_size = (self.target_resolution, self.target_resolution)
                smal_keypoints, smal_visibility = loader.map_keypoints_to_smal_model(
                    adjusted_keypoints, visibility, preprocessed_size
                )
                
                # If 3D keypoints had outliers, mark corresponding 2D joints as invisible
                if outlier_joint_mask is not None and outlier_joint_mask.any():
                    # Ensure mask matches the number of joints
                    if len(outlier_joint_mask) <= len(smal_visibility):
                        smal_visibility[:len(outlier_joint_mask)] = np.where(
                            outlier_joint_mask[:len(smal_visibility)],
                            0.0,  # Set visibility to 0 for outliers
                            smal_visibility[:len(outlier_joint_mask)]
                        )
                
                # Sanitize
                smal_keypoints = self._sanitize_array(smal_keypoints, 0.0)
                smal_visibility = self._sanitize_array(smal_visibility, 0.0)
                
                # Encode image
                jpeg_image = self._encode_image_jpeg(processed_image)
                
                view_images.append(jpeg_image)
                view_keypoints.append(smal_keypoints.astype(np.float32))
                view_visibility.append(smal_visibility.astype(np.float32))
                view_camera_names.append(camera_name)
                view_camera_indices.append(cam_idx)
                
                # Extract camera parameters if 3D data is available
                if has_3d_data and loader_3d is not None:
                    try:
                        # Get camera parameters from 3D loader
                        cam_params = loader_3d.get_camera_parameters(camera_name)
                        K = cam_params['intrinsic']['K']
                        R = cam_params['extrinsic']['R']
                        t = cam_params['extrinsic']['t']
                        img_size = cam_params['image_size']
                        
                        view_camera_intrinsics.append(K.astype(np.float32))
                        view_camera_extrinsics_R.append(R.astype(np.float32))
                        view_camera_extrinsics_t.append(t.astype(np.float32))
                        view_image_sizes.append(np.array([img_size['width'], img_size['height']], dtype=np.int32))
                    except Exception as e:
                        print(f"Warning: Failed to extract camera parameters for {camera_name}: {e}")
                        # Add placeholder camera parameters
                        view_camera_intrinsics.append(np.eye(3, dtype=np.float32))
                        view_camera_extrinsics_R.append(np.eye(3, dtype=np.float32))
                        view_camera_extrinsics_t.append(np.zeros(3, dtype=np.float32))
                        view_image_sizes.append(np.array([image_size[0], image_size[1]], dtype=np.int32))
                else:
                    # Add placeholder camera parameters if no 3D data
                    view_camera_intrinsics.append(np.eye(3, dtype=np.float32))
                    view_camera_extrinsics_R.append(np.eye(3, dtype=np.float32))
                    view_camera_extrinsics_t.append(np.zeros(3, dtype=np.float32))
                    view_image_sizes.append(np.array([image_size[0], image_size[1]], dtype=np.int32))
                
            except Exception as e:
                print(f"Warning: Failed to process camera {camera_name} frame {frame_idx}: {e}")
                continue
        
        # Check minimum views
        if len(view_images) < self.min_views_per_sample:
            self.stats['skipped_insufficient_views'] += 1
            return None
        
        # Create multi-view sample
        sample = {
            # Per-view data
            'images_jpeg': view_images,  # List of JPEG bytes
            'keypoints_2d': np.stack(view_keypoints, axis=0),  # (num_views, N_joints, 2)
            'keypoint_visibility': np.stack(view_visibility, axis=0),  # (num_views, N_joints)
            'camera_names': view_camera_names,  # List of camera names
            'camera_indices': np.array(view_camera_indices, dtype=np.int32),  # (num_views,)
            'num_views': len(view_images),
            
            # Camera parameters (per-view)
            'camera_intrinsics': np.stack(view_camera_intrinsics, axis=0) if view_camera_intrinsics else None,  # (num_views, 3, 3)
            'camera_extrinsics_R': np.stack(view_camera_extrinsics_R, axis=0) if view_camera_extrinsics_R else None,  # (num_views, 3, 3)
            'camera_extrinsics_t': np.stack(view_camera_extrinsics_t, axis=0) if view_camera_extrinsics_t else None,  # (num_views, 3)
            'image_sizes': np.stack(view_image_sizes, axis=0) if view_image_sizes else None,  # (num_views, 2) - (width, height)
            
            # 3D keypoints (shared across views)
            'keypoints_3d': keypoints_3d.astype(np.float32) if keypoints_3d is not None else None,  # (n_joints, 3)
            'has_3d_data': keypoints_3d is not None,
            
            # Shared body parameters (placeholders for SLEAP)
            'global_rot': np.zeros(3, dtype=np.float32),
            'joint_rot': np.zeros((config.N_POSE + 1, 3), dtype=np.float32),
            'betas': ground_truth_betas.astype(np.float32) if ground_truth_betas is not None else np.zeros(config.N_BETAS, dtype=np.float32),
            'trans': np.zeros(3, dtype=np.float32),
            
            # Metadata
            'session_name': Path(session_path).name,
            'frame_idx': frame_idx,
            'has_ground_truth_betas': ground_truth_betas is not None
        }
        
        return sample

    def _map_3d_keypoints_to_smal(self, loader: SLEAPDataLoader, keypoints_3d: np.ndarray) -> np.ndarray:
        """
        Map raw SLEAP 3D keypoints to SMAL joint order using the same mapping as 2D preprocessing.

        Args:
            loader: SLEAPDataLoader (contains `smal_n_joints` and `joint_mapping`)
            keypoints_3d: (N_sleap, 3) array in world coordinates (units as provided by SLEAP/anipose)

        Returns:
            (N_smal, 3) array where N_smal == loader.smal_n_joints.
            Unmapped joints are filled with zeros.
        """
        if keypoints_3d is None:
            return None
        smal_n = int(getattr(loader, 'smal_n_joints', 0) or 0)
        mapping = getattr(loader, 'joint_mapping', None)
        if smal_n <= 0 or not isinstance(mapping, dict) or len(mapping) == 0:
            # Fallback: if no mapping, try best-effort truncate/pad to config-defined joint count
            # (this should not happen in normal training runs where 2D mapping is enabled).
            from config import CANONICAL_MODEL_JOINTS
            target_n = len(CANONICAL_MODEL_JOINTS)
            out = np.zeros((target_n, 3), dtype=np.float32)
            n = min(target_n, keypoints_3d.shape[0])
            out[:n] = keypoints_3d[:n].astype(np.float32)
            return out

        out = np.zeros((smal_n, 3), dtype=np.float32)
        for smal_idx, sleap_idx in mapping.items():
            if sleap_idx is None or int(sleap_idx) < 0:
                continue
            sleap_idx_int = int(sleap_idx)
            if sleap_idx_int < keypoints_3d.shape[0]:
                out[int(smal_idx)] = keypoints_3d[sleap_idx_int].astype(np.float32)
        return out
    
    def _filter_outlier_3d_keypoints(self, keypoints_3d: np.ndarray, 
                                     joint_names: Optional[List[str]] = None,
                                     outlier_threshold_std: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter outlier 3D keypoints that are far from the median position.
        
        Keypoints that are more than `outlier_threshold_std` standard deviations away
        from the median of all keypoints are marked as invalid (set to NaN).
        
        Args:
            keypoints_3d: (N_joints, 3) array of 3D keypoint coordinates
            joint_names: Optional list of joint names for printing (default: None)
            outlier_threshold_std: Number of standard deviations for outlier detection (default: 2.0)
            
        Returns:
            Tuple of (filtered_keypoints_3d, outlier_mask) where:
            - filtered_keypoints_3d: (N_joints, 3) array with outliers set to NaN
            - outlier_mask: (N_joints,) boolean array, True for outliers
        """
        if keypoints_3d is None or keypoints_3d.shape[0] == 0:
            return keypoints_3d, np.zeros(keypoints_3d.shape[0], dtype=bool) if keypoints_3d is not None else None
        
        # Create a copy to avoid modifying the original
        filtered_kp3d = keypoints_3d.copy()
        outlier_mask = np.zeros(keypoints_3d.shape[0], dtype=bool)
        
        # Find valid keypoints (non-zero, non-NaN, non-inf)
        valid_mask = ~(np.isnan(keypoints_3d).any(axis=1) | 
                      np.isinf(keypoints_3d).any(axis=1) |
                      (keypoints_3d == 0).all(axis=1))
        
        if valid_mask.sum() < 2:
            # Need at least 2 valid keypoints to compute statistics
            return filtered_kp3d, outlier_mask
        
        # Get valid keypoints
        valid_keypoints = keypoints_3d[valid_mask]
        
        # Compute median position of all valid keypoints
        median_pos = np.median(valid_keypoints, axis=0)  # (3,)
        
        # Compute distances from each keypoint to the median
        distances = np.linalg.norm(keypoints_3d - median_pos, axis=1)  # (N_joints,)
        
        # Compute median and standard deviation of distances (using only valid keypoints)
        valid_distances = distances[valid_mask]
        if len(valid_distances) == 0:
            return filtered_kp3d, outlier_mask
        
        median_distance = np.median(valid_distances)
        std_distance = np.std(valid_distances)
        
        # If std is zero or very small, all points are at the same location (unlikely but handle it)
        if std_distance < 1e-6:
            return filtered_kp3d, outlier_mask
        
        # Identify outliers: keypoints more than threshold_std standard deviations away
        # Use median + threshold_std * std as the cutoff
        outlier_threshold = median_distance + outlier_threshold_std * std_distance
        
        # Mark outliers (only among valid keypoints)
        outlier_mask = valid_mask & (distances > outlier_threshold)
        
        # Print outlier joint names if available and debug mode is enabled
        if outlier_mask.any() and joint_names is not None and self.debug:
            outlier_indices = np.where(outlier_mask)[0]
            outlier_names = [joint_names[i] if i < len(joint_names) else f"joint_{i}" 
                           for i in outlier_indices]
            print(f"    Outlier joints filtered: {', '.join(outlier_names)}")
        
        # Set outlier keypoints to NaN
        filtered_kp3d[outlier_mask] = np.nan
        
        return filtered_kp3d, outlier_mask
    
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
    
    def _preprocess_image(self, image: np.ndarray, 
                          keypoints_2d: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess image for training."""
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
            
        else:  # 'default' mode
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
    
    def _sanitize_array(self, arr: np.ndarray, default_value: float = 0.0) -> np.ndarray:
        """Replace NaN/inf values with default."""
        if arr is None:
            return None
        return np.nan_to_num(arr, nan=default_value, posinf=default_value, neginf=default_value)
    
    def _encode_image_jpeg(self, image: np.ndarray) -> bytes:
        """Encode image as JPEG bytes."""
        image_uint8 = (image * 255).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, jpeg_bytes = cv2.imencode('.jpg', image_uint8, encode_param)
        return jpeg_bytes.tobytes()
    
    def process_dataset(self, sessions_dir: str, output_path: str, 
                        verbose: bool = True) -> Dict[str, Any]:
        """
        Process all SLEAP sessions into a single multi-view HDF5 dataset.
        
        Args:
            sessions_dir: Directory containing SLEAP sessions
            output_path: Output HDF5 file path
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing processing statistics
        """
        if verbose:
            print("Discovering SLEAP sessions...")
        
        sessions = self.discover_sleap_sessions(sessions_dir)
        self.stats['total_sessions'] = len(sessions)
        
        if len(sessions) == 0:
            raise ValueError(f"No SLEAP sessions found in {sessions_dir}")
        
        if verbose:
            print(f"Found {len(sessions)} SLEAP sessions")
            for session in sessions:
                print(f"  - {Path(session).name}")
        
        # Establish canonical camera order
        self.canonical_camera_order = self._establish_canonical_camera_order(sessions)
        
        # Process sessions
        all_samples = []
        
        if verbose:
            sessions_iter = tqdm(sessions, desc="Processing sessions")
        else:
            sessions_iter = sessions
        
        for session in sessions_iter:
            samples = self.process_single_session(session)
            all_samples.extend(samples)
            if verbose:
                print(f"Processed {len(samples)} multi-view samples from {Path(session).name}")
        
        self.stats['total_multiview_samples'] = len(all_samples)
        
        if len(all_samples) == 0:
            raise ValueError("No multi-view samples were successfully processed")
        
        if verbose:
            print(f"Total multi-view samples processed: {len(all_samples)}")
            print("Saving to HDF5...")
        
        # Save to HDF5
        self._save_to_hdf5(all_samples, output_path)
        
        if verbose:
            print(f"Dataset saved to: {output_path}")
        
        return self.stats
    
    def _save_to_hdf5(self, samples: List[Dict[str, Any]], output_path: str):
        """Save processed multi-view samples to HDF5 file."""
        with h5py.File(output_path, 'w') as f:
            # Create groups
            images_group = f.create_group('multiview_images')
            keypoints_group = f.create_group('multiview_keypoints')
            parameters_group = f.create_group('parameters')
            auxiliary_group = f.create_group('auxiliary')
            metadata_group = f.create_group('metadata')
            
            num_samples = len(samples)
            
            # Determine max views across all samples
            max_views = max(s['num_views'] for s in samples)
            n_joints = samples[0]['keypoints_2d'].shape[1]  # Get from first sample
            
            # Prepare storage for variable-length image data
            # Store images as list of lists (variable per sample)
            dt_vlen = h5py.special_dtype(vlen=np.uint8)
            
            # For images: store per-view datasets
            for v in range(max_views):
                images_group.create_dataset(
                    f'image_jpeg_view_{v}',
                    shape=(num_samples,),
                    dtype=dt_vlen,
                    compression=self.compression,
                    compression_opts=self.compression_level
                )
            
            # View mask: which views are valid per sample
            view_mask_data = np.zeros((num_samples, max_views), dtype=bool)
            
            # Keypoints: (num_samples, max_views, n_joints, 2)
            keypoints_2d_data = np.zeros((num_samples, max_views, n_joints, 2), dtype=np.float32)
            keypoint_visibility_data = np.zeros((num_samples, max_views, n_joints), dtype=np.float32)
            
            # Camera indices
            camera_indices_data = np.full((num_samples, max_views), -1, dtype=np.int32)
            
            # Camera parameters: (num_samples, max_views, ...)
            camera_intrinsics_data = np.zeros((num_samples, max_views, 3, 3), dtype=np.float32)
            camera_extrinsics_R_data = np.zeros((num_samples, max_views, 3, 3), dtype=np.float32)
            camera_extrinsics_t_data = np.zeros((num_samples, max_views, 3), dtype=np.float32)
            image_sizes_data = np.zeros((num_samples, max_views, 2), dtype=np.int32)
            
            # 3D keypoints: (num_samples, n_keypoints_3d, 3) - variable size, will determine from first sample
            keypoints_3d_list = []
            has_3d_data_list = []
            
            # Shared parameters
            global_rot_data = []
            joint_rot_data = []
            betas_data = []
            trans_data = []
            
            # Auxiliary data
            session_names = []
            frame_indices = []
            num_views_data = []
            has_gt_betas_data = []
            camera_names_data = []  # Variable length strings
            
            # Fill data
            for i, sample in enumerate(samples):
                num_views = sample['num_views']
                
                # Store images and mark valid views
                for v in range(num_views):
                    img_bytes = sample['images_jpeg'][v]
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    f[f'multiview_images/image_jpeg_view_{v}'][i] = img_array
                    view_mask_data[i, v] = True
                    
                    # Keypoints
                    keypoints_2d_data[i, v] = sample['keypoints_2d'][v]
                    keypoint_visibility_data[i, v] = sample['keypoint_visibility'][v]
                    camera_indices_data[i, v] = sample['camera_indices'][v]
                    
                    # Camera parameters
                    if sample.get('camera_intrinsics') is not None:
                        camera_intrinsics_data[i, v] = sample['camera_intrinsics'][v]
                        camera_extrinsics_R_data[i, v] = sample['camera_extrinsics_R'][v]
                        camera_extrinsics_t_data[i, v] = sample['camera_extrinsics_t'][v]
                        image_sizes_data[i, v] = sample['image_sizes'][v]
                
                # 3D keypoints
                if sample.get('keypoints_3d') is not None:
                    kp3d = sample['keypoints_3d'].copy()
                    # Replace NaN values (from outlier filtering) with zeros
                    # This prevents NaN propagation during training
                    kp3d = np.nan_to_num(kp3d, nan=0.0, posinf=0.0, neginf=0.0)
                    keypoints_3d_list.append(kp3d.astype(np.float32))
                    has_3d_data_list.append(True)
                else:
                    # Placeholder - use zeros with same shape as 2D keypoints
                    keypoints_3d_list.append(np.zeros((n_joints, 3), dtype=np.float32))
                    has_3d_data_list.append(False)
                
                # Shared parameters
                global_rot_data.append(sample['global_rot'])
                joint_rot_data.append(sample['joint_rot'])
                betas_data.append(sample['betas'])
                trans_data.append(sample['trans'])
                
                # Auxiliary
                session_names.append(sample['session_name'])
                frame_indices.append(sample['frame_idx'])
                num_views_data.append(num_views)
                has_gt_betas_data.append(sample['has_ground_truth_betas'])
                camera_names_data.append(','.join(sample['camera_names']))
            
            # Save view mask and keypoints
            images_group.create_dataset('view_mask', data=view_mask_data,
                                       compression=self.compression,
                                       compression_opts=self.compression_level)
            
            keypoints_group.create_dataset('keypoints_2d', data=keypoints_2d_data,
                                          compression=self.compression,
                                          compression_opts=self.compression_level)
            keypoints_group.create_dataset('keypoint_visibility', data=keypoint_visibility_data,
                                          compression=self.compression,
                                          compression_opts=self.compression_level)
            keypoints_group.create_dataset('camera_indices', data=camera_indices_data,
                                          compression=self.compression,
                                          compression_opts=self.compression_level)
            
            # Save camera parameters
            keypoints_group.create_dataset('camera_intrinsics', data=camera_intrinsics_data,
                                          compression=self.compression,
                                          compression_opts=self.compression_level)
            keypoints_group.create_dataset('camera_extrinsics_R', data=camera_extrinsics_R_data,
                                          compression=self.compression,
                                          compression_opts=self.compression_level)
            keypoints_group.create_dataset('camera_extrinsics_t', data=camera_extrinsics_t_data,
                                          compression=self.compression,
                                          compression_opts=self.compression_level)
            keypoints_group.create_dataset('image_sizes', data=image_sizes_data,
                                          compression=self.compression,
                                          compression_opts=self.compression_level)
            
            # Save 3D keypoints
            keypoints_3d_data = np.array(keypoints_3d_list)
            keypoints_group.create_dataset('keypoints_3d', data=keypoints_3d_data,
                                          compression=self.compression,
                                          compression_opts=self.compression_level)
            auxiliary_group.create_dataset('has_3d_data', data=np.array(has_3d_data_list),
                                          compression=self.compression,
                                          compression_opts=self.compression_level)
            
            # Save shared parameters
            parameters_group.create_dataset('global_rot', data=np.array(global_rot_data, dtype=np.float32),
                                           compression=self.compression,
                                           compression_opts=self.compression_level)
            parameters_group.create_dataset('joint_rot', data=np.array(joint_rot_data, dtype=np.float32),
                                           compression=self.compression,
                                           compression_opts=self.compression_level)
            parameters_group.create_dataset('betas', data=np.array(betas_data, dtype=np.float32),
                                           compression=self.compression,
                                           compression_opts=self.compression_level)
            parameters_group.create_dataset('trans', data=np.array(trans_data, dtype=np.float32),
                                           compression=self.compression,
                                           compression_opts=self.compression_level)
            
            # Save auxiliary data
            auxiliary_group.create_dataset('session_name',
                                          data=[s.encode('utf-8') for s in session_names],
                                          compression=self.compression,
                                          compression_opts=self.compression_level)
            auxiliary_group.create_dataset('frame_idx', data=np.array(frame_indices),
                                          compression=self.compression,
                                          compression_opts=self.compression_level)
            auxiliary_group.create_dataset('num_views', data=np.array(num_views_data),
                                          compression=self.compression,
                                          compression_opts=self.compression_level)
            auxiliary_group.create_dataset('has_ground_truth_betas', data=np.array(has_gt_betas_data),
                                          compression=self.compression,
                                          compression_opts=self.compression_level)
            auxiliary_group.create_dataset('camera_names',
                                          data=[s.encode('utf-8') for s in camera_names_data],
                                          compression=self.compression,
                                          compression_opts=self.compression_level)
            
            # Save metadata
            metadata_group.attrs['num_samples'] = num_samples
            metadata_group.attrs['max_views'] = max_views
            metadata_group.attrs['n_joints'] = n_joints
            metadata_group.attrs['target_resolution'] = self.target_resolution
            metadata_group.attrs['backbone_name'] = self.backbone_name
            metadata_group.attrs['jpeg_quality'] = self.jpeg_quality
            metadata_group.attrs['crop_mode'] = self.crop_mode
            metadata_group.attrs['dataset_type'] = 'sleap_multiview'
            metadata_group.attrs['is_multiview'] = True
            metadata_group.attrs['n_pose'] = config.N_POSE
            metadata_group.attrs['n_betas'] = config.N_BETAS
            metadata_group.attrs['min_views_per_sample'] = self.min_views_per_sample
            metadata_group.attrs['canonical_camera_order'] = json.dumps(self.canonical_camera_order)
            metadata_group.attrs['has_camera_parameters'] = True
            metadata_group.attrs['has_3d_keypoints'] = any(has_3d_data_list)
            metadata_group.attrs['load_3d_data'] = self.load_3d_data
            # Store mapping info for downstream consumers/debugging
            try:
                metadata_group.attrs['keypoints_3d_joint_order'] = 'smal'
            except Exception:
                pass
            
            # Save statistics
            metadata_group.attrs['total_sessions'] = self.stats['total_sessions']
            metadata_group.attrs['sessions_processed'] = self.stats['sessions_processed']
            metadata_group.attrs['sessions_failed'] = self.stats['sessions_failed']
            metadata_group.attrs['total_multiview_samples'] = self.stats['total_multiview_samples']
            metadata_group.attrs['processed_samples'] = self.stats['processed_samples']
            metadata_group.attrs['skipped_insufficient_views'] = self.stats['skipped_insufficient_views']
            metadata_group.attrs['failed_samples'] = self.stats['failed_samples']
            metadata_group.attrs['sessions_with_3d_data'] = self.stats['sessions_with_3d_data']
            metadata_group.attrs['samples_with_3d_data'] = self.stats['samples_with_3d_data']
            metadata_group.attrs['total_outlier_keypoints_filtered'] = self.stats['total_outlier_keypoints_filtered']
            metadata_group.attrs['samples_with_outliers_filtered'] = self.stats['samples_with_outliers_filtered']


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess multi-view SLEAP dataset into optimized HDF5 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic multi-view preprocessing
  python preprocess_sleap_multiview_dataset.py /path/to/sleap/sessions multiview_sleap.h5
  
  # With lookup tables and minimum 3 views per sample
  python preprocess_sleap_multiview_dataset.py /path/to/sleap/sessions multiview_sleap.h5 \\
    --joint_lookup_table /path/to/joint_lookup.csv \\
    --shape_betas_table /path/to/shape_betas.csv \\
    --min_views 3
        """
    )
    
    # Required arguments
    parser.add_argument("sessions_dir", help="Directory containing SLEAP sessions")
    parser.add_argument("output_path", help="Output HDF5 file path")
    
    # Lookup table options
    parser.add_argument("--joint_lookup_table", type=str, default=None,
                       help="Path to CSV lookup table for joint name mapping")
    parser.add_argument("--shape_betas_table", type=str, default=None,
                       help="Path to CSV lookup table for ground truth shape betas")
    
    # Processing options
    parser.add_argument("--target_resolution", type=int, default=224,
                       help="Target image resolution (default: 224)")
    parser.add_argument("--backbone", dest="backbone_name", default='vit_large_patch16_224',
                       help="Backbone network name (default: vit_large_patch16_224)")
    parser.add_argument("--jpeg_quality", type=int, default=95,
                       help="JPEG compression quality 1-100 (default: 95)")
    parser.add_argument("--crop_mode", type=str, default='default',
                       choices=['default', 'centred', 'bbox_crop'],
                       help="Image cropping mode (default: default)")
    
    # Multi-view specific options
    parser.add_argument("--min_views", type=int, default=2,
                       help="Minimum views required per sample (default: 2)")
    parser.add_argument("--max_frames_per_session", type=int, default=None,
                       help="Maximum frames to process per session (default: all)")
    
    # 3D data options
    parser.add_argument("--no_3d_data", action="store_true",
                       help="Skip loading 3D keypoints and camera parameters (default: load if available)")
    
    # Other options
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                       help="Keypoint confidence threshold (default: 0.5)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress output")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode to print detailed information about filtered outlier keypoints")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.sessions_dir):
        print(f"Error: Sessions directory does not exist: {args.sessions_dir}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Print configuration
    if not args.quiet:
        print("\n" + "="*60)
        print("MULTI-VIEW SLEAP DATASET PREPROCESSING")
        print("="*60)
        print(f"Sessions directory: {args.sessions_dir}")
        print(f"Output file: {args.output_path}")
        print(f"Minimum views per sample: {args.min_views}")
        print(f"Target resolution: {args.target_resolution}x{args.target_resolution}")
        print(f"Crop mode: {args.crop_mode}")
        print("="*60)
    
    # Start preprocessing
    start_time = time.time()
    
    try:
        # Load 3D data by default unless explicitly disabled
        load_3d = not args.no_3d_data
        
        preprocessor = SLEAPMultiViewPreprocessor(
            joint_lookup_table_path=args.joint_lookup_table,
            shape_betas_table_path=args.shape_betas_table,
            target_resolution=args.target_resolution,
            backbone_name=args.backbone_name,
            jpeg_quality=args.jpeg_quality,
            max_frames_per_session=args.max_frames_per_session,
            crop_mode=args.crop_mode,
            confidence_threshold=args.confidence_threshold,
            min_views_per_sample=args.min_views,
            load_3d_data=load_3d,
            debug=args.debug
        )
        
        stats = preprocessor.process_dataset(
            sessions_dir=args.sessions_dir,
            output_path=args.output_path,
            verbose=not args.quiet
        )
        
        end_time = time.time()
        
        if not args.quiet:
            print(f"\nProcessing completed in {end_time - start_time:.1f} seconds")
            print(f"\nStatistics:")
            print(f"  Total sessions: {stats['total_sessions']}")
            print(f"  Sessions processed: {stats['sessions_processed']}")
            print(f"  Multi-view samples: {stats['total_multiview_samples']}")
            print(f"  Skipped (insufficient views): {stats['skipped_insufficient_views']}")
            print(f"  Failed samples: {stats['failed_samples']}")
            print(f"  Sessions with 3D data: {stats['sessions_with_3d_data']}")
            print(f"  Samples with 3D data: {stats['samples_with_3d_data']}")
            if stats.get('samples_with_outliers_filtered', 0) > 0:
                print(f"  Outlier 3D keypoints filtered: {stats['total_outlier_keypoints_filtered']} "
                      f"across {stats['samples_with_outliers_filtered']} samples")
            print(f"\nViews per sample distribution:")
            for num_views, count in sorted(stats['views_per_sample_histogram'].items()):
                print(f"    {num_views} views: {count} samples")
        
        print(f"\nMulti-view dataset saved to: {args.output_path}")
        
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

