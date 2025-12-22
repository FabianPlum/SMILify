#!/usr/bin/env python3
"""
SLEAP Dataset Preprocessing Script

This script preprocesses SLEAP pose estimation datasets into optimized HDF5 format
compatible with the SMILify training pipeline.

Usage:
    python preprocess_sleap_dataset.py sessions_dir output.h5 [options]

Example:
    python preprocess_sleap_dataset.py /path/to/sleap/sessions optimized_sleap_dataset.h5 \\
        --joint_lookup_table /path/to/joint_lookup.csv \\
        --shape_betas_table /path/to/shape_betas.csv \\
        --backbone vit_large_patch16_224
"""

import os
import sys
import argparse
import time
import h5py
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
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

import config


class SLEAPDatasetPreprocessor:
    """
    Preprocesses SLEAP datasets into optimized HDF5 format compatible with SMILify training.
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
                max_frames_per_video: Optional[int] = None,
                crop_mode: str = 'default',
                use_reprojections: bool = False,
                confidence_threshold: float = 0.5):
        """
        Initialize the SLEAP dataset preprocessor.
        
        Args:
            joint_lookup_table_path: Path to CSV lookup table for joint name mapping
            shape_betas_table_path: Path to CSV lookup table for ground truth shape betas
            target_resolution: Target image resolution for preprocessing
            backbone_name: Backbone network name for resolution selection
            jpeg_quality: JPEG compression quality (1-100)
            chunk_size: HDF5 chunk size (should match training batch size)
            compression: HDF5 compression algorithm
            compression_level: Compression level (1-9)
            max_frames_per_video: Maximum number of frames to process per video (None for all frames)
            crop_mode: Image cropping mode:
                      'default' - direct resize that may distort aspect ratio
                      'centred' - aspect-ratio preserving center crop
                      'bbox_crop' - crop around instance based on 2D keypoints bounding box (1.05x padding)
            use_reprojections: If True, use reprojected 2D coordinates from reprojections.h5 
                             instead of raw SLEAP predictions for improved accuracy
            confidence_threshold: Minimum confidence score (0.0-1.0) for keypoints. Keypoints below 
                                this threshold will be marked as invisible. Default: 0.5
        """
        self.joint_lookup_table_path = joint_lookup_table_path
        self.shape_betas_table_path = shape_betas_table_path
        self.target_resolution = target_resolution
        self.backbone_name = backbone_name
        self.jpeg_quality = jpeg_quality
        self.chunk_size = chunk_size
        self.compression = compression
        self.compression_level = compression_level
        self.max_frames_per_video = max_frames_per_video
        self.crop_mode = crop_mode
        self.use_reprojections = use_reprojections
        self.confidence_threshold = confidence_threshold
        self._reproj_camera_cache: Dict[str, Any] = {}
        self._reproj_warned_missing: bool = False
        
        # Validate crop_mode
        if self.crop_mode not in ['default', 'centred', 'bbox_crop']:
            raise ValueError(f"crop_mode must be 'default', 'centred', or 'bbox_crop', got: {self.crop_mode}")
        
        # Statistics tracking
        self.stats = {
            'total_sessions': 0,
            'total_samples': 0,
            'processed_samples': 0,
            'failed_samples': 0,
            'sessions_processed': 0,
            'sessions_failed': 0
        }
    
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
                # Check if this looks like a SLEAP session
                if self._is_sleap_session(item):
                    sessions.append(str(item))
        
        sessions.sort()  # Ensure consistent ordering
        return sessions
    
    def _is_sleap_session(self, session_path: Path) -> bool:
        """
        Check if a directory looks like a SLEAP session.
        
        Args:
            session_path: Path to potential session directory
            
        Returns:
            True if it looks like a SLEAP session
        """
        # Check for common SLEAP files
        sleap_indicators = [
            'calibration.toml',
            'points3d.h5'
        ]
        
        # Check for session subdirectories (session_dirs structure)
        session_subdirs = [d for d in session_path.iterdir() if d.is_dir()]
        if session_subdirs:
            # Check if any subdirectory contains .h5 files
            for subdir in session_subdirs:
                h5_files = list(subdir.glob('*.h5'))
                if h5_files:
                    return True
        
        # Check for camera directories (camera_dirs structure)
        camera_dirs = [d for d in session_path.iterdir() if d.is_dir()]
        if camera_dirs:
            for cam_dir in camera_dirs:
                slp_files = list(cam_dir.glob('*.slp'))
                if slp_files:
                    return True
        
        # Check for direct SLEAP files
        for indicator in sleap_indicators:
            if (session_path / indicator).exists():
                return True
        
        return False
    
    def process_single_session(self, session_path: str) -> List[Dict[str, Any]]:
        """
        Process a single SLEAP session and extract all samples.
        
        OPTIMIZED VERSION: Minimizes I/O operations by:
        1. Loading camera data once per camera
        2. Opening video files once per camera
        3. Only processing frames with annotation data
        
        Args:
            session_path: Path to SLEAP session directory
            
        Returns:
            List of processed sample dictionaries
        """
        try:
            # Initialize SLEAP data loader for this session
            loader = SLEAPDataLoader(
                project_path=session_path,
                lookup_table_path=self.joint_lookup_table_path,
                shape_betas_path=self.shape_betas_table_path,
                confidence_threshold=self.confidence_threshold
            )
            
            samples = []
            reproj_file = None
            reproj_files_by_subdir = {}
            
            if self.use_reprojections:
                # Try to open reprojections*.h5 at session root level (optional)
                # Look for files matching pattern reprojections*.h5
                session_path_obj = Path(session_path)
                reproj_candidates = list(session_path_obj.glob('reprojections*.h5'))
                if reproj_candidates:
                    reproj_path = reproj_candidates[0]  # Use first match
                    try:
                        reproj_file = h5py.File(str(reproj_path), 'r')
                        print(f"Info: Using reprojections file: {reproj_path.name}")
                    except Exception as e:
                        print(f"Warning: Failed to open reprojections file {reproj_path}: {e}")
                        reproj_file = None
                else:
                    reproj_file = None
                
                # If not at root, find and open reprojections*.h5 files in subdirectories
                # This allows sharing one file handle across multiple cameras in the same subdir
                if reproj_file is None:
                    reproj_files_by_subdir = self._find_all_reprojection_files(session_path)
            
            # Process each camera view
            for camera_name in loader.camera_views:
                try:
                    # Determine which reprojection file to use for this camera
                    camera_reproj_file = reproj_file
                    if camera_reproj_file is None and reproj_files_by_subdir:
                        # Find the subdirectory containing this camera's annotations
                        subdir = self._find_camera_subdir(session_path, camera_name)
                        if subdir and subdir in reproj_files_by_subdir:
                            camera_reproj_file = reproj_files_by_subdir[subdir]
                    
                    camera_samples = self._process_camera_optimized(
                        loader, camera_name, session_path, camera_reproj_file
                    )
                    samples.extend(camera_samples)
                    
                except Exception as e:
                    print(f"Warning: Failed to process camera {camera_name} in {session_path}: {e}")
                    continue
            
            self.stats['sessions_processed'] += 1
            return samples
            
        except Exception as e:
            print(f"Error processing session {session_path}: {e}")
            self.stats['sessions_failed'] += 1
            return []
        finally:
            # Close all reprojection file handles
            try:
                if 'reproj_file' in locals() and reproj_file is not None:
                    reproj_file.close()
            except Exception:
                pass
            try:
                if 'reproj_files_by_subdir' in locals():
                    for f in reproj_files_by_subdir.values():
                        try:
                            f.close()
                        except Exception:
                            pass
            except Exception:
                pass
    
    def _estimate_num_frames(self, camera_data: Dict[str, Any], data_structure_type: str) -> int:
        """
        Estimate the number of frames in a camera's data.
        
        Args:
            camera_data: Camera data dictionary
            data_structure_type: Type of SLEAP data structure
            
        Returns:
            Estimated number of frames
        """
        if data_structure_type == 'camera_dirs':
            # For camera_dirs, estimate from instances data
            if 'instances' in camera_data:
                instances = camera_data['instances']
                if len(instances) > 0:
                    return int(instances['frame_id'].max()) + 1
        elif data_structure_type == 'session_dirs':
            # For session_dirs, estimate from tracks data
            if 'tracks' in camera_data:
                tracks = camera_data['tracks']
                if len(tracks.shape) >= 4:  # (n_tracks, n_instances, n_keypoints, n_frames)
                    return tracks.shape[3]
        
        # Default fallback
        return 100  # Conservative estimate
    
    def _process_camera_optimized(self, loader: SLEAPDataLoader, camera_name: str, 
                                session_path: str, reproj_file: Optional[h5py.File] = None) -> List[Dict[str, Any]]:
        """
        Optimized processing of a single camera.
        
        Loads camera data once and video file once, then processes all frames.
        """
        samples = []
        
        # Load camera data ONCE per camera
        camera_data = loader.load_camera_data(camera_name)
        
        # Get all frames with annotation data
        annotated_frames = self._get_annotated_frames(camera_data, loader.data_structure_type)
        
        if len(annotated_frames) == 0:
            return samples
        
        # CRITICAL: Sort frames to enable sequential reading (major performance fix)
        annotated_frames = sorted(annotated_frames)
        
        # Get image size for coordinate transformation (once per camera)
        image_size = loader.get_camera_image_size(camera_name)
        
        # Get ground truth shape betas (once per camera)
        ground_truth_betas = loader.get_ground_truth_shape_betas()
        
        # Open video file ONCE per camera
        video_cap = self._open_video_capture(loader, camera_name)
        if video_cap is None:
            return samples
        
        try:
            # Process all annotated frames with progress bar
            frame_pbar = tqdm(annotated_frames, 
                            desc=f"Processing {camera_name}", 
                            leave=False,
                            unit="frame")
            
            # Use sequential reading for optimal performance
            current_frame = 0
            for target_frame_idx in frame_pbar:
                try:
                    # Seek to target frame if needed (only when not sequential)
                    if current_frame != target_frame_idx:
                        video_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                        current_frame = target_frame_idx
                    
                    # Read the frame
                    ret, frame = video_cap.read()
                    if not ret:
                        print(f"Warning: Failed to read frame {target_frame_idx} for camera {camera_name}")
                        continue
                    
                    current_frame += 1
                    
                    # Process the frame
                    sample = self._process_frame_with_data(
                        loader, camera_data, camera_name, target_frame_idx, 
                        session_path, image_size, ground_truth_betas, frame,
                        reproj_handle=reproj_file
                    )
                    if sample is not None:
                        samples.append(sample)
                        
                except Exception as e:
                    print(f"Warning: Failed to process frame {target_frame_idx} for camera {camera_name}: {e}")
                    self.stats['failed_samples'] += 1
                    continue
                    
        finally:
            # Close video file
            video_cap.release()
        
        self.stats['processed_samples'] += len(samples)
        return samples
    
    def _get_annotated_frames(self, camera_data: Dict[str, Any], 
                            data_structure_type: str) -> List[int]:
        """
        Get list of frame indices that have annotation data.
        
        This avoids processing frames without keypoints.
        If max_frames_per_video is set, limits the number of frames returned.
        """
        annotated_frames = []
        
        if data_structure_type == 'camera_dirs':
            # For camera_dirs, get frames from instances data
            if 'instances' in camera_data:
                instances = camera_data['instances']
                if len(instances) > 0:
                    # Get unique frame IDs that have instances
                    # Use np.unique() since instances is a numpy structured array
                    frame_ids = np.unique(instances['frame_id'])
                    annotated_frames = sorted(frame_ids.tolist())
                    
        elif data_structure_type == 'session_dirs':
            # For session_dirs, get frames from tracks data
            if 'tracks' in camera_data:
                tracks = camera_data['tracks']
                if len(tracks.shape) >= 4:  # (n_tracks, n_instances, n_keypoints, n_frames)
                    num_frames = tracks.shape[3]
                    # Check which frames have non-zero keypoints
                    for frame_idx in range(num_frames):
                        frame_tracks = tracks[:, :, :, frame_idx]
                        if np.any(frame_tracks != 0):  # Has non-zero keypoints
                            annotated_frames.append(frame_idx)
        
        # Apply frame limit if specified
        if self.max_frames_per_video is not None and len(annotated_frames) > self.max_frames_per_video:
            annotated_frames = annotated_frames[:self.max_frames_per_video]
        
        return annotated_frames
    
    def _open_video_capture(self, loader: SLEAPDataLoader, camera_name: str) -> Optional[cv2.VideoCapture]:
        """
        Open video capture for a camera (optimized version).
        """
        video_file = self._find_video_file(loader, camera_name)
        if video_file is None:
            return None
        
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"Failed to open video file: {video_file}")
            return None
        
        return cap
    
    def _find_video_file(self, loader: SLEAPDataLoader, camera_name: str) -> Optional[Path]:
        """
        Find video file for a camera, mirroring the logic in SLEAPDataLoader
        so that the same videos are used for both visualization and preprocessing.
        """
        # camera_dirs structure: videos live in per-camera subdirectories
        if loader.data_structure_type == 'camera_dirs':
            camera_dir = loader.project_path / camera_name

            # Prefer the video path referenced inside an associated .h5 file, if available
            try:
                # First, look for an analysis file, then fall back to predictions.h5
                h5_candidates = list(camera_dir.glob("*.analysis.h5"))
                if not h5_candidates:
                    h5_candidates = list(camera_dir.glob("*.predictions.h5"))
                
                if h5_candidates:
                    h5_file = h5_candidates[0]
                    with h5py.File(h5_file, 'r') as f:
                        if 'video_path' in f:
                            raw_path = f['video_path'][()]
                            # Decode path from HDF5 and use only the filename, as the
                            # original absolute directory is not available on this system.
                            if isinstance(raw_path, bytes):
                                video_path_str = raw_path.decode('utf-8')
                            else:
                                video_path_str = str(raw_path)
                            video_filename = Path(video_path_str).name
                            candidate_path = camera_dir / video_filename

                            if candidate_path.exists():
                                return candidate_path
            except Exception as e:
                print(f"Warning: Failed to read video_path from h5 in {camera_dir}: {e}")

            # Fallback: find video file directly in camera directory
            video_files = list(camera_dir.glob("*.mp4"))
            return video_files[0] if video_files else None
            
        # session_dirs structure: per-camera .h5 lives in a session subdirectory
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
                                # Decode path from HDF5 and use only the filename
                                if isinstance(raw_path, bytes):
                                    video_path_str = raw_path.decode('utf-8')
                                else:
                                    video_path_str = str(raw_path)
                                video_filename = Path(video_path_str).name
                                candidate_path = camera_h5_file.parent / video_filename

                                if candidate_path.exists():
                                    return candidate_path
                    except Exception as e:
                        print(f"Warning: Failed to read video_path from {camera_h5_file}: {e}")

            # Fallback: by pattern in project root
            video_files = list(loader.project_path.glob(f"*_cam{camera_name}.mp4"))
            return video_files[0] if video_files else None
        
        return None
    
    def _process_frame_optimized(self, loader: SLEAPDataLoader, camera_data: Dict[str, Any],
                               camera_name: str, frame_idx: int, session_path: str,
                               image_size: Tuple[int, int], ground_truth_betas: Optional[np.ndarray],
                               video_cap: cv2.VideoCapture) -> Optional[Dict[str, Any]]:
        """
        Process a single frame (optimized version).
        
        Uses pre-loaded camera data and video capture.
        """
        try:
            # Extract 2D keypoints and visibility (from pre-loaded camera data)
            keypoints_2d, visibility = loader.extract_2d_keypoints(camera_data, frame_idx)
            
            if len(keypoints_2d) == 0:
                return None  # Skip samples with no keypoints
            
            # Read frame from pre-opened video capture
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video_cap.read()
            
            if not ret:
                return None  # Skip samples without video frames
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Preprocess image (crop/resize) and collect transform
            # Pass keypoints for bbox_crop mode
            processed_image, transform_info = self._preprocess_image(frame_rgb, keypoints_2d)
            
            # Adjust keypoints by transform
            adjusted_keypoints_2d = self._adjust_keypoints_for_transform(keypoints_2d, transform_info)
            
            # Map keypoints to SMAL format in preprocessed space
            preprocessed_image_size = (self.target_resolution, self.target_resolution)
            smal_keypoints, smal_visibility = loader.map_keypoints_to_smal_model(
                adjusted_keypoints_2d, visibility, preprocessed_image_size
            )
            
            # Encode image as JPEG
            jpeg_image = self._encode_image_jpeg(processed_image)
            
            # Derive transform arrays for storage
            if transform_info['mode'] == 'centred':
                scale_pair = (transform_info['scale_factor'], transform_info['scale_factor'])
            else:
                scale_pair = (transform_info['scale_factor'][0], transform_info['scale_factor'][1])
            crop_offset_yx = np.array(transform_info['crop_offset'], dtype=np.int32)
            crop_size_hw = np.array(transform_info['crop_size'], dtype=np.int32)
            scale_yx = np.array(scale_pair, dtype=np.float32)
            
            # Create sample data
            sample_data = {
                # Image data
                'image_jpeg': jpeg_image,
                'mask': np.zeros((self.target_resolution, self.target_resolution), dtype=np.uint8),
                
                # SMIL parameters (placeholders for missing data)
                'global_rot': np.zeros(3, dtype=np.float32),
                'joint_rot': np.zeros((config.N_POSE + 1, 3), dtype=np.float32),  # Include root joint for consistency
                'betas': ground_truth_betas if ground_truth_betas is not None else np.zeros(config.N_BETAS, dtype=np.float32),
                'trans': np.zeros(3, dtype=np.float32),
                'fov': np.array(45.0, dtype=np.float32),
                'cam_rot': np.eye(3, dtype=np.float32),
                'cam_trans': np.zeros(3, dtype=np.float32),
                
                # Keypoint data (actual data)
                'keypoints_2d': smal_keypoints.astype(np.float32),
                'keypoints_3d': np.zeros((config.N_POSE + 1, 3), dtype=np.float32),
                'keypoint_visibility': smal_visibility.astype(np.float32),
                
                # Optional parameters (placeholders)
                'scale_weights': np.zeros(config.N_BETAS, dtype=np.float32),
                'trans_weights': np.zeros(config.N_BETAS, dtype=np.float32),
                
                # Metadata
                'original_path': f"{session_path}/{camera_name}/frame_{frame_idx:04d}",
                'session_name': Path(session_path).name,
                'camera_name': camera_name,
                'frame_idx': frame_idx,
                'silhouette_coverage': 0.0,
                'visible_keypoints': int(np.sum(smal_visibility > 0.5)),
                'has_ground_truth_betas': ground_truth_betas is not None,
                # Transform info for traceability
                'crop_offset_yx': crop_offset_yx,
                'crop_size_hw': crop_size_hw,
                'scale_yx': scale_yx,
            }
            
            return sample_data
            
        except Exception as e:
            print(f"Error processing frame {frame_idx} for camera {camera_name}: {e}")
            return None
    
    def _sanitize_array(self, arr: np.ndarray, default_value: float = 0.0) -> np.ndarray:
        """
        Sanitize numpy array by replacing NaN/None/inf values with a default value.
        
        Args:
            arr: Input array
            default_value: Value to replace invalid entries with
            
        Returns:
            Sanitized array
        """
        if arr is None:
            return None
        # Replace NaN and inf with default value
        arr = np.nan_to_num(arr, nan=default_value, posinf=default_value, neginf=default_value)
        return arr
    
    def _process_frame_with_data(self, loader: SLEAPDataLoader, camera_data: Dict[str, Any],
                               camera_name: str, frame_idx: int, session_path: str,
                               image_size: Tuple[int, int], ground_truth_betas: Optional[np.ndarray],
                               frame: np.ndarray, reproj_handle: Optional[h5py.File] = None) -> Optional[Dict[str, Any]]:
        """
        Process a single frame with pre-read frame data (optimized version).
        
        Uses pre-loaded camera data and pre-read frame.
        """
        try:
            # Extract 2D keypoints and visibility (from reprojections or SLEAP predictions)
            # These should be in original video frame coordinates
            keypoints_2d, visibility = self._extract_2d_keypoints_for_frame(
                loader=loader,
                camera_data=camera_data,
                camera_name=camera_name,
                frame_idx=frame_idx,
                reproj_handle=reproj_handle
            )
            
            if len(keypoints_2d) == 0:
                return None  # Skip samples with no keypoints
            
            # Convert BGR to RGB (frame is already read)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Preprocess image (crop and/or resize)
            # Pass keypoints for bbox_crop mode
            processed_image, transform_info = self._preprocess_image(frame_rgb, keypoints_2d)
            
            # Adjust keypoints based on image transformations (crop offset + scaling)
            adjusted_keypoints_2d = self._adjust_keypoints_for_transform(keypoints_2d, transform_info)
            
            # Map keypoints to SMAL model format
            # Use target_resolution as the image size since keypoints are now in preprocessed image coordinates
            preprocessed_image_size = (self.target_resolution, self.target_resolution)
            smal_keypoints, smal_visibility = loader.map_keypoints_to_smal_model(
                adjusted_keypoints_2d, visibility, preprocessed_image_size
            )
            
            # Sanitize keypoints and visibility to remove NaN/inf
            smal_keypoints = self._sanitize_array(smal_keypoints, default_value=0.0)
            smal_visibility = self._sanitize_array(smal_visibility, default_value=0.0)
            
            # Sanitize ground truth betas if provided
            if ground_truth_betas is not None:
                ground_truth_betas = self._sanitize_array(ground_truth_betas, default_value=0.0)
            
            # Encode image as JPEG
            jpeg_image = self._encode_image_jpeg(processed_image)
            
            # Create sample data with sanitized values
            sample_data = {
                # Image data
                'image_jpeg': jpeg_image,
                'mask': np.zeros((self.target_resolution, self.target_resolution), dtype=np.uint8),
                
                # SMIL parameters (placeholders for missing data)
                'global_rot': np.zeros(3, dtype=np.float32),
                'joint_rot': np.zeros((config.N_POSE + 1, 3), dtype=np.float32),  # Include root joint for consistency
                'betas': ground_truth_betas if ground_truth_betas is not None else np.zeros(config.N_BETAS, dtype=np.float32),
                'trans': np.zeros(3, dtype=np.float32),
                'fov': np.array(60.0, dtype=np.float32),
                'cam_rot': np.eye(3, dtype=np.float32),
                'cam_trans': np.zeros(3, dtype=np.float32),
                
                # Keypoint data (actual data, sanitized)
                'keypoints_2d': smal_keypoints.astype(np.float32),
                'keypoints_3d': np.zeros((config.N_POSE + 1, 3), dtype=np.float32),
                'keypoint_visibility': smal_visibility.astype(np.float32),
                
                # Optional parameters (placeholders)
                'scale_weights': np.zeros(config.N_BETAS, dtype=np.float32),
                'trans_weights': np.zeros(config.N_BETAS, dtype=np.float32),
                
                # Metadata
                'original_path': f"{session_path}/{camera_name}/frame_{frame_idx:04d}",
                'session_name': Path(session_path).name,
                'camera_name': camera_name,
                'frame_idx': frame_idx,
                'silhouette_coverage': 0.0,
                'visible_keypoints': int(np.sum(smal_visibility > 0.5)),
                'has_ground_truth_betas': ground_truth_betas is not None
            }
            
            return sample_data
            
        except Exception as e:
            print(f"Error processing frame {frame_idx} for camera {camera_name}: {e}")
            return None

    def _extract_2d_keypoints_for_frame(self, loader: SLEAPDataLoader,
                                        camera_data: Dict[str, Any],
                                        camera_name: str,
                                        frame_idx: int,
                                        reproj_handle: Optional[h5py.File]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (keypoints_2d, visibility) for a given frame, using reprojected 2D
        coordinates when enabled and available, otherwise falling back to loader
        predictions.
        """
        # Try reprojections first if enabled
        if self.use_reprojections and reproj_handle is not None:
            ds = self._resolve_reprojection_dataset(reproj_handle, camera_name)
            if ds is not None:
                try:
                    # Expected shape: (frames, instances=1, num_keypoints, 2)
                    kp = ds[frame_idx, 0, :, :]
                    kp = np.asarray(kp)
                    # Visibility: finite and positive coordinates
                    vis = (~np.isnan(kp).any(axis=1)).astype(np.float32)
                    # Penalize zero coordinates as invisible
                    vis = (vis * (np.all(kp > 0, axis=1).astype(np.float32))).astype(np.float32)
                    return kp, vis
                except Exception as e:
                    print(f"Warning: Failed to read reprojection data for camera {camera_name} frame {frame_idx}: {e}")
                    # fall through to loader
        
        # Fallback to SLEAP predictions
        return loader.extract_2d_keypoints(camera_data, frame_idx)

    def _resolve_reprojection_dataset(self, reproj_handle: h5py.File, camera_name: str) -> Optional[h5py.Dataset]:
        """
        Map a loader camera_name to a dataset in reprojections.h5.
        Tries exact match, uppercase match, last-char match, or the only dataset.
        """
        try:
            keys = list(reproj_handle.keys())
            if camera_name in reproj_handle:
                return reproj_handle[camera_name]
            if camera_name.upper() in reproj_handle:
                return reproj_handle[camera_name.upper()]
            if len(camera_name) > 0 and camera_name[-1].upper() in reproj_handle:
                return reproj_handle[camera_name[-1].upper()]
            if len(keys) == 1:
                return reproj_handle[keys[0]]
            # Try partial contains (e.g., 'camA' -> 'A')
            for k in keys:
                if k in camera_name or camera_name in k:
                    return reproj_handle[k]
        except Exception:
            pass
        print(f"Warning: Could not resolve reprojection dataset for camera {camera_name}")
        return None

    def _find_all_reprojection_files(self, session_path: str) -> Dict[str, h5py.File]:
        """
        Find and open all reprojections*.h5 files in subdirectories of session_path.
        Returns a dictionary mapping subdirectory paths to open file handles.
        """
        reproj_files = {}
        base = Path(session_path)
        try:
            for sub in base.iterdir():
                if not sub.is_dir():
                    continue
                # Look for files matching pattern reprojections*.h5
                candidates = list(sub.glob('reprojections*.h5'))
                if candidates:
                    cand = candidates[0]  # Use first match
                    try:
                        reproj_files[str(sub)] = h5py.File(str(cand), 'r')
                        if not self._reproj_warned_missing:
                            print(f"Info: Found reprojections file in {sub.name}: {cand.name}")
                    except Exception as e:
                        print(f"Warning: Failed to open reprojections file {cand}: {e}")
        except Exception as e:
            print(f"Warning: Error searching for reprojections in {session_path}: {e}")
        
        if reproj_files and not self._reproj_warned_missing:
            print(f"Info: Found {len(reproj_files)} reprojections*.h5 file(s) in session subdirectories")
            self._reproj_warned_missing = True  # Prevent spam
        
        return reproj_files
    
    def _find_camera_subdir(self, session_path: str, camera_name: str) -> Optional[str]:
        """
        Find the subdirectory containing annotation files for the given camera.
        Returns the subdirectory path or None if not found.
        """
        base = Path(session_path)
        try:
            for sub in base.iterdir():
                if not sub.is_dir():
                    continue
                # Check if this subdir contains the annotation for this camera
                annotation_files = list(sub.glob(f"*_cam{camera_name}.h5"))
                if annotation_files:
                    return str(sub)
        except Exception:
            pass
        return None
    
    def _open_reprojections_in_annotation_dir(self, session_path: str, camera_name: str) -> Optional[h5py.File]:
        """
        Open reprojections.h5 located in the subdirectory that contains
        the camera annotation file "*_cam{camera_name}.h5".
        Only searches immediate subdirectories of session_path.
        """
        base = Path(session_path)
        try:
            for sub in base.iterdir():
                if not sub.is_dir():
                    continue
                # Does this subdir contain the annotation for this camera?
                annotation_files = list(sub.glob(f"*_cam{camera_name}.h5"))
                if annotation_files:
                    # Look for files matching pattern reprojections*.h5
                    candidates = list(sub.glob('reprojections*.h5'))
                    if candidates:
                        cand = candidates[0]  # Use first match
                        try:
                            return h5py.File(str(cand), 'r')
                        except Exception as e:
                            print(f"Warning: Failed to open reprojections file {cand}: {e}")
                            return None
                    else:
                        # Annotation dir found but no reprojections*.h5
                        if not self._reproj_warned_missing:
                            print(f"Info: Annotation directory found ({sub}) but no reprojections*.h5 for camera {camera_name}")
                            self._reproj_warned_missing = True
        except Exception as e:
            print(f"Warning: Error searching for reprojections in {session_path}: {e}")
        return None
    
    def _process_single_sample(self, loader: SLEAPDataLoader, camera_name: str, 
                             frame_idx: int, session_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single sample (camera + frame combination).
        
        Args:
            loader: SLEAP data loader instance
            camera_name: Name of the camera
            frame_idx: Frame index
            session_path: Path to the session
            
        Returns:
            Processed sample dictionary or None if failed
        """
        try:
            # Load camera data
            camera_data = loader.load_camera_data(camera_name)
            
            # Extract 2D keypoints and visibility (in original video frame coordinates)
            keypoints_2d, visibility = loader.extract_2d_keypoints(camera_data, frame_idx)
            
            if len(keypoints_2d) == 0:
                return None  # Skip samples with no keypoints
            
            # Load video frame
            frame = loader.load_video_frame(camera_name, frame_idx)
            if frame is None:
                return None  # Skip samples without video frames
            
            # Preprocess image (crop and/or resize)
            # Pass keypoints for bbox_crop mode
            processed_image, transform_info = self._preprocess_image(frame, keypoints_2d)
            
            # Adjust keypoints based on image transformations
            adjusted_keypoints_2d = self._adjust_keypoints_for_transform(keypoints_2d, transform_info)
            
            # Map keypoints to SMAL model format
            # Use target_resolution as the image size since keypoints are now in preprocessed image coordinates
            preprocessed_image_size = (self.target_resolution, self.target_resolution)
            smal_keypoints, smal_visibility = loader.map_keypoints_to_smal_model(
                adjusted_keypoints_2d, visibility, preprocessed_image_size
            )
            
            # Encode image as JPEG
            jpeg_image = self._encode_image_jpeg(processed_image)
            
            # Get ground truth shape betas if available
            ground_truth_betas = loader.get_ground_truth_shape_betas()
            
            # Create sample data
            sample_data = {
                # Image data
                'image_jpeg': jpeg_image,
                'mask': np.zeros((self.target_resolution, self.target_resolution), dtype=np.uint8),  # Placeholder
                
                # SMIL parameters (placeholders for missing data)
                'global_rot': np.zeros(3, dtype=np.float32),  # Placeholder
                'joint_rot': np.zeros((config.N_POSE + 1, 3), dtype=np.float32),  # Include root joint for consistency  # Placeholder
                'betas': ground_truth_betas if ground_truth_betas is not None else np.zeros(config.N_BETAS, dtype=np.float32),
                'trans': np.zeros(3, dtype=np.float32),  # Placeholder
                'fov': np.array(45.0, dtype=np.float32),  # Placeholder
                'cam_rot': np.eye(3, dtype=np.float32),  # Placeholder
                'cam_trans': np.zeros(3, dtype=np.float32),  # Placeholder
                
                # Keypoint data (actual data)
                'keypoints_2d': smal_keypoints.astype(np.float32),
                'keypoints_3d': np.zeros((config.N_POSE + 1, 3), dtype=np.float32),  # Placeholder
                'keypoint_visibility': smal_visibility.astype(np.float32),
                
                # Optional parameters (placeholders)
                'scale_weights': np.zeros(config.N_BETAS, dtype=np.float32),  # Placeholder
                'trans_weights': np.zeros(config.N_BETAS, dtype=np.float32),  # Placeholder
                
                # Metadata
                'original_path': f"{session_path}/{camera_name}/frame_{frame_idx:04d}",
                'session_name': Path(session_path).name,
                'camera_name': camera_name,
                'frame_idx': frame_idx,
                'silhouette_coverage': 0.0,  # Placeholder
                'visible_keypoints': int(np.sum(smal_visibility > 0.5)),
                'has_ground_truth_betas': ground_truth_betas is not None
            }
            
            self.stats['processed_samples'] += 1
            return sample_data
            
        except Exception as e:
            print(f"Error processing sample {camera_name}/{frame_idx}: {e}")
            self.stats['failed_samples'] += 1
            return None
    
    def _preprocess_image(self, image: np.ndarray, keypoints_2d: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess image for training with optional center cropping or bbox-based cropping.
        
        Args:
            image: Input image (H, W, C) in range [0, 255]
            keypoints_2d: Optional 2D keypoints (N, 2) for bbox_crop mode
            
        Returns:
            Tuple of (preprocessed_image, transform_info) where:
            - preprocessed_image: Preprocessed image (target_resolution, target_resolution, C) in range [0, 1]
            - transform_info: Dictionary containing transformation parameters for keypoint adjustment
        """
        original_h, original_w = image.shape[:2]
        transform_info = {
            'original_size': (original_h, original_w),
            'crop_offset': (0, 0),  # (y_offset, x_offset)
            'crop_size': (original_h, original_w),
            'scale_factor': 1.0,
            'mode': self.crop_mode
        }
        
        if self.crop_mode == 'centred':
            # Center crop to square based on shorter side
            crop_size = min(original_h, original_w)
            
            # Calculate crop offsets (centered)
            y_offset = (original_h - crop_size) // 2
            x_offset = (original_w - crop_size) // 2
            
            # Crop the image
            image = image[y_offset:y_offset + crop_size, x_offset:x_offset + crop_size]
            
            # Store transformation info
            transform_info['crop_offset'] = (y_offset, x_offset)
            transform_info['crop_size'] = (crop_size, crop_size)
            
            # Calculate scale factor for resizing
            scale_factor = self.target_resolution / crop_size
            transform_info['scale_factor'] = scale_factor
            
            # Resize to target resolution
            image = cv2.resize(image, (self.target_resolution, self.target_resolution))
            
        elif self.crop_mode == 'bbox_crop':
            # Crop around instance based on 2D keypoints bounding box
            if keypoints_2d is None or len(keypoints_2d) == 0:
                # Fall back to default mode if no keypoints available
                scale_y = self.target_resolution / original_h
                scale_x = self.target_resolution / original_w
                transform_info['scale_factor'] = (scale_y, scale_x)
                image = cv2.resize(image, (self.target_resolution, self.target_resolution))
            else:
                # Filter out invalid keypoints (zeros, NaNs)
                valid_kpts = keypoints_2d[~np.isnan(keypoints_2d).any(axis=1)]
                valid_kpts = valid_kpts[(valid_kpts[:, 0] > 0) & (valid_kpts[:, 1] > 0)]
                
                if len(valid_kpts) == 0:
                    # Fall back to default mode if no valid keypoints
                    scale_y = self.target_resolution / original_h
                    scale_x = self.target_resolution / original_w
                    transform_info['scale_factor'] = (scale_y, scale_x)
                    image = cv2.resize(image, (self.target_resolution, self.target_resolution))
                else:
                    # Step 1: Get min/max coordinates
                    x_min, y_min = valid_kpts.min(axis=0)
                    x_max, y_max = valid_kpts.max(axis=0)
                    
                    # Step 2: Calculate center of bounding box
                    center_x = (x_min + x_max) / 2.0
                    center_y = (y_min + y_max) / 2.0
                    
                    # Step 3: Get longest dimension and multiply by 1.05
                    width = x_max - x_min
                    height = y_max - y_min
                    bbox_size = max(width, height) * 1.05
                    
                    # Step 4: Create square bounding box centered on instance
                    half_size = bbox_size / 2.0
                    x_start = center_x - half_size
                    y_start = center_y - half_size
                    x_end = center_x + half_size
                    y_end = center_y + half_size
                    
                    # Adjust if bbox extends beyond image boundaries
                    # Ensure crop is within image bounds while maintaining square aspect
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
                    
                    # Calculate actual dimensions after boundary adjustments
                    actual_width = x_end - x_start
                    actual_height = y_end - y_start
                    
                    # Step 5: Crop the image
                    image = image[int(y_start):int(y_end), int(x_start):int(x_end)]
                    
                    # Store transformation info
                    transform_info['crop_offset'] = (int(y_start), int(x_start))
                    transform_info['crop_size'] = (int(actual_height), int(actual_width))
                    
                    # Calculate scale factor for resizing
                    scale_factor = self.target_resolution / max(actual_height, actual_width)
                    transform_info['scale_factor'] = scale_factor
                    
                    # Resize to target resolution
                    image = cv2.resize(image, (self.target_resolution, self.target_resolution))
            
        else:  # 'default' mode
            # Direct resize (may distort aspect ratio)
            scale_y = self.target_resolution / original_h
            scale_x = self.target_resolution / original_w
            
            transform_info['scale_factor'] = (scale_y, scale_x)  # Different scales for each axis
            
            image = cv2.resize(image, (self.target_resolution, self.target_resolution))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image, transform_info
    
    def _adjust_keypoints_for_transform(self, keypoints_2d: np.ndarray, 
                                       transform_info: Dict[str, Any]) -> np.ndarray:
        """
        Adjust 2D keypoints based on image preprocessing transformations.
        
        Args:
            keypoints_2d: Original keypoints in image coordinates (N, 2)
            transform_info: Transformation information from _preprocess_image
            
        Returns:
            Adjusted keypoints in preprocessed image coordinates (N, 2)
        """
        adjusted_keypoints = keypoints_2d.copy()
        
        if transform_info['mode'] in ['centred', 'bbox_crop']:
            # Apply crop offset
            y_offset, x_offset = transform_info['crop_offset']
            adjusted_keypoints[:, 0] -= x_offset  # X coordinate
            adjusted_keypoints[:, 1] -= y_offset  # Y coordinate
            
            # Apply scaling
            scale_factor = transform_info['scale_factor']
            adjusted_keypoints *= scale_factor
            
        else:  # 'default' mode
            # Apply different scale factors for each axis
            scale_y, scale_x = transform_info['scale_factor']
            adjusted_keypoints[:, 0] *= scale_x  # X coordinate
            adjusted_keypoints[:, 1] *= scale_y  # Y coordinate
        
        return adjusted_keypoints
    
    def _encode_image_jpeg(self, image: np.ndarray) -> bytes:
        """
        Encode image as JPEG bytes for storage efficiency.
        
        Args:
            image: Image array (H, W, C) in range [0, 1]
            
        Returns:
            JPEG encoded bytes
        """
        # Convert to uint8
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, jpeg_bytes = cv2.imencode('.jpg', image_uint8, encode_param)
        
        return jpeg_bytes.tobytes()
    
    def process_dataset(self, sessions_dir: str, output_path: str, 
                       num_workers: int = 4, verbose: bool = True) -> Dict[str, Any]:
        """
        Process all SLEAP sessions into a single HDF5 dataset.
        
        Args:
            sessions_dir: Directory containing SLEAP sessions
            output_path: Output HDF5 file path
            num_workers: Number of parallel workers
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing processing statistics
        """
        if verbose:
            print("Discovering SLEAP sessions...")
        
        # Discover all sessions
        sessions = self.discover_sleap_sessions(sessions_dir)
        self.stats['total_sessions'] = len(sessions)
        
        if len(sessions) == 0:
            raise ValueError(f"No SLEAP sessions found in {sessions_dir}")
        
        if verbose:
            print(f"Found {len(sessions)} SLEAP sessions")
            for session in sessions:
                print(f"  - {Path(session).name}")
        
        # Process sessions in parallel
        all_samples = []
        
        if num_workers > 1:
            if verbose:
                print(f"Processing sessions with {num_workers} workers...")
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all sessions
                future_to_session = {
                    executor.submit(self.process_single_session, session): session 
                    for session in sessions
                }
                
                # Collect results with progress bar
                if verbose:
                    futures = tqdm(as_completed(future_to_session), 
                                 total=len(sessions), 
                                 desc="Processing sessions")
                else:
                    futures = as_completed(future_to_session)
                
                for future in futures:
                    session = future_to_session[future]
                    try:
                        samples = future.result()
                        all_samples.extend(samples)
                        if verbose:
                            print(f"Processed {len(samples)} samples from {Path(session).name}")
                    except Exception as e:
                        print(f"Error processing session {session}: {e}")
        else:
            # Single-threaded processing
            if verbose:
                sessions_iter = tqdm(sessions, desc="Processing sessions")
            else:
                sessions_iter = sessions
            
            for session in sessions_iter:
                samples = self.process_single_session(session)
                all_samples.extend(samples)
                if verbose:
                    print(f"Processed {len(samples)} samples from {Path(session).name}")
        
        self.stats['total_samples'] = len(all_samples)
        
        if len(all_samples) == 0:
            raise ValueError("No samples were successfully processed")
        
        if verbose:
            print(f"Total samples processed: {len(all_samples)}")
            print("Saving to HDF5...")
        
        # Save to HDF5
        self._save_to_hdf5(all_samples, output_path)
        
        if verbose:
            print(f"Dataset saved to: {output_path}")
        
        return self.stats
    
    def _save_to_hdf5(self, samples: List[Dict[str, Any]], output_path: str):
        """
        Save processed samples to HDF5 file.
        
        Args:
            samples: List of processed sample dictionaries
            output_path: Output HDF5 file path
        """
        with h5py.File(output_path, 'w') as f:
            # Create groups
            images_group = f.create_group('images')
            parameters_group = f.create_group('parameters')
            keypoints_group = f.create_group('keypoints')
            auxiliary_group = f.create_group('auxiliary')
            metadata_group = f.create_group('metadata')
            
            # Prepare data arrays
            num_samples = len(samples)
            
            # Image data
            image_jpeg_data = []
            mask_data = []
            
            # Parameter data
            global_rot_data = []
            joint_rot_data = []
            betas_data = []
            trans_data = []
            fov_data = []
            cam_rot_data = []
            cam_trans_data = []
            scale_weights_data = []
            trans_weights_data = []
            
            # Keypoint data
            keypoints_2d_data = []
            keypoints_3d_data = []
            keypoint_visibility_data = []
            
            # Auxiliary data
            original_paths = []
            session_names = []
            camera_names = []
            frame_indices = []
            silhouette_coverage_data = []
            visible_keypoints_data = []
            has_ground_truth_betas_data = []
            crop_offset_yx_data = []
            crop_size_hw_data = []
            scale_yx_data = []
            
            # Collect data
            for sample in samples:
                # Image data
                image_jpeg_data.append(sample['image_jpeg'])
                mask_data.append(sample['mask'])
                
                # Parameter data
                global_rot_data.append(sample['global_rot'])
                joint_rot_data.append(sample['joint_rot'])
                betas_data.append(sample['betas'])
                trans_data.append(sample['trans'])
                fov_data.append(sample['fov'])
                cam_rot_data.append(sample['cam_rot'])
                cam_trans_data.append(sample['cam_trans'])
                scale_weights_data.append(sample['scale_weights'])
                trans_weights_data.append(sample['trans_weights'])
                
                # Keypoint data
                keypoints_2d_data.append(sample['keypoints_2d'])
                keypoints_3d_data.append(sample['keypoints_3d'])
                keypoint_visibility_data.append(sample['keypoint_visibility'])
                
                # Auxiliary data
                original_paths.append(sample['original_path'])
                session_names.append(sample['session_name'])
                camera_names.append(sample['camera_name'])
                frame_indices.append(sample['frame_idx'])
                silhouette_coverage_data.append(sample['silhouette_coverage'])
                visible_keypoints_data.append(sample['visible_keypoints'])
                has_ground_truth_betas_data.append(sample['has_ground_truth_betas'])
                # Transform info (optional for older samples)
                crop_offset_yx_data.append(sample.get('crop_offset_yx', np.array([0, 0], dtype=np.int32)))
                crop_size_hw_data.append(sample.get('crop_size_hw', np.array([self.target_resolution, self.target_resolution], dtype=np.int32)))
                scale_yx_data.append(sample.get('scale_yx', np.array([1.0, 1.0], dtype=np.float32)))
            
            # Save image data (handle binary JPEG data properly)
            # Convert bytes objects to numpy arrays for HDF5 storage
            image_jpeg_arrays = []
            for img_bytes in image_jpeg_data:
                # Convert bytes to numpy array of uint8
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                image_jpeg_arrays.append(img_array)
            
            # Store as variable-length binary data
            dt = h5py.special_dtype(vlen=np.uint8)
            images_group.create_dataset('image_jpeg', 
                                      data=image_jpeg_arrays,
                                      dtype=dt,
                                      compression=self.compression,
                                      compression_opts=self.compression_level,
                                      chunks=(self.chunk_size,))
            images_group.create_dataset('mask',
                                      data=np.array(mask_data),
                                      compression=self.compression,
                                      compression_opts=self.compression_level,
                                      chunks=(self.chunk_size, self.target_resolution, self.target_resolution))
            
            # Save parameter data
            parameters_group.create_dataset('global_rot',
                                          data=np.array(global_rot_data),
                                          compression=self.compression,
                                          compression_opts=self.compression_level,
                                          chunks=(self.chunk_size, 3))
            parameters_group.create_dataset('joint_rot',
                                          data=np.array(joint_rot_data),
                                          compression=self.compression,
                                          compression_opts=self.compression_level,
                                          chunks=(self.chunk_size, config.N_POSE + 1, 3))
            parameters_group.create_dataset('betas',
                                          data=np.array(betas_data),
                                          compression=self.compression,
                                          compression_opts=self.compression_level,
                                          chunks=(self.chunk_size, config.N_BETAS))
            parameters_group.create_dataset('trans',
                                          data=np.array(trans_data),
                                          compression=self.compression,
                                          compression_opts=self.compression_level,
                                          chunks=(self.chunk_size, 3))
            parameters_group.create_dataset('fov',
                                          data=np.array(fov_data),
                                          compression=self.compression,
                                          compression_opts=self.compression_level,
                                          chunks=(self.chunk_size,))
            parameters_group.create_dataset('cam_rot',
                                          data=np.array(cam_rot_data),
                                          compression=self.compression,
                                          compression_opts=self.compression_level,
                                          chunks=(self.chunk_size, 3, 3))
            parameters_group.create_dataset('cam_trans',
                                          data=np.array(cam_trans_data),
                                          compression=self.compression,
                                          compression_opts=self.compression_level,
                                          chunks=(self.chunk_size, 3))
            parameters_group.create_dataset('scale_weights',
                                          data=np.array(scale_weights_data),
                                          compression=self.compression,
                                          compression_opts=self.compression_level,
                                          chunks=(self.chunk_size, config.N_BETAS))
            parameters_group.create_dataset('trans_weights',
                                          data=np.array(trans_weights_data),
                                          compression=self.compression,
                                          compression_opts=self.compression_level,
                                          chunks=(self.chunk_size, config.N_BETAS))
            
            # Save keypoint data
            keypoints_group.create_dataset('keypoints_2d',
                                         data=np.array(keypoints_2d_data),
                                         compression=self.compression,
                                         compression_opts=self.compression_level,
                                         chunks=(self.chunk_size, config.N_POSE + 1, 2))
            keypoints_group.create_dataset('keypoints_3d',
                                         data=np.array(keypoints_3d_data),
                                         compression=self.compression,
                                         compression_opts=self.compression_level,
                                         chunks=(self.chunk_size, config.N_POSE + 1, 3))
            keypoints_group.create_dataset('keypoint_visibility',
                                         data=np.array(keypoint_visibility_data),
                                         compression=self.compression,
                                         compression_opts=self.compression_level,
                                         chunks=(self.chunk_size, config.N_POSE + 1))
            
            # Save auxiliary data
            auxiliary_group.create_dataset('original_path',
                                         data=[p.encode('utf-8') for p in original_paths],
                                         compression=self.compression,
                                         compression_opts=self.compression_level,
                                         chunks=(self.chunk_size,))
            auxiliary_group.create_dataset('session_name',
                                         data=[s.encode('utf-8') for s in session_names],
                                         compression=self.compression,
                                         compression_opts=self.compression_level,
                                         chunks=(self.chunk_size,))
            auxiliary_group.create_dataset('camera_name',
                                         data=[c.encode('utf-8') for c in camera_names],
                                         compression=self.compression,
                                         compression_opts=self.compression_level,
                                         chunks=(self.chunk_size,))
            auxiliary_group.create_dataset('frame_idx',
                                         data=np.array(frame_indices),
                                         compression=self.compression,
                                         compression_opts=self.compression_level,
                                         chunks=(self.chunk_size,))
            auxiliary_group.create_dataset('silhouette_coverage',
                                         data=np.array(silhouette_coverage_data),
                                         compression=self.compression,
                                         compression_opts=self.compression_level,
                                         chunks=(self.chunk_size,))
            auxiliary_group.create_dataset('visible_keypoints',
                                         data=np.array(visible_keypoints_data),
                                         compression=self.compression,
                                         compression_opts=self.compression_level,
                                         chunks=(self.chunk_size,))
            auxiliary_group.create_dataset('has_ground_truth_betas',
                                         data=np.array(has_ground_truth_betas_data),
                                         compression=self.compression,
                                         compression_opts=self.compression_level,
                                         chunks=(self.chunk_size,))
            auxiliary_group.create_dataset('crop_offset_yx',
                                         data=np.array(crop_offset_yx_data, dtype=np.int32),
                                         compression=self.compression,
                                         compression_opts=self.compression_level,
                                         chunks=(self.chunk_size, 2))
            auxiliary_group.create_dataset('crop_size_hw',
                                         data=np.array(crop_size_hw_data, dtype=np.int32),
                                         compression=self.compression,
                                         compression_opts=self.compression_level,
                                         chunks=(self.chunk_size, 2))
            auxiliary_group.create_dataset('scale_yx',
                                         data=np.array(scale_yx_data, dtype=np.float32),
                                         compression=self.compression,
                                         compression_opts=self.compression_level,
                                         chunks=(self.chunk_size, 2))
            
            # Save metadata
            metadata_group.attrs['num_samples'] = num_samples
            metadata_group.attrs['target_resolution'] = self.target_resolution
            metadata_group.attrs['backbone_name'] = self.backbone_name
            metadata_group.attrs['jpeg_quality'] = self.jpeg_quality
            metadata_group.attrs['chunk_size'] = self.chunk_size
            metadata_group.attrs['compression'] = self.compression
            metadata_group.attrs['compression_level'] = self.compression_level
            metadata_group.attrs['crop_mode'] = self.crop_mode
            metadata_group.attrs['dataset_type'] = 'sleap'
            metadata_group.attrs['is_sleap_dataset'] = True  # Flag for SLEAP-specific loss computation
            metadata_group.attrs['n_pose'] = config.N_POSE
            metadata_group.attrs['n_betas'] = config.N_BETAS
            metadata_group.attrs['joint_lookup_table_used'] = self.joint_lookup_table_path is not None
            metadata_group.attrs['shape_betas_table_used'] = self.shape_betas_table_path is not None
            metadata_group.attrs['used_reprojections'] = self.use_reprojections
            metadata_group.attrs['confidence_threshold'] = self.confidence_threshold
            
            # Save processing statistics
            metadata_group.attrs['total_sessions'] = self.stats['total_sessions']
            metadata_group.attrs['sessions_processed'] = self.stats['sessions_processed']
            metadata_group.attrs['sessions_failed'] = self.stats['sessions_failed']
            metadata_group.attrs['total_samples'] = self.stats['total_samples']
            metadata_group.attrs['processed_samples'] = self.stats['processed_samples']
            metadata_group.attrs['failed_samples'] = self.stats['failed_samples']


def validate_input_directory(sessions_dir: str) -> bool:
    """
    Validate that input directory exists and contains SLEAP sessions.
    
    Args:
        sessions_dir: Path to sessions directory
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(sessions_dir):
        print(f"Error: Sessions directory does not exist: {sessions_dir}")
        return False
    
    if not os.path.isdir(sessions_dir):
        print(f"Error: Sessions path is not a directory: {sessions_dir}")
        return False
    
    # Check for SLEAP sessions
    preprocessor = SLEAPDatasetPreprocessor()
    sessions = preprocessor.discover_sleap_sessions(sessions_dir)
    
    if len(sessions) == 0:
        print(f"Error: No SLEAP sessions found in directory: {sessions_dir}")
        return False
    
    print(f"Found {len(sessions)} SLEAP sessions in input directory")
    return True


def validate_output_path(output_path: str) -> bool:
    """
    Validate output path and create directory if needed.
    
    Args:
        output_path: Path to output HDF5 file
        
    Returns:
        True if valid, False otherwise
    """
    # Check file extension
    if not (output_path.endswith('.h5') or output_path.endswith('.hdf5')):
        print(f"Warning: Output file should have .h5 or .hdf5 extension: {output_path}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error: Could not create output directory {output_dir}: {e}")
            return False
    
    # Check if output file already exists - always overwrite
    if os.path.exists(output_path):
        print(f"Output file already exists: {output_path} - will overwrite")
    
    return True


def estimate_processing_time(num_sessions: int, num_workers: int) -> str:
    """
    Estimate processing time based on number of sessions.
    
    Args:
        num_sessions: Number of sessions to process
        num_workers: Number of parallel workers
        
    Returns:
        Estimated time string
    """
    # Rough estimate: 30 seconds per session per worker
    seconds_per_session = 30
    estimated_seconds = (num_sessions * seconds_per_session) / num_workers
    
    if estimated_seconds < 60:
        return f"{estimated_seconds:.0f} seconds"
    elif estimated_seconds < 3600:
        return f"{estimated_seconds/60:.1f} minutes"
    else:
        return f"{estimated_seconds/3600:.1f} hours"


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SLEAP dataset into optimized HDF5 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing with default settings
  python preprocess_sleap_dataset.py /path/to/sleap/sessions optimized_sleap.h5
  
  # With joint lookup table and shape betas
  python preprocess_sleap_dataset.py /path/to/sleap/sessions optimized_sleap.h5 \\
    --joint_lookup_table /path/to/joint_lookup.csv \\
    --shape_betas_table /path/to/shape_betas.csv \\
    --backbone vit_large_patch16_224
  
  # High-quality preprocessing with more workers
  python preprocess_sleap_dataset.py /path/to/sleap/sessions optimized_sleap.h5 \\
    --jpeg_quality 98 \\
    --num_workers 8 \\
    --chunk_size 16
  
  # Debug mode - process only first 1000 frames per video
  python preprocess_sleap_dataset.py /path/to/sleap/sessions debug_sleap.h5 \\
    --max-frames-per-video 1000 \\
    --num_workers 2
  
  # Bbox crop mode - crop around detected instance (best for single-instance mesh recovery)
  python preprocess_sleap_dataset.py /path/to/sleap/sessions optimized_sleap.h5 \\
    --crop_mode bbox_crop \\
    --target_resolution 224
        """
    )
    
    # Required arguments
    parser.add_argument("sessions_dir", 
                       help="Directory containing SLEAP sessions")
    parser.add_argument("output_path", 
                       help="Output HDF5 file path (e.g., optimized_sleap_dataset.h5)")
    
    # Lookup table options
    parser.add_argument("--joint_lookup_table", type=str, default=None,
                       help="Path to CSV lookup table for joint name mapping")
    parser.add_argument("--shape_betas_table", type=str, default=None,
                       help="Path to CSV lookup table for ground truth shape betas")
    
    # Image processing options
    parser.add_argument("--target_resolution", type=int, default=224,
                       help="Target image resolution in pixels (default: 224)")
    parser.add_argument("--backbone", dest="backbone_name", default='vit_large_patch16_224',
                       choices=['vit_large_patch16_224', 'vit_base_patch16_224', 'resnet152'],
                       help="Backbone network name (default: vit_large_patch16_224)")
    parser.add_argument("--jpeg_quality", type=int, default=95,
                       help="JPEG compression quality 1-100 (default: 95)")
    parser.add_argument("--crop_mode", type=str, default='default',
                       choices=['default', 'centred', 'bbox_crop'],
                       help="Image resizing mode: 'default' = direct resize (current behavior), "
                            "'centred' = centered square crop then resize (preserves aspect ratio), "
                            "'bbox_crop' = crop around instance based on 2D keypoints with 1.05x padding")
    
    # Performance options
    parser.add_argument("--chunk_size", type=int, default=8,
                       help="HDF5 chunk size - should match training batch size (default: 8)")
    parser.add_argument("--num_workers", type=int, default=1,
                       help="Number of parallel processing workers (default: 4)")
    parser.add_argument("--max-frames-per-video", type=int, default=None,
                       help="Maximum number of frames to process per video (default: None for all frames, useful for debugging)")
    
    # Output options
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress output")
    parser.add_argument("--validate", action="store_true",
                       help="Validate the output dataset after preprocessing")
    parser.add_argument("--use_reprojections", action="store_true",
                       help="Use per-session reprojections.h5 for 2D keypoints instead of raw predictions")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                       help="Minimum confidence score (0.0-1.0) for keypoints. Keypoints below this threshold "
                            "will be marked as invisible. Default: 0.5")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        print("Error: jpeg_quality must be between 1 and 100")
        sys.exit(1)
    
    if args.chunk_size < 1:
        print("Error: chunk_size must be at least 1")
        sys.exit(1)
    
    if args.num_workers < 1:
        print("Error: num_workers must be at least 1")
        sys.exit(1)
    
    if args.confidence_threshold < 0.0 or args.confidence_threshold > 1.0:
        print("Error: confidence_threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Validate input and output paths
    if not validate_input_directory(args.sessions_dir):
        sys.exit(1)
    
    if not validate_output_path(args.output_path):
        sys.exit(1)
    
    # Count sessions for time estimation
    preprocessor = SLEAPDatasetPreprocessor()
    sessions = preprocessor.discover_sleap_sessions(args.sessions_dir)
    num_sessions = len(sessions)
    
    # Print configuration summary
    if not args.quiet:
        print("\n" + "="*60)
        print("SLEAP DATASET PREPROCESSING CONFIGURATION")
        print("="*60)
        print(f"Sessions directory: {args.sessions_dir}")
        print(f"Output file: {args.output_path}")
        print(f"Number of sessions: {num_sessions}")
        print(f"Joint lookup table: {args.joint_lookup_table or 'None'}")
        print(f"Shape betas table: {args.shape_betas_table or 'None'}")
        print(f"Target resolution: {args.target_resolution}x{args.target_resolution}")
        print(f"Backbone: {args.backbone_name}")
        print(f"JPEG quality: {args.jpeg_quality}")
        print(f"Crop mode: {args.crop_mode}")
        print(f"Chunk size: {args.chunk_size}")
        print(f"Workers: {args.num_workers}")
        print(f"Max frames per video: {args.max_frames_per_video or 'All frames'}")
        print(f"Use reprojections: {args.use_reprojections}")
        print(f"Confidence threshold: {args.confidence_threshold}")
        print(f"Estimated time: {estimate_processing_time(num_sessions, args.num_workers)}")
        print("="*60)
        
        print("\nStarting preprocessing...")
    
    # Start preprocessing
    start_time = time.time()
    
    try:
        # Create preprocessor
        preprocessor = SLEAPDatasetPreprocessor(
            joint_lookup_table_path=args.joint_lookup_table,
            shape_betas_table_path=args.shape_betas_table,
            target_resolution=args.target_resolution,
            backbone_name=args.backbone_name,
            jpeg_quality=args.jpeg_quality,
            chunk_size=args.chunk_size,
            max_frames_per_video=args.max_frames_per_video,
            crop_mode=args.crop_mode,
            use_reprojections=args.use_reprojections,
            confidence_threshold=args.confidence_threshold
        )
        
        # Process dataset
        stats = preprocessor.process_dataset(
            sessions_dir=args.sessions_dir,
            output_path=args.output_path,
            num_workers=args.num_workers,
            verbose=not args.quiet
        )
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        if not args.quiet:
            print(f"\nProcessing completed in {processing_time:.1f} seconds")
            print(f"Average time per session: {processing_time/num_sessions:.2f} seconds")
            
            # Calculate file size
            if os.path.exists(args.output_path):
                file_size = os.path.getsize(args.output_path) / (1024 * 1024)  # MB
                print(f"Output file size: {file_size:.1f} MB")
                if stats['processed_samples'] > 0:
                    print(f"Size per sample: {file_size/stats['processed_samples']:.2f} MB")
            
            # Print statistics
            print("\nProcessing Statistics:")
            print(f"  Total sessions: {stats['total_sessions']}")
            print(f"  Sessions processed: {stats['sessions_processed']}")
            print(f"  Sessions failed: {stats['sessions_failed']}")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Processed samples: {stats['processed_samples']}")
            print(f"  Failed samples: {stats['failed_samples']}")
        
        print(f"\nPreprocessing successful! Output saved to: {args.output_path}")
        
    except KeyboardInterrupt:
        print("\nPreprocessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
