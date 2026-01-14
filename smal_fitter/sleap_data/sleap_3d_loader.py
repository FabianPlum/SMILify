#!/usr/bin/env python3
"""
SLEAP 3D Data Loader

This module provides functionality to load 3D keypoint coordinates and camera
calibration parameters from SLEAP datasets for multi-view model fitting.

Usage:
    from sleap_3d_loader import SLEAP3DDataLoader
    
    loader = SLEAP3DDataLoader(session_path="/path/to/session")
    keypoints_3d = loader.get_3d_keypoints(frame_idx=0)
    camera_params = loader.get_camera_parameters(camera_name="Camera0")
"""

import os
import sys
import h5py
import numpy as np
import toml
import re
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class SLEAP3DDataLoader:
    """
    Loader for 3D keypoint coordinates and camera calibration from SLEAP datasets.
    
    This class provides clean access to:
    - 3D keypoint trajectories from points3d.h5
    - Camera intrinsic parameters (K matrices)
    - Camera extrinsic parameters (rotation and translation)
    - Camera names and image sizes
    
    All data is formatted for use in multi-view model fitting pipelines.
    """
    
    def __init__(self, session_path: str, video_subdir: Optional[str] = None, session_idx: int = 0):
        """
        Initialize the 3D data loader.
        
        Args:
            session_path: Path to SLEAP session directory or project directory containing sessions.
                         If project directory, will discover sessions and use the specified one.
            video_subdir: Optional video subdirectory name (e.g., "PerShu_012").
                         If None, will search for points3d.h5 in subdirectories.
            session_idx: Index of session to use if session_path is a project directory (default: 0)
        """
        self.session_path = Path(session_path)
        
        if not self.session_path.exists():
            raise ValueError(f"Session path does not exist: {session_path}")
        
        # Determine if this is a project directory (contains sessions) or a session directory
        self.is_project_dir = self._is_project_directory()
        
        if self.is_project_dir:
            # Discover sessions and use the specified one
            sessions = self._discover_sessions()
            if len(sessions) == 0:
                raise ValueError(f"No SLEAP sessions found in {session_path}")
            if session_idx < 0 or session_idx >= len(sessions):
                raise ValueError(f"Session index {session_idx} out of range [0, {len(sessions)-1}]")
            self.session_path = Path(sessions[session_idx])
            print(f"Using session: {self.session_path.name} (index {session_idx}/{len(sessions)-1})")
        
        # Find points3d.h5 file (may be in video subdirectory)
        self.points3d_file = self._find_points3d_file(video_subdir)
        if self.points3d_file is None:
            raise FileNotFoundError(f"points3d.h5 not found in {self.session_path} or subdirectories")
        
        # Find calibration.toml (should be in session directory)
        self.calibration_file = self.session_path / "calibration.toml"
        if not self.calibration_file.exists():
            raise FileNotFoundError(f"calibration.toml not found in {self.session_path}")
        
        # Store video subdirectory if found
        self.video_subdir = self.points3d_file.parent.name if self.points3d_file.parent != self.session_path else None
        
        # Load data
        self._load_3d_data()
        self._load_calibration_data()
        
        # Validate consistency
        self._validate_data()
    
    def _is_project_directory(self) -> bool:
        """
        Check if the path is a project directory (contains multiple sessions)
        or a single session directory.
        
        Returns:
            True if project directory, False if session directory
        """
        # Check if this looks like a project directory (contains session subdirectories)
        session_count = 0
        for item in self.session_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if this looks like a session (has calibration.toml or subdirectories with .h5)
                if (item / "calibration.toml").exists():
                    session_count += 1
                else:
                    # Check for video subdirectories with .h5 files
                    for subdir in item.iterdir():
                        if subdir.is_dir() and list(subdir.glob("*.h5")):
                            session_count += 1
                            break
        
        return session_count > 1
    
    def _discover_sessions(self) -> List[str]:
        """
        Discover all SLEAP session directories in a project directory.
        Uses the same logic as preprocess_sleap_dataset.py
        """
        sessions = []
        
        for item in self.session_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if self._is_sleap_session(item):
                    sessions.append(str(item))
        
        sessions.sort()
        return sessions
    
    def _is_sleap_session(self, session_path: Path) -> bool:
        """Check if a directory looks like a SLEAP session."""
        # Check for calibration.toml
        if (session_path / "calibration.toml").exists():
            return True
        
        # Check for video subdirectories with .h5 files
        for subdir in session_path.iterdir():
            if subdir.is_dir() and list(subdir.glob("*.h5")):
                return True
        
        # Check for camera directories with .slp files
        for cam_dir in session_path.iterdir():
            if cam_dir.is_dir() and list(cam_dir.glob("*.slp")):
                return True
        
        return False
    
    def _find_points3d_file(self, video_subdir: Optional[str] = None) -> Optional[Path]:
        """
        Find points3d.h5 file. May be in session root or in video subdirectory.
        
        Args:
            video_subdir: Optional video subdirectory name to search in
            
        Returns:
            Path to points3d.h5 file or None if not found
        """
        # First check session root
        points3d_file = self.session_path / "points3d.h5"
        if points3d_file.exists():
            return points3d_file
        
        # If video_subdir specified, check there
        if video_subdir:
            points3d_file = self.session_path / video_subdir / "points3d.h5"
            if points3d_file.exists():
                return points3d_file
        
        # Search in all subdirectories for points3d.h5
        for item in self.session_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                points3d_file = item / "points3d.h5"
                if points3d_file.exists():
                    return points3d_file
        
        return None
    
    def _load_3d_data(self):
        """Load 3D keypoint data from points3d.h5."""
        with h5py.File(self.points3d_file, 'r') as f:
            if 'tracks' not in f:
                raise ValueError(f"No 'tracks' dataset in {self.points3d_file}")
            
            tracks = f['tracks'][:]  # Shape: (n_frames, n_tracks, n_keypoints, 3)
            
            # Extract first track (assuming single animal)
            if tracks.shape[1] == 0:
                raise ValueError(f"No tracks found in {self.points3d_file}")
            
            self.keypoints_3d = tracks[:, 0, :, :]  # (n_frames, n_keypoints, 3)
            self.n_frames = self.keypoints_3d.shape[0]
            self.n_keypoints = self.keypoints_3d.shape[1]
            
            print(f"Loaded 3D data: {self.n_frames} frames, {self.n_keypoints} keypoints")
    
    def _load_calibration_data(self):
        """Load camera calibration data from calibration.toml."""
        calibration_data = toml.load(self.calibration_file)
        
        # Extract camera information
        self.cameras = {}
        self.camera_names = []
        self.calibration_key_to_name = {}  # Map calibration keys to camera names
        
        for key, value in calibration_data.items():
            if key == 'metadata':
                continue
            
            camera_name = value.get('name', key)
            self.camera_names.append(camera_name)
            self.calibration_key_to_name[key] = camera_name
            
            # Extract intrinsic parameters
            matrix = np.array(value['matrix'], dtype=np.float32)  # 3x3 K matrix
            size = value['size']  # [width, height]
            distortions = np.array(value['distortions'], dtype=np.float32)  # Distortion coeffs
            
            # Extract extrinsic parameters
            rotation_vec = np.array(value['rotation'], dtype=np.float32)  # Axis-angle
            translation = np.array(value['translation'], dtype=np.float32)  # Translation vector
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
            rotation_matrix = rotation_matrix.astype(np.float32)
            
            # Store camera parameters
            self.cameras[camera_name] = {
                'name': camera_name,
                'calibration_key': key,  # Store original key for matching
                'intrinsic': {
                    'K': matrix,  # 3x3 camera matrix
                    'distortion': distortions,  # Distortion coefficients
                },
                'extrinsic': {
                    'R': rotation_matrix,  # 3x3 rotation matrix
                    't': translation,  # 3x1 translation vector
                    'rotation_vec': rotation_vec,  # Original axis-angle (for reference)
                },
                'image_size': {
                    'width': int(size[0]),
                    'height': int(size[1]),
                }
            }
        
        self.camera_names.sort()  # Ensure consistent ordering
        print(f"Loaded calibration for {len(self.cameras)} cameras: {self.camera_names}")
        
        # Discover actual camera directories/files and create mapping
        self._discover_camera_mapping()
    
    def _discover_camera_mapping(self):
        """
        Discover actual camera directories/files and map them to calibration camera names.
        Creates a mapping from calibration camera names to actual directory/file names.
        """
        data_structure = self._detect_data_structure()
        self.camera_name_to_dir = {}  # Map calibration name -> directory/file name
        
        if data_structure == 'camera_dirs':
            # Find camera directories
            for item in self.session_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    slp_files = list(item.glob("*.slp"))
                    if slp_files:
                        dir_name = item.name
                        # Try to match with calibration camera names
                        matched = False
                        for calib_name in self.camera_names:
                            # Try exact match
                            if dir_name == calib_name:
                                self.camera_name_to_dir[calib_name] = dir_name
                                matched = True
                                break
                            # Try case-insensitive match
                            if dir_name.lower() == calib_name.lower():
                                self.camera_name_to_dir[calib_name] = dir_name
                                matched = True
                                break
                            # Try extracting number/index
                            calib_num = None
                            dir_num = None
                            try:
                                if 'Camera' in calib_name:
                                    calib_num = int(calib_name.replace('Camera', ''))
                                if 'cam' in dir_name.lower():
                                    nums = re.findall(r'\d+', dir_name)
                                    if nums:
                                        dir_num = int(nums[0])
                                if calib_num is not None and dir_num is not None and calib_num == dir_num:
                                    self.camera_name_to_dir[calib_name] = dir_name
                                    matched = True
                                    break
                            except:
                                pass
                        
                        if not matched:
                            # If no match found, try to match by index order
                            # This is a fallback - assumes cameras are in same order
                            pass
            
            # If we have calibration keys like cam_0, cam_1, try matching by index
            for calib_key, calib_name in self.calibration_key_to_name.items():
                if calib_name not in self.camera_name_to_dir:
                    # Try to extract index from calibration key
                    try:
                        if 'cam_' in calib_key.lower():
                            idx = int(calib_key.split('_')[-1])
                            # Find directory with matching index
                            for item in self.session_path.iterdir():
                                if item.is_dir() and not item.name.startswith('.'):
                                    slp_files = list(item.glob("*.slp"))
                                    if slp_files:
                                        import re
                                        nums = re.findall(r'\d+', item.name)
                                        if nums and int(nums[0]) == idx:
                                            self.camera_name_to_dir[calib_name] = item.name
                                            break
                    except:
                        pass
        
        elif data_structure == 'session_dirs':
            # Find camera files in session subdirectories
            for item in self.session_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    h5_files = list(item.glob("*.h5"))
                    if h5_files:
                        for h5_file in h5_files:
                            filename = h5_file.stem
                            if '_cam' in filename:
                                file_cam_name = filename.split('_cam')[-1]
                                # Try to match with calibration names
                                for calib_name in self.camera_names:
                                    if file_cam_name.lower() == calib_name.lower() or \
                                       file_cam_name.lower() == calib_name.lower().replace('camera', 'cam'):
                                        self.camera_name_to_dir[calib_name] = file_cam_name
                                        break
        
        # Print mapping for debugging
        if self.camera_name_to_dir:
            print(f"Camera name mapping (calibration -> directory/file):")
            for calib_name, dir_name in sorted(self.camera_name_to_dir.items()):
                print(f"  {calib_name} -> {dir_name}")
        else:
            print("Warning: No camera directory/file mapping found. Using calibration names directly.")
        
        # List all available directories/files for debugging
        if data_structure == 'camera_dirs':
            available_dirs = [d.name for d in self.session_path.iterdir() 
                            if d.is_dir() and not d.name.startswith('.') and list(d.glob("*.slp"))]
            print(f"Available camera directories: {available_dirs}")
        elif data_structure == 'session_dirs':
            available_cams = set()
            for item in self.session_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    h5_files = list(item.glob("*_cam*.h5"))
                    for h5_file in h5_files:
                        filename = h5_file.stem
                        if '_cam' in filename:
                            cam_name = filename.split('_cam')[-1]
                            available_cams.add(cam_name)
            print(f"Available camera names in files: {sorted(available_cams)}")
    
    def _validate_data(self):
        """Validate that loaded data is consistent."""
        # Check for NaN or invalid values in 3D data
        if np.isnan(self.keypoints_3d).any():
            print("Warning: NaN values found in 3D keypoints")
        
        if np.isinf(self.keypoints_3d).any():
            print("Warning: Inf values found in 3D keypoints")
        
        # Validate camera parameters
        for camera_name, camera_data in self.cameras.items():
            K = camera_data['intrinsic']['K']
            R = camera_data['extrinsic']['R']
            t = camera_data['extrinsic']['t']
            
            # Check K matrix is valid (upper triangular, positive diagonal)
            if K[0, 0] <= 0 or K[1, 1] <= 0:
                print(f"Warning: Invalid intrinsic matrix for {camera_name}")
            
            # Check R is a valid rotation matrix (orthogonal, det=1)
            if not np.isclose(np.linalg.det(R), 1.0, atol=1e-3):
                print(f"Warning: Rotation matrix for {camera_name} may not be valid (det={np.linalg.det(R):.6f})")
    
    def get_3d_keypoints(self, frame_idx: int) -> np.ndarray:
        """
        Get 3D keypoints for a specific frame.
        
        Args:
            frame_idx: Frame index (0-based)
            
        Returns:
            Array of shape (n_keypoints, 3) with 3D coordinates in mm
        """
        if frame_idx < 0 or frame_idx >= self.n_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.n_frames-1}]")
        
        return self.keypoints_3d[frame_idx].copy()  # (n_keypoints, 3)
    
    def get_all_3d_keypoints(self) -> np.ndarray:
        """
        Get all 3D keypoints for all frames.
        
        Returns:
            Array of shape (n_frames, n_keypoints, 3) with 3D coordinates in mm
        """
        return self.keypoints_3d.copy()
    
    def get_camera_parameters(self, camera_name: str) -> Dict[str, Any]:
        """
        Get camera parameters for a specific camera.
        
        Args:
            camera_name: Name of the camera (e.g., "Camera0")
            
        Returns:
            Dictionary containing:
            - 'intrinsic': {'K': 3x3 matrix, 'distortion': array}
            - 'extrinsic': {'R': 3x3 rotation matrix, 't': 3x1 translation vector}
            - 'image_size': {'width': int, 'height': int}
            - 'name': str
        """
        if camera_name not in self.cameras:
            raise ValueError(f"Camera '{camera_name}' not found. Available: {self.camera_names}")
        
        return self.cameras[camera_name].copy()
    
    def get_camera_intrinsic(self, camera_name: str) -> np.ndarray:
        """
        Get camera intrinsic matrix (K) for a specific camera.
        
        Args:
            camera_name: Name of the camera
            
        Returns:
            3x3 camera intrinsic matrix
        """
        return self.get_camera_parameters(camera_name)['intrinsic']['K'].copy()
    
    def get_camera_extrinsic(self, camera_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera extrinsic parameters (rotation and translation).
        
        Args:
            camera_name: Name of the camera
            
        Returns:
            Tuple of (R, t) where:
            - R: 3x3 rotation matrix
            - t: 3x1 translation vector
        """
        params = self.get_camera_parameters(camera_name)
        return params['extrinsic']['R'].copy(), params['extrinsic']['t'].copy()
    
    def get_camera_projection_matrix(self, camera_name: str, use_alternative_convention: bool = False) -> np.ndarray:
        """
        Get camera projection matrix P = K [R | t].
        
        Args:
            camera_name: Name of the camera
            use_alternative_convention: If True, use R^T and -R^T*t (camera pose convention)
            
        Returns:
            3x4 projection matrix P
        """
        K = self.get_camera_intrinsic(camera_name)
        R, t = self.get_camera_extrinsic(camera_name)
        
        if use_alternative_convention:
            R_world_to_cam = R.T
            t_world_to_cam = -R.T @ t
            Rt = np.hstack([R_world_to_cam, t_world_to_cam.reshape(3, 1)])
        else:
            Rt = np.hstack([R, t.reshape(3, 1)])
        
        P = K @ Rt
        return P
    
    def get_camera_center(self, camera_name: str, use_alternative_convention: bool = False) -> np.ndarray:
        """
        Get camera center position in world coordinates.
        
        If R and t represent camera pose: camera_center = t (or -R @ t depending on convention)
        If R and t represent world-to-camera transform: camera_center = -R^T @ t
        
        Args:
            camera_name: Name of the camera
            use_alternative_convention: If True, assume R and t represent camera pose
            
        Returns:
            3D position of camera center in world coordinates
        """
        R, t = self.get_camera_extrinsic(camera_name)
        
        if use_alternative_convention:
            # R and t represent camera pose: camera center is at t in world coordinates
            return t.copy()
        else:
            # R and t represent world-to-camera transform: camera center = -R^T @ t
            return -R.T @ t
    
    def get_image_size(self, camera_name: str) -> Tuple[int, int]:
        """
        Get image size for a specific camera.
        
        Args:
            camera_name: Name of the camera
            
        Returns:
            Tuple of (width, height)
        """
        params = self.get_camera_parameters(camera_name)
        size = params['image_size']
        return size['width'], size['height']
    
    def project_3d_to_2d(self, points_3d: np.ndarray, camera_name: str, 
                        use_alternative_convention: bool = False) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates for a specific camera.
        
        In SLEAP/OpenCV conventions, the rotation and translation may represent the
        camera pose (position/orientation in world space). For projection, we need
        to transform FROM world coordinates TO camera coordinates.
        
        Args:
            points_3d: Array of shape (N, 3) with 3D points in mm
            camera_name: Name of the camera
            use_alternative_convention: If True, use R^T and -R^T*t (camera pose convention)
            
        Returns:
            Array of shape (N, 2) with 2D image coordinates in pixels
        """
        K = self.get_camera_intrinsic(camera_name)
        R, t = self.get_camera_extrinsic(camera_name)
        
        if use_alternative_convention:
            # Alternative convention: R and t represent camera pose in world space
            # Transform FROM world TO camera: X_cam = R^T * (X_world - t)
            # Or equivalently: X_cam = R^T * X_world - R^T * t
            R_world_to_cam = R.T
            t_world_to_cam = -R.T @ t
            
            points_3d_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
            R_t = np.hstack([R_world_to_cam, t_world_to_cam.reshape(3, 1)])
            points_cam = (R_t @ points_3d_homogeneous.T).T  # (N, 3)
        else:
            # Standard OpenCV convention: R and t transform FROM world TO camera
            # X_cam = R * X_world + t
            points_3d_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
            R_t = np.hstack([R, t.reshape(3, 1)])
            points_cam = (R_t @ points_3d_homogeneous.T).T  # (N, 3)
        
        # Project to image plane: x = K * X_cam
        points_2d_homogeneous = (K @ points_cam.T).T  # (N, 3)
        
        # Convert from homogeneous to 2D
        points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]
        
        return points_2d
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded data.
        
        Returns:
            Dictionary with summary information
        """
        return {
            'session_path': str(self.session_path),
            'n_frames': self.n_frames,
            'n_keypoints': self.n_keypoints,
            'n_cameras': len(self.cameras),
            'camera_names': self.camera_names,
            'keypoints_3d_shape': self.keypoints_3d.shape,
            'keypoints_3d_range': {
                'x': [float(self.keypoints_3d[:, :, 0].min()), 
                      float(self.keypoints_3d[:, :, 0].max())],
                'y': [float(self.keypoints_3d[:, :, 1].min()), 
                      float(self.keypoints_3d[:, :, 1].max())],
                'z': [float(self.keypoints_3d[:, :, 2].min()), 
                      float(self.keypoints_3d[:, :, 2].max())],
            },
            'has_nan': bool(np.isnan(self.keypoints_3d).any()),
            'has_inf': bool(np.isinf(self.keypoints_3d).any()),
        }
    
    def print_summary(self):
        """Print a formatted summary of loaded data."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("SLEAP 3D DATA LOADER SUMMARY")
        print("="*60)
        print(f"Session path: {summary['session_path']}")
        print(f"Frames: {summary['n_frames']}")
        print(f"Keypoints: {summary['n_keypoints']}")
        print(f"Cameras: {summary['n_cameras']}")
        print(f"Camera names: {summary['camera_names']}")
        print(f"\n3D coordinate ranges (mm):")
        print(f"  X: [{summary['keypoints_3d_range']['x'][0]:.2f}, {summary['keypoints_3d_range']['x'][1]:.2f}]")
        print(f"  Y: [{summary['keypoints_3d_range']['y'][0]:.2f}, {summary['keypoints_3d_range']['y'][1]:.2f}]")
        print(f"  Z: [{summary['keypoints_3d_range']['z'][0]:.2f}, {summary['keypoints_3d_range']['z'][1]:.2f}]")
        print(f"\nData quality:")
        print(f"  Has NaN: {summary['has_nan']}")
        print(f"  Has Inf: {summary['has_inf']}")
        print("="*60 + "\n")
    
    def _detect_data_structure(self) -> str:
        """Detect whether this is a camera_dirs or session_dirs structure."""
        # Check for camera_dirs structure (directories with .slp files)
        camera_dirs_found = 0
        for item in self.session_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                slp_files = list(item.glob("*.slp"))
                if slp_files:
                    camera_dirs_found += 1
        
        # Check for session_dirs structure (subdirectories with .h5 files)
        session_dirs_found = 0
        for item in self.session_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                h5_files = list(item.glob("*.h5"))
                if h5_files and not list(item.glob("*.slp")):
                    session_dirs_found += 1
        
        if camera_dirs_found > 0 and session_dirs_found == 0:
            return 'camera_dirs'
        elif session_dirs_found > 0:
            return 'session_dirs'
        else:
            return 'unknown'
    
    def load_2d_predictions(self, camera_name: str, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load 2D keypoint predictions from SLEAP prediction files.
        
        Uses SLEAPDataLoader to ensure consistent data access with preprocessing scripts.
        This matches exactly how preprocess_sleap_dataset.py and preprocess_sleap_multiview_dataset.py
        load 2D predictions.
        
        Args:
            camera_name: Name of the camera
            frame_idx: Frame index
            
        Returns:
            Tuple of (keypoints_2d, visibility) where:
            - keypoints_2d: (n_keypoints, 2) array of 2D coordinates
            - visibility: (n_keypoints,) boolean array
        """
        try:
            from sleap_data_loader import SLEAPDataLoader
        except ImportError:
            sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
            from sleap_data_loader import SLEAPDataLoader
        
        # Use SLEAPDataLoader to load data (same as preprocessing scripts)
        loader = SLEAPDataLoader(
            project_path=str(self.session_path),
            confidence_threshold=0.5
        )
        
        # Use mapped camera name if available
        actual_cam_name = self.camera_name_to_dir.get(camera_name, camera_name)
        
        # Try to load camera data with mapped name first, then original name
        camera_data = None
        used_cam_name = None
        
        # Try mapped name first
        if actual_cam_name in loader.camera_views:
            try:
                camera_data = loader.load_camera_data(actual_cam_name)
                used_cam_name = actual_cam_name
            except Exception:
                pass
        
        # Try original name
        if camera_data is None and camera_name in loader.camera_views:
            try:
                camera_data = loader.load_camera_data(camera_name)
                used_cam_name = camera_name
            except Exception:
                pass
        
        # Try case-insensitive matching
        if camera_data is None:
            for cam_view in loader.camera_views:
                if cam_view.lower() == actual_cam_name.lower() or \
                   cam_view.lower() == camera_name.lower():
                    try:
                        camera_data = loader.load_camera_data(cam_view)
                        used_cam_name = cam_view
                        break
                    except Exception:
                        continue
        
        if camera_data is None:
            raise ValueError(f"Could not find camera data for {camera_name} (tried: {actual_cam_name}). "
                            f"Available cameras: {loader.camera_views}")
        
        # Extract 2D keypoints using the same method as preprocessing scripts
        keypoints_2d, visibility = loader.extract_2d_keypoints(camera_data, frame_idx)
        
        if len(keypoints_2d) == 0:
            # Debug: check what frames are available
            if loader.data_structure_type == 'camera_dirs':
                if 'instances' in camera_data:
                    instances = camera_data['instances']
                    if len(instances) > 0:
                        available_frames = sorted(np.unique(instances['frame_id']).tolist())
                        print(f"  Debug: Available frames for {used_cam_name}: {available_frames[:10]}... (showing first 10)")
            elif loader.data_structure_type == 'session_dirs':
                if 'tracks' in camera_data:
                    tracks = camera_data['tracks']
                    if len(tracks.shape) >= 4:
                        num_frames = tracks.shape[3]
                        print(f"  Debug: Tracks data has {num_frames} frames for {used_cam_name}")
        
        return keypoints_2d, visibility


def test_loader_basic(session_path: str):
    """Basic test of the 3D data loader."""
    loader = SLEAP3DDataLoader(session_path)
    return test_loader_basic_with_loader(loader)

def test_loader_basic_with_loader(loader: SLEAP3DDataLoader):
    """Basic test of the 3D data loader with pre-initialized loader."""
    print("\n" + "="*60)
    print("TEST 1: Basic Loader Functionality")
    print("="*60)
    
    try:
        loader.print_summary()
        
        # Test getting 3D keypoints
        print("Testing get_3d_keypoints...")
        keypoints = loader.get_3d_keypoints(frame_idx=0)
        print(f"  Frame 0 keypoints shape: {keypoints.shape}")
        print(f"  First keypoint: {keypoints[0]}")
        
        # Test getting camera parameters
        if loader.camera_names:
            camera_name = loader.camera_names[0]
            print(f"\nTesting camera parameters for {camera_name}...")
            params = loader.get_camera_parameters(camera_name)
            print(f"  Intrinsic K shape: {params['intrinsic']['K'].shape}")
            print(f"  Extrinsic R shape: {params['extrinsic']['R'].shape}")
            print(f"  Extrinsic t shape: {params['extrinsic']['t'].shape}")
            print(f"  Image size: {params['image_size']}")
        
        print("\n✓ Basic test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_camera_parameters(session_path: str):
    """Test camera parameter access."""
    loader = SLEAP3DDataLoader(session_path)
    return test_camera_parameters_with_loader(loader)

def test_camera_parameters_with_loader(loader: SLEAP3DDataLoader):
    """Test camera parameter access with pre-initialized loader."""
    print("\n" + "="*60)
    print("TEST 2: Camera Parameters")
    print("="*60)
    
    try:
        for camera_name in loader.camera_names:
            print(f"\nCamera: {camera_name}")
            
            # Test intrinsic
            K = loader.get_camera_intrinsic(camera_name)
            print(f"  K matrix:\n{K}")
            
            # Test extrinsic
            R, t = loader.get_camera_extrinsic(camera_name)
            print(f"  R matrix:\n{R}")
            print(f"  t vector: {t}")
            
            # Test projection matrix
            P = loader.get_camera_projection_matrix(camera_name)
            print(f"  P matrix shape: {P.shape}")
            
            # Test image size
            width, height = loader.get_image_size(camera_name)
            print(f"  Image size: {width}x{height}")
            
            # Validate rotation matrix
            det_R = np.linalg.det(R)
            R_orthogonal = np.allclose(R @ R.T, np.eye(3))
            print(f"  R determinant: {det_R:.6f} (should be ~1.0)")
            print(f"  R is orthogonal: {R_orthogonal}")
        
        print("\n✓ Camera parameters test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Camera parameters test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_reprojection_comparison(loader: SLEAP3DDataLoader, camera_name: str, 
                                     frame_idx: int, output_path: str):
    """
    Visualize comparison between reprojected 2D coordinates and original predictions.
    
    Args:
        loader: SLEAP3DDataLoader instance
        camera_name: Name of the camera
        frame_idx: Frame index
        output_path: Path to save the plot
    """
    # Load original 2D predictions
    try:
        original_2d, visibility = loader.load_2d_predictions(camera_name, frame_idx)
    except Exception as e:
        print(f"Warning: Could not load 2D predictions for {camera_name}: {e}")
        return
    
    if len(original_2d) == 0:
        print(f"Warning: No 2D predictions found for {camera_name} frame {frame_idx}")
        return
    
    # Get 3D keypoints
    keypoints_3d = loader.get_3d_keypoints(frame_idx)
    
    # Project 3D to 2D - try both conventions and use the better one
    reprojected_2d_standard = loader.project_3d_to_2d(keypoints_3d, camera_name, use_alternative_convention=False)
    reprojected_2d_alternative = loader.project_3d_to_2d(keypoints_3d, camera_name, use_alternative_convention=True)
    
    # Determine which convention is better
    if len(original_2d) == len(reprojected_2d_standard):
        visible_mask = visibility > 0
        if visible_mask.sum() > 0:
            errors_standard = np.linalg.norm(original_2d[visible_mask] - reprojected_2d_standard[visible_mask], axis=1).mean()
            errors_alternative = np.linalg.norm(original_2d[visible_mask] - reprojected_2d_alternative[visible_mask], axis=1).mean()
            reprojected_2d = reprojected_2d_alternative if errors_alternative < errors_standard else reprojected_2d_standard
        else:
            reprojected_2d = reprojected_2d_standard
    else:
        reprojected_2d = reprojected_2d_standard
    
    # Get image size
    width, height = loader.get_image_size(camera_name)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot image bounds
    ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0], 'k--', alpha=0.3, linewidth=1)
    
    # Plot original predictions (visible keypoints)
    visible_mask = visibility > 0
    if np.any(visible_mask):
        ax.scatter(original_2d[visible_mask, 0], original_2d[visible_mask, 1],
                  c='blue', s=100, alpha=0.7, marker='o', label='Original (visible)', edgecolors='white', linewidth=2)
    
    # Plot original predictions (invisible keypoints)
    invisible_mask = ~visible_mask
    if np.any(invisible_mask):
        ax.scatter(original_2d[invisible_mask, 0], original_2d[invisible_mask, 1],
                  c='lightblue', s=50, alpha=0.5, marker='o', label='Original (invisible)', edgecolors='gray', linewidth=1)
    
    # Plot reprojected coordinates
    ax.scatter(reprojected_2d[:, 0], reprojected_2d[:, 1],
              c='red', s=100, alpha=0.7, marker='x', label='Reprojected', linewidth=3)
    
    # Draw lines connecting original and reprojected (for visible keypoints)
    if len(original_2d) == len(reprojected_2d):
        for i in range(len(original_2d)):
            if visible_mask[i]:
                ax.plot([original_2d[i, 0], reprojected_2d[i, 0]],
                       [original_2d[i, 1], reprojected_2d[i, 1]],
                       'g-', alpha=0.3, linewidth=1)
    else:
        # Add warning text if keypoint counts don't match
        warning_text = f'Warning: Keypoint count mismatch\n(Original: {len(original_2d)}, Reprojected: {len(reprojected_2d)})'
        ax.text(0.5, 0.02, warning_text, transform=ax.transAxes, fontsize=10,
               horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Calculate and display statistics
    if len(original_2d) == len(reprojected_2d):
        # Calculate errors for visible keypoints
        errors = np.linalg.norm(original_2d[visible_mask] - reprojected_2d[visible_mask], axis=1)
        mean_error = errors.mean() if len(errors) > 0 else 0.0
        max_error = errors.max() if len(errors) > 0 else 0.0
        
        # Add text box with statistics
        stats_text = f'Mean error: {mean_error:.2f} px\nMax error: {max_error:.2f} px\nVisible keypoints: {visible_mask.sum()}/{len(visibility)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        # Show basic info even if counts don't match
        stats_text = f'Visible keypoints: {visible_mask.sum()}/{len(visibility)}\n(Keypoint count mismatch)'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title(f'Reprojection Comparison: {camera_name} - Frame {frame_idx}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()  # Match image coordinate system
    
    # Set limits with padding
    all_x = np.concatenate([original_2d[:, 0], reprojected_2d[:, 0]])
    all_y = np.concatenate([original_2d[:, 1], reprojected_2d[:, 1]])
    
    if len(all_x) > 0 and len(all_y) > 0:
        x_padding = (all_x.max() - all_x.min()) * 0.1 if all_x.max() > all_x.min() else width * 0.1
        y_padding = (all_y.max() - all_y.min()) * 0.1 if all_y.max() > all_y.min() else height * 0.1
        ax.set_xlim(max(0, all_x.min() - x_padding), min(width, all_x.max() + x_padding))
        ax.set_ylim(min(height, all_y.max() + y_padding), max(0, all_y.min() - y_padding))
    else:
        # Fallback to image bounds
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization to: {output_path}")


def test_3d_to_2d_projection(session_path: str, plot_examples: bool = False):
    """Test 3D to 2D projection by comparing to ground truth predictions."""
    loader = SLEAP3DDataLoader(session_path)
    return test_3d_to_2d_projection_with_loader(loader, plot_examples=plot_examples)

def test_3d_to_2d_projection_with_loader(loader: SLEAP3DDataLoader, plot_examples: bool = False):
    """
    Test 3D to 2D projection by comparing to ground truth predictions.
    
    Args:
        loader: Pre-initialized SLEAP3DDataLoader
        plot_examples: If True, save visualization plots for first frame of each camera
    """
    print("\n" + "="*60)
    print("TEST 3: 3D to 2D Projection (vs Ground Truth)")
    print("="*60)
    
    try:
        # Get 3D keypoints from first frame
        keypoints_3d = loader.get_3d_keypoints(frame_idx=0)
        print(f"3D keypoints shape: {keypoints_3d.shape}")
        
        # Create output directory for plots if needed
        if plot_examples:
            plot_dir = loader.session_path / "reprojection_comparison_plots"
            plot_dir.mkdir(exist_ok=True)
            print(f"Plot directory: {plot_dir}")
        
        # Project to each camera and compare with original predictions
        # Try both standard and alternative conventions to find which works
        all_errors_standard = []
        all_errors_alternative = []
        
        for camera_name in loader.camera_names:
            print(f"\nCamera: {camera_name}")
            
            # Show mapping if available
            if hasattr(loader, 'camera_name_to_dir') and camera_name in loader.camera_name_to_dir:
                actual_name = loader.camera_name_to_dir[camera_name]
                if actual_name != camera_name:
                    print(f"  Mapped to directory/file: {actual_name}")
            
            width, height = loader.get_image_size(camera_name)
            print(f"  Image size: {width}x{height}")
            
            # Load original 2D predictions
            try:
                original_2d, visibility = loader.load_2d_predictions(camera_name, frame_idx=0)
                
                if len(original_2d) == 0:
                    print(f"  Warning: No 2D predictions found for frame 0")
                    continue
                
                print(f"  Original 2D points shape: {original_2d.shape}")
                print(f"  Visible keypoints: {visibility.sum()}/{len(visibility)}")
                
                # Try standard convention
                reprojected_2d_standard = loader.project_3d_to_2d(keypoints_3d, camera_name, use_alternative_convention=False)
                
                # Try alternative convention
                reprojected_2d_alternative = loader.project_3d_to_2d(keypoints_3d, camera_name, use_alternative_convention=True)
                
                # Compare both with original
                if len(original_2d) == len(reprojected_2d_standard):
                    visible_mask = visibility > 0
                    if visible_mask.sum() > 0:
                        # Standard convention errors
                        errors_standard = np.linalg.norm(original_2d[visible_mask] - reprojected_2d_standard[visible_mask], axis=1)
                        mean_error_standard = errors_standard.mean()
                        
                        # Alternative convention errors
                        errors_alternative = np.linalg.norm(original_2d[visible_mask] - reprojected_2d_alternative[visible_mask], axis=1)
                        mean_error_alternative = errors_alternative.mean()
                        
                        # Use the convention with lower error
                        if mean_error_alternative < mean_error_standard:
                            print(f"  Using ALTERNATIVE convention (R^T, -R^T*t)")
                            reprojected_2d = reprojected_2d_alternative
                            errors = errors_alternative
                            all_errors_alternative.extend(errors.tolist())
                        else:
                            print(f"  Using STANDARD convention (R, t)")
                            reprojected_2d = reprojected_2d_standard
                            errors = errors_standard
                            all_errors_standard.extend(errors.tolist())
                        
                        mean_error = errors.mean()
                        median_error = np.median(errors)
                        max_error = errors.max()
                        std_error = errors.std()
                        
                        print(f"  Reprojection errors (visible keypoints only):")
                        print(f"    Mean: {mean_error:.2f} px")
                        print(f"    Median: {median_error:.2f} px")
                        print(f"    Max: {max_error:.2f} px")
                        print(f"    Std: {std_error:.2f} px")
                    else:
                        print(f"  Warning: No visible keypoints to compare")
                        reprojected_2d = reprojected_2d_standard
                else:
                    print(f"  Warning: Mismatch in number of keypoints "
                          f"(original: {len(original_2d)}, reprojected: {len(reprojected_2d_standard)})")
                    reprojected_2d = reprojected_2d_standard
                
                # Create visualization if requested
                if plot_examples:
                    plot_path = plot_dir / f"{camera_name}_frame_0_reprojection.png"
                    visualize_reprojection_comparison(loader, camera_name, frame_idx=0, output_path=str(plot_path))
                
            except Exception as e:
                print(f"  Warning: Could not load 2D predictions: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Print overall statistics
        all_errors = all_errors_standard + all_errors_alternative
        if all_errors:
            print(f"\nOverall reprojection statistics (all cameras):")
            print(f"  Mean error: {np.mean(all_errors):.2f} px")
            print(f"  Median error: {np.median(all_errors):.2f} px")
            print(f"  Max error: {np.max(all_errors):.2f} px")
            print(f"  Std error: {np.std(all_errors):.2f} px")
            if all_errors_standard and all_errors_alternative:
                print(f"\nConvention usage:")
                print(f"  Standard (R, t): {len(all_errors_standard)} keypoints")
                print(f"  Alternative (R^T, -R^T*t): {len(all_errors_alternative)} keypoints")
        
        print("\n✓ 3D to 2D projection test completed!")
        return True
        
    except Exception as e:
        print(f"\n✗ 3D to 2D projection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_consistency(session_path: str):
    """Test data consistency across frames."""
    loader = SLEAP3DDataLoader(session_path)
    return test_data_consistency_with_loader(loader)

def test_data_consistency_with_loader(loader: SLEAP3DDataLoader):
    """Test data consistency across frames with pre-initialized loader."""
    print("\n" + "="*60)
    print("TEST 4: Data Consistency")
    print("="*60)
    
    try:
        # Check frame-to-frame consistency
        all_keypoints = loader.get_all_3d_keypoints()
        
        print(f"Total frames: {loader.n_frames}")
        print(f"Keypoints per frame: {loader.n_keypoints}")
        
        # Check for NaN/Inf
        nan_count = np.isnan(all_keypoints).sum()
        inf_count = np.isinf(all_keypoints).sum()
        print(f"NaN values: {nan_count}")
        print(f"Inf values: {inf_count}")
        
        # Check coordinate ranges
        print(f"\nCoordinate statistics:")
        for axis, idx in [('X', 0), ('Y', 1), ('Z', 2)]:
            coords = all_keypoints[:, :, idx]
            print(f"  {axis}: min={coords.min():.2f}, max={coords.max():.2f}, "
                  f"mean={coords.mean():.2f}, std={coords.std():.2f}")
        
        # Check frame-to-frame motion
        if loader.n_frames > 1:
            velocities = np.diff(all_keypoints, axis=0)
            speeds = np.linalg.norm(velocities, axis=2)
            print(f"\nMotion statistics:")
            print(f"  Mean speed: {speeds.mean():.2f} mm/frame")
            print(f"  Max speed: {speeds.max():.2f} mm/frame")
            print(f"  Std speed: {speeds.std():.2f} mm/frame")
        
        print("\n✓ Data consistency test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Data consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SLEAP 3D Data Loader")
    parser.add_argument("session_path", help="Path to SLEAP session directory or project directory")
    parser.add_argument("--test", type=str, default="all",
                       choices=["all", "basic", "camera", "projection", "consistency"],
                       help="Which test to run (default: all)")
    parser.add_argument("--plot_examples", action="store_true",
                       help="Save visualization plots for reprojection comparison")
    parser.add_argument("--session_idx", type=int, default=None,
                       help="Session index if path is a project directory (default: process all sessions)")
    parser.add_argument("--video_subdir", type=str, default=None,
                       help="Video subdirectory name (e.g., 'PerShu_012') if points3d.h5 is in a subdirectory")
    parser.add_argument("--list_sessions", action="store_true",
                       help="List all available sessions and exit")
    
    args = parser.parse_args()
    
    session_path = args.session_path
    
    if not Path(session_path).exists():
        print(f"Error: Session path does not exist: {session_path}")
        sys.exit(1)
    
    # Check if this is a project directory and handle session selection
    # Discover sessions using the same logic as the class
    session_path_obj = Path(session_path)
    sessions = []
    
    if session_path_obj.exists():
        for item in session_path_obj.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if this looks like a SLEAP session
                if (item / "calibration.toml").exists():
                    sessions.append(str(item))
                else:
                    # Check for video subdirectories with .h5 files or camera directories with .slp files
                    has_h5 = any(subdir.is_dir() and list(subdir.glob("*.h5")) 
                                for subdir in item.iterdir() if subdir.is_dir())
                    has_slp = any(cam_dir.is_dir() and list(cam_dir.glob("*.slp"))
                                for cam_dir in item.iterdir() if cam_dir.is_dir())
                    if has_h5 or has_slp:
                        sessions.append(str(item))
    
    sessions.sort()
    
    if len(sessions) > 1:
        # This is a project directory with multiple sessions
        if args.list_sessions:
            print(f"\nFound {len(sessions)} sessions:")
            for i, session in enumerate(sessions):
                print(f"  [{i}] {Path(session).name}")
            sys.exit(0)
        
        # If session_idx is specified, validate and use it
        if args.session_idx is not None:
            if args.session_idx < 0 or args.session_idx >= len(sessions):
                print(f"Error: Session index {args.session_idx} out of range [0, {len(sessions)-1}]")
                print(f"\nAvailable sessions:")
                for i, session in enumerate(sessions):
                    print(f"  [{i}] {Path(session).name}")
                sys.exit(1)
    elif len(sessions) == 1:
        # Single session found, use it
        pass  # sessions already contains the single session
    else:
        # No sessions found, treat as single session directory
        sessions = [session_path]
    
    # Determine which sessions to process
    if args.session_idx is not None and len(sessions) > 1:
        # User specified a session index, process only that one
        sessions_to_process = [sessions[args.session_idx]]
        print(f"Processing session {args.session_idx}: {Path(sessions[args.session_idx]).name}")
    else:
        # Process all sessions (or single session if only one found)
        sessions_to_process = sessions
        if len(sessions_to_process) > 1:
            print(f"Processing all {len(sessions_to_process)} sessions")
        else:
            print(f"Processing session: {Path(sessions_to_process[0]).name}")
    
    # Process each session
    all_results = {}
    
    for session_idx, current_session_path in enumerate(sessions_to_process):
        print("\n" + "="*60)
        print("SLEAP 3D DATA LOADER TESTS")
        print("="*60)
        print(f"Session [{session_idx+1}/{len(sessions_to_process)}]: {Path(current_session_path).name}")
        print(f"Session path: {current_session_path}")
        if args.video_subdir:
            print(f"Video subdirectory: {args.video_subdir}")
        print(f"Test: {args.test}")
        print(f"Plot examples: {args.plot_examples}")
        print("="*60)
        
        # Create loader for this session
        try:
            loader = SLEAP3DDataLoader(current_session_path, video_subdir=args.video_subdir, session_idx=0)
        except Exception as e:
            print(f"Error: Failed to initialize loader for {current_session_path}: {e}")
            import traceback
            traceback.print_exc()
            all_results[Path(current_session_path).name] = {'error': str(e)}
            continue
        
        # Run tests for this session
        session_results = {}
        
        if args.test in ["all", "basic"]:
            session_results['basic'] = test_loader_basic_with_loader(loader)
        
        if args.test in ["all", "camera"]:
            session_results['camera'] = test_camera_parameters_with_loader(loader)
        
        if args.test in ["all", "projection"]:
            session_results['projection'] = test_3d_to_2d_projection_with_loader(loader, plot_examples=args.plot_examples)
        
        if args.test in ["all", "consistency"]:
            session_results['consistency'] = test_data_consistency_with_loader(loader)
        
        all_results[Path(current_session_path).name] = session_results
    
    # Aggregate results
    results = {}
    for test_name in ['basic', 'camera', 'projection', 'consistency']:
        test_results = [r.get(test_name, None) for r in all_results.values() if isinstance(r, dict) and 'error' not in r]
        if test_results:
            # Test passes if all sessions pass
            results[test_name] = all(r for r in test_results if r is not None)
        else:
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if len(all_results) > 1:
        # Print per-session results
        print(f"\nResults for {len(all_results)} sessions:")
        for session_name, session_result in all_results.items():
            if 'error' in session_result:
                print(f"\n{session_name}: ERROR - {session_result['error']}")
            else:
                print(f"\n{session_name}:")
                for test_name, passed in session_result.items():
                    status = "✓ PASSED" if passed else "✗ FAILED"
                    print(f"  {test_name:20s}: {status}")
    
    # Print overall summary
    print(f"\nOverall Results:")
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    print("="*60)
    
    # Exit with error if any test failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
