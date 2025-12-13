"""
SLEAP Data Loader for SMILify Pipeline

This module provides functionality to load and process SLEAP pose estimation data
for integration with the SMILify training pipeline.
"""

import os
import h5py
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import toml
import pandas as pd
import sys

# Add the parent directories to the path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


class SLEAPDataLoader:
    """
    A modular data loader for SLEAP pose estimation datasets.
    
    This class handles loading of:
    - Multi-camera 2D keypoint predictions (.slp files)
    - 3D reconstructed keypoints (points3d.h5)
    - Camera calibration data (calibration.toml)
    - Video frames for visualization
    """
    
    def __init__(self, project_path: str, lookup_table_path: Optional[str] = None, 
                 shape_betas_path: Optional[str] = None):
        """
        Initialize the SLEAP data loader.
        
        Args:
            project_path (str): Path to the SLEAP project directory
            lookup_table_path (str, optional): Path to CSV lookup table for joint name mapping
            shape_betas_path (str, optional): Path to CSV lookup table for ground truth shape betas
        """
        self.project_path = Path(project_path)
        self.lookup_table_path = lookup_table_path
        self.shape_betas_path = shape_betas_path
        self.calibration_data = None
        self.camera_views = []
        self.keypoint_names = []
        self.data_structure_type = None  # 'camera_dirs' or 'session_dirs'
        self.session_name = None  # For session-based structure
        
        # SMAL model joint information
        self.smal_joint_names = []
        self.smal_n_joints = 0
        self.joint_mapping = {}  # Maps SMAL joint index to SLEAP keypoint index
        self.lookup_table = None
        
        # Ground truth shape betas
        self.shape_betas_table = None
        self.ground_truth_betas = None
        
        self._load_project_structure()
        self._load_smal_model_info()
        self._load_lookup_table()
        self._load_shape_betas()
        self._create_joint_mapping()
        
    def _load_project_structure(self):
        """Load the project structure and identify available camera views."""
        print(f"Loading SLEAP project from: {self.project_path}")
        
        # Load calibration data
        calibration_file = self.project_path / "calibration.toml"
        if calibration_file.exists():
            self.calibration_data = toml.load(calibration_file)
            print(f"Loaded calibration data for {len(self.calibration_data)} cameras")
        else:
            print("Warning: No calibration.toml found")
            
        # Detect data structure type
        self._detect_data_structure()
        
        # Identify camera views based on structure type
        if self.data_structure_type == 'camera_dirs':
            self._load_camera_dirs_structure()
        elif self.data_structure_type == 'session_dirs':
            self._load_session_dirs_structure()
        else:
            print("Warning: Could not determine data structure type")
            self.camera_views = []
            
        print(f"Found {len(self.camera_views)} camera views: {self.camera_views}")
        
        # Load 3D points if available
        points3d_file = self.project_path / "points3d.h5"
        if points3d_file.exists():
            print("Found 3D points file")
        else:
            print("Warning: No points3d.h5 found")
            
    def _detect_data_structure(self):
        """Detect whether this is a camera_dirs or session_dirs structure."""
        # Check for camera_dirs structure (original)
        # Look for directories that contain .slp files
        camera_dirs_found = 0
        for item in self.project_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                slp_files = list(item.glob("*.slp"))
                if slp_files:
                    camera_dirs_found += 1
                    
        # Check for session_dirs structure (new)
        # Look for directories that contain .h5 files and videos in main dir
        session_dirs_found = 0
        video_files_in_main = list(self.project_path.glob("*.mp4"))
        
        for item in self.project_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                h5_files = list(item.glob("*.h5"))
                if h5_files and not list(item.glob("*.slp")):
                    session_dirs_found += 1
                    
        # Determine structure type
        if camera_dirs_found > 0 and session_dirs_found == 0:
            self.data_structure_type = 'camera_dirs'
            print(f"Detected camera_dirs structure with {camera_dirs_found} camera directories")
        elif session_dirs_found > 0 and video_files_in_main:
            self.data_structure_type = 'session_dirs'
            print(f"Detected session_dirs structure with {session_dirs_found} session directories")
            # Find the session name (assuming there's one main session)
            for item in self.project_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    h5_files = list(item.glob("*.h5"))
                    if h5_files and not list(item.glob("*.slp")):
                        self.session_name = item.name
                        break
            print(f"Using session: {self.session_name}")
        else:
            print("Warning: Could not determine data structure type")
            self.data_structure_type = None
            
    def _load_camera_dirs_structure(self):
        """Load camera views for camera_dirs structure."""
        self.camera_views = []
        for item in self.project_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if this directory contains SLEAP prediction files
                slp_files = list(item.glob("*.slp"))
                if slp_files:
                    self.camera_views.append(item.name)
                    
    def _load_session_dirs_structure(self):
        """Load camera views for session_dirs structure."""
        self.camera_views = []
        if not self.session_name:
            return
            
        session_dir = self.project_path / self.session_name
        if not session_dir.exists():
            return
            
        # Look for camera-specific .h5 files in the session directory
        h5_files = list(session_dir.glob("*_cam*.h5"))
        for h5_file in h5_files:
            # Extract camera name from filename (e.g., "021_4th_camA.h5" -> "camA")
            filename = h5_file.stem
            if '_cam' in filename:
                camera_name = filename.split('_cam')[-1]
                self.camera_views.append(camera_name)
                
        # Also check for video files in main directory to get camera names
        video_files = list(self.project_path.glob("*_cam*.mp4"))
        for video_file in video_files:
            filename = video_file.stem
            if '_cam' in filename:
                camera_name = filename.split('_cam')[-1]
                if camera_name not in self.camera_views:
                    self.camera_views.append(camera_name)
                    
    def load_camera_data(self, camera_name: str) -> Dict[str, Any]:
        """
        Load data for a specific camera view.
        
        Args:
            camera_name (str): Name of the camera view (e.g., 'back', 'side', 'top', 'camA')
            
        Returns:
            Dict containing camera data including keypoints, frames, and metadata
        """
        if self.data_structure_type == 'camera_dirs':
            return self._load_camera_dirs_data(camera_name)
        elif self.data_structure_type == 'session_dirs':
            return self._load_session_dirs_data(camera_name)
        else:
            raise ValueError("Unknown data structure type")
            
    def _load_camera_dirs_data(self, camera_name: str) -> Dict[str, Any]:
        """Load data for camera_dirs structure."""
        camera_dir = self.project_path / camera_name
        
        if not camera_dir.exists():
            raise ValueError(f"Camera directory not found: {camera_dir}")
            
        # Find the main prediction file (prefer proofread version)
        prediction_files = list(camera_dir.glob("*.predictions.proofread.slp"))
        if not prediction_files:
            prediction_files = list(camera_dir.glob("*.predictions.slp"))
            
        if not prediction_files:
            raise ValueError(f"No prediction files found in {camera_dir}")
            
        prediction_file = prediction_files[0]
        print(f"Loading predictions from: {prediction_file}")
        
        # Load the SLEAP prediction file
        with h5py.File(prediction_file, 'r') as f:
            data = {}
            
            # Load key data structures
            if 'instances' in f:
                data['instances'] = f['instances'][:]
                
            if 'frames' in f:
                data['frames'] = f['frames'][:]
                
            if 'points' in f:
                data['points'] = f['points'][:]
                
            if 'pred_points' in f:
                data['pred_points'] = f['pred_points'][:]
                
            # Load metadata (optional, not required for keypoint extraction)
            if 'videos_json' in f:
                try:
                    videos_json = f['videos_json'][()]
                    data['videos'] = json.loads(self._decode_json_data(videos_json))
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Warning: Could not parse videos_json metadata: {e}")
                
            if 'tracks_json' in f:
                try:
                    tracks_json = f['tracks_json'][()]
                    data['tracks_metadata'] = json.loads(self._decode_json_data(tracks_json))
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Warning: Could not parse tracks_json metadata: {e}")
                
        # Load calibration data for this camera
        calibration_entry = self._get_calibration_entry(camera_name)
        if calibration_entry:
            data['calibration'] = calibration_entry
            
        return data
        
    def _load_session_dirs_data(self, camera_name: str) -> Dict[str, Any]:
        """Load data for session_dirs structure."""
        if not self.session_name:
            raise ValueError("No session name available")
            
        session_dir = self.project_path / self.session_name
        
        # Find the camera-specific .h5 file
        camera_h5_files = list(session_dir.glob(f"*_cam{camera_name}.h5"))
        if not camera_h5_files:
            raise ValueError(f"No .h5 file found for camera {camera_name} in {session_dir}")
            
        camera_h5_file = camera_h5_files[0]
        print(f"Loading camera data from: {camera_h5_file}")
        
        # Load the camera data file
        with h5py.File(camera_h5_file, 'r') as f:
            data = {}
            
            # Load key data structures - session_dirs uses different format
            if 'tracks' in f:
                data['tracks'] = f['tracks'][:]
                
            if 'point_scores' in f:
                data['point_scores'] = f['point_scores'][:]
                
            if 'instance_scores' in f:
                data['instance_scores'] = f['instance_scores'][:]
                
            if 'track_occupancy' in f:
                data['track_occupancy'] = f['track_occupancy'][:]
                
            if 'node_names' in f:
                data['node_names'] = f['node_names'][:]
                
            if 'edge_inds' in f:
                data['edge_inds'] = f['edge_inds'][:]
                
            if 'edge_names' in f:
                data['edge_names'] = f['edge_names'][:]
                
            # Load metadata
            if 'video_path' in f:
                data['video_path'] = f['video_path'][()]
                
            if 'labels_path' in f:
                data['labels_path'] = f['labels_path'][()]
                
        # Load calibration data for this camera
        calibration_entry = self._get_calibration_entry(camera_name)
        if calibration_entry:
            data['calibration'] = calibration_entry
            
        return data
        
    def _get_calibration_entry(self, camera_name: str) -> Optional[Dict]:
        """Get calibration data for a camera."""
        if not self.calibration_data:
            return None
            
        # Find the calibration entry that matches this camera name
        for key, value in self.calibration_data.items():
            if key != 'metadata' and value.get('name') == camera_name:
                return value
        return None
    
    def _decode_json_data(self, data) -> str:
        """
        Decode JSON data from various HDF5 storage formats.
        
        Args:
            data: Data loaded from HDF5, could be bytes, str, numpy array, etc.
            
        Returns:
            str: Decoded JSON string ready for json.loads()
        """
        # Handle bytes directly
        if isinstance(data, bytes):
            return data.decode('utf-8')
        
        # Handle string directly
        if isinstance(data, str):
            return data
        
        # Handle numpy array or HDF5 dataset
        if hasattr(data, 'shape'):
            # Single element array - use .item()
            if data.shape == () or (hasattr(data, 'size') and data.size == 1):
                item = data.item() if hasattr(data, 'item') else data[()]
                if isinstance(item, bytes):
                    return item.decode('utf-8')
                return str(item)
            
            # Array of bytes/strings - join them
            if len(data.shape) == 1:
                try:
                    # Try to decode and join if it's an array of bytes
                    decoded_parts = []
                    for item in data:
                        if isinstance(item, bytes):
                            decoded_parts.append(item.decode('utf-8'))
                        else:
                            decoded_parts.append(str(item))
                    return ''.join(decoded_parts)
                except Exception:
                    pass
        
        # Fallback: convert to string
        return str(data)
        
    def extract_2d_keypoints(self, camera_data: Dict[str, Any], frame_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 2D keypoints for a specific frame.
        
        Args:
            camera_data (Dict): Camera data loaded by load_camera_data()
            frame_idx (int): Frame index to extract
            
        Returns:
            Tuple of (keypoints_2d, visibility) where:
            - keypoints_2d: (N, 2) array of 2D coordinates
            - visibility: (N,) boolean array indicating keypoint visibility
        """
        if self.data_structure_type == 'camera_dirs':
            return self._extract_2d_keypoints_camera_dirs(camera_data, frame_idx)
        elif self.data_structure_type == 'session_dirs':
            return self._extract_2d_keypoints_session_dirs(camera_data, frame_idx)
        else:
            print("Unknown data structure type")
            return np.array([]), np.array([])
            
    def _extract_2d_keypoints_camera_dirs(self, camera_data: Dict[str, Any], frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 2D keypoints for camera_dirs structure."""
        instances = camera_data['instances']
        frames = camera_data['frames']
        
        # Use pred_points if available, otherwise fall back to points
        if 'pred_points' in camera_data and len(camera_data['pred_points']) > 0:
            points = camera_data['pred_points']
        elif 'points' in camera_data and len(camera_data['points']) > 0:
            points = camera_data['points']
        else:
            print(f"Warning: No point data available")
            return np.array([]), np.array([])
        
        # Find instances for the specified frame
        frame_instances = instances[instances['frame_id'] == frame_idx]
        
        if len(frame_instances) == 0:
            print(f"Warning: No instances found for frame {frame_idx}")
            return np.array([]), np.array([])
            
        # Get the first instance (assuming single animal tracking)
        instance = frame_instances[0]
        
        # Extract keypoints for this instance
        point_start = instance['point_id_start']
        point_end = instance['point_id_end']
        
        if point_end > len(points):
            print(f"Warning: Point indices out of range: {point_start}-{point_end} (max: {len(points)})")
            return np.array([]), np.array([])
            
        instance_points = points[point_start:point_end]
        
        # Extract 2D coordinates and visibility
        keypoints_2d = np.column_stack([instance_points['x'], instance_points['y']])
        visibility = instance_points['visible']
        
        return keypoints_2d, visibility
        
    def _extract_2d_keypoints_session_dirs(self, camera_data: Dict[str, Any], frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 2D keypoints for session_dirs structure."""
        if 'tracks' not in camera_data:
            print("Warning: No tracks data available")
            return np.array([]), np.array([])
            
        tracks = camera_data['tracks']  # Shape: (n_tracks, n_instances, n_keypoints, n_frames)
        # Where instance 0 = x coordinates, instance 1 = y coordinates
        
        if frame_idx >= tracks.shape[3]:
            print(f"Warning: Frame index {frame_idx} out of range (max: {tracks.shape[3]-1})")
            return np.array([]), np.array([])
            
        # Get x and y coordinates from the first track
        x_coords = tracks[0, 0, :, frame_idx]  # x coordinates for all keypoints
        y_coords = tracks[0, 1, :, frame_idx]  # y coordinates for all keypoints
        
        # Combine x and y coordinates
        keypoints_2d = np.column_stack([x_coords, y_coords])  # Shape: (n_keypoints, 2)
        
        # For visibility, we can use point scores if available
        if 'point_scores' in camera_data:
            point_scores = camera_data['point_scores']  # Shape: (n_tracks, n_keypoints, n_frames)
            visibility = point_scores[0, :, frame_idx] > 0.5  # Threshold for visibility
        else:
            # Fallback: assume keypoints are visible if they have non-NaN coordinates
            visibility = ~np.isnan(keypoints_2d).any(axis=1)
            
        return keypoints_2d, visibility
        
    def load_3d_keypoints(self, frame_idx: int = 0) -> np.ndarray:
        """
        Load 3D keypoints for a specific frame.
        
        Args:
            frame_idx (int): Frame index to extract
            
        Returns:
            np.ndarray: (N, 3) array of 3D coordinates
        """
        points3d_file = self.project_path / "points3d.h5"
        
        if not points3d_file.exists():
            print("Warning: No 3D points file found")
            return np.array([])
            
        with h5py.File(points3d_file, 'r') as f:
            tracks = f['tracks'][:]
            
        if frame_idx >= tracks.shape[0]:
            print(f"Warning: Frame index {frame_idx} out of range (max: {tracks.shape[0]-1})")
            return np.array([])
            
        # Extract 3D keypoints for the first track
        keypoints_3d = tracks[frame_idx, 0, :, :]  # (N, 3)
        
        return keypoints_3d
        
    def get_camera_image_size(self, camera_name: str) -> Tuple[int, int]:
        """
        Get the image size for a specific camera.
        
        Args:
            camera_name (str): Name of the camera
            
        Returns:
            Tuple of (width, height)
        """
        # Find the calibration entry that matches this camera name
        if self.calibration_data:
            for key, value in self.calibration_data.items():
                if key != 'metadata' and value.get('name') == camera_name:
                    size = value['size']
                    return size[0], size[1]  # width, height
                    
        print(f"Warning: No calibration data for camera {camera_name}")
        return 1280, 1024  # Default size
            
    def load_video_frame(self, camera_name: str, frame_idx: int = 0) -> Optional[np.ndarray]:
        """
        Load a specific frame from the camera's video file.
        
        Args:
            camera_name (str): Name of the camera view
            frame_idx (int): Frame index to load
            
        Returns:
            np.ndarray or None: Video frame as RGB image, or None if failed
        """
        if self.data_structure_type == 'camera_dirs':
            return self._load_video_frame_camera_dirs(camera_name, frame_idx)
        elif self.data_structure_type == 'session_dirs':
            return self._load_video_frame_session_dirs(camera_name, frame_idx)
        else:
            print("Unknown data structure type")
            return None
            
    def _load_video_frame_camera_dirs(self, camera_name: str, frame_idx: int) -> Optional[np.ndarray]:
        """Load video frame for camera_dirs structure."""
        camera_dir = self.project_path / camera_name
        
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
                            print(f"Using video file from {h5_file} for camera {camera_name}: {candidate_path}")
                            return self._read_video_frame(candidate_path, frame_idx)

                        print(f"Warning: Expected video file not found next to {h5_file}: {candidate_path}")
        except Exception as e:
            print(f"Warning: Failed to read video_path from h5 in {camera_dir}: {e}")
        
        # Fallback: find video file directly in camera directory
        video_files = list(camera_dir.glob("*.mp4"))
        if not video_files:
            print(f"No video file found in {camera_dir}")
            return None
            
        video_file = video_files[0]
        print(f"Using fallback video file for camera {camera_name}: {video_file}")
        return self._read_video_frame(video_file, frame_idx)
        
    def _load_video_frame_session_dirs(self, camera_name: str, frame_idx: int) -> Optional[np.ndarray]:
        """Load video frame for session_dirs structure."""
        # Prefer the video path referenced inside the per-camera .h5 file
        if self.session_name:
            session_dir = self.project_path / self.session_name
            camera_h5_files = list(session_dir.glob(f"*_cam{camera_name}.h5"))
            if camera_h5_files:
                camera_h5_file = camera_h5_files[0]
                try:
                    with h5py.File(camera_h5_file, 'r') as f:
                        if 'video_path' in f:
                            raw_path = f['video_path'][()]
                            # Decode path from HDF5 and use only the filename, as the
                            # original absolute directory is not available on this system.
                            if isinstance(raw_path, bytes):
                                video_path_str = raw_path.decode('utf-8')
                            else:
                                video_path_str = str(raw_path)
                            video_filename = Path(video_path_str).name
                            candidate_path = camera_h5_file.parent / video_filename

                            if candidate_path.exists():
                                print(f"Using video file from {camera_h5_file} for camera {camera_name}: {candidate_path}")
                                return self._read_video_frame(candidate_path, frame_idx)

                            print(f"Warning: Expected video file not found next to {camera_h5_file}: {candidate_path}")
                except Exception as e:
                    print(f"Warning: Failed to read video_path from {camera_h5_file}: {e}")

        # Fallback: find video file in main directory by pattern
        video_files = list(self.project_path.glob(f"*_cam{camera_name}.mp4"))
        if not video_files:
            print(f"No video file found for camera {camera_name} in {self.project_path}")
            return None
            
        video_file = video_files[0]
        print(f"Using fallback video file for camera {camera_name}: {video_file}")
        return self._read_video_frame(video_file, frame_idx)
        
    def _read_video_frame(self, video_file: Path, frame_idx: int) -> Optional[np.ndarray]:
        """Read a frame from a video file."""
        # Open video file
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"Failed to open video file: {video_file}")
            return None
            
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Failed to read frame {frame_idx} from {video_file}")
            return None
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame_rgb
        
    def plot_keypoints_2d(self, camera_name: str, frame_idx: int = 0, save_path: Optional[str] = None):
        """
        Plot 2D keypoints for a specific camera and frame.
        
        Args:
            camera_name (str): Name of the camera view
            frame_idx (int): Frame index to plot
            save_path (str, optional): Path to save the plot
        """
        # Load camera data
        camera_data = self.load_camera_data(camera_name)
        
        # Extract 2D keypoints
        keypoints_2d, visibility = self.extract_2d_keypoints(camera_data, frame_idx)
        
        if len(keypoints_2d) == 0:
            print(f"No keypoints found for camera {camera_name}, frame {frame_idx}")
            return
            
        # Get image size for proper scaling
        width, height = self.get_camera_image_size(camera_name)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot visible keypoints
        visible_mask = visibility > 0
        if np.any(visible_mask):
            plt.scatter(keypoints_2d[visible_mask, 0], keypoints_2d[visible_mask, 1], 
                       c='red', s=50, alpha=0.7, label='Visible keypoints')
            
        # Plot invisible keypoints
        invisible_mask = ~visible_mask
        if np.any(invisible_mask):
            plt.scatter(keypoints_2d[invisible_mask, 0], keypoints_2d[invisible_mask, 1], 
                       c='gray', s=30, alpha=0.5, label='Invisible keypoints')
            
        # Add keypoint indices
        for i, (x, y) in enumerate(keypoints_2d):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
            
        plt.xlim(0, width)
        plt.ylim(height, 0)  # Flip y-axis to match image coordinates
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.title(f'SLEAP 2D Keypoints - Camera: {camera_name}, Frame: {frame_idx}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
            
    def plot_keypoints_on_frame(self, camera_name: str, frame_idx: int = 0, save_path: Optional[str] = None):
        """
        Plot 2D keypoints overlaid on the actual video frame with skeleton connections.
        
        Args:
            camera_name (str): Name of the camera view
            frame_idx (int): Frame index to plot
            save_path (str, optional): Path to save the plot
        """
        # Load camera data
        camera_data = self.load_camera_data(camera_name)
        
        # Extract 2D keypoints
        keypoints_2d, visibility = self.extract_2d_keypoints(camera_data, frame_idx)
        
        if len(keypoints_2d) == 0:
            print(f"No keypoints found for camera {camera_name}, frame {frame_idx}")
            return
            
        # Load video frame
        frame = self.load_video_frame(camera_name, frame_idx)
        if frame is None:
            print(f"Failed to load video frame for camera {camera_name}, frame {frame_idx}")
            return
            
        # Get keypoint names and skeleton structure
        keypoint_names = self.get_keypoint_names()
        edge_indices, edge_names = self.get_skeleton_structure()
        
        # Create single plot with matplotlib's built-in legend
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Display the video frame
        ax.imshow(frame)
        
        # Define colors for each keypoint
        colors = plt.cm.tab20(np.linspace(0, 1, len(keypoint_names)))
        
        # Plot skeleton connections first (so they appear behind keypoints)
        for edge_idx, edge_name in zip(edge_indices, edge_names):
            from_idx, to_idx = edge_idx
            if from_idx < len(keypoints_2d) and to_idx < len(keypoints_2d):
                # Only draw connection if both keypoints are visible
                if visibility[from_idx] and visibility[to_idx]:
                    x_coords = [keypoints_2d[from_idx, 0], keypoints_2d[to_idx, 0]]
                    y_coords = [keypoints_2d[from_idx, 1], keypoints_2d[to_idx, 1]]
                    ax.plot(x_coords, y_coords, 'w-', linewidth=2, alpha=0.7)
        
        # Create legend entries by plotting invisible points
        legend_elements = []
        for i, (x, y) in enumerate(keypoints_2d):
            if i < len(keypoint_names):
                color = colors[i]
                if visibility[i]:
                    # Visible keypoint
                    scatter = ax.scatter(x, y, c=[color], s=75, alpha=0.9, 
                                       edgecolors='white', linewidth=2, zorder=5)
                    # Add to legend
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor=color, markersize=8,
                                                    label=f'{i}: {keypoint_names[i]}'))
                else:
                    # Invisible keypoint
                    ax.scatter(x, y, c=[color], s=50, alpha=0.6, 
                              edgecolors='white', linewidth=1, zorder=5)
                    # Add to legend with different style
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor=color, markersize=6,
                                                    alpha=0.6, linestyle='None',
                                                    label=f'{i}: {keypoint_names[i]} (hidden)'))
        
        # Add keypoint indices (smaller, less obtrusive)
        for i, (x, y) in enumerate(keypoints_2d):
            if visibility[i]:  # Only label visible keypoints
                ax.annotate(str(i), (x, y), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=10, alpha=0.9, color='white', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.8))
        
        ax.axis('off')  # Hide axes for cleaner look
        ax.set_title(f'SLEAP 2D Keypoints with Skeleton - Camera: {camera_name}, Frame: {frame_idx}', 
                    fontsize=14, color='white', 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
        
        # Create legend using matplotlib's built-in functionality
        legend = ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
                          fontsize=9, framealpha=0.9, fancybox=False, shadow=False)
        legend.set_title('Keypoints', prop={'size': 12, 'weight': 'bold'})
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
            
    def plot_all_cameras(self, frame_idx: int = 0, save_dir: Optional[str] = None, overlay_on_frames: bool = True):
        """
        Plot 2D keypoints for all available cameras.
        
        Args:
            frame_idx (int): Frame index to plot
            save_dir (str, optional): Directory to save plots
            overlay_on_frames (bool): Whether to overlay keypoints on video frames
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
        for camera_name in self.camera_views:
            print(f"Plotting keypoints for camera: {camera_name}")
            
            save_path = None
            if save_dir:
                if overlay_on_frames:
                    save_path = save_dir / f"{camera_name}_frame_{frame_idx:04d}_keypoints_on_frame.png"
                else:
                    save_path = save_dir / f"{camera_name}_frame_{frame_idx:04d}_keypoints.png"
                
            if overlay_on_frames:
                self.plot_keypoints_on_frame(camera_name, frame_idx, save_path)
            else:
                self.plot_keypoints_2d(camera_name, frame_idx, save_path)
            
    def get_keypoint_names(self) -> List[str]:
        """
        Get the keypoint names for the SLEAP model by extracting them from analysis files.
        
        Returns:
            List of keypoint names extracted dynamically from the analysis files
        """
        # Try to find an analysis file to extract keypoint names
        analysis_file = None
        
        if self.data_structure_type == 'camera_dirs':
            # Look for analysis files in any camera directory
            for camera_name in self.camera_views:
                camera_dir = self.project_path / camera_name
                analysis_files = list(camera_dir.glob("*.analysis.h5"))
                if analysis_files:
                    analysis_file = analysis_files[0]
                    break
        elif self.data_structure_type == 'session_dirs':
            # For session_dirs, keypoint names are stored in the camera .h5 files
            if self.session_name and self.camera_views:
                session_dir = self.project_path / self.session_name
                # Use the first camera's .h5 file to get keypoint names
                first_camera = self.camera_views[0]
                camera_h5_files = list(session_dir.glob(f"*_cam{first_camera}.h5"))
                if camera_h5_files:
                    analysis_file = camera_h5_files[0]
                
        # If no analysis file was found, try falling back to predictions files (camera_dirs only)
        if analysis_file is None and self.data_structure_type == 'camera_dirs':
            for camera_name in self.camera_views:
                camera_dir = self.project_path / camera_name
                # Use the first predictions file we find
                predictions_files = list(camera_dir.glob("*.predictions.h5"))
                if predictions_files:
                    analysis_file = predictions_files[0]
                    print(f"Using predictions file for keypoint names: {analysis_file}")
                    break

        if analysis_file is None:
            print("Warning: No analysis or predictions file found, using generic keypoint names")
            # Fallback to generic names if no analysis/predictions file found
            num_keypoints = 15  # Based on the data structure we observed
            return [f'keypoint_{i}' for i in range(num_keypoints)]
            
        try:
            # Extract keypoint names from the analysis file
            with h5py.File(analysis_file, 'r') as f:
                if 'node_names' in f:
                    node_names = f['node_names'][:]
                    keypoint_names = [name.decode('utf-8') for name in node_names]
                    print(f"Loaded keypoint names from: {analysis_file}")
                    return keypoint_names
                else:
                    print("Warning: No node_names found in analysis file")
                    num_keypoints = 15
                    return [f'keypoint_{i}' for i in range(num_keypoints)]
                    
        except Exception as e:
            print(f"Warning: Failed to read analysis file {analysis_file}: {e}")
            num_keypoints = 15
            return [f'keypoint_{i}' for i in range(num_keypoints)]
            
    def get_skeleton_structure(self) -> Tuple[List[Tuple[int, int]], List[Tuple[str, str]]]:
        """
        Get the skeleton structure (keypoint connections) from analysis files.
        
        Returns:
            Tuple of (edge_indices, edge_names) where:
            - edge_indices: List of (from_idx, to_idx) tuples
            - edge_names: List of (from_name, to_name) tuples
        """
        # Try to find an analysis file to extract skeleton structure
        analysis_file = None
        
        if self.data_structure_type == 'camera_dirs':
            # Look for analysis files in any camera directory
            for camera_name in self.camera_views:
                camera_dir = self.project_path / camera_name
                analysis_files = list(camera_dir.glob("*.analysis.h5"))
                if analysis_files:
                    analysis_file = analysis_files[0]
                    break
        elif self.data_structure_type == 'session_dirs':
            # For session_dirs, skeleton structure is stored in the camera .h5 files
            if self.session_name and self.camera_views:
                session_dir = self.project_path / self.session_name
                # Use the first camera's .h5 file to get skeleton structure
                first_camera = self.camera_views[0]
                camera_h5_files = list(session_dir.glob(f"*_cam{first_camera}.h5"))
                if camera_h5_files:
                    analysis_file = camera_h5_files[0]
                
        # If no analysis file was found, try falling back to predictions files
        if analysis_file is None:
            if self.data_structure_type == 'camera_dirs':
                # Look for *.predictions.h5 files inside camera directories
                for camera_name in self.camera_views:
                    camera_dir = self.project_path / camera_name
                    # Use the first predictions file we find
                    predictions_files = list(camera_dir.glob("*.predictions.h5"))
                    if predictions_files:
                        analysis_file = predictions_files[0]
                        print(f"Using predictions file for skeleton structure: {analysis_file}")
                        break

            if analysis_file is None:
                print("Warning: No analysis or predictions file found for skeleton structure")
                return [], []
            
        try:
            # Extract skeleton structure from the analysis file
            with h5py.File(analysis_file, 'r') as f:
                edge_indices = []
                edge_names = []
                
                if 'edge_inds' in f:
                    edge_inds_data = f['edge_inds'][:]
                    edge_indices = [(int(edge[0]), int(edge[1])) for edge in edge_inds_data]
                    
                if 'edge_names' in f:
                    edge_names_data = f['edge_names'][:]
                    edge_names = [(edge[0].decode('utf-8'), edge[1].decode('utf-8')) 
                                for edge in edge_names_data]
                    
                print(f"Loaded skeleton structure from: {analysis_file}")
                return edge_indices, edge_names
                
        except Exception as e:
            print(f"Warning: Failed to read skeleton structure from {analysis_file}: {e}")
            return [], []
        
    def print_data_summary(self):
        """Print a summary of the loaded data."""
        print("\n" + "="*50)
        print("SLEAP DATA SUMMARY")
        print("="*50)
        print(f"Project path: {self.project_path}")
        print(f"Data structure type: {self.data_structure_type}")
        if self.session_name:
            print(f"Session name: {self.session_name}")
        print(f"Number of camera views: {len(self.camera_views)}")
        print(f"Camera views: {self.camera_views}")
        
        # Get keypoint names dynamically
        keypoint_names = self.get_keypoint_names()
        print(f"Keypoint names: {keypoint_names}")
        
        # Get skeleton structure dynamically
        edge_indices, edge_names = self.get_skeleton_structure()
        if edge_indices:
            print(f"Skeleton connections: {len(edge_indices)} edges")
            print("Key connections:")
            for i, (edge_idx, edge_name) in enumerate(zip(edge_indices[:5], edge_names[:5])):
                print(f"  {edge_name[0]} -> {edge_name[1]} ({edge_idx[0]} -> {edge_idx[1]})")
            if len(edge_indices) > 5:
                print(f"  ... and {len(edge_indices) - 5} more connections")
        
        # Check 3D data
        points3d_file = self.project_path / "points3d.h5"
        if points3d_file.exists():
            with h5py.File(points3d_file, 'r') as f:
                tracks = f['tracks'][:]
                print(f"3D data shape: {tracks.shape}")
                print(f"Number of frames: {tracks.shape[0]}")
                print(f"Number of keypoints: {tracks.shape[2]}")
                
        # Check calibration data
        if self.calibration_data:
            print(f"Calibration data available for: {list(self.calibration_data.keys())}")
            
        print("="*50)
    
    def _load_smal_model_info(self):
        """Load SMAL model joint information from config."""
        try:
            # Load SMAL model data from config
            self.smal_joint_names = config.dd["J_names"]
            self.smal_n_joints = len(self.smal_joint_names)
            print(f"Loaded SMAL model with {self.smal_n_joints} joints")
            print(f"SMAL joint names: {self.smal_joint_names}")
        except Exception as e:
            print(f"Warning: Failed to load SMAL model info from config: {e}")
            # Fallback to generic joint names
            self.smal_joint_names = [f"joint_{i}" for i in range(50)]  # Default size
            self.smal_n_joints = len(self.smal_joint_names)
    
    def _load_lookup_table(self):
        """Load CSV lookup table for joint name mapping."""
        if not self.lookup_table_path:
            print("No lookup table provided, will use direct name matching")
            return
            
        try:
            self.lookup_table = pd.read_csv(self.lookup_table_path)
            print(f"Loaded lookup table from: {self.lookup_table_path}")
            print(f"Lookup table shape: {self.lookup_table.shape}")
            print(f"Columns: {list(self.lookup_table.columns)}")
            
            # Display first few mappings
            print("Sample mappings:")
            for i, row in self.lookup_table.head().iterrows():
                model_name = row.get('model', 'N/A')
                data_name = row.get('data', 'N/A')
                print(f"  {model_name} -> {data_name}")
                
        except Exception as e:
            print(f"Warning: Failed to load lookup table from {self.lookup_table_path}: {e}")
            self.lookup_table = None
    
    def _load_shape_betas(self):
        """Load ground truth shape betas from CSV lookup table."""
        if not self.shape_betas_path:
            print("No shape betas lookup table provided")
            return
            
        try:
            self.shape_betas_table = pd.read_csv(self.shape_betas_path)
            print(f"Loaded shape betas table from: {self.shape_betas_path}")
            print(f"Shape betas table shape: {self.shape_betas_table.shape}")
            print(f"Columns: {list(self.shape_betas_table.columns)}")
            
            # Validate shape betas table against SMAL model configuration
            self._validate_shape_betas_table()
            
            # Try to match the current dataset to a row in the shape betas table
            self._match_dataset_to_shape_betas()
            
        except Exception as e:
            print(f"Warning: Failed to load shape betas table from {self.shape_betas_path}: {e}")
            self.shape_betas_table = None
            self.ground_truth_betas = None
    
    def _validate_shape_betas_table(self):
        """Validate the shape betas table against SMAL model configuration."""
        if self.shape_betas_table is None:
            return
            
        # Get expected number of shape betas from config
        expected_n_betas = config.N_BETAS
        print(f"Expected number of shape betas from config: {expected_n_betas}")
        
        # Count PC columns in the table
        pc_columns = [col for col in self.shape_betas_table.columns if col.startswith('PC')]
        actual_n_betas = len(pc_columns)
        print(f"Number of PC columns in table: {actual_n_betas}")
        print(f"PC columns: {pc_columns}")
        
        # Validate the number of components
        if actual_n_betas != expected_n_betas:
            print(f"WARNING: Mismatch in number of shape betas!")
            print(f"  Expected (from config): {expected_n_betas}")
            print(f"  Found in table: {actual_n_betas}")
            print(f"  This may cause issues during training.")
            
            # Use the smaller number to avoid index errors
            self.n_shape_betas_to_use = min(expected_n_betas, actual_n_betas)
            print(f"  Using {self.n_shape_betas_to_use} shape betas to avoid errors.")
        else:
            print(f"✓ Shape betas count matches: {expected_n_betas}")
            self.n_shape_betas_to_use = expected_n_betas
    
    def _match_dataset_to_shape_betas(self):
        """Match the current dataset to ground truth shape betas."""
        if self.shape_betas_table is None:
            return
            
        # Extract dataset name from project path
        dataset_name = self.project_path.name
        print(f"Looking for shape betas for dataset: {dataset_name}")
        
        # Try to find a matching row in the shape betas table
        matched_row = None
        
        # First, try exact match
        for _, row in self.shape_betas_table.iterrows():
            label = str(row.get('label', '')).strip()
            if label == dataset_name:
                matched_row = row
                print(f"Found exact match: {label}")
                break
        
        # If no exact match, try partial matching
        if matched_row is None:
            for _, row in self.shape_betas_table.iterrows():
                label = str(row.get('label', '')).strip()
                if dataset_name in label or label in dataset_name:
                    matched_row = row
                    print(f"Found partial match: {label} for {dataset_name}")
                    break
        
        if matched_row is not None:
            # Extract shape betas dynamically based on validated number
            shape_betas = []
            for i in range(1, self.n_shape_betas_to_use + 1):  # PC1 to PC{n}
                pc_col = f'PC{i}'
                if pc_col in matched_row:
                    shape_betas.append(float(matched_row[pc_col]))
                else:
                    print(f"Warning: {pc_col} not found in shape betas table")
                    shape_betas.append(0.0)
            
            # Pad with zeros if we have fewer betas than expected by config
            if len(shape_betas) < config.N_BETAS:
                padding_needed = config.N_BETAS - len(shape_betas)
                shape_betas.extend([0.0] * padding_needed)
                print(f"Padded shape betas with {padding_needed} zeros to match config.N_BETAS ({config.N_BETAS})")
            
            self.ground_truth_betas = np.array(shape_betas, dtype=np.float32)
            print(f"Loaded ground truth shape betas: {self.ground_truth_betas}")
            print(f"Shape betas shape: {self.ground_truth_betas.shape}")
            print(f"Shape betas range: [{self.ground_truth_betas.min():.3f}, {self.ground_truth_betas.max():.3f}]")
        else:
            print(f"Warning: No matching shape betas found for dataset: {dataset_name}")
            print("Available labels in shape betas table:")
            for _, row in self.shape_betas_table.iterrows():
                label = str(row.get('label', '')).strip()
                print(f"  - {label}")
            self.ground_truth_betas = None
    
    def _create_joint_mapping(self):
        """Create mapping between SMAL model joints and SLEAP keypoints."""
        # Get SLEAP keypoint names
        sleap_keypoint_names = self.get_keypoint_names()
        
        print(f"Creating joint mapping between {len(self.smal_joint_names)} SMAL joints and {len(sleap_keypoint_names)} SLEAP keypoints")
        
        # Initialize mapping (all joints start as unmapped)
        self.joint_mapping = {i: -1 for i in range(self.smal_n_joints)}
        
        # Create mapping using lookup table if available
        if self.lookup_table is not None:
            self._create_mapping_with_lookup_table(sleap_keypoint_names)
        else:
            self._create_mapping_direct_matching(sleap_keypoint_names)
        
        # Print mapping summary
        mapped_count = sum(1 for idx in self.joint_mapping.values() if idx != -1)
        print(f"Joint mapping created: {mapped_count}/{self.smal_n_joints} joints mapped")
        
        # Show some example mappings
        print("Sample mappings:")
        for i, (smal_idx, sleap_idx) in enumerate(self.joint_mapping.items()):
            if sleap_idx != -1 and i < 10:  # Show first 10 mappings
                smal_name = self.smal_joint_names[smal_idx]
                sleap_name = sleap_keypoint_names[sleap_idx]
                print(f"  SMAL[{smal_idx}] {smal_name} -> SLEAP[{sleap_idx}] {sleap_name}")
    
    def _create_mapping_with_lookup_table(self, sleap_keypoint_names: List[str]):
        """Create joint mapping using lookup table."""
        print("Using lookup table for joint mapping")
        
        for _, row in self.lookup_table.iterrows():
            model_name = row.get('model', '')
            data_name = row.get('data', '')
            
            # Handle NaN values and convert to string
            if pd.isna(model_name) or pd.isna(data_name):
                continue
                
            model_name = str(model_name).strip()
            data_name = str(data_name).strip()
            
            if not model_name or not data_name:
                continue
                
            # Find SMAL joint index
            smal_idx = None
            for i, smal_name in enumerate(self.smal_joint_names):
                if smal_name == model_name:
                    smal_idx = i
                    break
            
            # Find SLEAP keypoint index
            sleap_idx = None
            for i, sleap_name in enumerate(sleap_keypoint_names):
                if sleap_name == data_name:
                    sleap_idx = i
                    break
            
            # Create mapping if both found
            if smal_idx is not None and sleap_idx is not None:
                self.joint_mapping[smal_idx] = sleap_idx
                print(f"  Mapped: {model_name} -> {data_name}")
            else:
                if smal_idx is None:
                    print(f"  Warning: SMAL joint '{model_name}' not found in model")
                if sleap_idx is None:
                    print(f"  Warning: SLEAP keypoint '{data_name}' not found in data")
    
    def _create_mapping_direct_matching(self, sleap_keypoint_names: List[str]):
        """Create joint mapping using direct name matching."""
        print("Using direct name matching for joint mapping")
        
        for smal_idx, smal_name in enumerate(self.smal_joint_names):
            for sleap_idx, sleap_name in enumerate(sleap_keypoint_names):
                if smal_name == sleap_name:
                    self.joint_mapping[smal_idx] = sleap_idx
                    print(f"  Direct match: {smal_name} -> {sleap_name}")
                    break
    
    def map_keypoints_to_smal_model(self, keypoints_2d: np.ndarray, visibility: np.ndarray, 
                                   image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map SLEAP keypoints to SMAL model joint order and convert to SMILify format.
        
        Args:
            keypoints_2d (np.ndarray): SLEAP 2D keypoints (N, 2) in pixel coordinates
            visibility (np.ndarray): SLEAP visibility flags (N,)
            image_size (Tuple[int, int]): Image size (width, height)
            
        Returns:
            Tuple of (smal_keypoints_2d, smal_visibility) where:
            - smal_keypoints_2d: (N_SMAL_JOINTS, 2) normalized coordinates [0,1] in [y, x] format
            - smal_visibility: (N_SMAL_JOINTS,) binary visibility flags
        """
        width, height = image_size
        
        # Initialize arrays for SMAL model joints
        smal_keypoints_2d = np.zeros((self.smal_n_joints, 2), dtype=np.float32)
        smal_visibility = np.zeros(self.smal_n_joints, dtype=np.float32)
        
        # Map each SMAL joint
        for smal_idx, sleap_idx in self.joint_mapping.items():
            if sleap_idx != -1 and sleap_idx < len(keypoints_2d):
                # Get SLEAP keypoint data
                sleap_x, sleap_y = keypoints_2d[sleap_idx]
                sleap_visible = visibility[sleap_idx]
                
                # Convert to normalized coordinates [0, 1]
                # Note: SLEAP uses [x, y] format, SMILify uses [y, x] format
                norm_x = sleap_x / width
                norm_y = sleap_y / height
                
                # Store in SMAL format: [y, x] normalized coordinates
                smal_keypoints_2d[smal_idx, 0] = norm_y  # y coordinate
                smal_keypoints_2d[smal_idx, 1] = norm_x  # x coordinate
                smal_visibility[smal_idx] = float(sleap_visible)
        
        return smal_keypoints_2d, smal_visibility
    
    def get_ground_truth_shape_betas(self) -> Optional[np.ndarray]:
        """
        Get the ground truth shape betas for the current dataset.
        
        Returns:
            np.ndarray or None: Ground truth shape betas if available
        """
        return self.ground_truth_betas
    
    def get_shape_betas_info(self) -> Dict[str, Any]:
        """
        Get information about shape betas loading and validation.
        
        Returns:
            Dict containing shape betas information
        """
        info = {
            'shape_betas_available': self.ground_truth_betas is not None,
            'shape_betas_table_loaded': self.shape_betas_table is not None,
            'expected_n_betas': config.N_BETAS,
            'actual_n_betas_in_table': getattr(self, 'n_shape_betas_to_use', 0),
            'shape_betas_shape': self.ground_truth_betas.shape if self.ground_truth_betas is not None else None
        }
        
        if self.ground_truth_betas is not None:
            info.update({
                'shape_betas_range': [float(self.ground_truth_betas.min()), float(self.ground_truth_betas.max())],
                'shape_betas_mean': float(self.ground_truth_betas.mean()),
                'shape_betas_std': float(self.ground_truth_betas.std())
            })
        
        return info
    
    def get_smal_model_info(self) -> Dict[str, Any]:
        """Get information about the SMAL model and joint mapping."""
        return {
            'smal_joint_names': self.smal_joint_names,
            'smal_n_joints': self.smal_n_joints,
            'joint_mapping': self.joint_mapping,
            'lookup_table_used': self.lookup_table is not None,
            'mapped_joints': sum(1 for idx in self.joint_mapping.values() if idx != -1),
            'ground_truth_betas_available': self.ground_truth_betas is not None,
            'ground_truth_betas': self.ground_truth_betas
        }


def main():
    """Test the SLEAP data loader with lookup table functionality."""
    # Test with session_dirs structure and lookup table
    print("="*60)
    print("TESTING SLEAP DATA LOADER WITH LOOKUP TABLE")
    print("="*60)
    
    project_path = "/home/fabi/DATA_LOCAL/2025-12-05-better-falkner-mice/TRAIN_SMIL_MICE/WhiteMouse"
    lookup_table_path = "/home/fabi/DATA_LOCAL/2025-12-05-better-falkner-mice/TRAIN_SMIL_MICE/lookup_table_names_Falkner_MICE.csv"
    shape_betas_path = "/home/fabi/DATA_LOCAL/2025-12-05-better-falkner-mice/TRAIN_SMIL_MICE/lookup_table_PCs_Falkner_MICE.csv"
    
    if Path(project_path).exists():
        print(f"Testing with project: {project_path}")
        if Path(lookup_table_path).exists():
            print(f"Using lookup table: {lookup_table_path}")
        else:
            print(f"Lookup table not found: {lookup_table_path}")
            lookup_table_path = None
            
        if Path(shape_betas_path).exists():
            print(f"Using shape betas table: {shape_betas_path}")
        else:
            print(f"Shape betas table not found: {shape_betas_path}")
            shape_betas_path = None
            
        # Initialize loader with lookup table and shape betas
        loader = SLEAPDataLoader(project_path, lookup_table_path, shape_betas_path)
        
        # Print data summary
        loader.print_data_summary()
        
        # Print SMAL model info
        print("\n" + "="*50)
        print("SMAL MODEL INTEGRATION INFO")
        print("="*50)
        smal_info = loader.get_smal_model_info()
        print(f"SMAL joints: {smal_info['smal_n_joints']}")
        print(f"Mapped joints: {smal_info['mapped_joints']}")
        print(f"Lookup table used: {smal_info['lookup_table_used']}")
        print(f"Ground truth betas available: {smal_info['ground_truth_betas_available']}")
        
        if smal_info['ground_truth_betas_available']:
            betas = smal_info['ground_truth_betas']
            print(f"Ground truth shape betas: {betas}")
            print(f"Shape betas shape: {betas.shape}")
            print(f"Shape betas range: [{betas.min():.3f}, {betas.max():.3f}]")
        
        # Print detailed shape betas validation info
        print("\n" + "="*50)
        print("SHAPE BETAS VALIDATION INFO")
        print("="*50)
        shape_betas_info = loader.get_shape_betas_info()
        print(f"Shape betas available: {shape_betas_info['shape_betas_available']}")
        print(f"Shape betas table loaded: {shape_betas_info['shape_betas_table_loaded']}")
        print(f"Expected N_BETAS (from config): {shape_betas_info['expected_n_betas']}")
        print(f"Actual betas in table: {shape_betas_info['actual_n_betas_in_table']}")
        
        if shape_betas_info['shape_betas_available']:
            print(f"Final shape betas shape: {shape_betas_info['shape_betas_shape']}")
            print(f"Shape betas statistics:")
            print(f"  Range: [{shape_betas_info['shape_betas_range'][0]:.3f}, {shape_betas_info['shape_betas_range'][1]:.3f}]")
            print(f"  Mean: {shape_betas_info['shape_betas_mean']:.3f}")
            print(f"  Std: {shape_betas_info['shape_betas_std']:.3f}")
        
        # Test keypoint mapping for first camera
        if loader.camera_views:
            camera_name = loader.camera_views[0]
            print(f"\nTesting keypoint mapping for camera: {camera_name}")
            
            # Load camera data
            camera_data = loader.load_camera_data(camera_name)
            
            # Extract 2D keypoints
            keypoints_2d, visibility = loader.extract_2d_keypoints(camera_data, frame_idx=0)
            
            if len(keypoints_2d) > 0:
                # Get image size
                image_size = loader.get_camera_image_size(camera_name)
                
                # Map to SMAL model
                smal_keypoints, smal_visibility = loader.map_keypoints_to_smal_model(
                    keypoints_2d, visibility, image_size
                )
                
                print(f"Original SLEAP keypoints shape: {keypoints_2d.shape}")
                print(f"Mapped SMAL keypoints shape: {smal_keypoints.shape}")
                print(f"Original visibility shape: {visibility.shape}")
                print(f"Mapped SMAL visibility shape: {smal_visibility.shape}")
                
                # Show some mapped keypoints
                print("\nSample mapped keypoints:")
                for i in range(min(10, len(smal_keypoints))):
                    if smal_visibility[i] > 0:
                        y_norm, x_norm = smal_keypoints[i]
                        smal_name = loader.smal_joint_names[i]
                        print(f"  {smal_name}: ({x_norm:.3f}, {y_norm:.3f}) - visible")
                    else:
                        smal_name = loader.smal_joint_names[i]
                        print(f"  {smal_name}: (0.000, 0.000) - hidden")
            else:
                print("No keypoints found for testing")

        # After all checks, plot all cameras for the first frame
        print("\nPlotting keypoints for all cameras (frame 0)...")
        loader.plot_all_cameras(frame_idx=0, save_dir=None, overlay_on_frames=True)
        
        
        print("\nSLEAP data loader with lookup table test completed!")
    else:
        print(f"Project path not found: {project_path}")
        print("Please update the path in the main() function to test with your data.")


if __name__ == "__main__":
    main()
