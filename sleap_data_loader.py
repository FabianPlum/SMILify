"""
SLEAP Data Loader for SMILify Pipeline

This module provides functionality to load and process SLEAP pose estimation data
for integration with the SMILify training pipeline.

Author: SMILify Team
Date: 2025
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


class SLEAPDataLoader:
    """
    A modular data loader for SLEAP pose estimation datasets.
    
    This class handles loading of:
    - Multi-camera 2D keypoint predictions (.slp files)
    - 3D reconstructed keypoints (points3d.h5)
    - Camera calibration data (calibration.toml)
    - Video frames for visualization
    """
    
    def __init__(self, project_path: str):
        """
        Initialize the SLEAP data loader.
        
        Args:
            project_path (str): Path to the SLEAP project directory
        """
        self.project_path = Path(project_path)
        self.calibration_data = None
        self.camera_views = []
        self.keypoint_names = []
        self._load_project_structure()
        
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
            
        # Identify camera views from directory structure
        self.camera_views = []
        for item in self.project_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if this directory contains SLEAP prediction files
                slp_files = list(item.glob("*.slp"))
                if slp_files:
                    self.camera_views.append(item.name)
                    
        print(f"Found {len(self.camera_views)} camera views: {self.camera_views}")
        
        # Load 3D points if available
        points3d_file = self.project_path / "points3d.h5"
        if points3d_file.exists():
            print("Found 3D points file")
        else:
            print("Warning: No points3d.h5 found")
            
    def load_camera_data(self, camera_name: str) -> Dict[str, Any]:
        """
        Load data for a specific camera view.
        
        Args:
            camera_name (str): Name of the camera view (e.g., 'back', 'side', 'top')
            
        Returns:
            Dict containing camera data including keypoints, frames, and metadata
        """
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
                
            # Load metadata
            if 'videos_json' in f:
                videos_json = f['videos_json'][()]
                data['videos'] = json.loads(videos_json.item())
                
            if 'tracks_json' in f:
                tracks_json = f['tracks_json'][()]
                data['tracks'] = json.loads(tracks_json.item())
                
        # Load calibration data for this camera
        # Find the calibration entry that matches this camera name
        calibration_entry = None
        if self.calibration_data:
            for key, value in self.calibration_data.items():
                if key != 'metadata' and value.get('name') == camera_name:
                    calibration_entry = value
                    break
                    
        if calibration_entry:
            data['calibration'] = calibration_entry
            
        return data
        
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
        camera_dir = self.project_path / camera_name
        
        # Find video file
        video_files = list(camera_dir.glob("*.mp4"))
        if not video_files:
            print(f"No video file found in {camera_dir}")
            return None
            
        video_file = video_files[0]
        
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
        
        # Look for analysis files in any camera directory
        for camera_name in self.camera_views:
            camera_dir = self.project_path / camera_name
            analysis_files = list(camera_dir.glob("*.analysis.h5"))
            if analysis_files:
                analysis_file = analysis_files[0]
                break
                
        if analysis_file is None:
            print("Warning: No analysis file found, using generic keypoint names")
            # Fallback to generic names if no analysis file found
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
        
        # Look for analysis files in any camera directory
        for camera_name in self.camera_views:
            camera_dir = self.project_path / camera_name
            analysis_files = list(camera_dir.glob("*.analysis.h5"))
            if analysis_files:
                analysis_file = analysis_files[0]
                break
                
        if analysis_file is None:
            print("Warning: No analysis file found for skeleton structure")
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


def main():
    """Test the SLEAP data loader with the provided dataset."""
    # Initialize the data loader
    project_path = "/home/fabi/DATA_LOCAL/MICE/10072022120554"
    loader = SLEAPDataLoader(project_path)
    
    # Print data summary
    loader.print_data_summary()
    
    # Plot keypoints for all cameras (first frame)
    print("\nPlotting 2D keypoints for all cameras...")
    loader.plot_all_cameras(frame_idx=0, save_dir="sleap_validation_plots")
    
    print("\nSLEAP data loader test completed!")


if __name__ == "__main__":
    main()
