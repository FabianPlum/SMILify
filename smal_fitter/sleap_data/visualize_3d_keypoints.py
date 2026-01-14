#!/usr/bin/env python3
"""
Visualize 3D Keypoints from SLEAP Datasets

This script loads 3D coordinates from SLEAP datasets and creates an animated
3D visualization to check data quality.

Usage:
    python visualize_3d_keypoints.py sessions_dir [options]

Example:
    python visualize_3d_keypoints.py /path/to/sleap/sessions --session_idx 0 --max_frames 500
"""

import os
import sys
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List, Optional, Tuple

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from sleap_data_loader import SLEAPDataLoader
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from sleap_data_loader import SLEAPDataLoader


def discover_sleap_sessions(sessions_dir: str) -> List[str]:
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
            # Check for points3d.h5 file (indicates 3D data available)
            if (item / "points3d.h5").exists():
                sessions.append(str(item))
    
    sessions.sort()
    return sessions


def load_3d_trajectory(session_path: str) -> Optional[np.ndarray]:
    """
    Load all 3D keypoints for a session.
    
    Args:
        session_path: Path to SLEAP session directory
        
    Returns:
        Array of shape (n_frames, n_keypoints, 3) or None if failed
    """
    points3d_file = Path(session_path) / "points3d.h5"
    
    if not points3d_file.exists():
        print(f"Warning: No points3d.h5 found in {session_path}")
        return None
    
    try:
        with h5py.File(points3d_file, 'r') as f:
            if 'tracks' not in f:
                print(f"Warning: No 'tracks' dataset in {points3d_file}")
                return None
            
            tracks = f['tracks'][:]  # Shape: (n_frames, n_tracks, n_keypoints, 3)
            
            # Extract first track (assuming single animal)
            if tracks.shape[1] > 0:
                trajectory = tracks[:, 0, :, :]  # (n_frames, n_keypoints, 3)
                return trajectory
            else:
                print(f"Warning: No tracks found in {points3d_file}")
                return None
                
    except Exception as e:
        print(f"Error loading 3D data from {session_path}: {e}")
        return None


def get_keypoint_names(session_path: str) -> List[str]:
    """
    Get keypoint names from a session.
    
    Args:
        session_path: Path to SLEAP session directory
        
    Returns:
        List of keypoint names
    """
    try:
        loader = SLEAPDataLoader(project_path=session_path)
        keypoint_names = loader.get_keypoint_names()
        return keypoint_names
    except Exception as e:
        print(f"Warning: Could not load keypoint names: {e}")
        # Return generic names based on number of keypoints
        trajectory = load_3d_trajectory(session_path)
        if trajectory is not None:
            n_keypoints = trajectory.shape[1]
            return [f'kp_{i}' for i in range(n_keypoints)]
        return []


def get_skeleton_edges(session_path: str) -> List[Tuple[int, int]]:
    """
    Get skeleton edge connections from a session.
    
    Args:
        session_path: Path to SLEAP session directory
        
    Returns:
        List of (from_idx, to_idx) tuples
    """
    try:
        loader = SLEAPDataLoader(project_path=session_path)
        edge_indices, _ = loader.get_skeleton_structure()
        return edge_indices
    except Exception as e:
        print(f"Warning: Could not load skeleton structure: {e}")
        return []


def visualize_3d_trajectory(trajectory: np.ndarray,
                            keypoint_names: List[str],
                            skeleton_edges: List[Tuple[int, int]],
                            session_name: str,
                            fps: int = 30,
                            save_path: Optional[str] = None):
    """
    Create animated 3D visualization of keypoint trajectory.
    
    Args:
        trajectory: Array of shape (n_frames, n_keypoints, 3)
        keypoint_names: List of keypoint names
        skeleton_edges: List of (from_idx, to_idx) tuples for skeleton connections
        session_name: Name of the session (for title)
        fps: Frames per second for animation
        save_path: Optional path to save animation as GIF
    """
    n_frames, n_keypoints, _ = trajectory.shape
    
    # Compute bounding box for consistent axes
    all_points = trajectory.reshape(-1, 3)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    
    # Add padding
    padding = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.1
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2 + padding
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis labels and limits (set once, will be maintained)
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.set_xlim(x_center - max_range, x_center + max_range)
    ax.set_ylim(y_center - max_range, y_center + max_range)
    ax.set_zlim(z_center - max_range, z_center + max_range)
    
    # Animation update function
    def update(frame):
        # Clear previous data
        ax.clear()
        
        # Get current frame data
        keypoints_3d = trajectory[frame]  # (n_keypoints, 3)
        
        # Plot keypoints
        ax.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2],
                  s=50, c='red', alpha=0.8, label='Keypoints')
        
        # Plot skeleton lines
        if skeleton_edges:
            for edge in skeleton_edges:
                from_idx, to_idx = edge
                if from_idx < n_keypoints and to_idx < n_keypoints:
                    x_coords = [keypoints_3d[from_idx, 0], keypoints_3d[to_idx, 0]]
                    y_coords = [keypoints_3d[from_idx, 1], keypoints_3d[to_idx, 1]]
                    z_coords = [keypoints_3d[from_idx, 2], keypoints_3d[to_idx, 2]]
                    ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=2, alpha=0.6)
        
        # Update axis labels and limits
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_zlabel('Z (mm)', fontsize=12)
        ax.set_title(f'3D Keypoint Trajectory: {session_name}\nFrame: {frame}/{n_frames-1}', 
                     fontsize=14, fontweight='bold')
        ax.set_xlim(x_center - max_range, x_center + max_range)
        ax.set_ylim(y_center - max_range, y_center + max_range)
        ax.set_zlim(z_center - max_range, z_center + max_range)
        
        # Add legend
        if skeleton_edges:
            ax.plot([], [], [], 'b-', linewidth=2, alpha=0.6, label='Skeleton')
        ax.legend(loc='upper right')
    
    # Create animation
    interval = 1000 / fps  # milliseconds per frame
    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, 
                        blit=False, repeat=True)
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Animation saved!")
    else:
        plt.show()
    
    return anim


def print_trajectory_stats(trajectory: np.ndarray, session_name: str):
    """
    Print statistics about the 3D trajectory.
    
    Args:
        trajectory: Array of shape (n_frames, n_keypoints, 3)
        session_name: Name of the session
    """
    n_frames, n_keypoints, _ = trajectory.shape
    
    print(f"\n{'='*60}")
    print(f"3D Trajectory Statistics: {session_name}")
    print(f"{'='*60}")
    print(f"Number of frames: {n_frames}")
    print(f"Number of keypoints: {n_keypoints}")
    
    # Compute statistics
    all_points = trajectory.reshape(-1, 3)
    
    print(f"\nCoordinate ranges:")
    print(f"  X: [{trajectory[:, :, 0].min():.2f}, {trajectory[:, :, 0].max():.2f}] mm")
    print(f"  Y: [{trajectory[:, :, 1].min():.2f}, {trajectory[:, :, 1].max():.2f}] mm")
    print(f"  Z: [{trajectory[:, :, 2].min():.2f}, {trajectory[:, :, 2].max():.2f}] mm")
    
    # Check for NaN or invalid values
    nan_count = np.isnan(all_points).sum()
    inf_count = np.isinf(all_points).sum()
    zero_count = (np.abs(all_points) < 1e-6).sum()
    
    print(f"\nData quality:")
    print(f"  NaN values: {nan_count} ({nan_count / all_points.size * 100:.2f}%)")
    print(f"  Inf values: {inf_count} ({inf_count / all_points.size * 100:.2f}%)")
    print(f"  Near-zero values: {zero_count} ({zero_count / all_points.size * 100:.2f}%)")
    
    # Compute velocity statistics (frame-to-frame differences)
    if n_frames > 1:
        velocities = np.diff(trajectory, axis=0)  # (n_frames-1, n_keypoints, 3)
        speeds = np.linalg.norm(velocities, axis=2)  # (n_frames-1, n_keypoints)
        mean_speed = speeds.mean()
        max_speed = speeds.max()
        
        print(f"\nMotion statistics:")
        print(f"  Mean speed: {mean_speed:.2f} mm/frame")
        print(f"  Max speed: {max_speed:.2f} mm/frame")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D keypoints from SLEAP datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize first session
  python visualize_3d_keypoints.py /path/to/sleap/sessions --session_idx 0
  
  # Visualize specific session with limited frames
  python visualize_3d_keypoints.py /path/to/sleap/sessions --session_idx 2 --max_frames 200
  
  # Save animation as GIF
  python visualize_3d_keypoints.py /path/to/sleap/sessions --session_idx 0 --save output.gif
        """
    )
    
    parser.add_argument("sessions_dir", help="Directory containing SLEAP sessions")
    parser.add_argument("--session_idx", type=int, default=0,
                       help="Index of session to visualize (default: 0)")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum number of frames to visualize (default: all)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second for animation (default: 30)")
    parser.add_argument("--save", type=str, default=None,
                       help="Path to save animation as GIF (default: display interactively)")
    parser.add_argument("--list_sessions", action="store_true",
                       help="List all available sessions and exit")
    
    args = parser.parse_args()
    
    # Discover sessions
    sessions = discover_sleap_sessions(args.sessions_dir)
    
    if len(sessions) == 0:
        print(f"Error: No SLEAP sessions with 3D data found in {args.sessions_dir}")
        print("Sessions must contain a points3d.h5 file.")
        sys.exit(1)
    
    # List sessions if requested
    if args.list_sessions:
        print(f"\nFound {len(sessions)} sessions with 3D data:")
        for i, session in enumerate(sessions):
            print(f"  [{i}] {Path(session).name}")
        sys.exit(0)
    
    # Validate session index
    if args.session_idx < 0 or args.session_idx >= len(sessions):
        print(f"Error: Session index {args.session_idx} out of range (0-{len(sessions)-1})")
        print(f"\nAvailable sessions:")
        for i, session in enumerate(sessions):
            print(f"  [{i}] {Path(session).name}")
        sys.exit(1)
    
    # Select session
    session_path = sessions[args.session_idx]
    session_name = Path(session_path).name
    
    print(f"\n{'='*60}")
    print(f"VISUALIZING 3D KEYPOINTS")
    print(f"{'='*60}")
    print(f"Session: {session_name}")
    print(f"Path: {session_path}")
    print(f"{'='*60}\n")
    
    # Load 3D trajectory
    print("Loading 3D trajectory...")
    trajectory = load_3d_trajectory(session_path)
    
    if trajectory is None:
        print(f"Error: Failed to load 3D trajectory from {session_path}")
        sys.exit(1)
    
    # Limit frames if requested
    if args.max_frames is not None and args.max_frames < trajectory.shape[0]:
        trajectory = trajectory[:args.max_frames]
        print(f"Limited to {args.max_frames} frames")
    
    # Print statistics
    print_trajectory_stats(trajectory, session_name)
    
    # Get keypoint names and skeleton structure
    print("Loading keypoint names and skeleton structure...")
    keypoint_names = get_keypoint_names(session_path)
    skeleton_edges = get_skeleton_edges(session_path)
    
    print(f"Keypoints: {len(keypoint_names)}")
    if skeleton_edges:
        print(f"Skeleton edges: {len(skeleton_edges)}")
    
    # Create visualization
    print("\nCreating 3D visualization...")
    print("Close the window or press Ctrl+C to stop the animation.")
    
    try:
        anim = visualize_3d_trajectory(
            trajectory=trajectory,
            keypoint_names=keypoint_names,
            skeleton_edges=skeleton_edges,
            session_name=session_name,
            fps=args.fps,
            save_path=args.save
        )
        
        if args.save is None:
            # Keep animation running
            plt.show()
        
    except KeyboardInterrupt:
        print("\nAnimation interrupted by user")
    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
