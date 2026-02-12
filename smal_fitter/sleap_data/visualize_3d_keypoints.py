#!/usr/bin/env python3
"""
3D Keypoints Visualization Script

Loads a preprocessed multi-view SLEAP dataset and visualizes 3D keypoints
as an animated 3D plot using matplotlib.

Usage:
    python visualize_3d_keypoints.py dataset.h5 [options]

Example:
    python visualize_3d_keypoints.py multiview_sleap.h5 --fps 10 --sample_skip 1
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from pathlib import Path


def load_3d_keypoints(h5_path: str):
    """
    Load 3D keypoints from preprocessed HDF5 dataset.
    
    Args:
        h5_path: Path to HDF5 file
        
    Returns:
        Tuple of (keypoints_3d, has_3d_data, num_samples, n_joints)
    """
    with h5py.File(h5_path, 'r') as f:
        # Load 3D keypoints
        keypoints_3d = f['multiview_keypoints/keypoints_3d'][:]  # (num_samples, n_joints, 3)
        has_3d_data = f['auxiliary/has_3d_data'][:]  # (num_samples,)
        
        # Get metadata
        num_samples = keypoints_3d.shape[0]
        n_joints = keypoints_3d.shape[1]
        
        # Filter to only samples with 3D data
        valid_mask = has_3d_data.astype(bool)
        keypoints_3d_valid = keypoints_3d[valid_mask]
        
        print(f"Loaded dataset: {h5_path}")
        print(f"  Total samples: {num_samples}")
        print(f"  Samples with 3D data: {valid_mask.sum()}")
        print(f"  Number of joints: {n_joints}")
        
        return keypoints_3d_valid, valid_mask, num_samples, n_joints


def compute_axis_limits(keypoints_3d: np.ndarray, padding: float = 0.1):
    """
    Compute axis limits with padding.
    
    Args:
        keypoints_3d: (num_samples, n_joints, 3) array of 3D keypoints
        padding: Padding factor (0.1 = 10%)
        
    Returns:
        Tuple of (xlim, ylim, zlim) where each is (min, max)
    """
    # Find valid (non-zero, non-NaN) keypoints across all samples
    valid_mask = ~(np.isnan(keypoints_3d).any(axis=2) | 
                   np.isinf(keypoints_3d).any(axis=2) |
                   (keypoints_3d == 0).all(axis=2))
    
    if valid_mask.sum() == 0:
        # Fallback: use all keypoints
        valid_keypoints = keypoints_3d.reshape(-1, 3)
    else:
        # Extract valid keypoints
        valid_keypoints = keypoints_3d[valid_mask].reshape(-1, 3)
    
    # Compute min/max for each axis
    x_min, y_min, z_min = valid_keypoints.min(axis=0)
    x_max, y_max, z_max = valid_keypoints.max(axis=0)
    
    # Compute ranges
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    # Use the maximum range for all axes (equal-sized axes)
    max_range = max(x_range, y_range, z_range)
    
    # Compute center
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    z_center = (z_min + z_max) / 2.0
    
    # Apply padding
    half_range = max_range / 2.0 * (1 + padding)
    
    xlim = (x_center - half_range, x_center + half_range)
    ylim = (y_center - half_range, y_center + half_range)
    zlim = (z_center - half_range, z_center + half_range)
    
    return xlim, ylim, zlim


def create_3d_animation(keypoints_3d: np.ndarray, 
                       fps: int = 10,
                       sample_skip: int = 1,
                       padding: float = 0.1,
                       point_size: float = 20.0):
    """
    Create animated 3D visualization of keypoints.
    
    Args:
        keypoints_3d: (num_samples, n_joints, 3) array of 3D keypoints
        fps: Frames per second for animation
        sample_skip: Process every Nth sample (default: 1, all samples)
        padding: Padding factor for axes (default: 0.1 = 10%)
        point_size: Size of scatter plot points (default: 20.0)
    """
    # Apply sample skip
    if sample_skip > 1:
        keypoints_3d = keypoints_3d[::sample_skip]
    
    num_samples = keypoints_3d.shape[0]
    n_joints = keypoints_3d.shape[1]
    
    # Compute axis limits
    xlim, ylim, zlim = compute_axis_limits(keypoints_3d, padding=padding)
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize scatter plot
    scatter = ax.scatter([], [], [], s=point_size, c='blue', alpha=0.7)
    
    # Set equal aspect ratio and limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Keypoints Animation')
    
    # Store current sample index
    current_sample = [0]
    
    def update(frame):
        """Update function for animation."""
        sample_idx = current_sample[0]
        
        if sample_idx >= num_samples:
            sample_idx = 0  # Loop back to start
        
        # Get keypoints for current sample
        kp3d = keypoints_3d[sample_idx]  # (n_joints, 3)
        
        # Filter out invalid keypoints (NaN, inf, or all zeros)
        valid_mask = ~(np.isnan(kp3d).any(axis=1) | 
                      np.isinf(kp3d).any(axis=1) |
                      (kp3d == 0).all(axis=1))
        
        if valid_mask.sum() > 0:
            valid_kp = kp3d[valid_mask]
            x, y, z = valid_kp[:, 0], valid_kp[:, 1], valid_kp[:, 2]
        else:
            x, y, z = [], [], []
        
        # Clear previous plot
        ax.clear()
        
        # Recreate scatter plot
        if len(x) > 0:
            ax.scatter(x, y, z, s=point_size, c='blue', alpha=0.7)
        
        # Reset limits and labels
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Keypoints - Sample {sample_idx + 1}/{num_samples} '
                    f'({valid_mask.sum()}/{n_joints} valid joints)')
        
        # Update sample index
        current_sample[0] = (sample_idx + 1) % num_samples
        
        return []
    
    # Create animation
    interval_ms = 1000 / fps  # Convert fps to interval in milliseconds
    anim = animation.FuncAnimation(
        fig, 
        update, 
        interval=interval_ms,
        blit=False,
        repeat=True
    )
    
    print(f"\nStarting animation...")
    print(f"  Samples: {num_samples}")
    print(f"  FPS: {fps}")
    print(f"  Close the window to stop")
    
    plt.show()
    
    return anim


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D keypoints from preprocessed multi-view SLEAP dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic visualization
  python visualize_3d_keypoints.py multiview_sleap.h5
  
  # Faster animation with sample skipping
  python visualize_3d_keypoints.py multiview_sleap.h5 --fps 20 --sample_skip 5
  
  # Larger points
  python visualize_3d_keypoints.py multiview_sleap.h5 --point_size 50
        """
    )
    
    parser.add_argument("dataset_path", help="Path to preprocessed HDF5 dataset")
    parser.add_argument("--fps", type=int, default=10,
                       help="Animation frames per second (default: 10)")
    parser.add_argument("--sample_skip", type=int, default=1,
                       help="Process every Nth sample (default: 1, all samples)")
    parser.add_argument("--padding", type=float, default=0.1,
                       help="Axis padding factor (default: 0.1 = 10%%)")
    parser.add_argument("--point_size", type=float, default=20.0,
                       help="Size of scatter plot points (default: 20.0)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.dataset_path).exists():
        print(f"Error: Dataset file does not exist: {args.dataset_path}")
        return 1
    
    # Load 3D keypoints
    try:
        keypoints_3d, has_3d_mask, num_samples, n_joints = load_3d_keypoints(args.dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    if len(keypoints_3d) == 0:
        print("Error: No samples with 3D data found in dataset")
        return 1
    
    # Create animation
    try:
        create_3d_animation(
            keypoints_3d,
            fps=args.fps,
            sample_skip=args.sample_skip,
            padding=args.padding,
            point_size=args.point_size
        )
    except KeyboardInterrupt:
        print("\nAnimation interrupted by user")
        return 0
    except Exception as e:
        print(f"Error creating animation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
