"""
Multi-View SLEAP Dataset Class for SMILify Training Pipeline

This module provides a PyTorch dataset class for loading preprocessed multi-view SLEAP datasets
that is compatible with the multi-view SMILify training pipeline.
"""

import os
import torch
import h5py
import numpy as np
import cv2
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config


class SLEAPMultiViewDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset class for loading preprocessed multi-view SLEAP datasets.
    
    Each sample contains multiple synchronized camera views of the same frame,
    enabling multi-view training for improved pose and shape estimation.
    
    Key Features:
    - Variable number of views per sample (no padding, use masking)
    - Optional view sampling when more views available than needed
    - Random or fixed view selection
    - Backward compatibility mode for single-view training
    """
    
    def __init__(self, 
                 hdf5_path: str, 
                 rotation_representation: str = '6d',
                 num_views_to_use: Optional[int] = None,
                 return_single_view: bool = False,
                 preferred_view: int = 0,
                 random_view_sampling: bool = True,
                 backbone_name: str = None,
                 **kwargs):
        """
        Initialize the multi-view SLEAP dataset.
        
        Args:
            hdf5_path: Path to the preprocessed multi-view HDF5 file
            rotation_representation: Rotation representation ('6d' or 'axis_angle')
            num_views_to_use: Maximum views to use per sample (None = use all available)
                             If sample has more views, sample this many
                             If sample has fewer views, use all available (no padding)
            return_single_view: If True, return only one view (backward compatible)
            preferred_view: Which view to return if return_single_view=True
            random_view_sampling: If True, randomly sample views when more available
            backbone_name: Backbone name (for compatibility)
            **kwargs: Additional keyword arguments (for compatibility)
        """
        self.hdf5_path = hdf5_path
        self.rotation_representation = rotation_representation
        self.num_views_to_use = num_views_to_use
        self.return_single_view = return_single_view
        self.preferred_view = preferred_view
        self.random_view_sampling = random_view_sampling
        
        # Load dataset metadata
        self._load_metadata()
        
        # File handle (lazy loading for multiprocessing compatibility)
        self._file = None
        
        # Validate dataset
        self._validate_dataset()
    
    def _load_metadata(self):
        """Load dataset metadata from HDF5 file."""
        with h5py.File(self.hdf5_path, 'r') as f:
            metadata = f['metadata']
            
            self.num_samples = metadata.attrs['num_samples']
            self.max_views = metadata.attrs['max_views']
            self.n_joints = metadata.attrs['n_joints']
            self.target_resolution = metadata.attrs['target_resolution']
            self.backbone_name = metadata.attrs['backbone_name']
            self.dataset_type = metadata.attrs['dataset_type']
            self.is_multiview = metadata.attrs.get('is_multiview', True)
            self.n_pose = metadata.attrs['n_pose']
            self.n_betas = metadata.attrs['n_betas']
            self.min_views_per_sample = metadata.attrs.get('min_views_per_sample', 2)
            self.crop_mode = metadata.attrs.get('crop_mode', 'bbox_crop')
            
            # Parse canonical camera order
            canonical_order_json = metadata.attrs.get('canonical_camera_order', '[]')
            self.canonical_camera_order = json.loads(canonical_order_json)
    
    def _validate_dataset(self):
        """Validate that the dataset is properly formatted."""
        with h5py.File(self.hdf5_path, 'r') as f:
            # Check required groups
            required_groups = ['multiview_images', 'multiview_keypoints', 'parameters', 
                             'auxiliary', 'metadata']
            for group in required_groups:
                if group not in f:
                    raise ValueError(f"Missing required group: {group}")
            
            # Verify view mask exists
            if 'view_mask' not in f['multiview_images']:
                raise ValueError("Missing view_mask in multiview_images group")
            
            # Verify keypoints exist
            if 'keypoints_2d' not in f['multiview_keypoints']:
                raise ValueError("Missing keypoints_2d in multiview_keypoints group")
    
    def _ensure_file_open(self):
        """Ensure HDF5 file is open in current process."""
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r')
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (x_data, y_data) where:
            - x_data: Input data dictionary with multi-view images
            - y_data: Target data dictionary with keypoints and parameters
        """
        self._ensure_file_open()
        
        if self.return_single_view:
            return self._get_single_view_sample(idx)
        else:
            return self._get_multiview_sample(idx)
    
    def _get_multiview_sample(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get a multi-view sample."""
        f = self._file
        
        # Get view mask and number of available views
        view_mask = f['multiview_images/view_mask'][idx]  # (max_views,)
        num_available = int(view_mask.sum())
        available_view_indices = np.where(view_mask)[0]
        
        # Determine which views to use
        if self.num_views_to_use is not None and num_available > self.num_views_to_use:
            # Sample views
            if self.random_view_sampling:
                selected_indices = np.random.choice(
                    available_view_indices, 
                    size=self.num_views_to_use, 
                    replace=False
                )
                selected_indices = np.sort(selected_indices)  # Keep canonical order
            else:
                selected_indices = available_view_indices[:self.num_views_to_use]
        else:
            # Use all available views
            selected_indices = available_view_indices
        
        num_views = len(selected_indices)
        
        # Load images for selected views
        images = []
        for v in selected_indices:
            image_jpeg_bytes = f[f'multiview_images/image_jpeg_view_{v}'][idx]
            image = self._decode_jpeg_image(image_jpeg_bytes)
            images.append(image)
        
        # Load keypoints for selected views
        keypoints_2d_all = f['multiview_keypoints/keypoints_2d'][idx]  # (max_views, n_joints, 2)
        keypoint_visibility_all = f['multiview_keypoints/keypoint_visibility'][idx]  # (max_views, n_joints)
        camera_indices_all = f['multiview_keypoints/camera_indices'][idx]  # (max_views,)
        
        keypoints_2d = keypoints_2d_all[selected_indices]
        keypoint_visibility = keypoint_visibility_all[selected_indices]
        camera_indices = camera_indices_all[selected_indices]
        
        # Load shared body parameters
        betas = f['parameters/betas'][idx]
        
        # Load auxiliary data
        session_name = f['auxiliary/session_name'][idx].decode('utf-8')
        frame_idx = f['auxiliary/frame_idx'][idx]
        has_ground_truth_betas = f['auxiliary/has_ground_truth_betas'][idx]
        camera_names_str = f['auxiliary/camera_names'][idx].decode('utf-8')
        all_camera_names = camera_names_str.split(',')
        
        # Filter camera names for selected views
        camera_names = [all_camera_names[i] if i < len(all_camera_names) else f'cam_{i}' 
                       for i in range(num_available)]
        camera_names = [camera_names[np.where(available_view_indices == v)[0][0]] 
                       for v in selected_indices if v in available_view_indices]
        
        # Create view valid mask (all True for selected views)
        view_valid = np.ones(num_views, dtype=bool)
        
        # Prepare x_data (input data)
        x_data = {
            'images': images,  # List of (H, W, C) arrays in [0, 1]
            'view_mask': view_valid,  # (num_views,) all True
            'camera_names': camera_names,  # List of camera name strings
            'camera_indices': camera_indices,  # (num_views,) canonical indices
            'num_active_views': num_views,
            'session_name': session_name,
            'frame_idx': frame_idx,
            'is_multiview': True,
            'is_sleap_dataset': True,
            'available_labels': {
                'global_rot': False,
                'joint_rot': False,
                'betas': bool(has_ground_truth_betas),
                'trans': False,
                'fov': False,
                'cam_rot': False,
                'cam_trans': False,
                'log_beta_scales': False,
                'betas_trans': False,
                'keypoint_2d': True,
                'keypoint_3d': False,
                'silhouette': False,
            },
        }
        
        # Prepare y_data (target data)
        y_data = {
            # Per-view keypoint data
            'keypoints_2d': keypoints_2d,  # (num_views, n_joints, 2)
            'keypoint_visibility': keypoint_visibility,  # (num_views, n_joints)
            'view_valid': view_valid,  # (num_views,)
            
            # Shared body parameters
            'shape_betas': betas if has_ground_truth_betas else None,
            
            # Placeholders (no ground truth for SLEAP)
            'root_rot': None,
            'joint_angles': None,
            'root_loc': None,
            'cam_fov': None,
            'cam_rot': None,
            'cam_trans': None,
            
            # Metadata
            'has_ground_truth_betas': has_ground_truth_betas,
            'num_views': num_views,
        }
        
        return x_data, y_data
    
    def _get_single_view_sample(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get a single-view sample (backward compatibility mode)."""
        f = self._file
        
        # Get view mask
        view_mask = f['multiview_images/view_mask'][idx]
        available_view_indices = np.where(view_mask)[0]
        
        if len(available_view_indices) == 0:
            raise ValueError(f"No valid views for sample {idx}")
        
        # Select the preferred view, or first available
        if self.preferred_view < len(available_view_indices):
            selected_view = available_view_indices[self.preferred_view]
        else:
            selected_view = available_view_indices[0]
        
        # Load image
        image_jpeg_bytes = f[f'multiview_images/image_jpeg_view_{selected_view}'][idx]
        image = self._decode_jpeg_image(image_jpeg_bytes)
        
        # Load keypoints
        keypoints_2d = f['multiview_keypoints/keypoints_2d'][idx, selected_view]
        keypoint_visibility = f['multiview_keypoints/keypoint_visibility'][idx, selected_view]
        
        # Load shared parameters
        betas = f['parameters/betas'][idx]
        
        # Load auxiliary data
        session_name = f['auxiliary/session_name'][idx].decode('utf-8')
        frame_idx = f['auxiliary/frame_idx'][idx]
        has_ground_truth_betas = f['auxiliary/has_ground_truth_betas'][idx]
        
        # Prepare x_data (single-view format compatible with existing pipeline)
        x_data = {
            'input_image': f"{session_name}/frame_{frame_idx:04d}",
            'input_image_data': image,
            'input_image_mask': None,
            'session_name': session_name,
            'frame_idx': frame_idx,
            'is_sleap_dataset': True,
            'is_multiview': False,
            'available_labels': {
                'global_rot': False,
                'joint_rot': False,
                'betas': bool(has_ground_truth_betas),
                'trans': False,
                'fov': False,
                'cam_rot': False,
                'cam_trans': False,
                'log_beta_scales': False,
                'betas_trans': False,
                'keypoint_2d': True,
                'keypoint_3d': False,
                'silhouette': False,
            },
        }
        
        # Prepare y_data (single-view format)
        y_data = {
            'keypoints_2d': keypoints_2d,
            'keypoint_visibility': keypoint_visibility,
            'shape_betas': betas if has_ground_truth_betas else None,
            'root_rot': None,
            'joint_angles': None,
            'root_loc': None,
            'cam_fov': None,
            'cam_rot': None,
            'cam_trans': None,
            'has_ground_truth_betas': has_ground_truth_betas,
        }
        
        return x_data, y_data
    
    def _decode_jpeg_image(self, jpeg_bytes) -> np.ndarray:
        """Decode JPEG bytes to image array."""
        if isinstance(jpeg_bytes, np.ndarray):
            jpeg_array = jpeg_bytes
        else:
            jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        
        image = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode JPEG image")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def get_max_views_in_dataset(self) -> int:
        """Return the maximum number of views across all samples."""
        return self.max_views
    
    def get_canonical_camera_order(self) -> List[str]:
        """Return the canonical ordering of cameras."""
        return self.canonical_camera_order
    
    def get_target_resolution(self) -> int:
        """Get the target resolution of the dataset."""
        return self.target_resolution
    
    def get_ue_scaling_flag(self) -> bool:
        """Get the UE scaling flag (always False for SLEAP data)."""
        return False
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        return {
            'dataset_type': self.dataset_type,
            'is_multiview': self.is_multiview,
            'num_samples': self.num_samples,
            'max_views': self.max_views,
            'n_joints': self.n_joints,
            'target_resolution': self.target_resolution,
            'backbone_name': self.backbone_name,
            'crop_mode': self.crop_mode,
            'n_pose': self.n_pose,
            'n_betas': self.n_betas,
            'min_views_per_sample': self.min_views_per_sample,
            'canonical_camera_order': self.canonical_camera_order,
            'rotation_representation': self.rotation_representation,
            'num_views_to_use': self.num_views_to_use,
        }
    
    def print_dataset_summary(self):
        """Print a summary of the dataset."""
        info = self.get_dataset_info()
        
        print("\n" + "="*50)
        print("MULTI-VIEW SLEAP DATASET SUMMARY")
        print("="*50)
        print(f"Dataset type: {info['dataset_type']}")
        print(f"Number of samples: {info['num_samples']}")
        print(f"Maximum views per sample: {info['max_views']}")
        print(f"Number of joints: {info['n_joints']}")
        print(f"Target resolution: {info['target_resolution']}x{info['target_resolution']}")
        print(f"Backbone: {info['backbone_name']}")
        print(f"Crop mode: {info['crop_mode']}")
        print(f"N_POSE: {info['n_pose']}")
        print(f"N_BETAS: {info['n_betas']}")
        print(f"Min views per sample: {info['min_views_per_sample']}")
        print(f"Canonical camera order: {info['canonical_camera_order']}")
        print(f"Num views to use: {info['num_views_to_use'] or 'all'}")
        print("="*50)
    
    def __del__(self):
        """Close file handle on deletion."""
        if hasattr(self, '_file') and self._file is not None:
            try:
                self._file.close()
            except:
                pass


def multiview_collate_fn(batch: List[Tuple[Dict, Dict]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Custom collate function for multi-view batches.
    
    Handles variable number of views per sample by keeping samples as lists
    rather than stacking into tensors.
    
    Args:
        batch: List of (x_data, y_data) tuples
        
    Returns:
        Tuple of (x_data_batch, y_data_batch) where each is a list of dicts
    """
    x_data_batch = []
    y_data_batch = []
    
    for x_data, y_data in batch:
        # Stack images into array if they're in list format
        if 'images' in x_data and isinstance(x_data['images'], list):
            # Keep as list for now - model will handle conversion
            pass
        
        x_data_batch.append(x_data)
        y_data_batch.append(y_data)
    
    return x_data_batch, y_data_batch


if __name__ == "__main__":
    # Test the multi-view dataset
    import argparse
    
    parser = argparse.ArgumentParser(description="Test multi-view SLEAP dataset")
    parser.add_argument("hdf5_path", help="Path to multi-view SLEAP HDF5 dataset")
    parser.add_argument("--num_views", type=int, default=None,
                       help="Number of views to use per sample")
    parser.add_argument("--single_view", action="store_true",
                       help="Test single-view mode")
    args = parser.parse_args()
    
    # Load dataset
    dataset = SLEAPMultiViewDataset(
        args.hdf5_path,
        num_views_to_use=args.num_views,
        return_single_view=args.single_view
    )
    
    # Print summary
    dataset.print_dataset_summary()
    
    # Test loading samples
    if len(dataset) > 0:
        print(f"\nTesting sample loading...")
        x_data, y_data = dataset[0]
        
        if args.single_view:
            print(f"Single-view sample 0:")
            print(f"  Image shape: {x_data['input_image_data'].shape}")
            print(f"  Keypoints 2D shape: {y_data['keypoints_2d'].shape}")
        else:
            print(f"Multi-view sample 0:")
            print(f"  Number of views: {x_data['num_active_views']}")
            print(f"  Image shapes: {[img.shape for img in x_data['images']]}")
            print(f"  Keypoints 2D shape: {y_data['keypoints_2d'].shape}")
            print(f"  Camera names: {x_data['camera_names']}")
            print(f"  Camera indices: {x_data['camera_indices']}")
        
        print(f"  Has ground truth betas: {y_data['has_ground_truth_betas']}")
    
    print("\nMulti-view SLEAP dataset test completed!")

