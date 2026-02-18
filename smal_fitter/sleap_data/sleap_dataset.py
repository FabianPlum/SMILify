"""
SLEAP Dataset Class for SMILify Training Pipeline

This module provides a PyTorch dataset class for loading preprocessed SLEAP datasets
that is compatible with the UnifiedSMILDataset interface.
"""

import os
import torch
import h5py
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config


class SLEAPDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset class for loading preprocessed SLEAP datasets.
    
    This class loads HDF5 files created by the SLEAP dataset preprocessor
    and provides data in the same format as the existing SMILify datasets.
    """
    
    def __init__(self, hdf5_path: str, rotation_representation: str = '6d', backbone_name: str = None, **kwargs):
        """
        Initialize the SLEAP dataset.
        
        Args:
            hdf5_path: Path to the preprocessed HDF5 file
            rotation_representation: Rotation representation ('6d' or 'axis_angle')
            backbone_name: Backbone name (for compatibility, not used)
            **kwargs: Additional keyword arguments (for compatibility with other datasets)
        """
        self.hdf5_path = hdf5_path
        self.rotation_representation = rotation_representation
        
        # Load dataset metadata
        self._load_metadata()
        
        # Validate dataset
        self._validate_dataset()
    
    def _load_metadata(self):
        """Load dataset metadata from HDF5 file."""
        with h5py.File(self.hdf5_path, 'r') as f:
            metadata = f['metadata']
            
            self.num_samples = metadata.attrs['num_samples']
            self.target_resolution = metadata.attrs['target_resolution']
            self.backbone_name = metadata.attrs['backbone_name']
            self.dataset_type = metadata.attrs['dataset_type']
            self.is_sleap_dataset = metadata.attrs.get('is_sleap_dataset', True)  # Flag for SLEAP-specific loss computation
            self.n_pose = metadata.attrs['n_pose']
            self.n_betas = metadata.attrs['n_betas']
            self.joint_lookup_table_used = metadata.attrs['joint_lookup_table_used']
            self.shape_betas_table_used = metadata.attrs['shape_betas_table_used']
            self.crop_mode = metadata.attrs.get('crop_mode', 'default')  # Crop mode used during preprocessing
            
            # Processing statistics
            self.total_sessions = metadata.attrs.get('total_sessions', 0)
            self.sessions_processed = metadata.attrs.get('sessions_processed', 0)
            self.sessions_failed = metadata.attrs.get('sessions_failed', 0)
            self.processed_samples = metadata.attrs.get('processed_samples', 0)
            self.failed_samples = metadata.attrs.get('failed_samples', 0)
    
    def _validate_dataset(self):
        """Validate that the dataset is properly formatted."""
        with h5py.File(self.hdf5_path, 'r') as f:
            # Check required groups
            required_groups = ['images', 'parameters', 'keypoints', 'auxiliary', 'metadata']
            for group in required_groups:
                if group not in f:
                    raise ValueError(f"Missing required group: {group}")
            
            # Check required datasets
            required_datasets = {
                'images': ['image_jpeg', 'mask'],
                'parameters': ['global_rot', 'joint_rot', 'betas', 'trans', 'fov', 'cam_rot', 'cam_trans'],
                'keypoints': ['keypoints_2d', 'keypoints_3d', 'keypoint_visibility'],
                'auxiliary': ['original_path', 'session_name', 'camera_name', 'frame_idx']
            }
            
            for group, datasets in required_datasets.items():
                for dataset in datasets:
                    if dataset not in f[group]:
                        raise ValueError(f"Missing required dataset: {group}/{dataset}")
            
            # Validate dimensions
            images_group = f['images']
            parameters_group = f['parameters']
            keypoints_group = f['keypoints']
            
            # Check that all datasets have the same number of samples
            num_samples_expected = self.num_samples
            
            for group in [images_group, parameters_group, keypoints_group]:
                for dataset_name in group.keys():
                    dataset = group[dataset_name]
                    if len(dataset) != num_samples_expected:
                        raise ValueError(f"Dataset {group.name}/{dataset_name} has {len(dataset)} samples, expected {num_samples_expected}")
    
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
            - x_data: Input data dictionary
            - y_data: Target data dictionary
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            # Load image data
            image_jpeg_bytes = f['images/image_jpeg'][idx]
            mask = f['images/mask'][idx]
            
            # Decode JPEG image
            image = self._decode_jpeg_image(image_jpeg_bytes)
            
            # Load parameter data
            global_rot = f['parameters/global_rot'][idx]
            joint_rot = f['parameters/joint_rot'][idx]
            betas = f['parameters/betas'][idx]
            trans = f['parameters/trans'][idx]
            fov = f['parameters/fov'][idx]
            cam_rot = f['parameters/cam_rot'][idx]
            cam_trans = f['parameters/cam_trans'][idx]
            scale_weights = f['parameters/scale_weights'][idx]
            trans_weights = f['parameters/trans_weights'][idx]
            
            # Load keypoint data
            keypoints_2d = f['keypoints/keypoints_2d'][idx]
            keypoints_3d = f['keypoints/keypoints_3d'][idx]
            keypoint_visibility = f['keypoints/keypoint_visibility'][idx]
            
            # Load auxiliary data
            original_path = f['auxiliary/original_path'][idx].decode('utf-8')
            session_name = f['auxiliary/session_name'][idx].decode('utf-8')
            camera_name = f['auxiliary/camera_name'][idx].decode('utf-8')
            frame_idx = f['auxiliary/frame_idx'][idx]
            silhouette_coverage = f['auxiliary/silhouette_coverage'][idx]
            visible_keypoints = f['auxiliary/visible_keypoints'][idx]
            has_ground_truth_betas = f['auxiliary/has_ground_truth_betas'][idx]
            # Optional transform info
            crop_offset_yx = f['auxiliary/crop_offset_yx'][idx] if 'crop_offset_yx' in f['auxiliary'] else np.array([0, 0], dtype=np.int32)
            crop_size_hw = f['auxiliary/crop_size_hw'][idx] if 'crop_size_hw' in f['auxiliary'] else np.array([self.target_resolution, self.target_resolution], dtype=np.int32)
            scale_yx = f['auxiliary/scale_yx'][idx] if 'scale_yx' in f['auxiliary'] else np.array([1.0, 1.0], dtype=np.float32)
        
        # Prepare x_data (input data)
        x_data = {
            'input_image': original_path,  # For compatibility
            'input_image_data': image,  # Preprocessed image (H, W, C) in range [0, 1]
            'input_image_mask': None,  # No silhouette ground truth for SLEAP
            'session_name': session_name,
            'camera_name': camera_name,
            'frame_idx': frame_idx,
            'is_sleap_dataset': self.is_sleap_dataset,  # Flag for SLEAP-specific loss computation
            # Optional transform info
            'crop_offset_yx': crop_offset_yx,
            'crop_size_hw': crop_size_hw,
            'scale_yx': scale_yx,
            # Per-sample available label flags to guide loss masking
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
        
        # Convert rotations if needed
        if self.rotation_representation == '6d':
            # Convert from axis-angle to 6D representation
            import torch
            from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
            
            # Convert global rotation
            global_rot_tensor = torch.from_numpy(global_rot).float()
            global_rot_matrix = axis_angle_to_matrix(global_rot_tensor)
            global_rot_6d = matrix_to_rotation_6d(global_rot_matrix)
            global_rot = global_rot_6d.numpy()
            
            # Convert joint rotations
            joint_rot_tensor = torch.from_numpy(joint_rot).float()
            joint_rot_matrix = axis_angle_to_matrix(joint_rot_tensor)
            joint_rot_6d = matrix_to_rotation_6d(joint_rot_matrix)
            joint_rot = joint_rot_6d.numpy()
        
        # Prepare y_data (target data)
        y_data = {
            # Original data structure for compatibility
            'pose_data': {},  # Empty for SLEAP data
            
            # Processed parameters (mark unavailable ones as None)
            'root_rot': None,
            'joint_angles': None,  # Not available for SLEAP
            'shape_betas': betas if has_ground_truth_betas else None,
            'root_loc': None,
            'cam_fov': None,
            'cam_rot': None,
            'cam_trans': None,
            
            # Optional parameters (unused for SLEAP)
            'scale_weights': None,
            'trans_weights': None,
            
            # Keypoint data (actual SLEAP data)
            'keypoints_2d': keypoints_2d,
            'keypoints_3d': keypoints_3d,
            'keypoint_visibility': keypoint_visibility,
            
            # Additional compatibility fields
            'propagate_scaling': True,
            'translation_factor': 0.01,
            
            # SLEAP-specific metadata
            'has_ground_truth_betas': has_ground_truth_betas,
            'visible_keypoints_count': visible_keypoints,
            'silhouette_coverage': silhouette_coverage
        }
        
        return x_data, y_data
    
    def _decode_jpeg_image(self, jpeg_bytes) -> np.ndarray:
        """
        Decode JPEG bytes to image array.
        
        Args:
            jpeg_bytes: JPEG encoded bytes (can be numpy array or bytes)
            
        Returns:
            Image array (H, W, C) in range [0, 1]
        """
        # Handle different input types (numpy array or bytes)
        if isinstance(jpeg_bytes, np.ndarray):
            # If it's already a numpy array, use it directly
            jpeg_array = jpeg_bytes
        else:
            # Convert bytes to numpy array
            jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        
        # Decode JPEG
        image = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode JPEG image")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def _reconstruct_joint_angles(self, joint_rot: np.ndarray) -> np.ndarray:
        """
        Reconstruct joint angles from stored format.
        
        Args:
            joint_rot: Joint rotations in stored format (N_POSE + 1, rot_dim) including root joint
            
        Returns:
            Joint angles in the format expected by the training pipeline
        """
        # For SLEAP data, joint_rot is now stored with root joint included for consistency
        # (N_POSE + 1, 3) for axis-angle or (N_POSE + 1, 6) for 6D representation
        return joint_rot
    
    def get_input_resolution(self) -> int:
        """Get the input resolution of the dataset."""
        return self.target_resolution
    
    def get_target_resolution(self) -> int:
        """Get the target resolution of the dataset."""
        return self.target_resolution
    
    def get_ue_scaling_flag(self) -> bool:
        """Get the UE scaling flag (always False for SLEAP data)."""
        return False  # SLEAP data doesn't use UE scaling
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            'dataset_type': self.dataset_type,
            'num_samples': self.num_samples,
            'target_resolution': self.target_resolution,
            'backbone_name': self.backbone_name,
            'crop_mode': self.crop_mode,
            'n_pose': self.n_pose,
            'n_betas': self.n_betas,
            'joint_lookup_table_used': self.joint_lookup_table_used,
            'shape_betas_table_used': self.shape_betas_table_used,
            'total_sessions': self.total_sessions,
            'sessions_processed': self.sessions_processed,
            'sessions_failed': self.sessions_failed,
            'processed_samples': self.processed_samples,
            'failed_samples': self.failed_samples,
            'rotation_representation': self.rotation_representation
        }
    
    def print_dataset_summary(self):
        """Print a summary of the dataset."""
        info = self.get_dataset_info()
        
        print("\n" + "="*50)
        print("SLEAP DATASET SUMMARY")
        print("="*50)
        print(f"Dataset type: {info['dataset_type']}")
        print(f"Number of samples: {info['num_samples']}")
        print(f"Target resolution: {info['target_resolution']}x{info['target_resolution']}")
        print(f"Backbone: {info['backbone_name']}")
        print(f"Crop mode: {info['crop_mode']}")
        print(f"Rotation representation: {info['rotation_representation']}")
        print(f"N_POSE: {info['n_pose']}")
        print(f"N_BETAS: {info['n_betas']}")
        print(f"Joint lookup table used: {info['joint_lookup_table_used']}")
        print(f"Shape betas table used: {info['shape_betas_table_used']}")
        print(f"UE scaling: {self.get_ue_scaling_flag()}")
        
        print("\nProcessing Statistics:")
        print(f"  Total sessions: {info['total_sessions']}")
        print(f"  Sessions processed: {info['sessions_processed']}")
        print(f"  Sessions failed: {info['sessions_failed']}")
        print(f"  Processed samples: {info['processed_samples']}")
        print(f"  Failed samples: {info['failed_samples']}")
        print("="*50)


# Add SLEAP dataset support to UnifiedSMILDataset
def _patch_unified_dataset(verbose: bool = False):
    """
    Patch the UnifiedSMILDataset to support SLEAP datasets.
    This function should be called after importing this module.
    """
    if verbose:
        print("Patching UnifiedSMILDataset to support SLEAP datasets...")
    try:
        import smal_fitter.neuralSMIL.smil_datasets as smil_datasets
    except ImportError:
        # Fallback for when running from different directory
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'neuralSMIL'))
        import smil_datasets

    # Check if already patched
    if hasattr(smil_datasets.UnifiedSMILDataset.from_path, '_sleap_patched'):
        if verbose:
            print("UnifiedSMILDataset already patched, skipping...")
        return
    
    # Store original from_path method
    original_from_path = smil_datasets.UnifiedSMILDataset.from_path
    
    @staticmethod
    def from_path_with_sleap(data_path: str, **kwargs):
        """
        Create appropriate dataset instance based on file path.
        
        Args:
            data_path: Path to dataset (directory for JSON format, .h5/.hdf5 file for optimized format)
            **kwargs: Additional arguments passed to dataset constructor
            
        Returns:
            Dataset instance (OptimizedSMILDataset, replicAntSMILDataset, or SLEAPDataset)
        """
        if data_path.endswith('.h5') or data_path.endswith('.hdf5'):
            # Check if it's a SLEAP dataset by examining metadata
            try:
                with h5py.File(data_path, 'r') as f:
                    if 'metadata' in f and 'dataset_type' in f['metadata'].attrs:
                        dataset_type = f['metadata'].attrs['dataset_type']
                        if dataset_type == 'sleap':
                            return SLEAPDataset(data_path, **kwargs)
            except Exception as e:
                pass  # Fall back to original logic
            
            # Load optimized HDF5 dataset (original logic)
            from optimized_dataset import OptimizedSMILDataset
            return OptimizedSMILDataset(data_path, **kwargs)
        else:
            # Load original JSON dataset
            return smil_datasets.replicAntSMILDataset(data_path, **kwargs)
    
    # Replace the method
    from_path_with_sleap._sleap_patched = True
    smil_datasets.UnifiedSMILDataset.from_path = from_path_with_sleap
    if verbose:
        print("UnifiedSMILDataset.from_path method patched successfully")


# Note: Patching is done manually in train_smil_regressor.py
# _patch_unified_dataset()


if __name__ == "__main__":
    # Test the SLEAP dataset
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SLEAP dataset")
    parser.add_argument("hdf5_path", help="Path to SLEAP HDF5 dataset")
    parser.add_argument("--rotation_representation", choices=['6d', 'axis_angle'], default='6d',
                       help="Rotation representation")
    args = parser.parse_args()
    
    # Load dataset
    dataset = SLEAPDataset(args.hdf5_path, args.rotation_representation)
    
    # Print summary
    dataset.print_dataset_summary()
    
    # Test loading a sample
    if len(dataset) > 0:
        print(f"\nTesting sample loading...")
        x_data, y_data = dataset[0]
        
        print(f"Sample 0:")
        print(f"  Image shape: {x_data['input_image_data'].shape}")
        print(f"  Mask shape: {x_data['input_image_mask'].shape}")
        print(f"  Keypoints 2D shape: {y_data['keypoints_2d'].shape}")
        print(f"  Keypoint visibility shape: {y_data['keypoint_visibility'].shape}")
        print(f"  Visible keypoints: {y_data['visible_keypoints_count']}")
        print(f"  Has ground truth betas: {y_data['has_ground_truth_betas']}")
        print(f"  Session: {x_data['session_name']}")
        print(f"  Camera: {x_data['camera_name']}")
        print(f"  Frame: {x_data['frame_idx']}")
    
    print("\nSLEAP dataset test completed!")
