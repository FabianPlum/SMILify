"""
Optimized Dataset Classes for HDF5 SMIL Data

This module provides optimized dataset classes for loading preprocessed SMIL data
from HDF5 files. The classes are designed for maximum I/O efficiency during training.
"""

import os
import sys
import h5py
import numpy as np
import torch
import cv2
from typing import Dict, Tuple, Any, Optional
import threading

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class OptimizedSMILDataset(torch.utils.data.Dataset):
    """
    Optimized dataset class for loading preprocessed SMIL data from HDF5 files.
    
    This class provides fast loading of preprocessed data with minimal I/O overhead.
    It's designed to be a drop-in replacement for replicAntSMILDataset.
    """
    
    def __init__(self, 
                 hdf5_path: str,
                 mode: str = 'train',
                 rotation_representation: str = '6d',
                 backbone_name: str = 'vit_large_patch16_224'):
        """
        Initialize the optimized dataset.
        
        Args:
            hdf5_path: Path to the preprocessed HDF5 file
            mode: Dataset mode ('train', 'val', 'test') - currently unused but kept for compatibility
            rotation_representation: Rotation representation ('6d' or 'axis_angle')
            backbone_name: Backbone name for compatibility
        """
        self.hdf5_path = hdf5_path
        self.mode = mode
        self.rotation_representation = rotation_representation
        self.backbone_name = backbone_name
        
        # Open HDF5 file
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        
        # Load metadata
        self.metadata = dict(self.hdf5_file['metadata'].attrs)
        self.target_resolution = self.metadata['target_resolution']
        self.original_backbone = self.metadata['backbone_name']
        self.original_rotation_repr = self.metadata['rotation_representation']
        
        # Validate compatibility
        if self.rotation_representation != self.original_rotation_repr:
            print(f"Warning: Requested rotation representation '{self.rotation_representation}' "
                  f"differs from preprocessed '{self.original_rotation_repr}'. "
                  f"Using preprocessed representation.")
            self.rotation_representation = self.original_rotation_repr
        
        # Get dataset size
        self.num_samples = self.metadata['total_samples']
        
        # Access to data groups
        self.images = self.hdf5_file['images']
        self.parameters = self.hdf5_file['parameters']
        self.keypoints = self.hdf5_file['keypoints']
        self.auxiliary = self.hdf5_file['auxiliary']
        
        # Thread safety for HDF5 access
        self._lock = threading.Lock()
        
        print(f"Loaded optimized dataset: {self.num_samples} samples, "
              f"resolution {self.target_resolution}x{self.target_resolution}, "
              f"rotation: {self.rotation_representation}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (x_data, y_data) compatible with existing training pipeline
        """
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset size {self.num_samples}")
        
        with self._lock:
            # Load image data
            jpeg_bytes = self.images['rgb_jpeg'][idx]
            silhouette_mask = self.images['silhouette_masks'][idx]
            
            # Load parameters
            global_rot = self.parameters['global_rot'][idx]
            joint_rot = self.parameters['joint_rot'][idx]
            betas = self.parameters['betas'][idx]
            trans = self.parameters['trans'][idx]
            fov = self.parameters['fov'][idx]
            cam_rot = self.parameters['cam_rot'][idx]
            cam_trans = self.parameters['cam_trans'][idx]
            log_beta_scales = self.parameters['log_beta_scales'][idx]
            betas_trans = self.parameters['betas_trans'][idx]
            
            # Load keypoints
            keypoints_2d = self.keypoints['keypoints_2d'][idx]
            keypoints_3d = self.keypoints['keypoints_3d'][idx]
            keypoint_visibility = self.keypoints['keypoint_visibility'][idx]
            
            # Load auxiliary data
            original_path = self.auxiliary['original_paths'][idx]
        
        # Decode JPEG image
        rgb_image = self._decode_jpeg_image(jpeg_bytes)
        
        # Prepare x_data (input data)
        x_data = {
            'input_image': original_path,  # Keep original path for compatibility
            'input_image_data': rgb_image,
            'input_image_mask': silhouette_mask.squeeze(0)  # Remove channel dimension
        }
        
        # Prepare y_data (target data)
        y_data = {
            # Original data structure for compatibility
            'pose_data': {},  # Empty for now, not used in current training
            
            # Processed parameters
            'root_rot': global_rot,
            'joint_angles': self._reconstruct_joint_angles(joint_rot),
            'shape_betas': betas,
            'root_loc': trans,
            'cam_fov': fov if fov.ndim == 0 else fov[0],  # Handle scalar vs array
            'cam_rot': cam_rot,
            'cam_trans': cam_trans,
            
            # Optional parameters
            'scale_weights': log_beta_scales if not np.allclose(log_beta_scales, 0) else None,
            'trans_weights': betas_trans if not np.allclose(betas_trans, 0) else None,
            
            # Keypoint data
            'keypoints_2d': keypoints_2d,
            'keypoints_3d': keypoints_3d,
            'keypoint_visibility': keypoint_visibility,
            
            # Additional compatibility fields
            'propagate_scaling': True,
            'translation_factor': 0.01
        }
        
        return x_data, y_data
    
    def _decode_jpeg_image(self, jpeg_bytes: np.ndarray) -> np.ndarray:
        """
        Decode JPEG bytes to image array.
        
        Args:
            jpeg_bytes: JPEG encoded bytes
            
        Returns:
            Image array (H, W, C) in range [0, 1]
        """
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
        Reconstruct full joint angles array including root joint.
        
        Args:
            joint_rot: Joint rotations excluding root (N-1, rot_dim)
            
        Returns:
            Full joint angles including root (N, rot_dim)
        """
        # Add zero rotation for root joint at the beginning
        if self.rotation_representation == '6d':
            root_rot = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)  # Identity in 6D
            full_joint_angles = np.vstack([root_rot.reshape(1, -1), joint_rot])
        else:
            root_rot = np.zeros(3, dtype=np.float32)  # Zero rotation in axis-angle
            full_joint_angles = np.vstack([root_rot.reshape(1, -1), joint_rot])
        
        return full_joint_angles
    
    def get_ue_scaling_flag(self) -> bool:
        """
        Get the UE scaling flag for compatibility.
        
        Returns:
            Always True for replicAnt data
        """
        return True
    
    def get_input_resolution(self) -> int:
        """Get the input resolution."""
        return self.target_resolution
    
    def get_target_resolution(self) -> int:
        """Get the target resolution."""
        return self.target_resolution
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata information for a specific sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing sample metadata
        """
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset size {self.num_samples}")
        
        with self._lock:
            info = {
                'index': idx,
                'original_path': self.auxiliary['original_paths'][idx],
                'silhouette_coverage': float(self.auxiliary['silhouette_coverage'][idx]),
                'visible_keypoints': int(self.auxiliary['visible_keypoints'][idx]),
                'resolution': self.target_resolution,
                'rotation_representation': self.rotation_representation
            }
        
        return info
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get overall dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = dict(self.metadata)
        
        # Add statistics from preprocessing
        if 'statistics' in self.hdf5_file['metadata']:
            preprocessing_stats = dict(self.hdf5_file['metadata']['statistics'].attrs)
            stats.update(preprocessing_stats)
        
        return stats
    
    def close(self):
        """Close the HDF5 file."""
        if hasattr(self, 'hdf5_file') and self.hdf5_file is not None:
            self.hdf5_file.close()
    
    def __del__(self):
        """Destructor to ensure HDF5 file is closed."""
        self.close()


class HDF5DatasetValidator:
    """
    Utility class for validating HDF5 datasets and comparing with original data.
    """
    
    def __init__(self, hdf5_path: str):
        """
        Initialize the validator.
        
        Args:
            hdf5_path: Path to HDF5 file to validate
        """
        self.hdf5_path = hdf5_path
        self.dataset = OptimizedSMILDataset(hdf5_path)
    
    def validate_sample(self, idx: int, verbose: bool = True) -> Dict[str, bool]:
        """
        Validate a specific sample.
        
        Args:
            idx: Sample index to validate
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary of validation results
        """
        try:
            x_data, y_data = self.dataset[idx]
            sample_info = self.dataset.get_sample_info(idx)
            
            results = {
                'image_loaded': x_data['input_image_data'] is not None,
                'image_shape_correct': x_data['input_image_data'].shape == (self.dataset.target_resolution, self.dataset.target_resolution, 3),
                'image_range_correct': 0.0 <= x_data['input_image_data'].min() and x_data['input_image_data'].max() <= 1.0,
                'mask_loaded': x_data['input_image_mask'] is not None,
                'parameters_present': all(key in y_data for key in ['root_rot', 'joint_angles', 'shape_betas'])
            }
            
            if verbose:
                print(f"Sample {idx} validation:")
                for key, value in results.items():
                    status = "✓" if value else "✗"
                    print(f"  {key}: {status}")
                print(f"  Original path: {sample_info['original_path']}")
                print(f"  Silhouette coverage: {sample_info['silhouette_coverage']:.3f}")
                print(f"  Visible keypoints: {sample_info['visible_keypoints']}")
            
            return results
            
        except Exception as e:
            if verbose:
                print(f"Error validating sample {idx}: {e}")
            return {'error': str(e)}
    
    def validate_dataset(self, num_samples: int = 10) -> Dict[str, Any]:
        """
        Validate multiple samples from the dataset.
        
        Args:
            num_samples: Number of samples to validate
            
        Returns:
            Dictionary containing validation summary
        """
        print(f"Validating {num_samples} samples from dataset...")
        
        total_samples = len(self.dataset)
        sample_indices = np.linspace(0, total_samples - 1, num_samples, dtype=int)
        
        all_results = []
        for idx in sample_indices:
            result = self.validate_sample(idx, verbose=False)
            all_results.append(result)
        
        # Summarize results
        summary = {
            'total_validated': len(all_results),
            'successful_validations': sum(1 for r in all_results if 'error' not in r),
            'failed_validations': sum(1 for r in all_results if 'error' in r)
        }
        
        # Count specific validation checks
        if summary['successful_validations'] > 0:
            successful_results = [r for r in all_results if 'error' not in r]
            for check_name in successful_results[0].keys():
                passed = sum(1 for r in successful_results if r.get(check_name, False))
                summary[f'{check_name}_pass_rate'] = passed / len(successful_results)
        
        return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test optimized SMIL dataset")
    parser.add_argument("hdf5_path", help="Path to HDF5 dataset file")
    parser.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to validate")
    parser.add_argument("--sample_idx", type=int, help="Specific sample index to test")
    
    args = parser.parse_args()
    
    if args.validate:
        validator = HDF5DatasetValidator(args.hdf5_path)
        
        if args.sample_idx is not None:
            # Validate specific sample
            validator.validate_sample(args.sample_idx, verbose=True)
        else:
            # Validate multiple samples
            summary = validator.validate_dataset(args.num_samples)
            print("\nValidation Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
    
    else:
        # Simple dataset test
        dataset = OptimizedSMILDataset(args.hdf5_path)
        print(f"Dataset loaded: {len(dataset)} samples")
        
        # Test loading first sample
        x_data, y_data = dataset[0]
        print(f"Sample 0 image shape: {x_data['input_image_data'].shape}")
        print(f"Sample 0 keypoints shape: {y_data['keypoints_2d'].shape}")
        
        # Print dataset statistics
        stats = dataset.get_dataset_statistics()
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        dataset.close()


