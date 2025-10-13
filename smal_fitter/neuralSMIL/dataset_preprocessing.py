"""
Dataset Preprocessing Pipeline for SMIL Training

This module provides functionality to preprocess replicAnt SMIL datasets into
optimized HDF5 format for faster training. The preprocessing includes:
1. Loading and parsing all samples using Unreal2Pytorch3D
2. Filtering samples based on silhouette coverage
3. Resizing images and masks to target resolution
4. Precomputing keypoint visibility
5. Storing in HDF5 format with batch-oriented chunking
"""

import os
import sys
import h5py
import numpy as np
import cv2
import json
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import argparse
from pathlib import Path
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Unreal2Pytorch3D import load_SMIL_Unreal_sample, compute_keypoint_visibility
import config


class DatasetPreprocessor:
    """
    Preprocesses replicAnt SMIL datasets into optimized HDF5 format.
    """
    
    def __init__(self, 
                 silhouette_threshold: float = 0.1,
                 target_resolution: int = 224,
                 backbone_name: str = 'vit_large_patch16_224',
                 rotation_representation: str = '6d',
                 min_visible_keypoints: int = 5,
                 chunk_size: int = 8,
                 compression: str = 'gzip',
                 compression_level: int = 6,
                 jpeg_quality: int = 95,
                 ignored_joints_config: dict = None):
        """
        Initialize the dataset preprocessor.
        
        Args:
            silhouette_threshold: Minimum fraction of image that must be occupied by subject
            target_resolution: Target image resolution (224 for ViT, 512 for ResNet)
            backbone_name: Backbone network name for resolution selection
            rotation_representation: '6d' or 'axis_angle'
            min_visible_keypoints: Minimum number of visible keypoints required
            chunk_size: Number of samples per HDF5 chunk (batch size)
            compression: HDF5 compression type
            compression_level: Compression level (1-9)
            jpeg_quality: JPEG compression quality for images (1-100)
            ignored_joints_config: Dictionary containing ignored joints configuration
        """
        self.silhouette_threshold = silhouette_threshold
        self.target_resolution = target_resolution
        self.backbone_name = backbone_name
        self.rotation_representation = rotation_representation
        self.min_visible_keypoints = min_visible_keypoints
        self.chunk_size = chunk_size
        self.compression = compression
        self.compression_level = compression_level
        self.jpeg_quality = jpeg_quality
        self.ignored_joints_config = ignored_joints_config or {'ignored_joint_names': [], 'verbose_ignored_joints': False}
        
        # Determine target resolution based on backbone
        if backbone_name.startswith('vit'):
            self.target_resolution = 224
        else:
            self.target_resolution = target_resolution
        
        # Statistics tracking
        self.stats = {
            'total_samples': 0,
            'filtered_samples': 0,
            'silhouette_filtered': 0,
            'keypoint_filtered': 0,
            'error_samples': 0,
            'final_samples': 0
        }
        
        # Thread-safe progress tracking
        self._lock = threading.Lock()
        
        # Create joint name to index mapping for ignored joints
        self._setup_ignored_joints_mapping()
    
    def _setup_ignored_joints_mapping(self):
        """Setup mapping from joint names to indices for ignored joints."""
        import config
        
        # Get joint names from config
        if hasattr(config, 'dd') and 'J_names' in config.dd:
            joint_names = config.dd['J_names']
            self.joint_name_to_index = {name: idx for idx, name in enumerate(joint_names)}
            
            # Find indices of ignored joints
            ignored_joint_names = self.ignored_joints_config.get('ignored_joint_names', [])
            self.ignored_joint_indices = []
            
            for joint_name in ignored_joint_names:
                if joint_name in self.joint_name_to_index:
                    self.ignored_joint_indices.append(self.joint_name_to_index[joint_name])
                    if self.ignored_joints_config.get('verbose_ignored_joints', False):
                        print(f"Will ignore joint '{joint_name}' (index {self.joint_name_to_index[joint_name]}) during preprocessing")
                else:
                    print(f"Warning: Ignored joint '{joint_name}' not found in model joint names")
            
            if self.ignored_joints_config.get('verbose_ignored_joints', False) and ignored_joint_names:
                print(f"Total ignored joints: {len(self.ignored_joint_indices)}")
        else:
            print("Warning: Could not load joint names from config, ignored joints will not be applied")
            self.joint_name_to_index = {}
            self.ignored_joint_indices = []
    
    def discover_samples(self, input_dir: str) -> List[str]:
        """
        Discover all JSON files in the input directory.
        
        Args:
            input_dir: Path to input dataset directory
            
        Returns:
            List of JSON file paths
        """
        json_files = []
        input_path = Path(input_dir)
        
        for json_file in input_path.glob("*.json"):
            # Skip batch data files
            if not json_file.name.startswith('_BatchData'):
                json_files.append(str(json_file))
        
        json_files.sort()  # Ensure consistent ordering
        return json_files
    
    def process_single_sample(self, json_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single sample and return processed data or None if filtered.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Dictionary containing processed sample data or None if filtered
        """
        try:
            # Load sample using existing Unreal2Pytorch3D function
            x_data, y_data = load_SMIL_Unreal_sample(
                json_path,
                plot_tests=False,
                propagate_scaling=True,
                translation_factor=0.01,
                load_image=True,
                verbose=False
            )
            
            with self._lock:
                self.stats['total_samples'] += 1
            
            # Check if image and mask are available
            if x_data['input_image_data'] is None:
                with self._lock:
                    self.stats['error_samples'] += 1
                return None
            
            # Apply silhouette coverage filter
            if not self._check_silhouette_coverage(x_data['input_image_mask']):
                with self._lock:
                    self.stats['silhouette_filtered'] += 1
                return None
            
            # Precompute keypoint visibility using dilated mask (same as training pipeline)
            visibility = self._compute_keypoint_visibility(
                y_data['keypoints_2d'], 
                x_data['input_image_mask']
            )
            
            # Apply keypoint visibility filter
            if not self._check_keypoint_visibility(visibility):
                with self._lock:
                    self.stats['keypoint_filtered'] += 1
                return None
            
            # Preprocess image and mask
            processed_image = self._preprocess_image(x_data['input_image_data'])
            processed_mask = self._preprocess_mask(x_data['input_image_mask'])
            
            # Convert image to JPEG bytes for storage efficiency
            jpeg_image = self._encode_image_jpeg(processed_image)
            
            # Prepare processed sample data
            sample_data = {
                # Images (as JPEG bytes)
                'image_jpeg': jpeg_image,
                'mask': processed_mask.astype(np.uint8),
                
                # Parameters (converted to target representation)
                'global_rot': self._convert_rotation(y_data['root_rot']),
                'joint_rot': self._convert_joint_rotations(y_data['joint_angles']),
                'betas': np.array(y_data['shape_betas'], dtype=np.float32),
                'trans': np.array(y_data['root_loc'], dtype=np.float32),
                'fov': np.array(y_data['cam_fov'], dtype=np.float32),
                'cam_rot': np.array(y_data['cam_rot'], dtype=np.float32),
                'cam_trans': np.array(y_data['cam_trans'], dtype=np.float32),
                
                # Keypoints (use precomputed visibility)
                'keypoints_2d': np.array(y_data['keypoints_2d'], dtype=np.float32),
                'keypoints_3d': np.array(y_data['keypoints_3d'], dtype=np.float32),
                'keypoint_visibility': visibility.astype(np.float32),
                
                # Optional parameters (if available)
                'scale_weights': np.array(y_data.get('scale_weights', np.zeros(config.N_BETAS)), dtype=np.float32),
                'trans_weights': np.array(y_data.get('trans_weights', np.zeros(config.N_BETAS)), dtype=np.float32),
                
                # Metadata
                'original_path': json_path,
                'silhouette_coverage': self._compute_silhouette_coverage(x_data['input_image_mask']),
                'visible_keypoints': int(np.sum(visibility > 0.5))
            }
            
            with self._lock:
                self.stats['final_samples'] += 1
            
            return sample_data
            
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            with self._lock:
                self.stats['error_samples'] += 1
            return None
    
    def _check_silhouette_coverage(self, mask: Optional[np.ndarray]) -> bool:
        """Check if silhouette coverage meets threshold."""
        if mask is None:
            return False
        
        coverage = self._compute_silhouette_coverage(mask)
        return coverage >= self.silhouette_threshold
    
    def _compute_silhouette_coverage(self, mask: np.ndarray) -> float:
        """Compute the fraction of image covered by silhouette."""
        if mask is None:
            return 0.0
        
        # Convert to binary if needed
        if mask.max() > 1.0:
            binary_mask = (mask > 127).astype(np.uint8)
        else:
            binary_mask = (mask > 0.5).astype(np.uint8)
        
        total_pixels = binary_mask.size
        subject_pixels = np.sum(binary_mask)
        
        return subject_pixels / total_pixels if total_pixels > 0 else 0.0
    
    def _compute_keypoint_visibility(self, keypoints_2d: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Compute keypoint visibility using dilated silhouette mask.
        
        This replicates the same visibility computation used in training to ensure consistency.
        
        Args:
            keypoints_2d: 2D keypoints array (N, 2)
            mask: Silhouette mask (H, W) or None
            
        Returns:
            Visibility array (N,) with values 0.0 or 1.0
        """
        if mask is None:
            # No mask available - mark all keypoints as invisible
            return np.zeros(len(keypoints_2d), dtype=np.float32)
        
        # Import here to avoid circular dependencies
        from Unreal2Pytorch3D import compute_keypoint_visibility
        
        # Use the same visibility computation as in training
        # Note: mask is already dilated when loaded by Unreal2Pytorch3D
        visibility = compute_keypoint_visibility(
            keypoints_2d, 
            mask, 
            image_width=mask.shape[1] if mask is not None else 1024,
            image_height=mask.shape[0] if mask is not None else 1024
        )
        
        # Apply ignored joints - set visibility to 0 for ignored joints
        for ignored_idx in self.ignored_joint_indices:
            if ignored_idx < len(visibility):
                visibility[ignored_idx] = 0.0
        
        return visibility
    
    def _check_keypoint_visibility(self, visibility: np.ndarray) -> bool:
        """Check if minimum number of keypoints are visible."""
        visible_count = np.sum(visibility > 0.5)
        return visible_count >= self.min_visible_keypoints
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to target resolution and normalize.
        
        Args:
            image: Input image array (H, W, C) in range [0, 255]
            
        Returns:
            Preprocessed image array (C, H, W) in range [0, 1]
        """
        # Resize to target resolution
        if image.shape[:2] != (self.target_resolution, self.target_resolution):
            image = cv2.resize(image, (self.target_resolution, self.target_resolution))
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Convert from HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def _preprocess_mask(self, mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Preprocess silhouette mask to target resolution.
        
        Args:
            mask: Input mask array (H, W) or None
            
        Returns:
            Preprocessed mask array (1, H, W) in range [0, 1]
        """
        if mask is None:
            # Create empty mask
            mask = np.zeros((self.target_resolution, self.target_resolution), dtype=np.float32)
        else:
            # Resize to target resolution
            if mask.shape[:2] != (self.target_resolution, self.target_resolution):
                mask = cv2.resize(mask, (self.target_resolution, self.target_resolution))
            
            # Normalize to [0, 1]
            if mask.max() > 1.0:
                mask = mask.astype(np.float32) / 255.0
            else:
                mask = mask.astype(np.float32)
        
        # Add channel dimension
        mask = np.expand_dims(mask, axis=0)
        
        return mask
    
    def _encode_image_jpeg(self, image: np.ndarray) -> bytes:
        """
        Encode image as JPEG bytes for storage efficiency.
        
        Args:
            image: Image array (C, H, W) in range [0, 1]
            
        Returns:
            JPEG encoded bytes
        """
        # Convert from CHW to HWC and scale to [0, 255]
        image_hwc = np.transpose(image, (1, 2, 0))
        image_uint8 = (image_hwc * 255).astype(np.uint8)
        
        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        success, encoded_img = cv2.imencode('.jpg', image_uint8, encode_param)
        
        if not success:
            raise ValueError("Failed to encode image as JPEG")
        
        return encoded_img.tobytes()
    
    def _convert_rotation(self, rotation: np.ndarray) -> np.ndarray:
        """Convert rotation to target representation."""
        if self.rotation_representation == '6d':
            # Convert axis-angle to 6D representation
            from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
            import torch
            
            rot_tensor = torch.from_numpy(rotation.astype(np.float32))
            rot_matrix = axis_angle_to_matrix(rot_tensor)
            rot_6d = matrix_to_rotation_6d(rot_matrix)
            return rot_6d.numpy().astype(np.float32)
        else:
            # Keep as axis-angle
            return rotation.astype(np.float32)
    
    def _convert_joint_rotations(self, joint_rotations: np.ndarray) -> np.ndarray:
        """Convert joint rotations to target representation."""
        if self.rotation_representation == '6d':
            # Convert axis-angle to 6D representation
            from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
            import torch
            
            # Exclude root joint (first joint)
            joint_rots = joint_rotations[1:].astype(np.float32)
            rot_tensor = torch.from_numpy(joint_rots)
            rot_matrices = axis_angle_to_matrix(rot_tensor)
            rot_6d = matrix_to_rotation_6d(rot_matrices)
            return rot_6d.numpy().astype(np.float32)
        else:
            # Keep as axis-angle, exclude root joint
            return joint_rotations[1:].astype(np.float32)
    
    def process_dataset(self, 
                       input_dir: str, 
                       output_path: str,
                       num_workers: int = 4,
                       batch_size: int = 32) -> Dict[str, Any]:
        """
        Process entire dataset and save to HDF5.
        
        Args:
            input_dir: Input dataset directory
            output_path: Output HDF5 file path
            num_workers: Number of parallel workers
            batch_size: Batch size for processing
            
        Returns:
            Dictionary containing processing statistics
        """
        print(f"Discovering samples in {input_dir}...")
        json_files = self.discover_samples(input_dir)
        print(f"Found {len(json_files)} JSON files")
        
        if len(json_files) == 0:
            raise ValueError(f"No JSON files found in {input_dir}")
        
        # Process samples in parallel
        print(f"Processing samples with {num_workers} workers...")
        processed_samples = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_single_sample, json_path): json_path 
                for json_path in json_files
            }
            
            # Collect results with progress bar
            with tqdm(total=len(json_files), desc="Processing samples") as pbar:
                for future in as_completed(future_to_path):
                    result = future.result()
                    if result is not None:
                        processed_samples.append(result)
                    pbar.update(1)
        
        if len(processed_samples) == 0:
            raise ValueError("No valid samples after filtering")
        
        print(f"Successfully processed {len(processed_samples)} samples")
        
        # Save to HDF5
        print(f"Saving to HDF5: {output_path}")
        self._save_to_hdf5(processed_samples, output_path)
        
        # Update final statistics
        self.stats['final_samples'] = len(processed_samples)
        
        return self.stats
    
    def _save_to_hdf5(self, samples: List[Dict[str, Any]], output_path: str):
        """
        Save processed samples to HDF5 file with batch chunking.
        
        Args:
            samples: List of processed sample dictionaries
            output_path: Output HDF5 file path
        """
        num_samples = len(samples)
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if there is one
            os.makedirs(output_dir, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # Create metadata group
            metadata = f.create_group('metadata')
            metadata.attrs['total_samples'] = num_samples
            metadata.attrs['target_resolution'] = self.target_resolution
            metadata.attrs['backbone_name'] = self.backbone_name
            metadata.attrs['rotation_representation'] = self.rotation_representation
            metadata.attrs['silhouette_threshold'] = self.silhouette_threshold
            metadata.attrs['min_visible_keypoints'] = self.min_visible_keypoints
            metadata.attrs['chunk_size'] = self.chunk_size
            metadata.attrs['jpeg_quality'] = self.jpeg_quality
            
            # Store processing statistics
            stats_group = metadata.create_group('statistics')
            for key, value in self.stats.items():
                stats_group.attrs[key] = value
            
            # Determine data shapes and types from first sample
            first_sample = samples[0]
            
            # Create groups for different data types
            images_group = f.create_group('images')
            params_group = f.create_group('parameters')
            keypoints_group = f.create_group('keypoints')
            aux_group = f.create_group('auxiliary')
            
            # Create datasets with batch chunking
            chunk_shape = (self.chunk_size,)
            
            # Images (variable length JPEG data)
            max_jpeg_size = max(len(sample['image_jpeg']) for sample in samples)
            images_group.create_dataset(
                'rgb_jpeg', 
                shape=(num_samples,), 
                dtype=h5py.special_dtype(vlen=np.uint8),
                chunks=chunk_shape,
                compression=self.compression,
                compression_opts=self.compression_level
            )
            
            # Silhouette masks
            mask_shape = first_sample['mask'].shape
            images_group.create_dataset(
                'silhouette_masks',
                shape=(num_samples,) + mask_shape,
                dtype=np.uint8,
                chunks=(self.chunk_size,) + mask_shape,
                compression=self.compression,
                compression_opts=self.compression_level
            )
            
            # Parameters
            param_datasets = {}
            for param_name in ['global_rot', 'joint_rot', 'betas', 'trans', 'fov', 
                              'cam_rot', 'cam_trans', 'log_beta_scales', 'betas_trans']:
                param_shape = first_sample[param_name].shape
                param_datasets[param_name] = params_group.create_dataset(
                    param_name,
                    shape=(num_samples,) + param_shape,
                    dtype=np.float32,
                    chunks=(self.chunk_size,) + param_shape,
                    compression=self.compression,
                    compression_opts=self.compression_level
                )
            
            # Keypoints
            keypoint_datasets = {}
            for kp_name in ['keypoints_2d', 'keypoints_3d', 'keypoint_visibility']:
                kp_shape = first_sample[kp_name].shape
                keypoint_datasets[kp_name] = keypoints_group.create_dataset(
                    kp_name,
                    shape=(num_samples,) + kp_shape,
                    dtype=np.float32,
                    chunks=(self.chunk_size,) + kp_shape,
                    compression=self.compression,
                    compression_opts=self.compression_level
                )
            
            # Auxiliary data
            aux_group.create_dataset(
                'original_paths',
                shape=(num_samples,),
                dtype=h5py.special_dtype(vlen=str),
                chunks=chunk_shape
            )
            
            aux_group.create_dataset(
                'silhouette_coverage',
                shape=(num_samples,),
                dtype=np.float32,
                chunks=chunk_shape,
                compression=self.compression,
                compression_opts=self.compression_level
            )
            
            aux_group.create_dataset(
                'visible_keypoints',
                shape=(num_samples,),
                dtype=np.int32,
                chunks=chunk_shape,
                compression=self.compression,
                compression_opts=self.compression_level
            )
            
            # Fill datasets
            print("Writing data to HDF5...")
            with tqdm(total=num_samples, desc="Writing samples") as pbar:
                for i, sample in enumerate(samples):
                    # Images
                    images_group['rgb_jpeg'][i] = np.frombuffer(sample['image_jpeg'], dtype=np.uint8)
                    images_group['silhouette_masks'][i] = sample['mask']
                    
                    # Parameters
                    for param_name, dataset in param_datasets.items():
                        dataset[i] = sample[param_name]
                    
                    # Keypoints
                    for kp_name, dataset in keypoint_datasets.items():
                        dataset[i] = sample[kp_name]
                    
                    # Auxiliary
                    aux_group['original_paths'][i] = sample['original_path']
                    aux_group['silhouette_coverage'][i] = sample['silhouette_coverage']
                    aux_group['visible_keypoints'][i] = sample['visible_keypoints']
                    
                    pbar.update(1)
        
        print(f"Successfully saved {num_samples} samples to {output_path}")
        
        # Print file size info
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"HDF5 file size: {file_size:.1f} MB ({file_size/num_samples:.2f} MB per sample)")


def print_statistics(stats: Dict[str, Any]):
    """Print processing statistics."""
    print("\n" + "="*50)
    print("DATASET PREPROCESSING STATISTICS")
    print("="*50)
    print(f"Total samples discovered: {stats['total_samples']}")
    print(f"Samples with errors: {stats['error_samples']}")
    print(f"Filtered by silhouette coverage: {stats['silhouette_filtered']}")
    print(f"Filtered by keypoint visibility: {stats['keypoint_filtered']}")
    print(f"Final valid samples: {stats['final_samples']}")
    
    if stats['total_samples'] > 0:
        success_rate = stats['final_samples'] / stats['total_samples'] * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SMIL dataset to HDF5 format")
    parser.add_argument("input_dir", help="Input dataset directory")
    parser.add_argument("output_path", help="Output HDF5 file path")
    parser.add_argument("--silhouette_threshold", type=float, default=0.1,
                       help="Minimum silhouette coverage (default: 0.1)")
    parser.add_argument("--target_resolution", type=int, default=224,
                       help="Target image resolution (default: 224)")
    parser.add_argument("--backbone_name", default='vit_large_patch16_224',
                       help="Backbone name for resolution selection")
    parser.add_argument("--rotation_representation", choices=['6d', 'axis_angle'], default='6d',
                       help="Rotation representation")
    parser.add_argument("--min_visible_keypoints", type=int, default=5,
                       help="Minimum visible keypoints required")
    parser.add_argument("--chunk_size", type=int, default=8,
                       help="HDF5 chunk size (batch size)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--jpeg_quality", type=int, default=95,
                       help="JPEG compression quality (1-100)")
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = DatasetPreprocessor(
        silhouette_threshold=args.silhouette_threshold,
        target_resolution=args.target_resolution,
        backbone_name=args.backbone_name,
        rotation_representation=args.rotation_representation,
        min_visible_keypoints=args.min_visible_keypoints,
        chunk_size=args.chunk_size,
        jpeg_quality=args.jpeg_quality
    )
    
    # Process dataset
    try:
        stats = preprocessor.process_dataset(
            args.input_dir,
            args.output_path,
            num_workers=args.num_workers
        )
        
        print_statistics(stats)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
