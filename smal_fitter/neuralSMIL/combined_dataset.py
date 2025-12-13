"""
Combined SMIL Dataset for Multi-Source Training

This module provides functionality for combining multiple SMIL datasets with different
available labels for stable mixed-dataset training.
"""

import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler, Subset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import copy


class CombinedSMILDataset:
    """
    Combines multiple SMIL datasets with different available labels.
    
    This class wraps PyTorch's ConcatDataset and adds:
    - Label availability tracking per dataset
    - Dataset source tracking per sample
    - Per-dataset validation splitting
    - Weighted random sampling
    """
    
    def __init__(self, dataset_configs: List[Dict[str, Any]], 
                 rotation_representation: str = '6d',
                 backbone_name: str = 'vit_large_patch16_224',
                 **dataset_kwargs):
        """
        Initialize combined dataset from multiple dataset configurations.
        
        Args:
            dataset_configs: List of dataset configuration dictionaries from TrainingConfig
            rotation_representation: Rotation representation for all datasets
            backbone_name: Backbone name for all datasets
            **dataset_kwargs: Additional keyword arguments passed to dataset constructors
        """
        self.dataset_configs = dataset_configs
        self.rotation_representation = rotation_representation
        self.backbone_name = backbone_name
        self.dataset_kwargs = dataset_kwargs
        
        # Load individual datasets
        self.datasets = []
        self.dataset_names = []
        self.dataset_lengths = []
        self.available_labels_per_dataset = []
        
        for config in dataset_configs:
            if not config.get('enabled', False):
                continue
            
            dataset_name = config['name']
            dataset_path = config['path']
            dataset_type = config.get('type', 'auto')
            available_labels = config.get('available_labels', {})
            
            print(f"Loading dataset '{dataset_name}' from {dataset_path}")
            
            # Load dataset based on type
            dataset = self._load_dataset(dataset_path, dataset_type)
            
            self.datasets.append(dataset)
            self.dataset_names.append(dataset_name)
            self.dataset_lengths.append(len(dataset))
            self.available_labels_per_dataset.append(available_labels)
            
            print(f"  Loaded {len(dataset)} samples from '{dataset_name}'")
        
        if not self.datasets:
            raise ValueError("No datasets were successfully loaded. Check your configuration.")
        
        # Create combined dataset
        self.combined_dataset = ConcatDataset(self.datasets)
        
        # Create mapping from global index to (dataset_idx, local_idx, dataset_name, available_labels)
        self._build_index_mapping()
        
        print(f"\nCombined dataset created with {len(self.combined_dataset)} total samples")
        print(f"  Datasets: {self.dataset_names}")
        print(f"  Lengths: {self.dataset_lengths}")
    
    def _load_dataset(self, dataset_path: str, dataset_type: str):
        """
        Load a dataset based on path and type.
        
        Args:
            dataset_path: Path to dataset
            dataset_type: Dataset type ('replicant', 'sleap', 'optimized_hdf5', or 'auto')
            
        Returns:
            Loaded dataset instance
        """
        # Import here to avoid circular dependencies
        from smil_datasets import UnifiedSMILDataset
        
        if dataset_type == 'auto':
            # Auto-detect based on path
            return UnifiedSMILDataset.from_path(
                dataset_path,
                rotation_representation=self.rotation_representation,
                backbone_name=self.backbone_name,
                **self.dataset_kwargs
            )
        elif dataset_type == 'replicant':
            from smil_datasets import replicAntSMILDataset
            return replicAntSMILDataset(
                dataset_path,
                rotation_representation=self.rotation_representation,
                backbone_name=self.backbone_name,
                **self.dataset_kwargs
            )
        elif dataset_type == 'sleap':
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sleap_data'))
            from sleap_dataset import SLEAPDataset
            return SLEAPDataset(
                dataset_path,
                rotation_representation=self.rotation_representation,
                backbone_name=self.backbone_name,
                **self.dataset_kwargs
            )
        elif dataset_type == 'optimized_hdf5':
            from optimized_dataset import OptimizedSMILDataset
            return OptimizedSMILDataset(
                dataset_path,
                rotation_representation=self.rotation_representation,
                backbone_name=self.backbone_name,
                **self.dataset_kwargs
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _build_index_mapping(self):
        """Build mapping from global index to dataset information."""
        self.index_to_dataset_info = {}
        global_idx = 0
        
        for dataset_idx, dataset in enumerate(self.datasets):
            dataset_name = self.dataset_names[dataset_idx]
            available_labels = self.available_labels_per_dataset[dataset_idx]
            
            for local_idx in range(len(dataset)):
                self.index_to_dataset_info[global_idx] = {
                    'dataset_idx': dataset_idx,
                    'local_idx': local_idx,
                    'dataset_name': dataset_name,
                    'available_labels': available_labels
                }
                global_idx += 1
    
    def __len__(self):
        """Return total number of samples across all datasets."""
        return len(self.combined_dataset)
    
    def __getitem__(self, idx):
        """
        Get a sample with added metadata about dataset source and available labels.
        
        Args:
            idx: Global sample index
            
        Returns:
            Tuple of (x_data, y_data) with added metadata
        """
        # Get the sample from combined dataset
        x_data, y_data = self.combined_dataset[idx]
        
        # Get dataset information for this index
        dataset_info = self.index_to_dataset_info[idx]
        
        # Add dataset source and available labels to x_data
        x_data['dataset_source'] = dataset_info['dataset_name']
        x_data['available_labels'] = copy.deepcopy(dataset_info['available_labels'])
        
        # Replace unavailable labels with None in y_data to catch leaks
        y_data = self._apply_availability_mask(y_data, dataset_info['available_labels'])
        
        return x_data, y_data
    
    def _apply_availability_mask(self, y_data: Dict[str, Any], 
                                 available_labels: Dict[str, bool]) -> Dict[str, Any]:
        """
        Replace unavailable labels with None to catch leaks.
        
        Args:
            y_data: Target data dictionary
            available_labels: Dictionary indicating which labels are available
            
        Returns:
            Modified y_data with None for unavailable labels
        """
        # Map from y_data keys to available_labels keys
        key_mapping = {
            'root_rot': 'global_rot',
            'joint_angles': 'joint_rot',
            'shape_betas': 'betas',
            'root_loc': 'trans',
            'cam_fov': 'fov',
            'cam_rot': 'cam_rot',
            'cam_trans': 'cam_trans',
            'scale_weights': 'log_beta_scales',
            'trans_weights': 'betas_trans',
            'keypoints_2d': 'keypoint_2d',
            'keypoints_3d': 'keypoint_3d'
        }
        
        # Replace unavailable labels with None
        for y_key, label_key in key_mapping.items():
            if y_key in y_data and label_key in available_labels:
                if not available_labels[label_key]:
                    y_data[y_key] = None
        
        return y_data
    
    def get_dataset_by_name(self, name: str) -> Optional[torch.utils.data.Dataset]:
        """
        Get a specific dataset by name.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset instance or None if not found
        """
        try:
            idx = self.dataset_names.index(name)
            return self.datasets[idx]
        except ValueError:
            return None
    
    def get_ue_scaling_flag(self) -> bool:
        """
        Get the UE scaling flag for the combined dataset.
        
        Note: This is a simplified implementation that always returns True.
        In reality, UE scaling depends on which dataset a sample comes from.
        However, since we don't have ground truth scaling information for SLEAP data,
        and the scaling is handled during data preprocessing, we assume True for all samples.
        
        TODO: When creating more complex mixed datasets with different scaling conventions,
        this should be made sample-aware by storing the UE scaling flag in x_data metadata
        and using it appropriately during loss computation.
        
        Returns:
            bool: Always True for now
        """
        return True
    
    def create_weighted_sampler(self, weights: List[float], 
                               train_indices: List[int],
                               num_samples: Optional[int] = None) -> WeightedRandomSampler:
        """
        Create a weighted random sampler for mixed-batch training.
        
        The weights control the relative sampling frequency of datasets, normalized by dataset size.
        For example, if dataset A has weight 1.0 and dataset B has weight 0.5, then samples from
        dataset B will be drawn half as frequently as samples from dataset A, regardless of
        dataset sizes.
        
        Args:
            weights: Per-dataset sampling weights (controls relative frequency)
            train_indices: List of training indices (from split_datasets)
            num_samples: Number of samples to draw (default: same as training set length)
            
        Returns:
            WeightedRandomSampler instance
        """
        if len(weights) != len(self.datasets):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of datasets ({len(self.datasets)})")
        
        # Normalize weights by dataset size to achieve desired sampling frequency
        # This ensures that weight=0.5 means "sample half as often", not "each sample gets 0.5 weight"
        normalized_weights = []
        for dataset_idx, dataset_length in enumerate(self.dataset_lengths):
            dataset_weight = weights[dataset_idx]
            # Divide by dataset size so that the total probability mass is proportional to the weight
            per_sample_weight = dataset_weight / dataset_length if dataset_length > 0 else 0.0
            normalized_weights.append(per_sample_weight)
        
        # Create per-sample weights for the FULL dataset
        full_sample_weights = []
        for dataset_idx, dataset_length in enumerate(self.dataset_lengths):
            per_sample_weight = normalized_weights[dataset_idx]
            full_sample_weights.extend([per_sample_weight] * dataset_length)
        
        # Extract weights only for training indices
        train_sample_weights = [full_sample_weights[idx] for idx in train_indices]
        train_sample_weights = torch.tensor(train_sample_weights, dtype=torch.float)
        
        if num_samples is None:
            num_samples = len(train_indices)
        
        print(f"\nWeighted Sampler Statistics:")
        print(f"  Dataset sizes: {self.dataset_lengths}")
        print(f"  User weights: {weights}")
        print(f"  Normalized per-sample weights: {normalized_weights}")
        print(f"  Expected sampling ratio (replicant:sleap): {normalized_weights[0]/normalized_weights[1]:.2f}:1")
        
        return WeightedRandomSampler(
            weights=train_sample_weights,
            num_samples=num_samples,
            replacement=True  # Allow sampling same sample multiple times per epoch
        )
    
    def split_datasets(self, train_size: float = 0.85, val_size: float = 0.05, 
                      test_size: float = 0.1, seed: int = 1234) -> Tuple:
        """
        Split datasets into train/val/test sets using per-dataset splitting.
        
        Args:
            train_size: Fraction for training (default: 0.85)
            val_size: Fraction for validation (default: 0.05)
            test_size: Fraction for testing (default: 0.1)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "Split sizes must sum to 1.0"
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        rng = np.random.RandomState(seed)
        global_offset = 0
        
        # Split each dataset separately, then combine
        for dataset_idx, dataset_length in enumerate(self.dataset_lengths):
            # Generate random permutation for this dataset
            indices = rng.permutation(dataset_length)
            
            # Calculate split points
            val_count = int(dataset_length * val_size)
            test_count = int(dataset_length * test_size)
            train_count = dataset_length - val_count - test_count
            
            # Split indices
            train_idx = indices[:train_count]
            val_idx = indices[train_count:train_count + val_count]
            test_idx = indices[train_count + val_count:]
            
            # Convert to global indices
            train_indices.extend(train_idx + global_offset)
            val_indices.extend(val_idx + global_offset)
            test_indices.extend(test_idx + global_offset)
            
            global_offset += dataset_length
            
            print(f"Dataset '{self.dataset_names[dataset_idx]}' split:")
            print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # Create subset datasets
        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        test_dataset = Subset(self, test_indices)
        
        print(f"\nTotal combined split:")
        print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        # Store training indices for weighted sampler
        self.train_indices = train_indices
        
        return train_dataset, val_dataset, test_dataset
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the combined dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_samples': len(self.combined_dataset),
            'num_datasets': len(self.datasets),
            'datasets': []
        }
        
        for idx, name in enumerate(self.dataset_names):
            dataset_stats = {
                'name': name,
                'num_samples': self.dataset_lengths[idx],
                'available_labels': self.available_labels_per_dataset[idx]
            }
            stats['datasets'].append(dataset_stats)
        
        return stats
    
    def print_statistics(self):
        """Print dataset statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("COMBINED DATASET STATISTICS")
        print("="*70)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Number of datasets: {stats['num_datasets']}")
        print()
        
        for ds_stats in stats['datasets']:
            print(f"Dataset: {ds_stats['name']}")
            print(f"  Samples: {ds_stats['num_samples']} ({ds_stats['num_samples']/stats['total_samples']*100:.1f}%)")
            print(f"  Available labels:")
            
            # Group labels by availability
            available = [k for k, v in ds_stats['available_labels'].items() if v]
            unavailable = [k for k, v in ds_stats['available_labels'].items() if not v]
            
            print(f"    Available ({len(available)}): {', '.join(available)}")
            print(f"    Unavailable ({len(unavailable)}): {', '.join(unavailable)}")
            print()
        
        print("="*70)

