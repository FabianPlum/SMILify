"""
Training Configuration for SMIL Image Regressor

This file contains all configuration parameters for training the SMILImageRegressor,
including dataset paths, training splits, loss curriculum, and hyperparameters.
"""

import os
from typing import Dict, Any, Optional, Tuple


class TrainingConfig:
    """Configuration class for SMIL training."""
    
    # Dataset configuration
    DATA_PATHS = {
        'masked_simple': "/media/fabi/Data/replicAnt-x-SMIL-OmniAnt-Masked-Simple",
        'pose_only_simple': "/media/fabi/Data/replicAnt-x-SMIL-OmniAnt-PoseOnly-Simple",
        'test_textured': "data/replicAnt_trials/replicAnt-x-SMIL-TEX"
    }
    
    # Default dataset to use
    DEFAULT_DATASET = 'pose_only_simple'
    
    # Training split configuration
    SPLIT_CONFIG = {
        'test_size': 0.1,      # 10% for testing
        'val_size': 0.05,      # 5% for validation
        # Training size is automatically: 1 - test_size - val_size = 0.85 (85%)
    }
    
    # Training hyperparameters
    TRAINING_PARAMS = {
        'batch_size': 32,
        'num_epochs': 500,
        'learning_rate': 0.0001,
        'seed': 0,
        'rotation_representation': '6d',  # '6d' or 'axis_angle'
        'resume_checkpoint': "/home/fabi/dev/SMILify/checkpoints/best_model.pth",  # Path to checkpoint file to resume training from (None for training from scratch)
    }
    
    # Model configuration
    MODEL_CONFIG = {
        'freeze_backbone': True,
        'hidden_dim': 2048,
        'rgb_only': False,
        'use_unity_prior': False,
    }
    
    # Loss curriculum configuration
    LOSS_CURRICULUM = {
        # Base loss weights (applied throughout training)
        'base_weights': {
            'global_rot': 0.02,
            'joint_rot': 0.02,
            'betas': 0.01,
            'trans': 0.001,
            'fov': 0.001,
            'cam_rot': 0.01,
            'cam_trans': 0.001,
            'log_beta_scales': 0.1,
            'betas_trans': 0.1,
            'keypoint_2d': 0.0,      # Start at zero
            'silhouette': 0.0        # Start at zero
        },
        
        # Curriculum stages: (epoch_threshold, weight_updates)
        'curriculum_stages': [
            # Stage 1: Introduce keypoint and silhouette losses gradually
            (20, {
                'keypoint_2d': 0.001,
                'silhouette': 0.0001
            }),
            
            # Stage 2: Increase keypoint and silhouette weights
            (25, {
                'keypoint_2d': 0.01,
                'silhouette': 0.001
            }),
            
            # Stage 3: Further increase and start reducing joint rotation influence
            (30, {
                'keypoint_2d': 0.05,
                'joint_rot': 0.01,      # Reduce to favor keypoint loss
                'silhouette': 0.001
            }),
            
            # Stage 4: High keypoint weight for fine-tuning
            (50, {
                'keypoint_2d': 0.2,
                'joint_rot': 0.01,      # Keep reduced
                'silhouette': 0.01
            })
        ]
    }
    
    # Checkpoint and output configuration
    OUTPUT_CONFIG = {
        'checkpoint_dir': 'checkpoints',
        'plots_dir': 'plots',
        'visualizations_dir': 'visualizations',
        'save_checkpoint_every': 10,        # Save checkpoint every N epochs
        'generate_visualizations_every': 1, # Generate visualizations every N epochs
        'plot_history_every': 10,          # Plot training history every N epochs
        'num_visualization_samples': 5,    # Number of samples to visualize
    }
    
    @classmethod
    def get_data_path(cls, dataset_name: Optional[str] = None) -> str:
        """
        Get the data path for the specified dataset.
        
        Args:
            dataset_name: Name of the dataset ('masked_simple', 'pose_only_simple', 'test_textured')
                         If None, uses DEFAULT_DATASET
        
        Returns:
            Path to the dataset
        """
        if dataset_name is None:
            dataset_name = cls.DEFAULT_DATASET
        
        if dataset_name not in cls.DATA_PATHS:
            available = list(cls.DATA_PATHS.keys())
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
        
        return cls.DATA_PATHS[dataset_name]
    
    @classmethod
    def get_train_val_test_sizes(cls, dataset_size: int) -> Tuple[int, int, int]:
        """
        Calculate training, validation, and test set sizes based on the dataset size.
        
        Args:
            dataset_size: Total size of the dataset
            
        Returns:
            Tuple of (train_size, val_size, test_size)
        """
        test_size = int(dataset_size * cls.SPLIT_CONFIG['test_size'])
        val_size = int(dataset_size * cls.SPLIT_CONFIG['val_size'])
        train_size = dataset_size - test_size - val_size
        
        return train_size, val_size, test_size
    
    @classmethod
    def get_loss_weights_for_epoch(cls, epoch: int) -> Dict[str, float]:
        """
        Get loss weights for a specific epoch based on the curriculum.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Dictionary of loss weights for the epoch
        """
        # Start with base weights
        weights = cls.LOSS_CURRICULUM['base_weights'].copy()
        
        # Apply curriculum stages
        for epoch_threshold, weight_updates in cls.LOSS_CURRICULUM['curriculum_stages']:
            if epoch >= epoch_threshold:
                weights.update(weight_updates)
        
        return weights
    
    @classmethod
    def get_all_config(cls, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all configuration as a dictionary.
        
        Args:
            dataset_name: Name of the dataset to use
            
        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            'data_path': cls.get_data_path(dataset_name),
            'split_config': cls.SPLIT_CONFIG,
            'training_params': cls.TRAINING_PARAMS,
            'model_config': cls.MODEL_CONFIG,
            'loss_curriculum': cls.LOSS_CURRICULUM,
            'output_config': cls.OUTPUT_CONFIG,
        }
    
    @classmethod
    def print_config_summary(cls, dataset_name: Optional[str] = None):
        """Print a summary of the current configuration."""
        config = cls.get_all_config(dataset_name)
        
        print("=" * 60)
        print("SMIL Training Configuration Summary")
        print("=" * 60)
        print(f"Dataset: {dataset_name or cls.DEFAULT_DATASET}")
        print(f"Data Path: {config['data_path']}")
        print(f"Train/Val/Test Split: {1 - cls.SPLIT_CONFIG['test_size'] - cls.SPLIT_CONFIG['val_size']:.1%}/{cls.SPLIT_CONFIG['val_size']:.1%}/{cls.SPLIT_CONFIG['test_size']:.1%}")
        print()
        
        print("Training Parameters:")
        for key, value in config['training_params'].items():
            print(f"  {key}: {value}")
        print()
        
        print("Model Configuration:")
        for key, value in config['model_config'].items():
            print(f"  {key}: {value}")
        print()
        
        print("Loss Curriculum:")
        print("  Base weights:")
        for param, weight in config['loss_curriculum']['base_weights'].items():
            print(f"    {param}: {weight}")
        print("  Curriculum stages:")
        for epoch_threshold, updates in config['loss_curriculum']['curriculum_stages']:
            print(f"    Epoch {epoch_threshold}+: {updates}")
        print()
        
        print("Output Configuration:")
        for key, value in config['output_config'].items():
            print(f"  {key}: {value}")
        print("=" * 60)


# Example usage and validation
if __name__ == "__main__":
    # Print configuration summary
    TrainingConfig.print_config_summary()
    
    # Test loss weights for different epochs
    print("\nLoss weights at different epochs:")
    test_epochs = [0, 10, 25, 35, 55]
    for epoch in test_epochs:
        weights = TrainingConfig.get_loss_weights_for_epoch(epoch)
        print(f"Epoch {epoch}: keypoint_2d={weights['keypoint_2d']}, silhouette={weights['silhouette']}, joint_rot={weights['joint_rot']}")
