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
    # TODO: remove all that were used for debugging only during development
    DATA_PATHS = {
        'masked_simple': "/media/fabi/Data/replicAnt-x-SMIL-OmniAnt-Masked-Simple",
        'pose_only_simple': "/media/fabi/Data/replicAnt-x-SMIL-OmniAnt-PoseOnly-Simple",
        'test_textured': "data/replicAnt_trials/replicAnt-x-SMIL-TEX",
        'full_dataset': "/media/fabi/Data/replicAnt-x-SMIL-OmniAnt-Masked",
        'SHAPE-n-POSE': "/media/fabi/Data/replicAnt-x-SMIL-OmniAnt-SHAPE-n-POSE",
        'simple': "/media/fabi/Data/replicAnt-x-SMIL-OmniAnt-Simple",
        'simple100k': "/media/fabi/Data/replicAnt-x-SMIL-OmniAnt-Simple100k-noScale",
        'simple100k_local': "/home/fabi/DATA_LOCAL/replicAnt-x-SMIL-OmniAnt-Simple100k-noScale"
    }
    
    # Default dataset to use
    DEFAULT_DATASET = 'simple100k_local'
    
    # Training split configuration
    SPLIT_CONFIG = {
        'test_size': 0.1,      # 10% for testing
        'val_size': 0.05,      # 5% for validation
        # Training size is automatically: 1 - test_size - val_size = 0.85 (85%)
    }
    
    # Training hyperparameters (AniMer-style conservative settings)
    TRAINING_PARAMS = {
        'batch_size': 8,
        'num_epochs': 200,
        'learning_rate': 1.25e-6,  # AniMer-style very conservative learning rate
        'weight_decay': 1e-4,  # Add weight decay for AdamW
        'seed': 1234,
        'rotation_representation': '6d',  # '6d' or 'axis_angle'
        'resume_checkpoint': None, # Path to checkpoint file to resume training from (None for training from scratch)
        'num_workers': 16,  # Number of data loading workers (reduced to prevent tkinter issues)
        'pin_memory': True,  # Faster GPU transfer
        'prefetch_factor': 8,  # Prefetch batches
    }
    
    # Model configuration
    MODEL_CONFIG = {
        'backbone_name': 'vit_large_patch16_224',  # 'resnet152', 'vit_base_patch16_224', etc.
        'freeze_backbone': True,
        'hidden_dim': 1024,  # Deprecated, Will be adjusted based on backbone
        'rgb_only': False,
        'use_unity_prior': False,
        'head_type': 'transformer_decoder',  # 'mlp' or 'transformer_decoder'
        'transformer_config': {
            'hidden_dim': 1024,
            'depth': 6,
            'heads': 8,
            'dim_head': 64,
            'mlp_dim': 1024,
            'dropout': 0.1,  # Add some dropout for stability
            'ief_iters': 3,
            'scales_scale_factor': 0.001,  # Scale factor for log_beta_scales (encourages values close to zero)
            'trans_scale_factor': 0.001,   # Scale factor for betas_trans (encourages values close to zero)
        }
    }
    
    # Joint visibility configuration for preprocessing
    # Joints that should be ignored during training due to mesh vs. training data misalignment
    IGNORED_JOINTS_CONFIG = {
        # List of joint names to ignore (set visibility to 0) during preprocessing
        # These joints often have misalignment between the parametric model and ground truth
        'ignored_joint_names': [
            # Add joint names here that should be ignored
            # Example: 'head_tip', 'antenna_tip', etc.
            "b_a_5"
        ],
        
        # Whether to print information about ignored joints during preprocessing
        'verbose_ignored_joints': True,
    }
    
    # Loss curriculum configuration (AniMer-style conservative weights)
    LOSS_CURRICULUM = {
        # Base loss weights (applied throughout training) - AniMer-style conservative
        'base_weights': {
            'global_rot': 0.001,     # AniMer: 0.001
            'joint_rot': 0.001,      # AniMer: 0.001  
            'betas': 0.0005,         # AniMer: 0.0005
            'trans': 0.0005,         # AniMer: 0.0005
            'fov': 0.001,
            'cam_rot': 0.01,
            'cam_trans': 0.001,
            'log_beta_scales': 0.0005,  # Conservative
            'betas_trans': 0.0005,      # Conservative
            'keypoint_2d': 0.0,        # AniMer: 0.01
            'keypoint_3d': 0.0,        # 3D keypoint loss - start at zero
            'silhouette': 0.0           # Start at zero
        },
        
        # Curriculum stages: (epoch_threshold, weight_updates) - AniMer-style conservative
        'curriculum_stages': [
            # Stage 1: Introduce keypoint losses gradually (AniMer-style)
            (10, {
                'cam_trans': 0.00001, # decrease cam_trans once suitable extrinsics are found otherwise the error is overpowering
                'joint_rot': 0.002,
            }),
            # Stage 2: Introduce keypoint losses gradually (AniMer-style)
            (20, {
                'cam_trans': 0.00001,
                'joint_rot': 0.001,
                'keypoint_2d': 0.01,    # AniMer: 0.01
                'keypoint_3d': 0.005,   # 
                'silhouette': 0.01,
            }),
            (100, {
                'cam_trans': 0.0001,
                'joint_rot': 0.01,
                'keypoint_2d': 0.01,    # AniMer: 0.01
                'keypoint_3d': 0.005,   # 
                'silhouette': 0.1,
            })
        ]
    }
    
    # Learning rate curriculum configuration (AniMer-style conservative)
    LEARNING_RATE_CURRICULUM = {
        # Base learning rate (applied at epoch 0) - AniMer-style
        'base_learning_rate': 1.25e-6,
        
        # Learning rate stages: (epoch_threshold, learning_rate) - Very conservative
        'lr_stages': [
            # Stage 1: Slight reduction for fine-tuning
            (40, 1e-6),      # 1e-6
            
            # Stage 2: Further reduce
            (60, 5e-7),      # 5e-7
            
            # Stage 3: Very low learning rate for final convergence
            (80, 1e-7),      # 1e-7
        ]
    }
    
    # Checkpoint and output configuration
    OUTPUT_CONFIG = {
        'checkpoint_dir': 'checkpoints',
        'plots_dir': 'plots',
        'visualizations_dir': 'visualizations',
        'save_checkpoint_every': 2,        # Save checkpoint every N epochs
        'generate_visualizations_every': 1, # Generate visualizations every N epochs
        'plot_history_every': 10,          # Plot training history every N epochs
        'num_visualization_samples': 10,    # Number of samples to visualize
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
    def get_learning_rate_for_epoch(cls, epoch: int) -> float:
        """
        Get learning rate for a specific epoch based on the curriculum.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Learning rate for the epoch
        """
        # Start with base learning rate
        lr = cls.LEARNING_RATE_CURRICULUM['base_learning_rate']
        
        # Apply learning rate stages
        for epoch_threshold, learning_rate in cls.LEARNING_RATE_CURRICULUM['lr_stages']:
            if epoch >= epoch_threshold:
                lr = learning_rate
        
        return lr
    
    @classmethod
    def get_all_config(cls, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all configuration as a dictionary.
        
        Args:
            dataset_name: Name of the dataset to use
            
        Returns:
            Dictionary containing all configuration parameters
        """
        config = {
            'data_path': cls.get_data_path(dataset_name),
            'split_config': cls.SPLIT_CONFIG,
            'training_params': cls.TRAINING_PARAMS,
            'model_config': cls.MODEL_CONFIG.copy(),  # Make a copy to avoid modifying the original
            'ignored_joints_config': cls.IGNORED_JOINTS_CONFIG,
            'loss_curriculum': cls.LOSS_CURRICULUM,
            'learning_rate_curriculum': cls.LEARNING_RATE_CURRICULUM,
            'output_config': cls.OUTPUT_CONFIG,
        }
        
        # Adjust hidden_dim based on backbone
        backbone_name = config['model_config']['backbone_name']
        if backbone_name.startswith('vit'):
            if 'base' in backbone_name:
                config['model_config']['hidden_dim'] = 768
            elif 'large' in backbone_name:
                config['model_config']['hidden_dim'] = 1024
        elif backbone_name.startswith('resnet'):
            config['model_config']['hidden_dim'] = 2048
        
        # Adjust transformer config based on backbone
        if config['model_config']['head_type'] == 'transformer_decoder':
            # For ViT backbones, use spatial features
            if backbone_name.startswith('vit'):
                # ViT provides spatial features, so we can use them as context
                pass  # Keep default config
            else:
                # For ResNet, we don't have spatial features
                # Adjust context_dim to match feature_dim
                if 'base' in backbone_name:
                    config['model_config']['transformer_config']['context_dim'] = 768
                elif 'large' in backbone_name:
                    config['model_config']['transformer_config']['context_dim'] = 1024
                elif backbone_name.startswith('resnet'):
                    config['model_config']['transformer_config']['context_dim'] = 2048
        
        return config
    
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
            if key != 'transformer_config':  # Print transformer config separately
                print(f"  {key}: {value}")
        
        # Print backbone-specific information
        backbone_name = config['model_config']['backbone_name']
        print(f"  Backbone type: {'Vision Transformer' if backbone_name.startswith('vit') else 'ResNet'}")
        print(f"  Feature dimension: {config['model_config']['hidden_dim']}")
        
        # Print head-specific information
        head_type = config['model_config']['head_type']
        print(f"  Regression head: {head_type}")
        if head_type == 'transformer_decoder':
            print("  Transformer decoder config:")
            for key, value in config['model_config']['transformer_config'].items():
                print(f"    {key}: {value}")
        print()
        
        print("Loss Curriculum:")
        print("  Base weights:")
        for param, weight in config['loss_curriculum']['base_weights'].items():
            print(f"    {param}: {weight}")
        print("  Curriculum stages:")
        for epoch_threshold, updates in config['loss_curriculum']['curriculum_stages']:
            print(f"    Epoch {epoch_threshold}+: {updates}")
        print()
        
        print("Learning Rate Curriculum:")
        print(f"  Base learning rate: {config['learning_rate_curriculum']['base_learning_rate']}")
        print("  Learning rate stages:")
        for epoch_threshold, lr in config['learning_rate_curriculum']['lr_stages']:
            print(f"    Epoch {epoch_threshold}+: {lr}")
        print()
        
        print("Output Configuration:")
        for key, value in config['output_config'].items():
            print(f"  {key}: {value}")
        print("=" * 60)


# Example usage and validation
if __name__ == "__main__":
    # Print configuration summary
    TrainingConfig.print_config_summary()
    
    # Test loss weights and learning rates for different epochs
    print("\nLoss weights and learning rates at different epochs:")
    test_epochs = [0, 10, 25, 35, 55, 75, 105, 155]
    for epoch in test_epochs:
        weights = TrainingConfig.get_loss_weights_for_epoch(epoch)
        lr = TrainingConfig.get_learning_rate_for_epoch(epoch)
        print(f"Epoch {epoch}: keypoint_2d={weights['keypoint_2d']}, silhouette={weights['silhouette']}, joint_rot={weights['joint_rot']}, lr={lr}")
