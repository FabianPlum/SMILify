"""
Training Configuration for SMIL Image Regressor

This file contains all configuration parameters for training the SMILImageRegressor,
including dataset paths, training splits, loss curriculum, and hyperparameters.
"""

import os
from typing import Dict, Any, Optional, Tuple, List


class TrainingConfig:
    """Configuration class for SMIL training."""
    
    # Camera head initialization
    # If True, use GT camera params (when available) as base and predict deltas.
    use_gt_camera_init = True
    
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
        'simple100k_local': "/home/fabi/DATA_LOCAL/replicAnt-x-SMIL-OmniAnt-Simple100k-noScale",
        'SMILySTICK_precomp': "/home/fabi/dev/SMILify/SMILySTICK100k.h5",
        'SMILySLEAPySTICK-test': "/home/fabi/dev/SMILify/test_sleap_dataset.h5",
        'SMILySLEAPySTICKS': "/home/fabi/dev/SMILify/SMILySLEAPySTICKS.h5",
        'SMILySLEAPySTICKS-cropped': "/home/fabi/dev/SMILify/SMILySLEAPySTICKS-cropped.h5",
        'STICKyreprojected': "/home/fabi/dev/SMILify/STICKyReprojections.h5",
        'bbox_center_STICKY': "/home/fabi/dev/SMILify/bbox_center_STICKY.h5",
        'bbox_center_test': "/home/fabi/dev/SMILify/bbox_center_STICKY_test.h5",
        'bbox_center_STICKY_single_animal': "/home/fabi/dev/SMILify/bbox_center_STICKY_single_animal.h5",
        'bbox_center_PERU': "/home/fabi/dev/SMILify/bbox_center_PERU_test.h5",
        'SMILyMouseSYNTH': "SMILyMouseSYNTH.h5",
        'RealSMILyMouse': "/home/fabi/dev/SMILify/RealSMILyMouse.h5",
        'RealSMILyMouseFalkner': "RealSMILyMouseFalknerFROM3D.h5",
        'RealSMILyMouseFalknerFROM3D_no_crop': "RealSMILyMouseFalknerFROM3D_no_crop.h5",
        'RealSMILyMouseFalknerREAL3D': "RealSMILyMouseFalknerFROM3D_no_crop_REAL3D.h5"

    }
    
    # Default dataset to use (legacy single-dataset mode)
    DEFAULT_DATASET = 'RealSMILyMouseFalknerFROM3D_no_crop'#'bbox_center_STICKY' #'SMILySLEAPySTICKS-cropped'
    
    # Multi-dataset configuration for combined training
    # Set use_multi_dataset=True to enable training with multiple datasets
    # 
    # IMPORTANT NOTES:
    # 1. UE Scaling: Currently assumes all datasets use UE scaling (10x scale).
    #    This is handled uniformly across all datasets. For SLEAP data without
    #    ground truth scaling info, this is acceptable. Future: make this per-sample.
    # 2. Label Availability: Each dataset specifies which labels are real vs placeholder.
    #    Placeholder labels (False) contribute ZERO to loss (complete masking).
    # 3. Weighted Sampling: Dataset weights control relative sampling frequency.
    #    weight=1.0 means normal sampling, weight=0.5 means sample half as often.
    MULTI_DATASET_CONFIG = {
        'enabled': False,  # Set to True to use multi-dataset training
        'datasets': [
            {
                'name': 'replicant_main',
                'path': "SMILyMouseSYNTH.h5",
                'type': 'optimized_hdf5',  # 'replicant', 'sleap', 'optimized_hdf5', or 'auto'
                'weight': 1.0,  # Sampling weight (higher = more frequent sampling)
                'enabled': True,
                'available_labels': {
                    # Which ground truth labels are available in this dataset
                    'global_rot': True,
                    'joint_rot': True,
                    'betas': True,
                    'trans': True,
                    'fov': True,
                    'cam_rot': True,
                    'cam_trans': True,
                    'log_beta_scales': True,
                    'betas_trans': True,
                    'keypoint_2d': True,
                    'keypoint_3d': True,
                    'silhouette': True
                }
            },
            {
                'name': 'sleap_data',
                #'path': "/home/fabi/dev/SMILify/STICKyReprojections.h5",
                'path': "RealSMILyMouseFalkner.h5",
                'type': 'optimized_hdf5',
                'weight': 1.0,  # Sample 20% as often as replicant data
                'enabled': True,
                'available_labels': {
                    # SLEAP data only has keypoints and betas
                    'global_rot': False,  # Placeholder values only
                    'joint_rot': False,   # Placeholder values only
                    'betas': True,        # Real ground truth from lookup table
                    'trans': False,       # Placeholder
                    'fov': False,         # Placeholder
                    'cam_rot': False,     # Placeholder
                    'cam_trans': False,   # Placeholder
                    'log_beta_scales': False,  # Placeholder
                    'betas_trans': False,      # Placeholder
                    'keypoint_2d': True,  # Real ground truth from SLEAP
                    'keypoint_3d': False, # Placeholder
                    'silhouette': False   # Placeholder
                }
            }
        ],
        # Validation split strategy: 'per_dataset' or 'combined'
        # 'per_dataset': Split each dataset separately, then combine
        # 'combined': Combine datasets first, then split
        'validation_split_strategy': 'per_dataset'
    }
    
    # Training split configuration
    SPLIT_CONFIG = {
        'test_size': 0.1,      # 10% for testing
        'val_size': 0.05,      # 5% for validation
        # Training size is automatically: 1 - test_size - val_size = 0.85 (85%)
        
        # Dataset fraction for large datasets: fraction of training data to use per epoch
        # If 1.0, use full training dataset each epoch (default behavior)
        # If < 1.0 (e.g., 0.1), randomly sample that fraction of training examples at the
        # start of each epoch. Different samples are drawn each epoch for diversity.
        # Validation and test sets are NOT affected - they always use full data.
        # Note: The random sampling is deterministic per epoch (uses epoch as seed),
        # ensuring all DDP processes use the same subset.
        'dataset_fraction': 0.5,
    }
    
    # Training hyperparameters (AniMer-style conservative settings)
    TRAINING_PARAMS = {
        'batch_size': 1,
        'num_epochs': 1000,
        'learning_rate': 1.25e-6,  # AniMer-style very conservative learning rate
        'weight_decay': 1e-4,  # Add weight decay for AdamW
        'seed': 1234,
        'rotation_representation': '6d',  # '6d' or 'axis_angle'
        'resume_checkpoint': 'multiview_checkpoints/checkpoint_epoch_0574.pth', # Path to checkpoint file to resume training from (None for training from scratch)
        'num_workers': 8,  # Number of data loading workers (reduced to prevent tkinter issues)
        'pin_memory': True,  # Faster GPU transfer
        'prefetch_factor': 4,  # Prefetch batches
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
            'trans_scale_factor': 1,   # Scale factor for betas_trans (encourages values close to zero)
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
            #"b_a_5"
        ],
        
        # Whether to print information about ignored joints during preprocessing
        'verbose_ignored_joints': True,
    }
    
    # Scale and Translation Beta Handling Configuration
    SCALE_TRANS_BETA_CONFIG = {
        'mode': 'separate',  # Options: 'ignore', 'separate', 'entangled_with_betas'
        
        # Mode-specific configurations
        'ignore': {
            'use_zero_scales': True,
            'use_zero_trans': True,
            'loss_weights': {
                'log_beta_scales': 0.0,
                'betas_trans': 0.0
            }
        },
        
        'separate': {
            'use_pca_transformation': False,
            'transformer_scale_factors': {
                'trans_scale_factor': 0.01
            },
            'loss_weights': {
                'log_beta_scales': 0.0005,
                'betas_trans': 0.0005
            }
        },
        
        'entangled_with_betas': {
            'use_unified_betas': True,  # Same betas for shape, scale, and trans
            'loss_weights': {
                'betas': 0.0005,  # Single weight for all three components
                'log_beta_scales': 0.0,  # Disabled
                'betas_trans': 0.0       # Disabled
            }
        }
    }
    
    # Global Mesh Scaling Configuration
    # This allows the network to predict a single global scale factor that scales
    # the entire mesh uniformly (similar to how UE scaling applies a fixed 10x).
    # This is useful when 3D ground truth data has a different scale than the model.
    # The scale is trained IMPLICITLY through 3D keypoint losses - no supervision needed.
    MESH_SCALING_CONFIG = {
        'allow_mesh_scaling': True,  # If True, network predicts a global mesh scale
        'init_mesh_scale': 1.0,  # Initial scale value (1.0 = no scaling)
        'use_log_scale': True,  # Predict log(scale) for numerical stability
        # Note: mesh_scale is trained implicitly via 3D keypoint losses (when available).
        # The 3D ground truth naturally constrains the scale to the correct value.
    }
    
    # Joint Importance Weighting Configuration
    # Allows certain joints to have higher weight in 2D and 3D keypoint losses.
    # This is crucial when learning joint angles implicitly from keypoint supervision,
    # as it incentivizes the model to prioritize getting specific joints correct.
    JOINT_IMPORTANCE_CONFIG = {
        'enabled': True,  # Set to True to enable per-joint importance weighting
        
        # List of joint names (from the SMAL model pkl file) to emphasize.
        # These joints will receive higher loss weight in 2D and 3D keypoint losses.
        # Example for mouse: ['nose', 'l_ear', 'r_ear', 'tail_tip']
        # Example for ant: ['b_h', 'ma_l', 'ma_r', 'b_a_5']
        # Check config.joint_names for available names in your model.
        'important_joint_names': ["Nose", "paw_L_tip", "paw_R_tip", "Foot_L_tip", "Foot_R_tip", "Tail_07"],
        
        # Weight multiplier applied to important joints.
        # Other joints receive weight 1.0, important joints receive this value.
        # Higher values = stronger emphasis on these joints.
        # Typical range: 2.0 - 10.0
        'weight_multiplier': 10.0,
    }
    
    # Loss curriculum configuration (AniMer-style conservative weights)
    LOSS_CURRICULUM = {
        # Base loss weights (applied throughout training) - AniMer-style conservative
        'base_weights': {
            'global_rot': 0.0,     # AniMer: 0.001
            'joint_rot': 0.001,      # AniMer: 0.001  
            'betas': 0.0005,         # AniMer: 0.0005
            'trans': 0.0005,         # AniMer: 0.0005
            'fov': 0.001,
            'cam_rot': 0.01,
            'cam_trans': 0.01,
            'log_beta_scales': 0.0005,  # Conservative
            'betas_trans': 0.0005,      # Conservative
            'keypoint_2d': 0.1,        # AniMer: 0.01
            'keypoint_3d': 0.25,        # 3D keypoint loss - start at zero
            'silhouette': 0.0,          # Start at zero
            'joint_angle_regularization': 0.001,  # Penalty for large joint angles (excluding root)
            'limb_scale_regularization': 0.01,    # Penalty for deviations from scale=1 (log_beta_scales)
            'limb_trans_regularization': 1      # Heavy penalty for translation changes (betas_trans) - prevents artifacts
        },
        
        # Curriculum stages: (epoch_threshold, weight_updates) - AniMer-style conservative
        'curriculum_stages': [
            (1, {
                'joint_angle_regularization': 0.01,
                'limb_scale_regularization': 0.1,   # Start with higher penalty
                'limb_trans_regularization': 1     # Very high penalty for translation early on, essentially prohibit any translation at thsi stage
            }),
            (10, {
                'keypoint_2d': 0.1,    # AniMer: 0.01
                'joint_angle_regularization': 0.005,
                'limb_scale_regularization': 0.05,
                'limb_trans_regularization': 1
            }),
            (25, {
                'keypoint_2d': 0.2,    # AniMer: 0.01
                'joint_angle_regularization': 0.0025,
                'limb_scale_regularization': 0.02,
                'limb_trans_regularization': 1
            }),
            (35, {
                'keypoint_3d': 1,    # AniMer: 0.01
                'joint_angle_regularization': 0.001,
                'limb_scale_regularization': 0.01,
                'limb_trans_regularization': 1
            }),
            (45, {
                'keypoint_3d': 1,    # AniMer: 0.01
                'joint_angle_regularization': 0.0001,
                'limb_scale_regularization': 0.005,  # Keep at reasonable level to allow scales while preventing extreme values
                'limb_trans_regularization': 1
            }),
            (50, {
                'keypoint_3d': 2,    # AniMer: 0.01
                'joint_angle_regularization': 0.00005,
                'limb_scale_regularization': 0.001,  # Gradually reduce but keep meaningful
                'limb_trans_regularization': 0.5
            }),
            (100, {
                'keypoint_3d': 2,    # AniMer: 0.01
                'keypoint_2d': 0.2,    # AniMer: 0.01
                'joint_angle_regularization': 0.00001,
                'limb_scale_regularization': 0.0000001,  # Keep small but non-zero to prevent extreme scales
                'limb_trans_regularization': 0.1
            }),
            (300, {
                'keypoint_3d': 2,    # AniMer: 0.01
                'keypoint_2d': 0.2,    # AniMer: 0.01
                'joint_angle_regularization': 0.00001,
                'limb_scale_regularization': 0.0000001,  # Keep small but non-zero to prevent extreme scales
                'limb_trans_regularization': 0.1,
                'fov': 0.0000001, # reduce influence to allow for looser camera parameters
                'cam_rot': 0.00000001, # reduce influence to allow for looser camera parameters
                'cam_trans': 0.00000001 # reduce influence to allow for looser camera parameters
            }),

            (400, {
                'keypoint_3d': 2,    # AniMer: 0.01
                'keypoint_2d': 0.4,    # AniMer: 0.01
                'joint_angle_regularization': 0.0001,
                'limb_scale_regularization': 0.00001,  # Keep small but non-zero to prevent extreme scales
                'limb_trans_regularization': 0.1,
                'fov': 0.0000001, # reduce influence to allow for looser camera parameters
                'cam_rot': 0.00000001, # reduce influence to allow for looser camera parameters
                'cam_trans': 0.00000001 # reduce influence to allow for looser camera parameters
            }),

            (460, {
                'keypoint_3d': 2,    # AniMer: 0.01
                'keypoint_2d': 0.2,    # AniMer: 0.01
                'joint_angle_regularization': 0.00001,
                'limb_scale_regularization': 0.001,  # Keep small but non-zero to prevent extreme scales
                'limb_trans_regularization': 0.1,
                'fov': 0.0000001, # reduce influence to allow for looser camera parameters
                'cam_rot': 0.00000001, # reduce influence to allow for looser camera parameters
                'cam_trans': 0.00000001 # reduce influence to allow for looser camera parameters
            })

            ,

            (490, {
                'keypoint_3d': 2,    # AniMer: 0.01
                'keypoint_2d': 0.2,    # AniMer: 0.01
                'joint_angle_regularization': 0.000001,
                'limb_scale_regularization': 0.0001,  # Keep small but non-zero to prevent extreme scales
                'limb_trans_regularization': 0.1,
                'fov': 0.0000001, # reduce influence to allow for looser camera parameters
                'cam_rot': 0.00000001, # reduce influence to allow for looser camera parameters
                'cam_trans': 0.00000001 # reduce influence to allow for looser camera parameters
            }),
            (500, {
                'keypoint_3d': 20,    # AniMer: 0.01
                'keypoint_2d': 0.2,    # AniMer: 0.01
                'joint_angle_regularization': 0.0000001,
                'limb_scale_regularization': 0.00001,  # Keep small but non-zero to prevent extreme scales
                'limb_trans_regularization': 0.1,
                'fov': 0.000001, # reduce influence to allow for looser camera parameters
                'cam_rot': 0.0000001, # reduce influence to allow for looser camera parameters
                'cam_trans': 0.0000001 # reduce influence to allow for looser camera parameters
            }),

            (560, {
                'keypoint_3d': 20,    # AniMer: 0.01
                'keypoint_2d': 0.2,    # AniMer: 0.01
                'joint_angle_regularization': 0.0000001,
                'limb_scale_regularization': 0.001,  # Keep small but non-zero to prevent extreme scales
                'limb_trans_regularization': 1.0,
                'fov': 0.000001, # reduce influence to allow for looser camera parameters
                'cam_rot': 0.0000001, # reduce influence to allow for looser camera parameters
                'cam_trans': 0.0000001 # reduce influence to allow for looser camera parameters
            }),

            (575, {
                'keypoint_3d': 20,    # AniMer: 0.01
                'keypoint_2d': 0.2,    # AniMer: 0.01
                'joint_angle_regularization': 0.0000001,
                'limb_scale_regularization': 0.0025,  # Keep small but non-zero to prevent extreme scales
                'limb_trans_regularization': 1.0,
                'fov': 0.000001, # reduce influence to allow for looser camera parameters
                'cam_rot': 0.0000001, # reduce influence to allow for looser camera parameters
                'cam_trans': 0.0000001 # reduce influence to allow for looser camera parameters
            })
        ]
    }
    
    # Learning rate curriculum configuration (AniMer-style conservative)
    LEARNING_RATE_CURRICULUM = {
        # Base learning rate (applied at epoch 0) - AniMer-style
        'base_learning_rate': 5e-5, #previous value: 1.25e-6
        
        # Learning rate stages: (epoch_threshold, learning_rate) - Very conservative
        'lr_stages': [
            # Stage 1: Slight reduction for fine-tuning
            (10, 3e-5),      # 1e-5

            # Stage 1: Slight reduction for fine-tuning
            (20, 2e-5),      # 1e-5
            
            # Stage 2: Further reduce
            (60, 1e-5),      # 5e-7

            # Stage 3: Very low learning rate for final convergence
            (100, 1e-5),      # 1e-7

            # Stage 3: Very low learning rate for final convergence
            (150, 2e-6),      # 1e-7

            # Stage 3: Very low learning rate for final convergence
            (200, 2e-6),      # 1e-7

            # Stage 3: Very low learning rate for final convergence
            (250, 1e-6),      # 1e-7

            (300, 1e-5),      # 1e-7

            (350, 1e-6),      # 1e-7

            (400, 1e-5),      # 1e-7

            (475, 1e-6),      # 1e-7

            (490, 5e-7),      # 1e-7

            # It's save to use high learning rates when using ONLY 2D loss here!
            (500, 5e-6),

            (550, 1e-6),

            # It's save to use high learning rates when using ONLY 2D loss here!
            (718, 1e-5)      # 1e-5
        ]
    }
    
    # Checkpoint and output configuration
    OUTPUT_CONFIG = {
        'checkpoint_dir': 'checkpoints',
        'plots_dir': 'plots',
        'visualizations_dir': 'visualizations',
        'train_visualizations_dir': 'visualizations_train',
        'save_checkpoint_every': 10,        # Save checkpoint every N epochs
        'generate_visualizations_every': 10, # Generate visualizations every N epochs
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
        
        # Apply scale/trans mode specific weights
        scale_trans_config = cls.get_scale_trans_config()
        mode = scale_trans_config['mode']
        if mode in scale_trans_config and 'loss_weights' in scale_trans_config[mode]:
            weights.update(scale_trans_config[mode]['loss_weights'])
        
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
    def get_scale_trans_config(cls):
        """Get scale and translation beta configuration."""
        return cls.SCALE_TRANS_BETA_CONFIG

    @classmethod
    def get_scale_trans_mode(cls):
        """Get current scale and translation mode."""
        return cls.SCALE_TRANS_BETA_CONFIG['mode']
    
    @classmethod
    def get_mesh_scaling_config(cls):
        """Get mesh scaling configuration."""
        return cls.MESH_SCALING_CONFIG
    
    @classmethod
    def is_mesh_scaling_enabled(cls) -> bool:
        """Check if global mesh scaling is enabled."""
        return cls.MESH_SCALING_CONFIG.get('allow_mesh_scaling', False)
    
    @classmethod
    def get_joint_importance_config(cls) -> Dict[str, Any]:
        """Get joint importance weighting configuration."""
        return cls.JOINT_IMPORTANCE_CONFIG
    
    @classmethod
    def is_joint_importance_enabled(cls) -> bool:
        """Check if joint importance weighting is enabled."""
        config = cls.JOINT_IMPORTANCE_CONFIG
        return (config.get('enabled', False) and 
                len(config.get('important_joint_names', [])) > 0 and
                config.get('weight_multiplier', 1.0) != 1.0)
    
    @classmethod
    def get_important_joint_names(cls) -> List[str]:
        """Get list of joint names that should have higher importance."""
        return cls.JOINT_IMPORTANCE_CONFIG.get('important_joint_names', [])
    
    @classmethod
    def get_joint_importance_multiplier(cls) -> float:
        """Get the weight multiplier for important joints."""
        return cls.JOINT_IMPORTANCE_CONFIG.get('weight_multiplier', 1.0)
    
    @classmethod
    def get_dataset_fraction(cls) -> float:
        """
        Get the fraction of training data to use per epoch.
        
        Returns:
            Float between 0 and 1 (1.0 = full dataset)
        """
        return cls.SPLIT_CONFIG.get('dataset_fraction', 1.0)
    
    @classmethod
    def is_multi_dataset_enabled(cls) -> bool:
        """Check if multi-dataset training is enabled."""
        return cls.MULTI_DATASET_CONFIG.get('enabled', False)
    
    @classmethod
    def get_enabled_datasets(cls) -> List[Dict[str, Any]]:
        """
        Get list of enabled dataset configurations.
        
        Returns:
            List of dataset configuration dictionaries
        """
        if not cls.is_multi_dataset_enabled():
            return []
        
        datasets = cls.MULTI_DATASET_CONFIG.get('datasets', [])
        return [ds for ds in datasets if ds.get('enabled', False)]
    
    @classmethod
    def get_dataset_config_by_name(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        Get dataset configuration by name.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset configuration dictionary or None if not found
        """
        for dataset in cls.get_enabled_datasets():
            if dataset['name'] == name:
                return dataset
        return None
    
    @classmethod
    def get_dataset_weights(cls) -> List[float]:
        """
        Get sampling weights for all enabled datasets.
        
        Returns:
            List of weights (one per dataset)
        """
        return [ds.get('weight', 1.0) for ds in cls.get_enabled_datasets()]
    
    @classmethod
    def get_validation_split_strategy(cls) -> str:
        """
        Get validation split strategy for multi-dataset training.
        
        Returns:
            'per_dataset' or 'combined'
        """
        return cls.MULTI_DATASET_CONFIG.get('validation_split_strategy', 'per_dataset')
    
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