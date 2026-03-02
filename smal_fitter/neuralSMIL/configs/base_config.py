"""
Base Configuration for SMIL Training

Defines shared configuration parameters for both single-view and multi-view
training of the SMIL image regressor. Uses Python dataclasses for type safety
and IDE support.

Configuration Precedence (highest to lowest):
    1. CLI arguments
    2. JSON config file
    3. Mode-specific defaults (SingleViewConfig / MultiViewConfig)
    4. Base defaults (this file)
    5. Legacy config.py (only for SHAPE_FAMILY, SMAL_FILE, etc.)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class DatasetConfig:
    """Dataset paths and split configuration."""
    data_path: Optional[str] = None  # Required at runtime
    train_ratio: float = 0.85
    val_ratio: float = 0.05
    test_ratio: float = 0.1
    dataset_fraction: float = 0.5  # Fraction of training data used per epoch

    def validate(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if total > 1.0 + 1e-9:
            raise ValueError(f"Split ratios sum to {total}, must be <= 1.0")

    def get_split_sizes(self, dataset_size: int) -> Tuple[int, int, int]:
        """Calculate train, val, test sizes from total dataset size."""
        test_size = int(dataset_size * self.test_ratio)
        val_size = int(dataset_size * self.val_ratio)
        train_size = dataset_size - test_size - val_size
        return train_size, val_size, test_size


@dataclass
class ModelConfig:
    """Neural network model architecture."""
    backbone_name: str = 'vit_large_patch16_224'
    freeze_backbone: bool = True
    hidden_dim: int = 1024  # Auto-adjusted based on backbone in validate()
    head_type: str = 'transformer_decoder'  # 'mlp' or 'transformer_decoder'
    use_unity_prior: bool = False
    rgb_only: bool = False

    # Transformer decoder configuration
    transformer_depth: int = 6
    transformer_heads: int = 8
    transformer_dim_head: int = 64
    transformer_mlp_dim: int = 1024
    transformer_dropout: float = 0.1
    transformer_ief_iters: int = 3
    transformer_trans_scale_factor: int = 1  # Scale factor for betas_trans

    def validate(self):
        if self.head_type not in ('mlp', 'transformer_decoder'):
            raise ValueError(f"Invalid head_type '{self.head_type}', must be 'mlp' or 'transformer_decoder'")

    def get_adjusted_hidden_dim(self) -> int:
        """Return hidden_dim adjusted for backbone architecture."""
        if self.backbone_name.startswith('vit'):
            if 'base' in self.backbone_name:
                return 768
            elif 'large' in self.backbone_name:
                return 1024
        elif self.backbone_name.startswith('resnet'):
            return 2048
        return self.hidden_dim


@dataclass
class OptimizerConfig:
    """Optimizer and learning rate scheduling."""
    learning_rate: float = 5e-5  # Base LR (epoch 0)
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    optimizer_type: str = 'adamw'

    # Learning rate curriculum: epoch threshold -> learning rate
    # JSON keys are strings, converted to int on load via config_utils
    lr_schedule: Dict[int, float] = field(default_factory=lambda: {
        0: 5e-5,
        10: 3e-5,
        20: 2e-5,
        60: 1e-5,
        100: 1e-5,
        150: 2e-6,
        200: 2e-6,
        250: 1e-6,
        300: 1e-5,
        350: 1e-6,
        400: 1e-5,
        475: 1e-6,
        490: 5e-7,
        500: 5e-6,
        550: 1e-6,
        718: 1e-5,
    })

    def get_learning_rate_for_epoch(self, epoch: int) -> float:
        """Get learning rate for given epoch following curriculum."""
        lr = self.learning_rate
        for threshold in sorted(self.lr_schedule.keys()):
            if epoch >= threshold:
                lr = self.lr_schedule[threshold]
        return lr


@dataclass
class LossCurriculumConfig:
    """Loss weights and curriculum stages for progressive training."""

    # Base loss weights (applied throughout training)
    base_weights: Dict[str, float] = field(default_factory=lambda: {
        'global_rot': 0.0,
        'joint_rot': 0.001,
        'betas': 0.0005,
        'trans': 0.0005,
        'fov': 0.001,
        'cam_rot': 0.01,
        'cam_trans': 0.01,
        'log_beta_scales': 0.0005,
        'betas_trans': 0.0005,
        'keypoint_2d': 0.1,
        'keypoint_3d': 0.25,
        'silhouette': 0.0,
        'joint_angle_regularization': 0.001,
        'limb_scale_regularization': 0.01,
        'limb_trans_regularization': 1,
    })

    # Curriculum stages: epoch threshold -> dict of weight overrides
    # JSON keys are strings, converted to int on load via config_utils
    curriculum_stages: Dict[int, Dict[str, float]] = field(default_factory=lambda: {
        1: {
            'joint_angle_regularization': 0.01,
            'limb_scale_regularization': 0.1,
            'limb_trans_regularization': 1,
        },
        10: {
            'keypoint_2d': 0.1,
            'joint_angle_regularization': 0.005,
            'limb_scale_regularization': 0.05,
            'limb_trans_regularization': 1,
        },
        25: {
            'keypoint_2d': 0.2,
            'joint_angle_regularization': 0.0025,
            'limb_scale_regularization': 0.02,
            'limb_trans_regularization': 1,
        },
        35: {
            'keypoint_3d': 1,
            'joint_angle_regularization': 0.001,
            'limb_scale_regularization': 0.01,
            'limb_trans_regularization': 1,
        },
        45: {
            'keypoint_3d': 1,
            'joint_angle_regularization': 0.0001,
            'limb_scale_regularization': 0.005,
            'limb_trans_regularization': 1,
        },
        50: {
            'keypoint_3d': 2,
            'joint_angle_regularization': 0.00005,
            'limb_scale_regularization': 0.001,
            'limb_trans_regularization': 0.5,
        },
        100: {
            'keypoint_3d': 2,
            'keypoint_2d': 0.2,
            'joint_angle_regularization': 0.00001,
            'limb_scale_regularization': 0.0000001,
            'limb_trans_regularization': 0.1,
        },
        300: {
            'keypoint_3d': 2,
            'keypoint_2d': 0.2,
            'joint_angle_regularization': 0.00001,
            'limb_scale_regularization': 0.0000001,
            'limb_trans_regularization': 0.1,
            'fov': 0.0000001,
            'cam_rot': 0.00000001,
            'cam_trans': 0.00000001,
        },
        400: {
            'keypoint_3d': 2,
            'keypoint_2d': 0.4,
            'joint_angle_regularization': 0.0001,
            'limb_scale_regularization': 0.00001,
            'limb_trans_regularization': 0.1,
            'fov': 0.0000001,
            'cam_rot': 0.00000001,
            'cam_trans': 0.00000001,
        },
        460: {
            'keypoint_3d': 2,
            'keypoint_2d': 0.2,
            'joint_angle_regularization': 0.00001,
            'limb_scale_regularization': 0.001,
            'limb_trans_regularization': 0.1,
            'fov': 0.0000001,
            'cam_rot': 0.00000001,
            'cam_trans': 0.00000001,
        },
        490: {
            'keypoint_3d': 2,
            'keypoint_2d': 0.2,
            'joint_angle_regularization': 0.000001,
            'limb_scale_regularization': 0.0001,
            'limb_trans_regularization': 0.1,
            'fov': 0.0000001,
            'cam_rot': 0.00000001,
            'cam_trans': 0.00000001,
        },
        500: {
            'keypoint_3d': 20,
            'keypoint_2d': 0.2,
            'joint_angle_regularization': 0.0000001,
            'limb_scale_regularization': 0.00001,
            'limb_trans_regularization': 0.1,
            'fov': 0.000001,
            'cam_rot': 0.0000001,
            'cam_trans': 0.0000001,
        },
        560: {
            'keypoint_3d': 20,
            'keypoint_2d': 0.2,
            'joint_angle_regularization': 0.0000001,
            'limb_scale_regularization': 0.001,
            'limb_trans_regularization': 1.0,
            'fov': 0.000001,
            'cam_rot': 0.0000001,
            'cam_trans': 0.0000001,
        },
        575: {
            'keypoint_3d': 20,
            'keypoint_2d': 0.2,
            'joint_angle_regularization': 0.0000001,
            'limb_scale_regularization': 0.0025,
            'limb_trans_regularization': 1.0,
            'fov': 0.000001,
            'cam_rot': 0.0000001,
            'cam_trans': 0.0000001,
        },
    })

    def get_weights_for_epoch(self, epoch: int) -> Dict[str, float]:
        """Get loss weights for given epoch, applying curriculum overrides."""
        weights = self.base_weights.copy()
        for threshold in sorted(self.curriculum_stages.keys()):
            if epoch >= threshold:
                weights.update(self.curriculum_stages[threshold])
        return weights


@dataclass
class ScaleTransBetaConfig:
    """Scale and translation beta handling configuration."""
    mode: str = 'entangled_with_betas'  # 'ignore', 'separate', or 'entangled_with_betas'

    # Per-mode loss weight overrides (applied on top of base loss weights)
    ignore_loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'log_beta_scales': 0.0,
        'betas_trans': 0.0,
    })
    separate_loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'log_beta_scales': 0.0005,
        'betas_trans': 0.0005,
    })
    entangled_loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'betas': 0.0005,
        'log_beta_scales': 0.01,
        'betas_trans': 0.01,
    })
    separate_trans_scale_factor: float = 0.01

    def get_mode_loss_weights(self) -> Dict[str, float]:
        """Get loss weight overrides for the current mode."""
        if self.mode == 'ignore':
            return self.ignore_loss_weights
        elif self.mode == 'separate':
            return self.separate_loss_weights
        elif self.mode == 'entangled_with_betas':
            return self.entangled_loss_weights
        return {}

    def validate(self):
        if self.mode not in ('ignore', 'separate', 'entangled_with_betas'):
            raise ValueError(f"Invalid scale_trans_beta mode '{self.mode}'")


@dataclass
class MeshScalingConfig:
    """Global mesh scaling configuration."""
    allow_mesh_scaling: bool = True
    init_mesh_scale: float = 1.0
    use_log_scale: bool = True


@dataclass
class IgnoredJointLocationsConfig:
    """Loss-level joint location exclusion for 2D and 3D keypoint losses.

    Unlike IgnoredJointsConfig (data preprocessing), this operates during loss
    computation so joints remain in the dataset but are simply not supervised.
    """
    enabled: bool = True
    ignored_joint_names: List[str] = field(default_factory=list)


@dataclass
class JointImportanceConfig:
    """Per-joint importance weighting for keypoint losses."""
    enabled: bool = True
    important_joint_names: List[str] = field(default_factory=lambda: [])
    weight_multiplier: float = 10.0

    def is_active(self) -> bool:
        return (self.enabled
                and len(self.important_joint_names) > 0
                and self.weight_multiplier != 1.0)


@dataclass
class IgnoredJointsConfig:
    """Joints to ignore during training due to mesh vs data misalignment."""
    ignored_joint_names: List[str] = field(default_factory=list)
    verbose: bool = True


@dataclass
class MultiDatasetEntry:
    """Configuration for a single dataset in multi-dataset training."""
    name: str = ''
    path: str = ''
    type: str = 'optimized_hdf5'  # 'replicant', 'sleap', 'optimized_hdf5', 'auto'
    weight: float = 1.0
    enabled: bool = True
    available_labels: Dict[str, bool] = field(default_factory=lambda: {
        'global_rot': True, 'joint_rot': True, 'betas': True,
        'trans': True, 'fov': True, 'cam_rot': True, 'cam_trans': True,
        'log_beta_scales': True, 'betas_trans': True,
        'keypoint_2d': True, 'keypoint_3d': True, 'silhouette': True,
    })


@dataclass
class MultiDatasetConfig:
    """Multi-dataset training configuration."""
    enabled: bool = False
    datasets: List[Dict[str, Any]] = field(default_factory=list)
    validation_split_strategy: str = 'per_dataset'  # 'per_dataset' or 'combined'


@dataclass
class OutputConfig:
    """Checkpoint and visualization output settings."""
    checkpoint_dir: str = 'checkpoints'
    plots_dir: str = 'plots'
    visualizations_dir: str = 'visualizations'
    train_visualizations_dir: str = 'visualizations_train'
    save_checkpoint_every: int = 10
    generate_visualizations_every: int = 10
    plot_history_every: int = 10
    num_visualization_samples: int = 10


@dataclass
class TrainingHyperparameters:
    """General training hyperparameters."""
    batch_size: int = 8
    num_epochs: int = 1000
    seed: int = 1234
    rotation_representation: str = '6d'  # '6d' or 'axis_angle'
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 4
    resume_checkpoint: Optional[str] = None
    reset_ief_token_embedding: bool = False
    use_gt_camera_init: bool = True


@dataclass
class SmalModelConfig:
    """
    SMAL/SMIL model configuration overrides for inference and training.

    Allows specifying the SMAL model file and shape family without modifying
    `config.py` itself. These values are used by `apply_smal_file_override()`
    to reload `config.dd`, `config.N_POSE`, `config.N_BETAS`, etc.

    Notes:
    - `smal_file`: path to the SMAL/SMIL model pickle used by `config.py` to
      populate `dd`, `N_POSE`, `N_BETAS`, etc. If you override this at runtime,
      you must reload `config` for derived fields to update.
    - `shape_family`: the shape family passed into SMAL/SMIL fitter code.
    """
    smal_file: Optional[str] = None
    shape_family: Optional[int] = None


@dataclass
class BaseTrainingConfig:
    """
    Complete base configuration for all training modes.

    This is the single source of truth for shared training parameters.
    SingleViewConfig and MultiViewConfig extend this with mode-specific settings.
    """
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss_curriculum: LossCurriculumConfig = field(default_factory=LossCurriculumConfig)
    scale_trans_beta: ScaleTransBetaConfig = field(default_factory=ScaleTransBetaConfig)
    mesh_scaling: MeshScalingConfig = field(default_factory=MeshScalingConfig)
    joint_importance: JointImportanceConfig = field(default_factory=JointImportanceConfig)
    ignored_joint_locations: IgnoredJointLocationsConfig = field(default_factory=IgnoredJointLocationsConfig)
    ignored_joints: IgnoredJointsConfig = field(default_factory=IgnoredJointsConfig)
    multi_dataset: MultiDatasetConfig = field(default_factory=MultiDatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    training: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)
    smal_model: SmalModelConfig = field(default_factory=SmalModelConfig)

    def validate(self):
        """Validate entire configuration for consistency."""
        self.dataset.validate()
        self.model.validate()
        self.scale_trans_beta.validate()
        if self.training.rotation_representation not in ('6d', 'axis_angle'):
            raise ValueError(
                f"Invalid rotation_representation '{self.training.rotation_representation}'"
            )

    def get_loss_weights_for_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Get fully resolved loss weights for a given epoch.

        Applies curriculum stages and scale_trans_beta mode overrides.
        """
        weights = self.loss_curriculum.get_weights_for_epoch(epoch)
        weights.update(self.scale_trans_beta.get_mode_loss_weights())
        return weights

    def get_learning_rate_for_epoch(self, epoch: int) -> float:
        """Get learning rate for a given epoch from curriculum."""
        return self.optimizer.get_learning_rate_for_epoch(epoch)

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to the legacy dict format expected by existing training code.

        This bridges the new config system with the existing main() functions
        in train_smil_regressor.py and train_multiview_regressor.py, allowing
        incremental migration without rewriting all training logic at once.

        Returns:
            Dictionary matching the format of TrainingConfig.get_all_config()
        """
        hidden_dim = self.model.get_adjusted_hidden_dim()

        legacy_shape_family = (
            self.smal_model.shape_family
            if self.smal_model is not None and self.smal_model.shape_family is not None
            else None
        )
        legacy_smal_file = self.smal_model.smal_file if self.smal_model is not None else None

        return {
            'data_path': self.dataset.data_path,
            # Legacy overrides (consumed by callers; not part of TrainingConfig.get_all_config())
            'shape_family': legacy_shape_family,
            'smal_file': legacy_smal_file,
            'split_config': {
                'test_size': self.dataset.test_ratio,
                'val_size': self.dataset.val_ratio,
                'dataset_fraction': self.dataset.dataset_fraction,
            },
            'training_params': {
                'batch_size': self.training.batch_size,
                'num_epochs': self.training.num_epochs,
                'learning_rate': self.optimizer.learning_rate,
                'weight_decay': self.optimizer.weight_decay,
                'seed': self.training.seed,
                'rotation_representation': self.training.rotation_representation,
                'resume_checkpoint': self.training.resume_checkpoint,
                'num_workers': self.training.num_workers,
                'pin_memory': self.training.pin_memory,
                'prefetch_factor': self.training.prefetch_factor,
            },
            'model_config': {
                'backbone_name': self.model.backbone_name,
                'freeze_backbone': self.model.freeze_backbone,
                'hidden_dim': hidden_dim,
                'rgb_only': self.model.rgb_only,
                'use_unity_prior': self.model.use_unity_prior,
                'head_type': self.model.head_type,
                'transformer_config': {
                    'hidden_dim': hidden_dim,
                    'depth': self.model.transformer_depth,
                    'heads': self.model.transformer_heads,
                    'dim_head': self.model.transformer_dim_head,
                    'mlp_dim': self.model.transformer_mlp_dim,
                    'dropout': self.model.transformer_dropout,
                    'ief_iters': self.model.transformer_ief_iters,
                    'trans_scale_factor': self.model.transformer_trans_scale_factor,
                },
            },
            'ignored_joints_config': {
                'ignored_joint_names': self.ignored_joints.ignored_joint_names,
                'verbose_ignored_joints': self.ignored_joints.verbose,
            },
            'loss_curriculum': {
                'base_weights': dict(self.loss_curriculum.base_weights),
                'curriculum_stages': [
                    (epoch, dict(updates))
                    for epoch, updates in sorted(self.loss_curriculum.curriculum_stages.items())
                ],
            },
            'learning_rate_curriculum': {
                'base_learning_rate': self.optimizer.learning_rate,
                'lr_stages': [
                    (epoch, lr)
                    for epoch, lr in sorted(self.optimizer.lr_schedule.items())
                ],
            },
            'output_config': {
                'checkpoint_dir': self.output.checkpoint_dir,
                'plots_dir': self.output.plots_dir,
                'visualizations_dir': self.output.visualizations_dir,
                'train_visualizations_dir': self.output.train_visualizations_dir,
                'save_checkpoint_every': self.output.save_checkpoint_every,
                'generate_visualizations_every': self.output.generate_visualizations_every,
                'plot_history_every': self.output.plot_history_every,
                'num_visualization_samples': self.output.num_visualization_samples,
            },
            'scale_trans_beta': {
                'mode': self.scale_trans_beta.mode,
            },
            'joint_importance': {
                'enabled': self.joint_importance.enabled,
                'important_joint_names': list(self.joint_importance.important_joint_names),
                'weight_multiplier': self.joint_importance.weight_multiplier,
            },
            'ignored_joint_locations': {
                'enabled': self.ignored_joint_locations.enabled,
                'ignored_joint_names': list(self.ignored_joint_locations.ignored_joint_names),
            },
        }
