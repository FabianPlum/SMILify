"""
Multi-View Training Configuration

Extends BaseTrainingConfig with parameters specific to multi-view
SMIL regression training with cross-view attention.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .base_config import BaseTrainingConfig, OutputConfig


@dataclass
class MultiViewOutputConfig(OutputConfig):
    """Output config with multi-view specific defaults."""
    checkpoint_dir: str = 'multiview_checkpoints'
    visualizations_dir: str = 'multiview_visualizations'
    singleview_visualizations_dir: str = 'multiview_singleview_renders'
    save_checkpoint_every: int = 10
    generate_visualizations_every: int = 10
    num_visualization_samples: int = 3


@dataclass
class MultiViewConfig(BaseTrainingConfig):
    """Configuration for multi-view training."""

    # Multi-view specific parameters
    num_views_to_use: Optional[int] = None  # None = use all available views
    min_views_per_sample: int = 2

    # Cross-attention configuration
    cross_attention_layers: int = 2
    cross_attention_heads: int = 8
    cross_attention_dropout: float = 0.1

    # Override output defaults for multi-view
    output: MultiViewOutputConfig = field(default_factory=MultiViewOutputConfig)

    def validate(self):
        super().validate()
        if self.min_views_per_sample < 1:
            raise ValueError("min_views_per_sample must be >= 1")

    def to_multiview_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to the flat dict format expected by the existing multi-view main().

        The multiview training script uses a flat dict with keys like 'seed',
        'dataset_path', 'backbone_name', 'cross_attention_layers', etc.
        This bridges the new config system with that existing interface.
        """
        hidden_dim = self.model.get_adjusted_hidden_dim()

        d = {
            # Training params
            'batch_size': self.training.batch_size,
            'num_epochs': self.training.num_epochs,
            'learning_rate': self.optimizer.learning_rate,
            'weight_decay': self.optimizer.weight_decay,
            'seed': self.training.seed,
            'rotation_representation': self.training.rotation_representation,
            'resume_checkpoint': self.training.resume_checkpoint,
            'num_workers': self.training.num_workers,
            'pin_memory': self.training.pin_memory,
            'gradient_clip_norm': self.optimizer.gradient_clip_norm,

            # Model config
            'backbone_name': self.model.backbone_name,
            'freeze_backbone': self.model.freeze_backbone,
            'head_type': self.model.head_type,
            'hidden_dim': hidden_dim,
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
            'use_unity_prior': self.model.use_unity_prior,

            # Multi-view specific
            'dataset_path': self.dataset.data_path,
            'num_views_to_use': self.num_views_to_use,
            'min_views_per_sample': self.min_views_per_sample,
            'cross_attention_layers': self.cross_attention_layers,
            'cross_attention_heads': self.cross_attention_heads,
            'cross_attention_dropout': self.cross_attention_dropout,

            # Output directories
            'checkpoint_dir': self.output.checkpoint_dir,
            'visualizations_dir': self.output.visualizations_dir,
            'singleview_visualizations_dir': getattr(self.output, 'singleview_visualizations_dir', 'multiview_singleview_renders'),
            'save_every_n_epochs': self.output.save_checkpoint_every,
            'validate_every_n_epochs': 1,
            'visualize_every_n_epochs': self.output.generate_visualizations_every,
            'num_visualization_samples': self.output.num_visualization_samples,

            # Split ratios
            'train_ratio': self.dataset.train_ratio,
            'val_ratio': self.dataset.val_ratio,
            'test_ratio': self.dataset.test_ratio,

            # Scale/trans
            'scale_trans_mode': self.scale_trans_beta.mode,
            'allow_mesh_scaling': self.mesh_scaling.allow_mesh_scaling,
            'mesh_scale_init': self.mesh_scaling.init_mesh_scale,

            # Dataset fraction
            'dataset_fraction': self.dataset.dataset_fraction,

            # Legacy fields
            'shape_family': (
                self.legacy.shape_family
                if self.legacy is not None and self.legacy.shape_family is not None
                else -1
            ),  # May be overridden by training script at runtime
            'smal_file': self.legacy.smal_file if self.legacy is not None else None,
            'loss_weights': self.get_loss_weights_for_epoch(0),
            'use_gt_camera_init': self.training.use_gt_camera_init,
        }
        return d
