"""
Multi-View Training Configuration

Extends BaseTrainingConfig with parameters specific to multi-view
SMIL regression training with cross-view attention.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from .base_config import BaseTrainingConfig, OutputConfig


@dataclass
class MultiViewOutputConfig(OutputConfig):
    """Output config with multi-view specific defaults."""

    checkpoint_dir: str = "multiview_checkpoints"
    visualizations_dir: str = "multiview_visualizations"
    singleview_visualizations_dir: str = "multiview_singleview_renders"
    save_checkpoint_every: int = 10
    generate_visualizations_every: int = 10
    num_visualization_samples: int = 3


@dataclass
class SilhouetteConfig:
    """Rendering for the multi-view silhouette loss (loss weight 'silhouette' > 0, dataset.silhouette_path set).

    All valid views are rendered with a soft silhouette shader and compared to the targets with a masked soft IoU.
    """

    render_size: int = 128  # square render resolution; targets are area-resampled to it
    # Soft-rasterizer edge width. The soft edge makes the render slightly larger than the hard silhouette, so the
    # IoU optimum sits about 2-3% (up to 4% in some views) below the true mesh scale at 1e-5, about 3% at 3e-5;
    # 1e-5 still renders the dense mouse mesh without holes at faces_per_pixel 64.
    sigma: float = 1e-5
    faces_per_pixel: int = 64
    camera: str = (
        "gt"  # "gt": calibrated cameras (camera heads get no silhouette gradient); "pred": predicted, detached
    )
    cull_backfaces: bool = False

    def validate(self):
        if self.camera not in ("gt", "pred"):
            raise ValueError(f"silhouette.camera must be 'gt' or 'pred', got {self.camera!r}")


@dataclass
class MultiViewConfig(BaseTrainingConfig):
    """Configuration for multi-view training."""

    # Multi-view specific parameters
    num_views_to_use: Optional[int] = None  # Training only: at most N random views per sample (None = all)
    min_views_per_sample: int = 2
    # Temporary (issue #134): True reproduces the pre-#132 handling of padded view slots (backbone runs on the zero
    # padding, decoder attends to it). Only for legacy checkpoints, until they are replaced.
    legacy_view_padding: bool = False

    # Cross-attention configuration
    cross_attention_layers: int = 2
    cross_attention_heads: int = 8
    cross_attention_dropout: float = 0.1

    # Override output defaults for multi-view
    output: MultiViewOutputConfig = field(default_factory=MultiViewOutputConfig)

    # Silhouette loss rendering
    silhouette: SilhouetteConfig = field(default_factory=SilhouetteConfig)

    def validate(self):
        super().validate()
        self.silhouette.validate()
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
            "batch_size": self.training.batch_size,
            "num_epochs": self.training.num_epochs,
            "learning_rate": self.optimizer.learning_rate,
            "weight_decay": self.optimizer.weight_decay,
            "seed": self.training.seed,
            "rotation_representation": self.training.rotation_representation,
            "resume_checkpoint": self.training.resume_checkpoint,
            "init_checkpoint": self.training.init_checkpoint,
            "num_workers": self.training.num_workers,
            "pin_memory": self.training.pin_memory,
            "gradient_clip_norm": self.optimizer.gradient_clip_norm,
            # Model config
            "backbone_name": self.model.backbone_name,
            "freeze_backbone": self.model.freeze_backbone,
            "backbone_unfreeze_epoch": self.model.backbone_unfreeze_epoch,
            "backbone_lr_multiplier": self.model.backbone_lr_multiplier,
            "head_type": self.model.head_type,
            "hidden_dim": hidden_dim,
            "transformer_config": {
                "hidden_dim": hidden_dim,
                "depth": self.model.transformer_depth,
                "heads": self.model.transformer_heads,
                "dim_head": self.model.transformer_dim_head,
                "mlp_dim": self.model.transformer_mlp_dim,
                "dropout": self.model.transformer_dropout,
                "ief_iters": self.model.transformer_ief_iters,
                "trans_scale_factor": self.model.transformer_trans_scale_factor,
            },
            "use_unity_prior": self.model.use_unity_prior,
            # Multi-view specific
            "dataset_path": self.dataset.data_path,
            "num_views_to_use": self.num_views_to_use,
            "min_views_per_sample": self.min_views_per_sample,
            "legacy_view_padding": self.legacy_view_padding,
            "cross_attention_layers": self.cross_attention_layers,
            "cross_attention_heads": self.cross_attention_heads,
            "cross_attention_dropout": self.cross_attention_dropout,
            # Output directories
            "checkpoint_dir": self.output.checkpoint_dir,
            "visualizations_dir": self.output.visualizations_dir,
            "singleview_visualizations_dir": getattr(
                self.output, "singleview_visualizations_dir", "multiview_singleview_renders"
            ),
            "plots_dir": self.output.plots_dir,
            "save_every_n_epochs": self.output.save_checkpoint_every,
            "validate_every_n_epochs": 1,
            "visualize_every_n_epochs": self.output.generate_visualizations_every,
            "plot_history_every": self.output.plot_history_every,
            "num_visualization_samples": self.output.num_visualization_samples,
            # Split ratios
            "train_ratio": self.dataset.train_ratio,
            "val_ratio": self.dataset.val_ratio,
            "test_ratio": self.dataset.test_ratio,
            "split_method": self.dataset.split_method,
            "holdout_fold": self.dataset.holdout_fold,
            "keypoint_source": self.dataset.keypoint_source,
            "raw_keypoint_min_confidence": self.dataset.raw_keypoint_min_confidence,
            "allow_split_change": self.dataset.allow_split_change,
            # Silhouettes
            "silhouette_path": self.dataset.silhouette_path,
            "silhouette_exclude_label_flag": self.dataset.silhouette_exclude_label_flag,
            "silhouette_min_score": self.dataset.silhouette_min_score,
            "silhouette_config": asdict(self.silhouette),
            # Scale/trans
            "scale_trans_mode": self.scale_trans_beta.mode,
            "allow_mesh_scaling": self.mesh_scaling.allow_mesh_scaling,
            "mesh_scale_init": self.mesh_scaling.init_mesh_scale,
            # Training-split sub-sampling per epoch
            "train_sub_sampling": self.dataset.train_sub_sampling,
            "train_sub_sampling_by_session": dict(self.dataset.train_sub_sampling_by_session),
            # SMAL model configuration
            "shape_family": (
                self.smal_model.shape_family
                if self.smal_model is not None and self.smal_model.shape_family is not None
                else -1
            ),  # May be overridden by training script at runtime
            "smal_file": self.smal_model.smal_file if self.smal_model is not None else None,
            "loss_weights": self.get_loss_weights_for_epoch(0),
            "reset_ief_token_embedding": self.training.reset_ief_token_embedding,
            "use_gt_camera_init": self.training.use_gt_camera_init,
            "use_mixed_precision": self.training.use_mixed_precision,
            "backbone_chunk_size": self.training.backbone_chunk_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            # Joint importance / ignored joints / loss curriculum — must be carried
            # through so that DDP worker processes (mp.spawn / torchrun) can re-sync
            # TrainingConfig after being spawned as fresh Python processes.
            "joint_importance_config": {
                "enabled": self.joint_importance.enabled,
                "important_joint_names": list(self.joint_importance.important_joint_names),
                "weight_multiplier": self.joint_importance.weight_multiplier,
            },
            "ignored_joint_locations_config": {
                "enabled": self.ignored_joint_locations.enabled,
                "ignored_joint_names": list(self.ignored_joint_locations.ignored_joint_names),
            },
            "loss_curriculum_config": {
                "joint_twist_free_deg": self.loss_curriculum.joint_twist_free_deg,
                "base_weights": dict(self.loss_curriculum.base_weights),
                "curriculum_stages": [
                    (epoch, dict(updates)) for epoch, updates in sorted(self.loss_curriculum.curriculum_stages.items())
                ],
            },
            "lr_curriculum_config": {
                "base_learning_rate": self.optimizer.learning_rate,
                "lr_stages": [(epoch, lr) for epoch, lr in sorted(self.optimizer.lr_schedule.items())],
            },
            # Augmentation
            "augmentation": {
                "enabled": self.augmentation.enabled,
                "geometric_enabled": self.augmentation.geometric_enabled,
                "color_jitter_brightness": self.augmentation.color_jitter_brightness,
                "color_jitter_contrast": self.augmentation.color_jitter_contrast,
                "color_jitter_saturation": self.augmentation.color_jitter_saturation,
                "gaussian_noise_std": self.augmentation.gaussian_noise_std,
                "gaussian_blur_prob": self.augmentation.gaussian_blur_prob,
                "gaussian_blur_kernel_range": list(self.augmentation.gaussian_blur_kernel_range),
                "random_erasing_prob": self.augmentation.random_erasing_prob,
                "random_erasing_scale_range": list(self.augmentation.random_erasing_scale_range),
                "crop_jitter_fraction": self.augmentation.crop_jitter_fraction,
                "scale_jitter_range": list(self.augmentation.scale_jitter_range),
            },
        }
        return d
