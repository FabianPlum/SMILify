"""
Single-View Training Configuration

Extends BaseTrainingConfig with parameters specific to single-view
SMIL image regression training.
"""

from dataclasses import dataclass

from .base_config import BaseTrainingConfig


@dataclass
class SingleViewConfig(BaseTrainingConfig):
    """Configuration for single-view training."""

    def validate(self):
        super().validate()
        if self.dataset.split_method == "blocked" and not self.dataset.from_multiview:
            raise ValueError(
                "split_method='blocked' needs per-sample session_name / frame_idx, which only "
                "multi-view HDF5s carry; set dataset.from_multiview=true"
            )
