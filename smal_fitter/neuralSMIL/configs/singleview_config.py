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
