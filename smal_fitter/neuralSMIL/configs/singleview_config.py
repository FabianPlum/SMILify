"""
Single-View Training Configuration

Extends BaseTrainingConfig with parameters specific to single-view
SMIL image regression training.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from .base_config import BaseTrainingConfig, MultiDatasetConfig


@dataclass
class SingleViewConfig(BaseTrainingConfig):
    """Configuration for single-view training."""

    def validate(self):
        super().validate()
