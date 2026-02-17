"""
SMILify Unified Configuration System

Supports single-view and multi-view training with JSON config files,
CLI overrides, and backward compatibility with legacy config.py.

Configuration Precedence (highest to lowest):
    1. CLI arguments
    2. JSON config file (--config path/to/config.json)
    3. Mode-specific defaults (SingleViewConfig / MultiViewConfig)
    4. Base defaults (BaseTrainingConfig)
    5. Legacy config.py (SHAPE_FAMILY, SMAL_FILE, etc.)

Example:
    from smal_fitter.neuralSMIL.configs import load_config, SingleViewConfig

    # Load from JSON
    config = load_config(config_file='experiments/baseline.json')

    # Create with defaults
    config = SingleViewConfig()
    config.training.batch_size = 4
    config.validate()
"""

from .base_config import (
    BaseTrainingConfig,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    LossCurriculumConfig,
    ScaleTransBetaConfig,
    MeshScalingConfig,
    JointImportanceConfig,
    IgnoredJointsConfig,
    MultiDatasetConfig,
    OutputConfig,
    TrainingHyperparameters,
    LegacyOverridesConfig,
)
from .singleview_config import SingleViewConfig
from .multiview_config import MultiViewConfig, MultiViewOutputConfig
from .config_utils import (
    load_config,
    load_from_json,
    save_config_json,
    validate_json_mode,
    apply_smal_file_override,
    ConfigurationError,
)

__all__ = [
    # Base classes
    'BaseTrainingConfig',
    'DatasetConfig',
    'ModelConfig',
    'OptimizerConfig',
    'LossCurriculumConfig',
    'ScaleTransBetaConfig',
    'MeshScalingConfig',
    'JointImportanceConfig',
    'IgnoredJointsConfig',
    'MultiDatasetConfig',
    'OutputConfig',
    'TrainingHyperparameters',
    'LegacyOverridesConfig',
    # Mode-specific
    'SingleViewConfig',
    'MultiViewConfig',
    'MultiViewOutputConfig',
    # Utilities
    'load_config',
    'load_from_json',
    'save_config_json',
    'validate_json_mode',
    'apply_smal_file_override',
    'ConfigurationError',
]
