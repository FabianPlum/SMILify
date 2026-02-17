# SMILify Configuration System Refactor Plan

## Executive Summary

This document outlines a comprehensive refactor to consolidate and clarify the SMILify configuration system. Currently, configuration is scattered across 4+ files with unclear precedence rules. The new system provides:

- **Single source of truth** for each parameter (dataclass-based)
- **Explicit precedence hierarchy** (CLI > JSON > mode-specific > base > legacy)
- **Backward compatibility** with existing `config.py` for legacy code
- **JSON config files** for reproducibility and experimentation
- **Type safety** with dataclasses and validation
- **Clear separation** between single-view and multi-view training configurations

---

## Current State Analysis

### Problem Overview

The current configuration system is fragmented:

```
config.py (226 lines)
├── Legacy SMAL model paths (REQUIRED by fitter_3d)
├── Joint configurations
├── GPU settings
└── Optimization weights (only for 3D fitting)

training_config.py (769 lines)
├── TRAINING_PARAMS (batch_size, learning_rate, num_epochs)
├── MODEL_CONFIG (backbone, transformer settings)
├── LOSS_CURRICULUM (13+ stages, 15+ weights)
├── LEARNING_RATE_CURRICULUM (13 stages)
└── Helper methods (get_loss_weights_for_epoch, etc.)

train_smil_regressor.py (1900+ lines)
├── CLI argument parsing (batch_size, learning_rate, seed, etc.)
├── config_override creation
└── Partial use of training_config.py

train_multiview_regressor.py (2800+ lines)
├── Different CLI arguments
├── MultiViewTrainingConfig class (in-file)
├── MULTIVIEW_DEFAULTS dict
└── Different precedence rules
```

### Key Issues

1. **Parameter Duplication**: `batch_size`, `learning_rate`, `num_epochs` defined in multiple places
2. **Unclear Precedence**: No single documented source for which value "wins"
3. **Curriculum Opacity**: Loss and LR curriculum buried in training_config.py, hard to modify
4. **Mode Inconsistency**: Single-view and multi-view use different config mechanisms
5. **Legacy Coupling**: Hard to separate concerns between modern training and legacy fitter_3d code
6. **JSON Schema Constraints**: Curriculum dicts use string keys (JSON limitation) without conversion

### Legacy Constraints

**These config.py parameters CANNOT be changed** (required by `fitter_3d/optimise.py` and `smal_fitter/optimize_to_joints.py`):

- `SMAL_FILE` - Path to SMAL/SMIL model file
- `SHAPE_FAMILY` - Animal family selection
- `N_BETAS`, `N_POSE` - Shape/pose parameter counts
- `GPU_IDS` - GPU selection for legacy optimization
- `PLOT_RESULTS` - Visualization flag for 3D fitting

---

## Proposed Architecture

### Design Principles

1. **Separation of Concerns**: Legacy (config.py) ≠ Training (new system)
2. **Single Source of Truth**: Each parameter defined once
3. **Explicit Hierarchy**: Documented, validated precedence rules
4. **Type Safety**: Dataclasses with type hints for IDE support
5. **Validation**: Config validation at creation time (fail fast)
6. **Backward Compatibility**: Existing code continues to work
7. **Reproducibility**: Configs saved alongside checkpoints

### New Directory Structure

```
smal_fitter/neuralSMIL/
├── configs/                          # NEW: Unified config system
│   ├── __init__.py                   # Re-export public APIs
│   ├── base_config.py                # Shared base configuration
│   ├── singleview_config.py          # Single-view specific config
│   ├── multiview_config.py           # Multi-view specific config
│   ├── config_utils.py               # Loading, merging, validation
│   ├── README.md                     # Configuration system docs
│   └── examples/
│       ├── README.md                 # Guide to config files
│       ├── singleview_baseline.json  # Example single-view config
│       └── multiview_6cam.json       # Example multi-view config
│
├── training_config.py                # DEPRECATED (becomes compat shim)
├── train_smil_regressor.py           # MODIFIED (uses new config system)
├── train_multiview_regressor.py      # MODIFIED (uses new config system)
└── config.py                         # UNCHANGED (legacy compatibility)
```

---

## Detailed Design

### 1. Base Configuration (base_config.py)

Defines all shared parameters for both training modes.

**Key Features**:
- Dataclass-based for type safety and IDE support
- Nested dataclasses for logical grouping
- Validation methods for consistency checks
- Curriculum handling with epoch-based lookups
- JSON-compatible dict structures with automatic key conversion

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass
class DatasetConfig:
    """Dataset paths and split configuration"""
    data_path: Optional[str] = None  # Required at runtime
    train_ratio: float = 0.85
    val_ratio: float = 0.05
    test_ratio: float = 0.1
    dataset_fraction: float = 1.0  # Fraction of training data used per epoch

    def validate(self):
        """Validate that ratios sum to <= 1.0"""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        assert total <= 1.0, f"Ratios sum to {total}, must be <= 1.0"

@dataclass
class ModelConfig:
    """Neural network model architecture"""
    backbone_name: str = 'vit_large_patch16_224'
    freeze_backbone: bool = True
    hidden_dim: int = 1024
    head_type: str = 'transformer_decoder'  # 'mlp' or 'transformer_decoder'
    use_unity_prior: bool = False
    rgb_only: bool = False

    # Transformer decoder configuration
    transformer_depth: int = 6
    transformer_heads: int = 8
    transformer_dim_head: int = 64
    transformer_mlp_dim: int = 1024
    transformer_dropout: float = 0.1
    transformer_ief_iters: int = 3  # Iterative Error Feedback iterations

@dataclass
class OptimizerConfig:
    """Optimizer and learning rate scheduling"""
    learning_rate: float = 5e-5  # Base LR (epoch 0)
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    optimizer_type: str = 'adamw'

    # Learning rate curriculum: epoch -> learning_rate
    # JSON keys are strings, converted to int on load
    lr_schedule: Dict[int, float] = field(default_factory=lambda: {
        0: 5e-5, 10: 3e-5, 20: 2e-5, 60: 1e-5,
        # ... full schedule from training_config.py
    })

    def get_learning_rate_for_epoch(self, epoch: int) -> float:
        """Get learning rate for given epoch following curriculum"""
        lr = self.learning_rate
        for threshold in sorted(self.lr_schedule.keys()):
            if epoch >= threshold:
                lr = self.lr_schedule[threshold]
        return lr

@dataclass
class LossCurriculumConfig:
    """Loss weights and curriculum stages for progressive training"""
    # Base loss weights (always applied)
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

    # Curriculum stages: epoch -> dict of weight overrides
    # JSON keys are strings, converted to int on load
    curriculum_stages: Dict[int, Dict[str, float]] = field(default_factory=lambda: {
        1: {'joint_angle_regularization': 0.01},
        10: {'keypoint_2d': 0.1, 'keypoint_3d': 0.5},
        50: {'keypoint_3d': 2.0, 'cam_rot': 0.05},
        # ... rest of stages from training_config.py
    })

    def get_weights_for_epoch(self, epoch: int) -> Dict[str, float]:
        """Get loss weights for given epoch, applying curriculum overrides"""
        weights = self.base_weights.copy()
        for threshold in sorted(self.curriculum_stages.keys()):
            if epoch >= threshold:
                weights.update(self.curriculum_stages[threshold])
        return weights

@dataclass
class TrainingConfig:
    """General training hyperparameters"""
    batch_size: int = 1
    num_epochs: int = 1000
    seed: int = 1234
    rotation_representation: str = '6d'  # '6d' or 'axis_angle'
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 4

    # Checkpoint and output
    checkpoint_dir: str = 'checkpoints'
    save_checkpoint_every: int = 10
    visualize_every_n_epochs: int = 10
    num_visualization_samples: int = 10

@dataclass
class BaseTrainingConfig:
    """Complete base configuration for all training modes"""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss_curriculum: LossCurriculumConfig = field(default_factory=LossCurriculumConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Legacy compatibility (read from config.py)
    shape_family: int = field(default_factory=lambda: __import__('config').SHAPE_FAMILY)

    def validate(self):
        """Validate entire config for consistency"""
        self.dataset.validate()
        # Add more validation as needed
        pass
```

### 2. Single-View Configuration (singleview_config.py)

Extends base config with single-view training specific settings.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from base_config import BaseTrainingConfig

@dataclass
class SingleViewConfig(BaseTrainingConfig):
    """Configuration for single-view training"""

    # Single-view specific parameters
    resume_checkpoint: Optional[str] = None
    use_multi_dataset: bool = False
    multi_dataset_enabled: bool = False
    multi_dataset_configs: List[Dict] = field(default_factory=list)

    def validate(self):
        """Validate single-view specific constraints"""
        super().validate()
        # Add single-view specific validation if needed
        pass
```

### 3. Multi-View Configuration (multiview_config.py)

Extends base config with multi-view training specific settings.

```python
from dataclasses import dataclass
from typing import Optional
from base_config import BaseTrainingConfig

@dataclass
class MultiViewConfig(BaseTrainingConfig):
    """Configuration for multi-view training"""

    # Multi-view specific parameters
    max_views: int = 6
    num_views_to_use: Optional[int] = None  # Use all if None
    min_views_per_sample: int = 2

    # Cross-attention configuration
    cross_attention_layers: int = 2
    cross_attention_heads: int = 8
    cross_attention_dropout: float = 0.1

    # Per-view parameters
    predict_per_view_cameras: bool = True
    use_gt_camera_init: bool = True

    def __post_init__(self):
        """Auto-set checkpoint dir for multi-view if using default"""
        if self.training.checkpoint_dir == 'checkpoints':
            self.training.checkpoint_dir = 'multiview_checkpoints'

    def validate(self):
        """Validate multi-view specific constraints"""
        super().validate()
        assert self.max_views > 0, "max_views must be > 0"
        assert self.min_views_per_sample > 0, "min_views_per_sample must be > 0"
        assert self.min_views_per_sample <= self.max_views
```

### 4. Configuration Utilities (config_utils.py)

Handles loading, merging, and validation with JSON support and dict key conversion.

```python
from typing import Dict, Any, Optional, Union
import json
from pathlib import Path
from dataclasses import asdict, fields, is_dataclass

from base_config import BaseTrainingConfig
from singleview_config import SingleViewConfig
from multiview_config import MultiViewConfig

class ConfigurationError(Exception):
    """Raised when configuration is invalid or incompatible"""
    pass

def _parse_curriculum_keys(schedule: Dict[str, Any]) -> Dict[int, Any]:
    """
    Convert string keys to int keys for curriculum dicts.

    JSON requires string keys, so curriculum epochs are stored as strings.
    This converts them to integers for internal use.

    Example:
        {"0": 5e-5, "10": 3e-5} -> {0: 5e-5, 10: 3e-5}
    """
    if not isinstance(schedule, dict):
        return schedule

    result = {}
    for key, value in schedule.items():
        try:
            int_key = int(key)
            result[int_key] = value
        except (ValueError, TypeError):
            # Non-integer key, keep as-is
            result[key] = value
    return result

def load_from_json(path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    JSON file MUST include a 'mode' field:
        "mode": "singleview" or "multiview"

    JSON keys in curriculum dicts (lr_schedule, curriculum_stages) are
    automatically converted from strings to integers.

    Args:
        path: Path to JSON config file

    Returns:
        Dictionary with config data (mode + all config parameters)

    Raises:
        ConfigurationError: If mode field is missing or invalid
        FileNotFoundError: If file not found
        json.JSONDecodeError: If file is invalid JSON
    """
    with open(path) as f:
        config_dict = json.load(f)

    # Validate mode field exists
    if 'mode' not in config_dict:
        raise ConfigurationError(
            f"JSON config file {path} missing required 'mode' field. "
            "Must be 'singleview' or 'multiview'."
        )

    mode = config_dict['mode']
    if mode not in ['singleview', 'multiview']:
        raise ConfigurationError(
            f"Invalid mode '{mode}' in {path}. Must be 'singleview' or 'multiview'."
        )

    # Convert curriculum dict string keys to integers
    if 'optimizer' in config_dict and isinstance(config_dict['optimizer'], dict):
        if 'lr_schedule' in config_dict['optimizer']:
            config_dict['optimizer']['lr_schedule'] = _parse_curriculum_keys(
                config_dict['optimizer']['lr_schedule']
            )

    if 'loss_curriculum' in config_dict and isinstance(config_dict['loss_curriculum'], dict):
        if 'curriculum_stages' in config_dict['loss_curriculum']:
            config_dict['loss_curriculum']['curriculum_stages'] = _parse_curriculum_keys(
                config_dict['loss_curriculum']['curriculum_stages']
            )

    return config_dict

def merge_configs(base: BaseTrainingConfig, override: Dict[str, Any]) -> BaseTrainingConfig:
    """
    Deep merge override dictionary into base config.

    Handles nested dataclasses (e.g., config.dataset.data_path).
    Skips None values in override dict.

    Precedence:
        override dict values > base config values

    Args:
        base: Base config object to merge into
        override: Override values as dictionary (possibly from JSON or CLI)

    Returns:
        Modified base config object
    """
    # Filter out None values and 'mode' key
    override = {k: v for k, v in override.items() if v is not None and k != 'mode'}

    for field in fields(base):
        field_name = field.name
        if field_name not in override:
            continue

        override_value = override[field_name]

        # If field is a dataclass and override is a dict, recursively merge
        if is_dataclass(field.type) and isinstance(override_value, dict):
            current_value = getattr(base, field_name)
            merged_value = merge_configs(current_value, override_value)
            setattr(base, field_name, merged_value)
        else:
            # Direct assignment for primitives
            setattr(base, field_name, override_value)

    return base

def load_config(
    config_file: Optional[str] = None,
    cli_args: Optional[Dict[str, Any]] = None,
    expected_mode: Optional[str] = None
) -> Union[SingleViewConfig, MultiViewConfig]:
    """
    Load configuration with proper precedence:

    Precedence (highest to lowest):
        1. CLI arguments
        2. JSON config file (if provided)
        3. Mode-specific defaults (SingleViewConfig/MultiViewConfig)
        4. Base training defaults (BaseTrainingConfig)
        5. Legacy config.py (only for specific params like SHAPE_FAMILY)

    Args:
        config_file: Path to JSON config file (must include 'mode' field)
        cli_args: Dictionary of CLI arguments (e.g., from argparse.Namespace)
        expected_mode: If provided, validates JSON mode matches this mode

    Returns:
        Instantiated SingleViewConfig or MultiViewConfig

    Raises:
        ConfigurationError: If JSON mode doesn't match expected_mode
        FileNotFoundError: If config_file not found
    """
    mode = expected_mode
    json_config = None

    # Load JSON config if provided
    if config_file:
        json_config = load_from_json(config_file)
        mode = json_config['mode']

        # Validate mode matches expected if provided
        if expected_mode and mode != expected_mode:
            raise ConfigurationError(
                f"JSON config file is for '{mode}' but script expects '{expected_mode}'. "
                f"Use train_{mode}_regressor.py instead."
            )

    # Create mode-specific config with defaults
    if mode == 'singleview':
        config = SingleViewConfig()
    elif mode == 'multiview':
        config = MultiViewConfig()
    else:
        raise ConfigurationError(f"Invalid mode '{mode}'. Must be 'singleview' or 'multiview'.")

    # Merge JSON file (if provided)
    if json_config:
        config = merge_configs(config, json_config)

    # Apply CLI overrides (highest priority)
    if cli_args:
        cli_dict = {k: v for k, v in cli_args.items() if v is not None}
        config = merge_configs(config, cli_dict)

    # Validate final config
    config.validate()

    return config

def save_config_json(config: BaseTrainingConfig, path: str):
    """
    Save configuration to JSON for reproducibility.

    Includes 'mode' field to indicate single-view vs multi-view.
    Curriculum dict keys are automatically converted to strings (JSON requirement).

    Args:
        config: Config object to save
        path: Path where JSON will be written
    """
    config_dict = asdict(config)

    # Add mode field
    if isinstance(config, SingleViewConfig):
        config_dict['mode'] = 'singleview'
    elif isinstance(config, MultiViewConfig):
        config_dict['mode'] = 'multiview'

    # Convert int keys back to strings for JSON
    if 'optimizer' in config_dict and 'lr_schedule' in config_dict['optimizer']:
        config_dict['optimizer']['lr_schedule'] = {
            str(k): v for k, v in config_dict['optimizer']['lr_schedule'].items()
        }

    if 'loss_curriculum' in config_dict and 'curriculum_stages' in config_dict['loss_curriculum']:
        config_dict['loss_curriculum']['curriculum_stages'] = {
            str(k): v for k, v in config_dict['loss_curriculum']['curriculum_stages'].items()
        }

    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def validate_json_mode(json_path: str, expected_mode: str):
    """
    Quick validation of JSON config mode without full loading.

    Useful for early validation in training scripts before creating
    large data loaders or models.

    Args:
        json_path: Path to JSON config file
        expected_mode: Expected mode ('singleview' or 'multiview')

    Raises:
        ConfigurationError: If actual mode doesn't match expected
    """
    config_dict = load_from_json(json_path)
    actual_mode = config_dict['mode']

    if actual_mode != expected_mode:
        raise ConfigurationError(
            f"JSON config is for '{actual_mode}' training, but you're running "
            f"train_{expected_mode}_regressor.py. Please use the correct script "
            f"or update the JSON 'mode' field."
        )
```

### 5. Config Package Init (__init__.py)

Re-exports public APIs for clean imports.

```python
"""SMILify unified configuration system.

Supports single-view and multi-view training with JSON config files,
CLI overrides, and backward compatibility with legacy config.py.

Example:
    from configs import load_config, SingleViewConfig

    # Load from JSON
    config = load_config(config_file='experiments/baseline.json')

    # Create from scratch
    config = SingleViewConfig()
    config.training.batch_size = 4
"""

from base_config import (
    BaseTrainingConfig,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    LossCurriculumConfig,
    TrainingConfig,
)
from singleview_config import SingleViewConfig
from multiview_config import MultiViewConfig
from config_utils import (
    load_config,
    load_from_json,
    save_config_json,
    merge_configs,
    validate_json_mode,
    ConfigurationError,
)

__all__ = [
    # Base classes
    'BaseTrainingConfig',
    'DatasetConfig',
    'ModelConfig',
    'OptimizerConfig',
    'LossCurriculumConfig',
    'TrainingConfig',
    # Mode-specific classes
    'SingleViewConfig',
    'MultiViewConfig',
    # Utilities
    'load_config',
    'load_from_json',
    'save_config_json',
    'merge_configs',
    'validate_json_mode',
    'ConfigurationError',
]
```

---

## Implementation Roadmap

### Phase 1: Create New Config System (Parallel Development)

**Step 1.1**: Create directory and package
```bash
mkdir -p smal_fitter/neuralSMIL/configs/examples
touch smal_fitter/neuralSMIL/configs/__init__.py
```

**Step 1.2**: Implement base_config.py
- Port all curriculum data from training_config.py
- Create dataclass hierarchy
- Implement validation methods
- Add curriculum lookup methods (get_weights_for_epoch, get_learning_rate_for_epoch)

**Step 1.3**: Implement singleview_config.py
- Create SingleViewConfig extending BaseTrainingConfig
- Add single-view specific parameters
- Add from_training_config() class method for backward compatibility

**Step 1.4**: Implement multiview_config.py
- Create MultiViewConfig extending BaseTrainingConfig
- Port all MULTIVIEW_DEFAULTS from train_multiview_regressor.py
- Add multi-view specific parameters
- Implement __post_init__() for auto-configuration

**Step 1.5**: Implement config_utils.py
- Implement load_from_json() with mode validation
- Implement _parse_curriculum_keys() for dict key conversion
- Implement merge_configs() with deep merge logic
- Implement load_config() with full precedence
- Implement save_config_json()
- Add comprehensive error handling

**Step 1.6**: Create example configs
- Generate singleview_baseline.json from current training_config.py defaults
- Generate multiview_6cam.json from current multiview defaults
- Create examples/README.md with usage instructions

**Step 1.7**: Create package documentation
- Write configs/README.md explaining:
  - Configuration precedence rules
  - How to create/modify JSON configs
  - Curriculum parameter format
  - Legacy compatibility constraints

### Phase 2: Update Training Scripts

**Step 2.1**: Update train_smil_regressor.py
- Import config system: `from configs import SingleViewConfig, load_config`
- Replace manual TrainingConfig usage with load_config()
- Update CLI argument handling to work with new system:
  ```python
  parser.add_argument('--config', type=str, help='Path to JSON config file')
  parser.add_argument('--batch_size', type=int, help='Override batch size')
  parser.add_argument('--learning_rate', type=float, help='Override learning rate')
  # ... other args
  ```
- In main(), call:
  ```python
  cli_args = vars(args)
  config = load_config(
      config_file=args.config,
      cli_args=cli_args,
      expected_mode='singleview'
  )
  ```
- Save config at start of training: `save_config_json(config, os.path.join(config.training.checkpoint_dir, 'config.json'))`

**Step 2.2**: Update train_multiview_regressor.py
- Same approach as train_smil_regressor.py but with expected_mode='multiview'
- Remove inline MultiViewTrainingConfig class
- Remove MULTIVIEW_DEFAULTS dict
- Update all references to use new config object

**Step 2.3**: Add config export
- Both scripts save final config as JSON at checkpoint dir start
- Include timestamp or epoch info for tracking

### Phase 3: Integration Testing

**Step 3.1**: Test backward compatibility
- Verify fitter_3d/optimise.py still works (uses config.py)
- Verify smal_fitter/optimize_to_joints.py still works
- Verify config.py values are still accessible

**Step 3.2**: Test single-view training
- Run train_smil_regressor.py with JSON config: `--config configs/examples/singleview_baseline.json`
- Run with CLI overrides: `--batch_size 4`
- Verify CLI > JSON > defaults precedence
- Verify config.json is saved to checkpoint dir

**Step 3.3**: Test multi-view training
- Run train_multiview_regressor.py with JSON config
- Test mode validation (JSON says multiview, script is multiview)
- Test error case (JSON says singleview, script is multiview)

**Step 3.4**: Test curriculum application
- Verify loss weights change at correct epochs
- Verify learning rate changes follow curriculum
- Verify CLI lr override only affects epoch 0

### Phase 4: Deprecation and Documentation

**Step 4.1**: Deprecate training_config.py
- Add deprecation warning at module level
- Create compatibility shim that imports from new system
- Update module docstring to point to new configs/

**Step 4.2**: Update project documentation
- Update README to point to configs/README.md
- Add migration guide for users of old system
- Include example JSON configs in documentation

**Step 4.3**: Clean up
- Remove unused code from train_smil_regressor.py (old config handling)
- Remove unused code from train_multiview_regressor.py

---

## Configuration Precedence (Formal Specification)

### Resolution Order

For any parameter, the value is resolved in this order (first match wins):

1. **CLI Arguments** (highest priority)
   - Example: `--batch_size 4`
   - Applied last, overrides all other sources
   - Only affects parameters explicitly passed via CLI

2. **JSON Config File**
   - Example: `--config experiments/baseline.json`
   - Must include `"mode": "singleview"` or `"mode": "multiview"`
   - Merged with mode-specific defaults below

3. **Mode-Specific Defaults**
   - SingleViewConfig (from singleview_config.py)
   - MultiViewConfig (from multiview_config.py)
   - Inherits all BaseTrainingConfig defaults

4. **Base Training Defaults**
   - BaseTrainingConfig (from base_config.py)
   - Shared across both modes

5. **Legacy config.py**
   - Only for parameters like SHAPE_FAMILY, SMAL_FILE
   - Read at config initialization time
   - Cannot be overridden by JSON or CLI

### Special Cases: Curriculum Parameters

**Learning Rate**:
- CLI `--learning_rate X` sets epoch-0 value
- Training loop queries `config.optimizer.get_learning_rate_for_epoch(epoch)`
- After epoch 0, curriculum takes over (CLI value ignored)
- To override entire schedule, provide JSON with custom `lr_schedule`

**Loss Weights**:
- Cannot be overridden via CLI (too complex)
- Can be overridden via JSON by specifying `loss_curriculum.curriculum_stages`
- Training loop queries `config.loss_curriculum.get_weights_for_epoch(epoch)`

### JSON Format for Curriculum Parameters

**Learning Rate Curriculum** (lr_schedule):
```json
{
  "mode": "singleview",
  "optimizer": {
    "learning_rate": 1e-4,
    "lr_schedule": {
      "0": 1e-4,
      "10": 5e-5,
      "20": 3e-5,
      "50": 1e-5
    }
  }
}
```

**Loss Weight Curriculum** (curriculum_stages):
```json
{
  "mode": "singleview",
  "loss_curriculum": {
    "base_weights": {
      "keypoint_2d": 0.1,
      "keypoint_3d": 0.25,
      "joint_rot": 0.001
    },
    "curriculum_stages": {
      "1": {
        "joint_angle_regularization": 0.01
      },
      "10": {
        "keypoint_2d": 0.1,
        "keypoint_3d": 0.5
      },
      "50": {
        "keypoint_3d": 2.0,
        "cam_rot": 0.05
      }
    }
  }
}
```

---

## Example JSON Configuration Files

### Single-View Baseline (singleview_baseline.json)

```json
{
  "mode": "singleview",
  "dataset": {
    "data_path": "data/RealSMILyMouseFalkner.h5",
    "train_ratio": 0.85,
    "val_ratio": 0.05,
    "test_ratio": 0.1,
    "dataset_fraction": 0.5
  },
  "model": {
    "backbone_name": "vit_large_patch16_224",
    "freeze_backbone": true,
    "head_type": "transformer_decoder",
    "transformer_depth": 6,
    "transformer_heads": 8,
    "transformer_dropout": 0.1,
    "transformer_ief_iters": 3
  },
  "training": {
    "batch_size": 1,
    "num_epochs": 1000,
    "seed": 1234,
    "rotation_representation": "6d",
    "num_workers": 8,
    "checkpoint_dir": "checkpoints",
    "save_checkpoint_every": 10,
    "visualize_every_n_epochs": 10,
    "num_visualization_samples": 10
  },
  "optimizer": {
    "learning_rate": 5e-5,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 1.0,
    "lr_schedule": {
      "0": 5e-5,
      "10": 3e-5,
      "20": 2e-5,
      "60": 1e-5
    }
  },
  "loss_curriculum": {
    "base_weights": {
      "keypoint_2d": 0.1,
      "keypoint_3d": 0.25,
      "joint_rot": 0.001
    },
    "curriculum_stages": {
      "1": {"joint_angle_regularization": 0.01},
      "10": {"keypoint_2d": 0.1, "keypoint_3d": 0.5},
      "50": {"keypoint_3d": 2.0}
    }
  }
}
```

### Multi-View 6-Camera (multiview_6cam.json)

```json
{
  "mode": "multiview",
  "dataset": {
    "data_path": "data/multiview_dataset.h5",
    "train_ratio": 0.85,
    "val_ratio": 0.05,
    "test_ratio": 0.1,
    "dataset_fraction": 1.0
  },
  "model": {
    "backbone_name": "vit_large_patch16_224",
    "freeze_backbone": true,
    "head_type": "transformer_decoder"
  },
  "training": {
    "batch_size": 3,
    "num_epochs": 600,
    "seed": 1234,
    "checkpoint_dir": "multiview_checkpoints"
  },
  "optimizer": {
    "learning_rate": 5e-5,
    "weight_decay": 1e-4,
    "lr_schedule": {
      "0": 5e-5,
      "50": 1e-5
    }
  },
  "max_views": 6,
  "min_views_per_sample": 2,
  "cross_attention_layers": 2,
  "cross_attention_heads": 8,
  "cross_attention_dropout": 0.1,
  "predict_per_view_cameras": true,
  "use_gt_camera_init": true
}
```

---

## Usage Examples

### Command-Line Usage

**Single-view with JSON config**:
```bash
python train_smil_regressor.py --config configs/examples/singleview_baseline.json
```

**Single-view with JSON + CLI override**:
```bash
python train_smil_regressor.py \
    --config configs/examples/singleview_baseline.json \
    --batch_size 4 \
    --num_epochs 500
```

**Single-view with CLI only (uses defaults)**:
```bash
python train_smil_regressor.py \
    --dataset_path data/my_dataset.h5 \
    --batch_size 2 \
    --learning_rate 1e-4
```

**Multi-view with JSON config**:
```bash
python train_multiview_regressor.py --config configs/examples/multiview_6cam.json
```

### Programmatic Usage

```python
from smal_fitter.neuralSMIL.configs import (
    SingleViewConfig,
    MultiViewConfig,
    load_config,
    save_config_json
)

# Load from JSON
config = load_config(config_file='experiments/baseline.json')

# Create from scratch with defaults
config = SingleViewConfig()
config.dataset.data_path = 'data/my_dataset.h5'
config.training.batch_size = 4
config.validate()

# Access parameters
print(config.model.backbone_name)
print(config.optimizer.learning_rate)

# Query curriculum
weights = config.loss_curriculum.get_weights_for_epoch(50)
lr = config.optimizer.get_learning_rate_for_epoch(50)

# Save for reproducibility
save_config_json(config, 'checkpoints/config.json')
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_configs.py

def test_config_loading_from_json():
    """Test loading config from JSON file"""
    config = load_config(config_file='configs/examples/singleview_baseline.json')
    assert isinstance(config, SingleViewConfig)
    assert config.dataset.data_path == 'data/RealSMILyMouseFalkner.h5'

def test_config_mode_validation():
    """Test that mismatched mode raises error"""
    with pytest.raises(ConfigurationError):
        load_config(
            config_file='configs/examples/multiview_6cam.json',
            expected_mode='singleview'
        )

def test_cli_overrides_json():
    """Test that CLI args override JSON config"""
    config = load_config(
        config_file='configs/examples/singleview_baseline.json',
        cli_args={'batch_size': 8}
    )
    assert config.training.batch_size == 8

def test_curriculum_epoch_lookup():
    """Test loss weight curriculum application"""
    config = SingleViewConfig()
    weights_epoch_0 = config.loss_curriculum.get_weights_for_epoch(0)
    weights_epoch_50 = config.loss_curriculum.get_weights_for_epoch(50)
    # Verify curriculum stage was applied
    assert weights_epoch_50 != weights_epoch_0

def test_lr_curriculum_epoch_lookup():
    """Test learning rate curriculum"""
    config = SingleViewConfig()
    lr_epoch_0 = config.optimizer.get_learning_rate_for_epoch(0)
    lr_epoch_50 = config.optimizer.get_learning_rate_for_epoch(50)
    # Verify curriculum stage was applied
    assert lr_epoch_50 != lr_epoch_0
```

### Integration Tests

- Run train_smil_regressor.py with example config
- Run train_multiview_regressor.py with example config
- Verify saved config.json matches initial config
- Verify fitter_3d/optimise.py still works

### Regression Tests

- Compare loss/accuracy with old config system
- Ensure identical hyperparameters produce identical results

---

## Backward Compatibility

### What Stays the Same

- `config.py` unchanged (legacy compatibility)
- `fitter_3d/optimise.py` continues to use `config.SMAL_FILE`, etc.
- `smal_fitter/optimize_to_joints.py` continues to use `config.GPU_IDS`, etc.

### What Changes

- `training_config.py` becomes deprecated (but still works via compatibility shim)
- `train_smil_regressor.py` uses new config system internally
- `train_multiview_regressor.py` uses new config system internally

### Migration Path

Users currently using training_config.py directly:

**Before**:
```python
from training_config import TrainingConfig
config_dict = TrainingConfig.get_all_config('my_dataset')
```

**After**:
```python
from configs import SingleViewConfig, load_config
config = SingleViewConfig()
# or
config = load_config(config_file='experiments/my_config.json')
```

---

## Benefits of New System

| Benefit | Details |
|---------|---------|
| **Single Source of Truth** | Each parameter defined once in dataclass |
| **Type Safety** | IDE autocomplete, type hints catch errors early |
| **Clear Hierarchy** | Explicit precedence: CLI > JSON > mode-specific > base > legacy |
| **Validation** | Config validated at creation time, fail fast |
| **Reproducibility** | Configs saved as JSON alongside checkpoints |
| **Maintainability** | Changes to config structure in one place |
| **Testability** | Dataclasses easily unit testable |
| **Backward Compatible** | Legacy code (fitter_3d) continues to work |
| **Self-Documenting** | Dataclass structure documents parameters |
| **Flexible** | Supports CLI, JSON, and programmatic configuration |

---

## Migration Checklist

- [ ] Phase 1.1: Create directory structure
- [ ] Phase 1.2-1.6: Implement all config modules
- [ ] Phase 2.1-2.3: Update training scripts
- [ ] Phase 3.1-3.4: Integration testing
- [ ] Phase 4.1-4.3: Deprecation and cleanup
- [ ] Documentation review
- [ ] Deployment to production

---

## Key Design Decisions

### 1. JSON for Config Files (Not YAML)
- **Why**: JSON built-in to Python, no extra dependencies
- **Trade-off**: No comments, but offset by clear structure and examples

### 2. Dataclasses (Not Pydantic)
- **Why**: Built-in, no external dependencies, type hints sufficient
- **Trade-off**: Less validation, but explicit validation methods added

### 3. String Keys in JSON Curriculum → Int Keys in Python
- **Why**: JSON doesn't support integer keys
- **Implementation**: Automatic conversion in load_from_json() and save_config_json()

### 4. Nested Dataclasses (Not Flat Dict)
- **Why**: Better IDE support, logical organization, clear structure
- **Trade-off**: Slightly more verbose in code (`config.model.backbone_name` vs `config['model_backbone_name']`)

### 5. Mode Detection from JSON (Not Script Argument)
- **Why**: Config file self-documents intended mode, catches mismatches
- **Implementation**: JSON must include `"mode"` field, validated against script

---

## Implementation Notes (Actual Strategy Employed)

This section records what was actually implemented on the `config-refactor` branch: which files changed and how the plan was applied.

### Files Changed

| File | Change |
|------|--------|
| **New: `smal_fitter/neuralSMIL/configs/__init__.py`** | Package init; re-exports all public config classes and utilities. |
| **New: `smal_fitter/neuralSMIL/configs/base_config.py`** | Shared dataclasses: `DatasetConfig`, `ModelConfig`, `OptimizerConfig`, `LossCurriculumConfig`, `ScaleTransBetaConfig`, `MeshScalingConfig`, `JointImportanceConfig`, `IgnoredJointsConfig`, `MultiDatasetConfig`, `OutputConfig`, `TrainingHyperparameters`, `BaseTrainingConfig`, plus **`LegacyOverridesConfig`** (`legacy.smal_file`, `legacy.shape_family`). Full loss/LR curriculum data ported from `training_config.py`. `BaseTrainingConfig.to_legacy_dict()` bridges to the dict format expected by `train_smil_regressor.main()`, and now also exposes `smal_file` / `shape_family` for checkpoint + inference use. |
| **New: `smal_fitter/neuralSMIL/configs/singleview_config.py`** | `SingleViewConfig(BaseTrainingConfig)`; minimal extension, inherits base and validation. |
| **New: `smal_fitter/neuralSMIL/configs/multiview_config.py`** | `MultiViewConfig(BaseTrainingConfig)` with multi-view fields (`num_views_to_use`, `min_views_per_sample`, cross-attention params). `MultiViewOutputConfig` for output dirs. `to_multiview_legacy_dict()` produces the flat dict expected by `train_multiview_regressor` main, and now also includes `smal_file` / `shape_family` from `legacy` (when provided). |
| **New: `smal_fitter/neuralSMIL/configs/config_utils.py`** | `load_from_json()` (mode required, epoch key string→int conversion), `_deep_merge_into_dataclass()`, `load_config(config_file=..., cli_overrides=...)`, `save_config_json()`, `validate_json_mode(path, expected_mode)`, `ConfigurationError`. |
| **New: `smal_fitter/neuralSMIL/configs/README.md`** | User-facing docs: precedence, quick start, JSON requirements, API table, backward compatibility. Now documents optional `legacy` overrides (`smal_file`, `shape_family`). |
| **New: `smal_fitter/neuralSMIL/configs/examples/README.md`** | Guide to example JSON configs and top-level sections. Now includes `legacy` overrides section. |
| **New: `smal_fitter/neuralSMIL/configs/examples/singleview_baseline.json`** | Full single-view example (mode, **legacy overrides**, dataset, model, optimizer, loss_curriculum, training, output, scale_trans_beta, mesh_scaling). |
| **New: `smal_fitter/neuralSMIL/configs/examples/multiview_6cam.json`** | Full multi-view example (mode, **legacy overrides**, dataset, model, optimizer, loss_curriculum, training, output, scale_trans_beta, mesh_scaling) plus cross-attention params and multi-view output dirs. |
| **New: `smal_fitter/neuralSMIL/configs/test_config_load.py`** | Minimal test: load singleview + multiview JSON, validate, call `to_legacy_dict` / `to_multiview_legacy_dict`; plus test that JSON without `mode` raises `ConfigurationError`. Run from `smal_fitter/neuralSMIL`: `python configs/test_config_load.py`. |
| **Modified: `smal_fitter/neuralSMIL/train_smil_regressor.py`** | Added `--config` CLI arg. When `--config` is set: load via `load_config()`, apply optional `legacy` overrides (reload `config` if `smal_file` is set; override `SHAPE_FAMILY`), convert to legacy dict with `to_legacy_dict()`, pass to existing `main()`; saves resolved config.json. **Checkpoints now include `checkpoint[\"config\"]`** (model_config, rotation_representation, scale_trans_mode/config, shape_family, smal_file) so inference can be checkpoint-driven. When `--config` is omitted, existing legacy path unchanged. |
| **Modified: `smal_fitter/neuralSMIL/train_multiview_regressor.py`** | Config loading detects new-style JSON by presence of `"mode"` in file. When new-style: load with `load_config()`, apply optional `legacy` overrides (reload `config` if `smal_file` is set; override `SHAPE_FAMILY`), convert with `to_multiview_legacy_dict()` and ensure `shape_family`, `smal_file`, and `scale_trans_config` are present for checkpoints. When legacy (no `mode`): keep `MultiViewTrainingConfig.from_file()` / from_args path, but still inject `shape_family`, `smal_file`, and `scale_trans_config` so checkpoints are self-contained. |
| **Modified: `smal_fitter/neuralSMIL/training_config.py`** | Deprecation notice in module docstring: point users to `configs/` and JSON configs. No functional removal; legacy code still works. |
| **Modified: `smal_fitter/neuralSMIL/run_inference.py`** | Now prefers config from `checkpoint[\"config\"]` (model_config, rotation_representation, scale_trans_mode/config, shape_family, smal_file). If `smal_file` is present, reloads `config` so `dd`, `N_POSE`, `N_BETAS` reflect the checkpoint model. Falls back to `TrainingConfig.get_all_config()` only if checkpoint config is missing/incomplete. |
| **Modified: `smal_fitter/neuralSMIL/run_inference_BBOX.py`** | Same as `run_inference.py`: checkpoint-first config load with fallback to `training_config.py`, and `config` reload when `smal_file` is specified in the checkpoint. |
| **Unchanged: `config.py`** | No edits; remains the single source for legacy SMAL paths, `SHAPE_FAMILY`, `N_BETAS`, `N_POSE`, etc., for fitter_3d and optimize_to_joints. |

### Implementation Strategy Summary

1. **Incremental, backward-compatible**  
   No breaking changes. Both training scripts support the new path only when the user opts in (single-view: `--config <file>`; multi-view: config file contains `"mode"`). Without that, behavior matches the pre-refactor code.

2. **Bridge pattern**  
   New config is never forced into the rest of the training pipeline. `to_legacy_dict()` and `to_multiview_legacy_dict()` produce the exact dict shapes that existing `main()` already expect. Training loops, dataset construction, and checkpointing are unchanged.

3. **JSON as source of truth**  
   Mode is taken from JSON (`"mode": "singleview"` or `"multiview"`). Curriculum epoch keys are strings in JSON and converted to integers in `load_from_json()` so `lr_schedule` and `curriculum_stages` work as intended.

4. **CLI overrides**  
   Single-view maps flat CLI args (e.g. `--batch-size`, `--dataset`) into a nested override dict that `load_config()` merges on top of the JSON-loaded config before conversion to legacy dict. Multi-view can be extended the same way if needed.

5. **Validation**  
   Config dataclasses expose `validate()` (split ratios, head_type, scale_trans mode, etc.). Called after load/merge; optional save of resolved config supports reproducibility.

6. **Documentation and testing**  
   `configs/README.md` and `configs/examples/README.md` document usage and structure. `configs/test_config_load.py` verifies load + validate + legacy dict conversion for both modes and the required `mode` field.

### Inference and benchmark scripts (impact of new config system)

| Script | How it gets config | Affected? | Notes |
|--------|--------------------|-----------|--------|
| **`benchmark_multiview_model.py`** | Reads `checkpoint["config"]`; if missing, falls back to `MultiViewTrainingConfig.get_config()`. Uses `config` (legacy) for `SHAPE_FAMILY`. | **No change required.** | Multiview training saves the same flat dict into the checkpoint (from `to_multiview_legacy_dict()` when using new JSON). So `config_from_ckpt` has the same keys as before. The fallback to `MultiViewTrainingConfig.get_config()` still works because that class was not removed. Optional future: add `--config` to run benchmark with a JSON config when no checkpoint config is present. |
| **`run_multiview_inference.py`** | Reads all model/config from `checkpoint["config"]` (flat dict), including `shape_family`, `smal_file`, and `scale_trans_config` (now stored). One visualization path still calls `TrainingConfig.get_scale_trans_config()` because that function doesn’t have checkpoint context. | **No change required.** | Checkpoint config now contains `shape_family`, `smal_file`, and `scale_trans_config`. Optional future: thread `scale_trans_config` through the loader/model so visualization can avoid importing `training_config.py`. |
| **`run_inference.py`** | Prefers `checkpoint["config"]` (model_config, rotation_representation, scale_trans, shape_family, smal_file). Falls back to `TrainingConfig.get_all_config()` only if missing. Reloads `config` if `smal_file` is specified so `N_POSE`/`N_BETAS` match the model file. | **Updated.** | This enables fully checkpoint-driven inference when training saved `checkpoint["config"]` (new single-view and multi-view checkpoints). |
| **`run_inference_BBOX.py`** | Same as `run_inference.py`: checkpoint-first config load with fallback, plus `config` reload when `smal_file` is present. | **Updated.** | Same behavior for BBOX pipeline. |

**Summary**

- **Multiview**: Benchmark and run_multiview_inference are **not broken** by the new config system; they consume the flat dict in the checkpoint, which the new system still produces. Checkpoints now also include `shape_family`, `smal_file`, and `scale_trans_config` for inference parity.
- **Single-view**: run_inference and run_inference_BBOX are now **checkpoint-first** (when `checkpoint["config"]` exists) with a safe fallback to `training_config.py` for older checkpoints.
- **Optional follow-ups**: Thread `scale_trans_config` through `run_multiview_inference.py` visualization helpers (e.g. attach to model) to remove the remaining `training_config` import in that visualization path.

---

## Glossary

| Term | Definition |
|------|-----------|
| **Base Config** | BaseTrainingConfig, contains all shared parameters |
| **Mode** | 'singleview' or 'multiview', determines training type |
| **Curriculum** | Loss weights or LR that change per epoch for progressive training |
| **Precedence** | Order in which config sources are checked (CLI > JSON > defaults) |
| **Epoch** | Training iteration threshold where curriculum weights change |
| **Legacy** | Original config.py used by fitter_3d and optimize_to_joints.py |

---

## FAQ

**Q: Can I use the old training_config.py?**
A: Yes, it becomes a compatibility shim that imports from the new system.

**Q: What if I want to keep YAML instead of JSON?**
A: The structure is identical; you can add YAML support by creating a load_from_yaml() function.

**Q: Can I modify curriculum mid-training?**
A: The config object is immutable during training. Create a new config if you need changes.

**Q: Will my old experiments still work?**
A: Yes. Legacy code using config.py is unaffected. Existing checkpoints can be re-evaluated with new config system.

**Q: How do I debug config issues?**
A: Print the config object: `print(config)` shows all values. Check the saved `config.json` in checkpoint directory.

**Q: Can I share configs between single-view and multi-view?**
A: The JSON must specify the mode, but most parameters (dataset, model, optimizer) are shared. For Multiview models, additional arguments are required regarding the number of views and cross-attention layers.
