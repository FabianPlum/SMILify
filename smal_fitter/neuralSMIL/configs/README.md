# SMILify Unified Configuration System

This package provides a single, type-safe configuration system for both **single-view** and **multi-view** SMIL image regressor training. It replaces the previous mix of `training_config.py`, in-script defaults, and CLI-only overrides with JSON config files and dataclasses.

## Precedence (highest to lowest)

1. **CLI arguments** (e.g. `--batch-size 4`, `--config my.json`)
2. **JSON config file** (when using `--config path/to/config.json`)
3. **Mode-specific defaults** (`SingleViewConfig` / `MultiViewConfig`)
4. **Base defaults** (`BaseTrainingConfig` in `base_config.py`)
5. **Legacy `config.py`** (only for SMAL model paths, `SHAPE_FAMILY`, `N_BETAS`, `N_POSE`, etc.—unchanged for fitter_3d compatibility)

## Quick start

### Single-view training with a JSON config

```bash
python train_smil_regressor.py --config configs/examples/singleview_baseline.json
```

### Multi-view training with a JSON config

```bash
python train_multiview_regressor.py --config configs/examples/multiview_6cam.json
```

### Override from CLI

```bash
python train_smil_regressor.py --config configs/examples/singleview_baseline.json --batch-size 8 --dataset my_data.h5
```

### Without a config file (legacy behavior)

If you omit `--config`, the scripts fall back to the previous behavior (e.g. `TrainingConfig` / `MultiViewTrainingConfig` and their defaults). No JSON file is required.

## JSON config requirements

- The JSON file **must** include a top-level `"mode"` field: `"singleview"` or `"multiview"`.
- Optional: include a top-level `"smal_model"` section to override values otherwise sourced from `config.py`:
  - `"smal_file"`: SMAL/SMIL model pickle path. When used, callers should reload `config` so `dd`, `N_POSE`, `N_BETAS`, etc. match that file.
  - `"shape_family"`: integer shape family passed into SMAL/SMIL code (overrides `config.SHAPE_FAMILY` for the run).
- Curriculum parameters (loss weights and learning rate schedules) use **string keys** for epoch thresholds in JSON (e.g. `"10": 3e-5`). These are converted to integer keys when loaded.
- See `examples/` for full example configs and `examples/README.md` for a field-by-field guide.

## Public API

| Symbol | Description |
|--------|-------------|
| `load_config(config_file=..., cli_overrides=...)` | Load and merge JSON + CLI into a `SingleViewConfig` or `MultiViewConfig`. |
| `load_from_json(path)` | Load raw dict from JSON (with epoch key conversion). |
| `save_config_json(config, path)` | Serialize a config dataclass to JSON. |
| `validate_json_mode(path)` | Check that the JSON has a valid `mode` field. |
| `SingleViewConfig` | Default config for single-view training. |
| `MultiViewConfig` | Default config for multi-view training (adds cross-attention and multi-view output paths). |
| `BaseTrainingConfig` | Shared base; use mode-specific classes in practice. |
| `ConfigurationError` | Raised for invalid or incompatible config. |

## Structure

- **`base_config.py`** — Shared dataclasses: `DatasetConfig`, `ModelConfig`, `OptimizerConfig`, `LossCurriculumConfig`, `ScaleTransBetaConfig`, `MeshScalingConfig`, `OutputConfig`, etc., and `BaseTrainingConfig`.
- **`singleview_config.py`** — `SingleViewConfig` (minimal extension of base).
- **`multiview_config.py`** — `MultiViewConfig`, `MultiViewOutputConfig`, and multi-view–specific fields (cross-attention, output dirs).
- **`config_utils.py`** — JSON load/save, deep merge into dataclasses, mode validation.
- **`examples/`** — Example JSON configs and a short guide.

## Backward compatibility

- **Legacy scripts**: `train_smil_regressor.py` and `train_multiview_regressor.py` convert the new config to the dict format expected by their existing `main()` via `to_legacy_dict()` and `to_multiview_legacy_dict()`. No change to the rest of the training loop is required.
- **`config.py`**: Unchanged; still used for SMAL file paths, shape family, joint counts, and GPU settings by `fitter_3d` and `optimize_to_joints`.
- **`training_config.py`**: Deprecated; new setups should use this package and JSON configs.

## Validation

Configs can be validated programmatically:

```python
from configs import SingleViewConfig, load_config

config = load_config(config_file="experiments/baseline.json")
config.validate()
```

Validation checks split ratios, head type, scale/trans mode, and (for multi-view) minimum views per sample.
