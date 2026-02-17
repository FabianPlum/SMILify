"""
Configuration Utilities

Handles loading, merging, and validation of configuration objects.
Supports JSON config files with automatic curriculum key conversion.
"""

from typing import Dict, Any, Optional, Union
import json
import os
import copy
from pathlib import Path
from dataclasses import asdict, fields, is_dataclass

from .base_config import BaseTrainingConfig
from .singleview_config import SingleViewConfig
from .multiview_config import MultiViewConfig


class ConfigurationError(Exception):
    """Raised when configuration is invalid or incompatible."""
    pass


def _parse_epoch_keys(d: dict) -> dict:
    """
    Convert string keys to int keys for curriculum dicts.

    JSON requires string keys, so curriculum epochs are stored as "10", "50", etc.
    This converts them to integers for internal use.

    Example:
        {"0": 5e-5, "10": 3e-5} -> {0: 5e-5, 10: 3e-5}
    """
    if not isinstance(d, dict):
        return d
    result = {}
    for key, value in d.items():
        try:
            result[int(key)] = value
        except (ValueError, TypeError):
            result[key] = value
    return result


def _convert_epoch_keys_to_str(d: dict) -> dict:
    """Convert int keys back to strings for JSON serialization."""
    if not isinstance(d, dict):
        return d
    return {str(k): v for k, v in d.items()}


def load_from_json(path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    The JSON file MUST include a 'mode' field set to 'singleview' or 'multiview'.
    Curriculum dict keys (lr_schedule, curriculum_stages) are automatically
    converted from strings to integers.

    Args:
        path: Path to JSON config file

    Returns:
        Dictionary with config data

    Raises:
        ConfigurationError: If mode field is missing or invalid
        FileNotFoundError: If file not found
        json.JSONDecodeError: If file is invalid JSON
    """
    with open(path) as f:
        config_dict = json.load(f)

    if 'mode' not in config_dict:
        raise ConfigurationError(
            f"JSON config file '{path}' missing required 'mode' field. "
            "Must be 'singleview' or 'multiview'."
        )

    mode = config_dict['mode']
    if mode not in ('singleview', 'multiview'):
        raise ConfigurationError(
            f"Invalid mode '{mode}' in '{path}'. Must be 'singleview' or 'multiview'."
        )

    # Convert curriculum string keys to integers
    if 'optimizer' in config_dict and isinstance(config_dict['optimizer'], dict):
        if 'lr_schedule' in config_dict['optimizer']:
            config_dict['optimizer']['lr_schedule'] = _parse_epoch_keys(
                config_dict['optimizer']['lr_schedule']
            )

    if 'loss_curriculum' in config_dict and isinstance(config_dict['loss_curriculum'], dict):
        if 'curriculum_stages' in config_dict['loss_curriculum']:
            config_dict['loss_curriculum']['curriculum_stages'] = _parse_epoch_keys(
                config_dict['loss_curriculum']['curriculum_stages']
            )

    return config_dict


def _deep_merge_into_dataclass(target, overrides: Dict[str, Any]):
    """
    Recursively merge override dict into a dataclass instance.

    For nested dataclass fields, recursively merges the override dict.
    For primitive fields, directly assigns the override value.
    Skips keys not present in the dataclass fields.
    """
    if not is_dataclass(target):
        return

    field_names = {f.name for f in fields(target)}

    for key, value in overrides.items():
        if key not in field_names or value is None:
            continue

        current = getattr(target, key)

        if is_dataclass(current) and isinstance(value, dict):
            _deep_merge_into_dataclass(current, value)
        else:
            setattr(target, key, value)


def load_config(
    config_file: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    expected_mode: Optional[str] = None,
) -> Union[SingleViewConfig, MultiViewConfig]:
    """
    Load configuration with proper precedence:

        1. CLI arguments          (highest priority)
        2. JSON config file
        3. Mode-specific defaults (SingleViewConfig / MultiViewConfig)
        4. Base defaults          (BaseTrainingConfig)
        5. Legacy config.py       (lowest priority, specific params only)

    Args:
        config_file: Path to JSON config file (must include 'mode' field)
        cli_overrides: Dictionary of CLI argument overrides
        expected_mode: If set, validates JSON mode matches this value

    Returns:
        Fully merged SingleViewConfig or MultiViewConfig

    Raises:
        ConfigurationError: If mode is invalid or mismatched
    """
    mode = expected_mode
    json_config = None

    # Load JSON config if provided
    if config_file:
        json_config = load_from_json(config_file)
        mode = json_config['mode']

        if expected_mode and mode != expected_mode:
            raise ConfigurationError(
                f"JSON config is for '{mode}' but script expects '{expected_mode}'. "
                f"Use train_{'smil' if expected_mode == 'singleview' else 'multiview'}_regressor.py instead."
            )

    # Instantiate mode-specific config with defaults
    if mode == 'singleview':
        config = SingleViewConfig()
    elif mode == 'multiview':
        config = MultiViewConfig()
    else:
        raise ConfigurationError(
            f"Cannot determine mode. Provide --config with a 'mode' field "
            f"or ensure expected_mode is set."
        )

    # Merge JSON overrides
    if json_config:
        clean = {k: v for k, v in json_config.items() if k != 'mode'}
        # Support "smal_model" as alias for "legacy" (example configs use smal_model)
        if 'smal_model' in clean and clean['smal_model'] is not None:
            clean['legacy'] = clean.pop('smal_model')
        _deep_merge_into_dataclass(config, clean)

    # Merge CLI overrides (highest priority)
    if cli_overrides:
        clean = {k: v for k, v in cli_overrides.items() if v is not None and k != 'config'}
        _deep_merge_into_dataclass(config, clean)

    config.validate()
    return config


def save_config_json(config: BaseTrainingConfig, path: str):
    """
    Save configuration to JSON for reproducibility.

    Includes a 'mode' field and converts int curriculum keys to strings.

    Args:
        config: Config object to save
        path: Output JSON path
    """
    config_dict = asdict(config)

    # Add mode field
    if isinstance(config, MultiViewConfig):
        config_dict['mode'] = 'multiview'
    else:
        config_dict['mode'] = 'singleview'

    # Convert int keys to strings for JSON compatibility
    if 'optimizer' in config_dict and 'lr_schedule' in config_dict['optimizer']:
        config_dict['optimizer']['lr_schedule'] = _convert_epoch_keys_to_str(
            config_dict['optimizer']['lr_schedule']
        )

    if 'loss_curriculum' in config_dict and 'curriculum_stages' in config_dict['loss_curriculum']:
        config_dict['loss_curriculum']['curriculum_stages'] = _convert_epoch_keys_to_str(
            config_dict['loss_curriculum']['curriculum_stages']
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def validate_json_mode(json_path: str, expected_mode: str):
    """
    Quick validation of JSON config mode without full loading.

    Args:
        json_path: Path to JSON config file
        expected_mode: Expected mode ('singleview' or 'multiview')

    Raises:
        ConfigurationError: If modes don't match
    """
    config_dict = load_from_json(json_path)
    actual_mode = config_dict['mode']
    if actual_mode != expected_mode:
        raise ConfigurationError(
            f"JSON config is for '{actual_mode}' training, but you're running "
            f"train_{expected_mode}_regressor.py. Please use the correct script "
            f"or update the JSON 'mode' field."
        )


def apply_smal_file_override(smal_file: str, shape_family: Optional[int] = None):
    """
    Override config.SMAL_FILE and re-derive all dependent globals.

    config.py derives ``dd``, ``N_POSE``, ``N_BETAS``, ``joint_names``,
    ``ROOT_JOINT``, ``STATIC_JOINT_LOCATIONS``, and ``CANONICAL_MODEL_JOINTS``
    from the SMAL pickle at import time.  Simply assigning
    ``config.SMAL_FILE = "..."`` does NOT update those.

    This function re-reads the pickle and patches every global that the
    training scripts depend on.

    Args:
        smal_file:    Path to the SMAL/SMIL model pickle.
        shape_family: If not None, also override ``config.SHAPE_FAMILY``.
    """
    import pickle as pkl
    import numpy as np
    import config  # top-level config.py (must be on sys.path)

    config.SMAL_FILE = smal_file
    config.ignore_hardcoded_body = True

    with open(smal_file, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        dd = u.load()

    config.dd = dd
    config.joint_names = dd["J_names"]
    config.N_POSE = len(config.joint_names) - 1
    config.ROOT_JOINT = dd["J_names"][
        np.where(dd["kintree_table"][0] == -1)[0][0]
    ]
    config.STATIC_JOINT_LOCATIONS = bool(dd.get("static_joint_locs", False))
    config.CANONICAL_MODEL_JOINTS = list(range(len(config.joint_names)))

    try:
        config.N_BETAS = dd["shapedirs"].shape[2]
    except (IndexError, KeyError):
        config.N_BETAS = 20

    if shape_family is not None:
        config.SHAPE_FAMILY = int(shape_family)

    print(f"[config] SMAL_FILE overridden to: {config.SMAL_FILE}")
    print(f"[config] N_POSE={config.N_POSE}, N_BETAS={config.N_BETAS}, "
          f"joints={len(config.joint_names)}")
