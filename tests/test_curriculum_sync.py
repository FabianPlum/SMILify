#!/usr/bin/env python3
"""
Test script to verify that the loss curriculum from JSON config files
is correctly synced to the legacy TrainingConfig class.
"""

import json
import sys
import os

import pytest

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'smal_fitter', 'neuralSMIL'))

from training_config import TrainingConfig
from configs import load_config

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..',
    'smal_fitter', 'neuralSMIL', 'configs', 'examples', 'multiview_sticks.json'
)


def test_curriculum_loading():
    """Test that the multiview_sticks.json curriculum is loaded correctly."""
    if not os.path.exists(CONFIG_PATH):
        pytest.skip(f"Config file not found: {CONFIG_PATH}")

    new_config = load_config(
        config_file=CONFIG_PATH,
        cli_overrides={},
        expected_mode='multiview'
    )

    assert len(new_config.loss_curriculum.base_weights) > 0
    assert len(new_config.loss_curriculum.curriculum_stages) > 0

    # Simulate the sync that happens in train_multiview_regressor.py
    TrainingConfig.LOSS_CURRICULUM = {
        'base_weights': dict(new_config.loss_curriculum.base_weights),
        'curriculum_stages': [
            (epoch, dict(updates))
            for epoch, updates in sorted(new_config.loss_curriculum.curriculum_stages.items())
        ],
    }

    TrainingConfig.LEARNING_RATE_CURRICULUM = {
        'base_learning_rate': new_config.optimizer.learning_rate,
        'lr_stages': [
            (epoch, lr)
            for epoch, lr in sorted(new_config.optimizer.lr_schedule.items())
        ],
    }

    assert len(TrainingConfig.LOSS_CURRICULUM['curriculum_stages']) > 0

    # Load the raw JSON to get expected values
    with open(CONFIG_PATH) as f:
        json_config = json.load(f)

    # Verify epoch 0 returns base weights
    weights_epoch_0 = TrainingConfig.get_loss_weights_for_epoch(0)
    assert isinstance(weights_epoch_0, dict)
    assert 'limb_scale_regularization' in weights_epoch_0

    # Verify epoch 150 value matches JSON config
    weights_epoch_150 = TrainingConfig.get_loss_weights_for_epoch(150)
    limb_scale_150 = weights_epoch_150.get('limb_scale_regularization')
    assert limb_scale_150 is not None, "limb_scale_regularization not found at epoch 150"
    assert isinstance(limb_scale_150, (int, float)), f"Expected numeric value, got {type(limb_scale_150)}"

    expected_150 = (
        json_config.get('loss_curriculum', {})
        .get('curriculum_stages', {})
        .get('150', {})
        .get('limb_scale_regularization')
    )
    if expected_150 is not None:
        assert limb_scale_150 == expected_150, (
            f"Epoch 150 limb_scale_regularization mismatch: JSON={expected_150}, got={limb_scale_150}"
        )
