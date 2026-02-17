"""
Tests for the unified configuration system (smal_fitter/neuralSMIL/configs/).

Verifies JSON loading, dataclass merging, validation, legacy dict conversion,
curriculum application, CLI override precedence, and round-trip serialization.
"""

import json
import os
import sys
import tempfile

import pytest

# Ensure neuralSMIL is importable
_neural_smil = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "smal_fitter", "neuralSMIL")
)
if _neural_smil not in sys.path:
    sys.path.insert(0, _neural_smil)

from configs import (
    BaseTrainingConfig,
    SingleViewConfig,
    MultiViewConfig,
    MultiViewOutputConfig,
    load_config,
    load_from_json,
    save_config_json,
    validate_json_mode,
    ConfigurationError,
)

EXAMPLES_DIR = os.path.join(_neural_smil, "configs", "examples")
SINGLEVIEW_JSON = os.path.join(EXAMPLES_DIR, "singleview_baseline.json")
MULTIVIEW_JSON = os.path.join(EXAMPLES_DIR, "multiview_6cam.json")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def singleview_config():
    """Load singleview example config."""
    return load_config(config_file=SINGLEVIEW_JSON)


@pytest.fixture
def multiview_config():
    """Load multiview example config."""
    return load_config(config_file=MULTIVIEW_JSON)


def _write_tmp_json(data: dict) -> str:
    """Write a dict to a temp JSON file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Loading & type checks
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_singleview_returns_correct_type(self, singleview_config):
        assert isinstance(singleview_config, SingleViewConfig)

    def test_multiview_returns_correct_type(self, multiview_config):
        assert isinstance(multiview_config, MultiViewConfig)

    def test_singleview_validates(self, singleview_config):
        singleview_config.validate()

    def test_multiview_validates(self, multiview_config):
        multiview_config.validate()


class TestModeRequired:
    def test_missing_mode_raises(self):
        path = _write_tmp_json({"dataset": {"data_path": "x.h5"}})
        try:
            with pytest.raises(ConfigurationError, match="mode"):
                load_from_json(path)
        finally:
            os.unlink(path)

    def test_invalid_mode_raises(self):
        path = _write_tmp_json({"mode": "unknown"})
        try:
            with pytest.raises(ConfigurationError, match="unknown"):
                load_from_json(path)
        finally:
            os.unlink(path)

    def test_mode_mismatch_raises(self):
        path = _write_tmp_json({"mode": "multiview"})
        try:
            with pytest.raises(ConfigurationError):
                load_config(config_file=path, expected_mode="singleview")
        finally:
            os.unlink(path)


class TestValidateJsonMode:
    def test_singleview_ok(self):
        validate_json_mode(SINGLEVIEW_JSON, "singleview")

    def test_multiview_ok(self):
        validate_json_mode(MULTIVIEW_JSON, "multiview")

    def test_wrong_mode_raises(self):
        with pytest.raises(ConfigurationError):
            validate_json_mode(SINGLEVIEW_JSON, "multiview")


# ---------------------------------------------------------------------------
# Legacy dict conversion
# ---------------------------------------------------------------------------

class TestLegacyDictSingleView:
    def test_has_required_keys(self, singleview_config):
        d = singleview_config.to_legacy_dict()
        for key in ("data_path", "training_params", "model_config",
                     "loss_curriculum", "learning_rate_curriculum",
                     "output_config", "split_config", "shape_family", "smal_file"):
            assert key in d, f"Missing key: {key}"

    def test_model_backbone(self, singleview_config):
        d = singleview_config.to_legacy_dict()
        assert d["model_config"]["backbone_name"] == "vit_large_patch16_224"

    def test_curriculum_stages_are_tuples(self, singleview_config):
        d = singleview_config.to_legacy_dict()
        stages = d["loss_curriculum"]["curriculum_stages"]
        assert isinstance(stages, list)
        assert all(isinstance(s, tuple) and len(s) == 2 for s in stages)

    def test_lr_stages_are_tuples(self, singleview_config):
        d = singleview_config.to_legacy_dict()
        lr_stages = d["learning_rate_curriculum"]["lr_stages"]
        assert isinstance(lr_stages, list)
        assert all(isinstance(s, tuple) and len(s) == 2 for s in lr_stages)


class TestLegacyDictMultiView:
    def test_has_required_keys(self, multiview_config):
        d = multiview_config.to_multiview_legacy_dict()
        for key in ("batch_size", "dataset_path", "backbone_name",
                     "cross_attention_layers", "cross_attention_heads",
                     "checkpoint_dir", "shape_family", "smal_file",
                     "loss_weights", "use_gt_camera_init"):
            assert key in d, f"Missing key: {key}"

    def test_cross_attention_defaults(self, multiview_config):
        d = multiview_config.to_multiview_legacy_dict()
        assert d["cross_attention_layers"] == 2
        assert d["cross_attention_heads"] == 8

    def test_multiview_output_dirs(self, multiview_config):
        d = multiview_config.to_multiview_legacy_dict()
        assert d["checkpoint_dir"] == "multiview_checkpoints"
        assert d["visualizations_dir"] == "multiview_visualizations"


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

class TestCurriculum:
    def test_epoch_0_uses_base_weights(self):
        config = SingleViewConfig()
        w = config.get_loss_weights_for_epoch(0)
        assert w["keypoint_3d"] == 0.25

    def test_epoch_50_applies_curriculum(self):
        config = SingleViewConfig()
        w = config.get_loss_weights_for_epoch(50)
        assert w["keypoint_3d"] == 2

    def test_lr_schedule_epoch_0(self):
        config = SingleViewConfig()
        lr = config.get_learning_rate_for_epoch(0)
        assert lr == 5e-5

    def test_lr_schedule_epoch_60(self):
        config = SingleViewConfig()
        lr = config.get_learning_rate_for_epoch(60)
        assert lr == 1e-5


# ---------------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------------

class TestCLIOverrides:
    def test_cli_overrides_batch_size(self):
        config = load_config(
            config_file=SINGLEVIEW_JSON,
            cli_overrides={"training": {"batch_size": 16}},
        )
        assert config.training.batch_size == 16

    def test_cli_overrides_learning_rate(self):
        config = load_config(
            config_file=SINGLEVIEW_JSON,
            cli_overrides={"optimizer": {"learning_rate": 1e-3}},
        )
        assert config.optimizer.learning_rate == 1e-3

    def test_cli_overrides_nested_model(self):
        config = load_config(
            config_file=SINGLEVIEW_JSON,
            cli_overrides={"model": {"backbone_name": "resnet50"}},
        )
        assert config.model.backbone_name == "resnet50"

    def test_cli_overrides_multiview_top_level(self):
        config = load_config(
            config_file=MULTIVIEW_JSON,
            cli_overrides={"cross_attention_layers": 4},
        )
        assert config.cross_attention_layers == 4


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_bad_split_ratios(self):
        config = SingleViewConfig()
        config.dataset.train_ratio = 0.9
        config.dataset.val_ratio = 0.1
        config.dataset.test_ratio = 0.1
        with pytest.raises(ValueError, match="sum"):
            config.validate()

    def test_bad_head_type(self):
        config = SingleViewConfig()
        config.model.head_type = "invalid"
        with pytest.raises(ValueError, match="head_type"):
            config.validate()

    def test_bad_rotation_representation(self):
        config = SingleViewConfig()
        config.training.rotation_representation = "quaternion"
        with pytest.raises(ValueError, match="rotation_representation"):
            config.validate()

    def test_bad_scale_trans_mode(self):
        config = SingleViewConfig()
        config.scale_trans_beta.mode = "invalid"
        with pytest.raises(ValueError):
            config.validate()

    def test_bad_min_views(self):
        config = MultiViewConfig()
        config.min_views_per_sample = 0
        with pytest.raises(ValueError, match="min_views"):
            config.validate()


# ---------------------------------------------------------------------------
# Round-trip save / load
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_singleview_round_trip(self, singleview_config):
        path = _write_tmp_json({})  # placeholder, overwritten by save
        try:
            save_config_json(singleview_config, path)
            reloaded = load_config(config_file=path)
            assert isinstance(reloaded, SingleViewConfig)
            assert reloaded.model.backbone_name == singleview_config.model.backbone_name
            assert reloaded.training.batch_size == singleview_config.training.batch_size
            assert reloaded.optimizer.learning_rate == singleview_config.optimizer.learning_rate
        finally:
            os.unlink(path)

    def test_multiview_round_trip(self, multiview_config):
        path = _write_tmp_json({})
        try:
            save_config_json(multiview_config, path)
            reloaded = load_config(config_file=path)
            assert isinstance(reloaded, MultiViewConfig)
            assert reloaded.cross_attention_layers == multiview_config.cross_attention_layers
            assert reloaded.training.seed == multiview_config.training.seed
        finally:
            os.unlink(path)

    def test_curriculum_survives_round_trip(self, singleview_config):
        path = _write_tmp_json({})
        try:
            save_config_json(singleview_config, path)
            reloaded = load_config(config_file=path)
            # Curriculum stages should have int keys after round-trip
            assert all(isinstance(k, int) for k in reloaded.loss_curriculum.curriculum_stages)
            assert all(isinstance(k, int) for k in reloaded.optimizer.lr_schedule)
            # Values should match
            orig_w = singleview_config.get_loss_weights_for_epoch(50)
            reload_w = reloaded.get_loss_weights_for_epoch(50)
            assert orig_w == reload_w
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_singleview_default_creates(self):
        config = SingleViewConfig()
        config.validate()

    def test_multiview_default_creates(self):
        config = MultiViewConfig()
        config.validate()

    def test_multiview_output_defaults(self):
        config = MultiViewConfig()
        assert config.output.checkpoint_dir == "multiview_checkpoints"
        assert isinstance(config.output, MultiViewOutputConfig)

    def test_hidden_dim_vit_large(self):
        config = SingleViewConfig()
        assert config.model.get_adjusted_hidden_dim() == 1024

    def test_hidden_dim_vit_base(self):
        config = SingleViewConfig()
        config.model.backbone_name = "vit_base_patch16_224"
        assert config.model.get_adjusted_hidden_dim() == 768

    def test_hidden_dim_resnet(self):
        config = SingleViewConfig()
        config.model.backbone_name = "resnet50"
        assert config.model.get_adjusted_hidden_dim() == 2048