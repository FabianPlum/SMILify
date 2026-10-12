"""Tests for the single-/multi-view inference scripts (PR #101).

Covers the logic that does not need a GPU, a trained checkpoint or a dataset:

- ``resolve_smal_file_for_checkpoint``: precedence ``--smal_file`` > checkpoint
  > config.py, and warn-vs-raise for model files that don't exist.
- ``load_model_from_checkpoint`` (single-view): ``--smal_file`` is not
  overridden by the checkpoint's own ``smal_file`` afterwards.
- ``apply_scale_trans_to_fitter``: the scale and translation heads are applied
  independently, with PCA expansion in 'separate' mode.
- ``--dataset`` indexing: per-camera item selection and subclip ranges.
- ``inference_utils``: temporal smoothing and frame padding.

Rendering and real checkpoints are checked by a local GPU run of the scripts.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smal_fitter.neuralSMIL import configs  # noqa: E402
from smal_fitter.neuralSMIL.configs import config_utils, resolve_smal_file_for_checkpoint  # noqa: E402
from smal_fitter.neuralSMIL import run_singleview_inference as svi  # noqa: E402
from smal_fitter.neuralSMIL.inference_utils import PredictionSmoother, pad_or_resize  # noqa: E402


# ---------------------------------------------------------------------------
# SMAL/SMIL model file resolution
# ---------------------------------------------------------------------------


@pytest.fixture
def applied(monkeypatch):
    """Record apply_smal_file_override calls instead of reloading config.py globals."""
    calls = []

    def record(f, shape_family=None):
        calls.append(f)

    # Patch every name it can be reached through, so a re-introduced direct
    # call in the inference scripts is recorded too.
    monkeypatch.setattr(config_utils, "apply_smal_file_override", record)
    monkeypatch.setattr(configs, "apply_smal_file_override", record)
    monkeypatch.setattr(svi, "apply_smal_file_override", record, raising=False)
    return calls


@pytest.fixture
def model_files(tmp_path):
    ckpt_model = tmp_path / "ckpt_model.pkl"
    cli_model = tmp_path / "cli_model.pkl"
    ckpt_model.touch()
    cli_model.touch()
    return str(ckpt_model), str(cli_model)


def test_checkpoint_smal_file_is_applied(applied, model_files):
    ckpt_model, _ = model_files
    resolve_smal_file_for_checkpoint({"smal_file": ckpt_model}, None, None)
    assert applied == [ckpt_model]


def test_cli_smal_file_beats_checkpoint(applied, model_files):
    ckpt_model, cli_model = model_files
    resolve_smal_file_for_checkpoint({"smal_file": ckpt_model}, cli_model, None)
    assert applied == [cli_model]


def test_no_smal_file_anywhere_leaves_config_alone(applied):
    resolve_smal_file_for_checkpoint({}, None, None)
    assert applied == []


def test_missing_checkpoint_smal_file_warns(applied, capsys):
    resolve_smal_file_for_checkpoint({"smal_file": "/nope/missing.pkl"}, None, None)
    assert applied == []
    assert "WARNING" in capsys.readouterr().out


def test_missing_cli_smal_file_raises(applied, model_files):
    ckpt_model, _ = model_files
    with pytest.raises(FileNotFoundError):
        resolve_smal_file_for_checkpoint({"smal_file": ckpt_model}, "/nope/missing.pkl", None)
    assert applied == []


def test_singleview_loader_keeps_cli_smal_file(applied, model_files, tmp_path, monkeypatch):
    """The checkpoint's smal_file must not be re-applied over --smal_file."""
    ckpt_model, cli_model = model_files
    ckpt_path = tmp_path / "model.pth"
    ckpt_path.touch()
    fake_checkpoint = {"config": {"smal_file": ckpt_model}, "model_state_dict": {}}
    monkeypatch.setattr(svi.torch, "load", lambda *a, **k: fake_checkpoint)

    class StopBeforeModelBuild(Exception):
        pass

    def stop(*a, **k):
        raise StopBeforeModelBuild

    monkeypatch.setattr(svi, "SMILImageRegressor", stop)

    # load_model_from_checkpoint wraps every error in RuntimeError
    with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
        svi.load_model_from_checkpoint(str(ckpt_path), "cpu", cli_smal_file=cli_model)
    assert applied == [cli_model]


# ---------------------------------------------------------------------------
# apply_scale_trans_to_fitter
# ---------------------------------------------------------------------------


class FakeModel:
    """Only what apply_scale_trans_to_fitter reads from the model."""

    def __init__(self, mode):
        self.scale_trans_mode = mode
        self.expanded_with = None

    def _transform_separate_pca_weights_to_joint_values(self, scale_weights, trans_weights):
        self.expanded_with = (scale_weights, trans_weights)
        return torch.full((1, 4, 3), 2.0), torch.full((1, 4, 3), 3.0)


def fake_fitter():
    return SimpleNamespace(log_beta_scales=SimpleNamespace(data=None), betas_trans=SimpleNamespace(data=None))


@pytest.fixture
def use_pca(monkeypatch):
    monkeypatch.setattr(
        svi.TrainingConfig,
        "get_scale_trans_config",
        staticmethod(lambda: {"separate": {"use_pca_transformation": True}}),
    )


def test_translation_only_head_is_applied_per_joint():
    fitter = fake_fitter()
    trans = torch.ones(1, 4, 3)
    svi.apply_scale_trans_to_fitter(FakeModel("entangled_with_betas"), fitter, {"betas_trans": trans}, "cpu")
    assert torch.equal(fitter.betas_trans.data, trans)
    assert fitter.log_beta_scales.data is None


def test_translation_only_head_is_expanded_with_zero_scale_weights(use_pca):
    model, fitter = FakeModel("separate"), fake_fitter()
    svi.apply_scale_trans_to_fitter(model, fitter, {"betas_trans": torch.ones(1, 5)}, "cpu")
    scale_weights, trans_weights = model.expanded_with
    assert torch.equal(scale_weights, torch.zeros(1, 5))
    assert torch.equal(trans_weights, torch.ones(1, 5))
    assert torch.equal(fitter.betas_trans.data, torch.full((1, 4, 3), 3.0))
    assert fitter.log_beta_scales.data is None


def test_both_heads_are_expanded(use_pca):
    model, fitter = FakeModel("separate"), fake_fitter()
    params = {"log_beta_scales": torch.ones(1, 5), "betas_trans": torch.ones(1, 5)}
    svi.apply_scale_trans_to_fitter(model, fitter, params, "cpu")
    assert torch.equal(fitter.log_beta_scales.data, torch.full((1, 4, 3), 2.0))
    assert torch.equal(fitter.betas_trans.data, torch.full((1, 4, 3), 3.0))


def test_ignore_mode_applies_nothing():
    fitter = fake_fitter()
    params = {"log_beta_scales": torch.ones(1, 4, 3), "betas_trans": torch.ones(1, 4, 3)}
    svi.apply_scale_trans_to_fitter(FakeModel("ignore"), fitter, params, "cpu")
    assert fitter.log_beta_scales.data is None
    assert fitter.betas_trans.data is None


# ---------------------------------------------------------------------------
# --dataset indexing
# ---------------------------------------------------------------------------


class FakeMultiViewDataset:
    """3 frames x camera slots [0, 1, 2]; frame 1 has no camera 1."""

    _sv_items = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]

    def __len__(self):
        return len(self._sv_items)


def test_view_items_are_per_camera_and_frame_ordered():
    got = svi.compute_view_item_indices(FakeMultiViewDataset(), [0, 2])
    assert got == {0: [0, 3, 5], 2: [2, 4, 7]}


def test_missing_camera_frames_are_skipped():
    assert svi.compute_view_item_indices(FakeMultiViewDataset(), [1]) == {1: [1, 6]}


def test_singleview_dataset_maps_to_one_slot():
    dataset = type("FakeSingleViewDataset", (), {"__len__": lambda self: 5})()
    assert svi.compute_view_item_indices(dataset, [0]) == {0: [0, 1, 2, 3, 4]}


@pytest.mark.parametrize(
    "num_items, max_frames, num_subclips, expected",
    [
        (100, None, 1, [(0, 100)]),  # full clip
        (100, 30, 1, [(0, 30)]),  # max_frames caps a single clip
        (10, 999, 1, [(0, 10)]),  # max_frames beyond the dataset is clamped
        (100, 20, 3, [(0, 20), (33, 53), (66, 86)]),  # evenly spaced subclips
        (50, 40, 3, [(0, 50)]),  # subclips that don't fit -> one full clip
    ],
)
def test_subclip_ranges(num_items, max_frames, num_subclips, expected):
    assert svi.compute_subclip_ranges(num_items, max_frames, num_subclips) == expected


# ---------------------------------------------------------------------------
# inference_utils
# ---------------------------------------------------------------------------


def test_smoother_averages_over_window():
    smoother = PredictionSmoother(window_size=3)
    out = [smoother({"trans": torch.tensor([v])})["trans"].item() for v in (1.0, 2.0, 3.0, 4.0)]
    assert out == [1.0, 1.5, 2.0, 3.0]


def test_smoother_handles_per_view_and_metadata_keys():
    smoother = PredictionSmoother(window_size=2)
    smoother({"fov_per_view": [torch.tensor(10.0), torch.tensor(20.0)], "num_views": 2})
    out = smoother({"fov_per_view": [torch.tensor(30.0), torch.tensor(40.0)], "num_views": 3})
    assert [t.item() for t in out["fov_per_view"]] == [20.0, 30.0]
    assert out["num_views"] == 3  # metadata comes from the latest frame


def test_smoother_disabled_returns_input():
    params = {"trans": torch.tensor([1.0])}
    assert PredictionSmoother(window_size=0)(params) is params


def test_pad_or_resize():
    small = np.full((2, 2, 3), 255, dtype=np.uint8)
    padded = pad_or_resize(small, (4, 3))  # (width, height)
    assert padded.shape == (3, 4, 3)
    assert (padded[:2, :2] == 255).all() and (padded[2:, :] == 40).all()

    large = np.zeros((10, 10, 3), dtype=np.uint8)
    assert pad_or_resize(large, (4, 4)).shape == (4, 4, 3)
