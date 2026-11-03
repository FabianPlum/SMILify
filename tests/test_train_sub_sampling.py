"""
Tests for the training-split sub-sampling of the trainers
(dataset.train_sub_sampling and the multi-view dataset.train_sub_sampling_by_session).

Each source of a merged dataset draws its own subset of the training split per
epoch: the same subset for the same epoch (every DDP rank seeds with seed +
epoch), a new one for the next epoch, and sessions without a key keep
train_sub_sampling. Entries that cannot apply stop the run at start-up. The old
name dataset_fraction still works as a deprecated alias (with a printed
WARNING), in a JSON and on the multi-view command line.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from smal_fitter.neuralSMIL.configs import load_config
from smal_fitter.neuralSMIL.train_multiview_regressor import (
    MultiViewTrainingConfig,
    _check_fraction_by_session,
    create_fractional_train_loader,
)

SIZES = {"rig_a_s1": 400, "rig_b_s1": 120, "sparse_s1": 80}
REPO = Path(__file__).resolve().parents[1]


class _FakeDataset:
    """Stands in for SLEAPMultiViewDataset: the loader only reads the HDF5 path and the length."""

    def __init__(self, hdf5_path, n_samples):
        self.hdf5_path = hdf5_path
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples


@pytest.fixture
def merged(tmp_path):
    """(session name per HDF5 row, training rows, training Subset) of a merged file."""
    # Rows interleaved: two densely sampled sources and one already subsampled source.
    names = np.asarray([name for name, n in SIZES.items() for _ in range(n)], dtype=object)
    names = names[np.random.default_rng(0).permutation(len(names))]
    h5 = tmp_path / "merged.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("auxiliary/session_name", data=names, dtype=h5py.string_dtype())
        f.create_dataset("auxiliary/frame_idx", data=np.arange(len(names)))

    # The training split is a Subset in shuffled row order, as split_dataset returns it.
    rows = np.random.default_rng(1).permutation(len(names))[:540]
    return names, rows, torch.utils.data.Subset(_FakeDataset(str(h5), len(names)), rows.tolist())


def test_fraction_by_session_draws_exact_counts_and_a_fresh_subset_each_epoch(merged):
    names, rows, train_set = merged
    config = {
        "train_sub_sampling": 1.0,
        "train_sub_sampling_by_session": {"rig_a": 0.25, "rig_b": 0.5},
        "seed": 7,
        "batch_size": 4,
        "num_workers": 0,
        "pin_memory": False,
    }
    in_train = {s: int((names[rows] == s).sum()) for s in SIZES}
    expected = {s: int(in_train[s] * f) for s, f in (("rig_a_s1", 0.25), ("rig_b_s1", 0.5), ("sparse_s1", 1.0))}

    def drawn(epoch):
        loader = create_fractional_train_loader(train_set, epoch, config, is_distributed=False, collate_fn=None)
        drawn_rows = rows[np.asarray(loader.dataset.indices)]  # epoch subset -> training-set positions -> HDF5 rows
        assert len(set(drawn_rows.tolist())) == len(drawn_rows)
        return {s: set(drawn_rows[names[drawn_rows] == s].tolist()) for s in SIZES}

    epochs = [drawn(0), drawn(0), drawn(1)]
    for per_session in epochs:
        assert {s: len(r) for s, r in per_session.items()} == expected
    assert epochs[0] == epochs[1]
    assert epochs[1]["rig_a_s1"] != epochs[2]["rig_a_s1"] and epochs[1]["rig_b_s1"] != epochs[2]["rig_b_s1"]
    assert epochs[2]["sparse_s1"] == set(rows[names[rows] == "sparse_s1"].tolist())


def test_fraction_by_session_names_every_entry_that_cannot_apply(merged, tmp_path):
    # A typo, fractions above 1 and of 0, and an empty key; "rig_a" is fine and must not be named.
    with pytest.raises(ValueError) as err:
        _check_fraction_by_session(merged[2], {"rig_a": 0.25, "rig_c": 0.5, "rig_b": 1.5, "sparse": 0.0, "": 0.5})
    msg = str(err.value)
    assert "'rig_c' matches no training session" in msg and "'rig_b': 1.5" in msg and "'' is empty" in msg
    assert "'sparse': 0.0" in msg
    assert "'rig_a'" not in msg
    # A null fraction on a key that takes sessions is named as well (not a bare TypeError).
    with pytest.raises(ValueError, match=r"'rig_a': None is not in \(0, 1\]"):
        _check_fraction_by_session(merged[2], {"rig_a": None})
    # A file without session names is named by the setting that needs them, not by the blocked split.
    no_sessions = tmp_path / "no_sessions.h5"
    h5py.File(no_sessions, "w").close()
    with pytest.raises(ValueError, match="train_sub_sampling_by_session needs auxiliary/session_name"):
        _check_fraction_by_session(torch.utils.data.Subset(_FakeDataset(str(no_sessions), 4), [0]), {"a": 1})


def _mode_json(tmp_path, dataset, mode="multiview"):
    path = tmp_path / "config.json"
    out = {"checkpoint_dir": str(tmp_path / "checkpoints")}
    path.write_text(json.dumps({"mode": mode, "dataset": dataset, "output": out}))
    return str(path)


def test_dataset_fraction_is_a_deprecated_alias_of_train_sub_sampling(tmp_path, capsys):
    cfg = load_config(_mode_json(tmp_path, {"dataset_fraction": 0.2, "train_sub_sampling_by_session": {"a": 0.5}}))
    out = capsys.readouterr().out
    assert "WARNING: dataset.dataset_fraction is deprecated and will be removed; use dataset.train_sub_sampling" in out
    assert cfg.dataset.train_sub_sampling == 0.2
    # The trainers' main() read the legacy dicts, not the dataclass.
    legacy = cfg.to_multiview_legacy_dict()
    assert legacy["train_sub_sampling"] == 0.2 and legacy["train_sub_sampling_by_session"] == {"a": 0.5}
    cfg = load_config(_mode_json(tmp_path, {"dataset_fraction": 0.2}, mode="singleview"))
    assert "WARNING: dataset.dataset_fraction is deprecated" in capsys.readouterr().out
    assert cfg.to_legacy_dict()["split_config"]["train_sub_sampling"] == 0.2


def test_dataset_fraction_alias_in_a_legacy_flat_json(tmp_path, capsys):
    # A JSON without "mode" is merged over defaults that already set train_sub_sampling: resolved before the merge.
    (tmp_path / "legacy.json").write_text(json.dumps({"dataset_fraction": 0.3}))
    assert MultiViewTrainingConfig.from_file(str(tmp_path / "legacy.json"))["train_sub_sampling"] == 0.3
    assert "WARNING: dataset_fraction is deprecated" in capsys.readouterr().out


def test_dataset_fraction_and_train_sub_sampling_together_stop_the_run(tmp_path):
    with pytest.raises(ValueError, match="both set"):
        load_config(_mode_json(tmp_path, {"dataset_fraction": 0.2, "train_sub_sampling": 0.2}))


@pytest.mark.parametrize(
    "mode, flag",
    [
        ("multiview", "--train_sub_sampling"),
        ("multiview", "--dataset_fraction"),
        ("singleview", "--train_sub_sampling"),
    ],
)
def test_train_sub_sampling_flag_overrides_a_mode_json(tmp_path, mode, flag):
    # The trainer saves the resolved config, then stops on the missing dataset before any model is built.
    trainer, stop = {
        "multiview": ("train_multiview_regressor", "Dataset path does not exist"),
        "singleview": ("train_smil_regressor", "Could not open HDF5 file"),
    }[mode]
    cfg = _mode_json(tmp_path, {"data_path": str(tmp_path / "missing.h5"), "train_sub_sampling": 0.5}, mode=mode)
    run = subprocess.run(
        [sys.executable, "-m", f"smal_fitter.neuralSMIL.{trainer}", "--config", cfg, flag, "0.3"],
        cwd=REPO,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert stop in run.stderr, run.stdout[-2000:] + run.stderr[-2000:]
    resolved = json.loads((tmp_path / "checkpoints" / "config.json").read_text())
    assert resolved["dataset"]["train_sub_sampling"] == 0.3
    assert ("WARNING: --dataset_fraction is deprecated" in run.stdout) == (flag == "--dataset_fraction")
