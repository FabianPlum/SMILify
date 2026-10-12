"""
Tests for the shared train/val/test split helper (smal_fitter/neuralSMIL/data_splits.py).

The random split must reproduce the trainers' original split index for index,
so existing checkpoints benchmark on the same test set. The blocked split must
hold out one contiguous block per session along time, rotate it over the folds,
and keep every view of a frame in the same split.
"""

import json

import h5py
import numpy as np
import pytest
import torch

from smal_fitter.neuralSMIL.configs import SingleViewConfig, load_config
from smal_fitter.neuralSMIL.data_splits import blocked_sample_split, split_dataset, validate_split_settings


class _FakeDataset:
    """Stands in for SLEAPMultiViewDataset: one item per sample, or one per view."""

    def __init__(self, n_samples, hdf5_path=None, views_per_sample=None):
        self.num_samples = n_samples
        self.hdf5_path = hdf5_path
        self.item_sample_indices = None
        if views_per_sample is not None:
            self.item_sample_indices = np.repeat(np.arange(n_samples), views_per_sample)

    def __len__(self):
        return self.num_samples if self.item_sample_indices is None else len(self.item_sample_indices)


def _messy_sessions():
    """Three sessions of unequal length, rows interleaved, frame_idx with gaps."""
    rng = np.random.default_rng(1)
    names, frames = [], []
    for name, n in (("a", 103), ("b", 250), ("c", 41)):
        names += [name] * n
        frames += sorted(rng.choice(10 * n, size=n, replace=False).tolist())
    perm = rng.permutation(len(names))
    return np.asarray(names, dtype=object)[perm], np.asarray(frames)[perm]


def test_random_matches_legacy_splits():
    """Old multi-view call (random_split over the dataset) and old single-view
    call (random_split over samples, expanded to view-items in dataset order)."""
    seed, n = 42, 500
    n_train, n_val = int(n * 0.85), int(n * 0.05)
    lengths = [n_train, n_val, n - n_train - n_val]
    ratios = dict(train_ratio=0.85, val_ratio=0.05, test_ratio=0.1, seed=seed)

    mv = _FakeDataset(n)
    legacy_mv = torch.utils.data.random_split(mv, lengths, generator=torch.Generator().manual_seed(seed))
    new_mv, _ = split_dataset(mv, **ratios)
    assert [s.indices for s in new_mv] == [list(s.indices) for s in legacy_mv]

    sv = _FakeDataset(n, views_per_sample=np.random.default_rng(0).integers(2, 6, size=n))
    legacy_samples = torch.utils.data.random_split(range(n), lengths, generator=torch.Generator().manual_seed(seed))
    isi = sv.item_sample_indices
    legacy_sv = [[i for i, s in enumerate(isi) if int(s) in keep] for keep in map(set, legacy_samples)]
    new_sv, _ = split_dataset(sv, **ratios)
    assert [s.indices for s in new_sv] == legacy_sv


def test_blocked_folds_rotate_one_contiguous_block_per_session():
    names, frames = _messy_sessions()
    held_out_count = np.zeros(len(names), dtype=int)
    for fold in range(5):
        split = blocked_sample_split(names, frames, 0.05, 0.15, holdout_fold=fold)
        train, val, test = set(split.train), set(split.val), set(split.test)
        assert len(train) + len(val) + len(test) == len(names) == len(train | val | test)
        held_out_count[split.val + split.test] += 1
        for name in ("a", "b", "c"):
            rows = np.flatnonzero(names == name)
            rows = rows[np.argsort(frames[rows])]
            n = len(rows)
            labels = ["train" if r in train else "val" if r in val else "test" for r in rows]
            held = [i for i, lab in enumerate(labels) if lab != "train"]
            assert held == list(range(fold * n // 5, (fold + 1) * n // 5))
            n_val = [labels[i] for i in held].count("val")
            assert [labels[i] for i in held] == ["val"] * n_val + ["test"] * (len(held) - n_val)
    assert np.all(held_out_count == 1)
    # Default fold is the last block: train on the start, test on the end of each session.
    default = blocked_sample_split(names, frames, 0.05, 0.15)
    assert sorted(default.test) == sorted(split.test)


def test_blocked_split_from_hdf5_keeps_views_together(tmp_path):
    names, frames = _messy_sessions()
    h5 = tmp_path / "toy.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("auxiliary/session_name", data=names, dtype=h5py.string_dtype())
        f.create_dataset("auxiliary/frame_idx", data=frames)

    views = np.random.default_rng(0).integers(1, 5, size=len(names))
    ds = _FakeDataset(len(names), hdf5_path=str(h5), views_per_sample=views)
    kwargs = dict(split_method="blocked", train_ratio=0.8, val_ratio=0.05, test_ratio=0.15, holdout_fold=1)
    subsets, split = split_dataset(ds, seed=1, **kwargs)
    for subset, samples in zip(subsets, (split.train, split.val, split.test)):
        assert set(ds.item_sample_indices[subset.indices].tolist()) == set(samples)
        assert len(subset.indices) == int(views[samples].sum())
    assert split == blocked_sample_split(names, frames, 0.05, 0.15, holdout_fold=1)
    # The training seed does not move the blocked split.
    assert split_dataset(ds, seed=99, **kwargs)[1] == split


def test_validate_split_settings_rejects_bad_settings():
    validate_split_settings("blocked", 0.8, 0.05, 0.15, 0)
    for args, match in (
        (("kfold", 0.8, 0.05, 0.15, None), "split_method"),
        (("blocked", 0.85, 0.05, 0.1, None), "whole folds"),
        (("blocked", 0.7, 0.05, 0.15, None), "train_ratio"),
        (("blocked", 0.8, 0.05, 0.15, 5), r"\[0, 4\]"),
    ):
        with pytest.raises(ValueError, match=match):
            validate_split_settings(*args)


def test_split_settings_reach_the_trainers(tmp_path):
    dataset = {"train_ratio": 0.8, "val_ratio": 0.05, "test_ratio": 0.15, "split_method": "blocked"}
    cfg = tmp_path / "mv.json"
    cfg.write_text(json.dumps({"mode": "multiview", "dataset": dataset}))
    config = load_config(config_file=str(cfg), cli_overrides={"dataset": {"holdout_fold": 0}})
    flat = config.to_multiview_legacy_dict()
    assert (flat["split_method"], flat["holdout_fold"], flat["test_ratio"]) == ("blocked", 0, 0.15)

    sv = SingleViewConfig()
    for key, value in dataset.items():
        setattr(sv.dataset, key, value)
    with pytest.raises(ValueError, match="from_multiview"):
        sv.validate()
    sv.dataset.from_multiview = True
    sv.validate()
    assert sv.to_legacy_dict()["dataset"]["split_method"] == "blocked"
