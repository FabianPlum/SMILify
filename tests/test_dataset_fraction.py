"""
Tests for the per-source dataset fraction of the multi-view trainer
(dataset.dataset_fraction_by_session in create_fractional_train_loader).

Each source of a merged dataset draws its own subset of the training split per
epoch: the same subset for the same epoch (every DDP rank seeds with seed +
epoch), a new one for the next epoch, and sessions without a key keep
dataset_fraction. Entries that cannot apply stop the run at start-up.
"""

import h5py
import numpy as np
import pytest
import torch

from smal_fitter.neuralSMIL.train_multiview_regressor import _check_fraction_by_session, create_fractional_train_loader

SIZES = {"rig_a_s1": 400, "rig_b_s1": 120, "sparse_s1": 80}


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
        "dataset_fraction": 1.0,
        "dataset_fraction_by_session": {"rig_a": 0.25, "rig_b": 0.5},
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


def test_fraction_by_session_names_every_entry_that_cannot_apply(merged):
    # A typo, fractions above 1 and of 0, and an empty key; "rig_a" is fine and must not be named.
    with pytest.raises(ValueError) as err:
        _check_fraction_by_session(merged[2], {"rig_a": 0.25, "rig_c": 0.5, "rig_b": 1.5, "sparse": 0.0, "": 0.5})
    msg = str(err.value)
    assert "'rig_c' matches no training session" in msg and "'rig_b': 1.5" in msg and "'' is empty" in msg
    assert "'sparse': 0.0" in msg
    assert "'rig_a'" not in msg
