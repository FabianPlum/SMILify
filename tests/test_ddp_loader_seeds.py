"""Under multi-view DDP every rank's loader workers draw their own augmentation stream (issue #150)."""

import random

import numpy as np
import pytest
import torch

import smal_fitter.neuralSMIL.train_multiview_regressor as trainer
from smal_fitter.sleap_data.sleap_multiview_dataset import SLEAPMultiViewDataset, multiview_collate_fn
from tests.multiview_h5 import write_multiview_h5

# One random family each, so that an augmented image shows that family's draws only and the cross-rank checks
# cover the workers' numpy and torch streams separately. The dataset turns every photometric op on by default.
_OFF = {
    "color_jitter_brightness": 0.0,
    "color_jitter_contrast": 0.0,
    "color_jitter_saturation": 0.0,
    "gaussian_noise_std": 0.0,
    "gaussian_blur_prob": 0.0,
    "random_erasing_prob": 0.0,
}
AUGS = {
    "photometric": dict(_OFF, color_jitter_brightness=0.1),  # parameters drawn with numpy in the workers
    "noise": dict(_OFF, gaussian_noise_std=0.02),  # per-view noise drawn with torch in the workers
}
CONFIG = {"seed": 0, "batch_size": 2, "num_workers": 2, "pin_memory": False, "train_sub_sampling": 1.0}
PATHS = {  # the three train loaders of create_fractional_train_loader
    "full": {},
    "fractional": {"train_sub_sampling": 0.5},
    "by_session": {"train_sub_sampling_by_session": {"s": 0.5}},
}


@pytest.fixture(autouse=True)
def _keep_global_rng():
    states = torch.get_rng_state(), np.random.get_state(), random.getstate()
    yield
    torch.set_rng_state(states[0])
    np.random.set_state(states[1])
    random.setstate(states[2])


@pytest.fixture(params=AUGS)
def dataset(tmp_path, request):
    # Identical mid-gray samples with one view count, so that an augmented image depends on its random draws only.
    # With one view count the ranks' streams stayed in step even before #149.
    path = tmp_path / "multiview.h5"
    write_multiview_h5(path, views_per_sample=[3] * 8, image_size=16, fill=128)
    return SLEAPMultiViewDataset(str(path), augmentation_config=AUGS[request.param])


def _loader(monkeypatch, dataset, rank, epoch=0, world=2, **config):
    """The train loader that rank `rank` of `world` builds, after main()'s seeding (alike on every rank)."""
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: world)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: rank)
    torch.manual_seed(CONFIG["seed"])  # set_random_seeds seeds every rank's default generator alike
    return trainer.create_fractional_train_loader(dataset, epoch, dict(CONFIG, **config), True, multiview_collate_fn)


def _augmented(loader, dataset):
    with trainer.training_sampling(dataset, augment=True, num_views_to_use=None):
        return [np.stack(x["images"]) for x_batch, _ in loader for x in x_batch]


def test_ranks_draw_their_own_augmentation(monkeypatch, dataset):
    loaders = [_loader(monkeypatch, dataset, rank) for rank in (0, 1)]
    order = list(loaders[0].sampler) + list(loaders[1].sampler)
    assert sorted(order) == list(range(8))  # the ranks still split the samples between them
    images0, images1 = (_augmented(loader, dataset) for loader in loaders)
    assert len(images0) == 4
    assert not any(np.array_equal(a, b) for a, b in zip(images0, images1, strict=True))  # no draw shared anywhere
    # Reproducible, with fresh draws every epoch
    again = _augmented(_loader(monkeypatch, dataset, 0), dataset)
    assert all(np.array_equal(a, b) for a, b in zip(images0, again, strict=True))
    next_epoch = _augmented(_loader(monkeypatch, dataset, 0, epoch=1), dataset)
    assert not any(np.array_equal(a, b) for a, b in zip(images0, next_epoch, strict=True))


def test_the_default_generator_shares_the_draws_across_ranks(monkeypatch, dataset):
    """What #150 fixes: seeded from the default generator, worker w of every rank draws the same stream."""
    monkeypatch.setattr(trainer, "_train_loader_generator", lambda *args: None)
    images0, images1 = (_augmented(_loader(monkeypatch, dataset, rank), dataset) for rank in (0, 1))
    assert len(images0) == 4 and all(np.array_equal(a, b) for a, b in zip(images0, images1, strict=True))


@pytest.mark.parametrize("path", PATHS)
def test_every_train_loader_gets_its_own_seed_per_rank_and_epoch(monkeypatch, dataset, path):
    loaders = {(r, e): _loader(monkeypatch, dataset, r, e, world=3, **PATHS[path]) for r in range(3) for e in range(3)}
    seeds = [loader.generator.initial_seed() for loader in loaders.values()]
    assert len(set(seeds)) == len(seeds)  # a different seed for every (rank, epoch)
    if path != "full":  # the epoch's subset is the same on every rank
        for e in range(3):
            assert loaders[0, e].dataset.indices == loaders[1, e].dataset.indices == loaders[2, e].dataset.indices
    single = trainer.create_fractional_train_loader(
        dataset, 0, dict(CONFIG, **PATHS[path]), False, multiview_collate_fn
    )
    assert single.generator is None  # runs without DDP draw exactly what they drew before
