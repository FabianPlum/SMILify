"""The multi-view loader downscales to the model input in its workers (issue #118); the model input stays bit-identical."""

import types

import cv2
import numpy as np
import pytest
import torch

from smal_fitter.neuralSMIL.smil_image_regressor import SMILImageRegressor
from smal_fitter.sleap_data.sleap_multiview_dataset import SLEAPMultiViewDataset
from tests.multiview_h5 import write_multiview_h5

STORED = 64  # 64 -> 28 is the same 2.29x downscale as 512 -> 224
ALL_OPS = {  # every photometric operation, on every sample
    "color_jitter_brightness": 0.1,
    "color_jitter_contrast": 0.1,
    "color_jitter_saturation": 0.1,
    "gaussian_noise_std": 0.02,
    "gaussian_blur_prob": 1.0,
    "gaussian_blur_kernel_range": (3, 7),
    "random_erasing_prob": 1.0,
    "random_erasing_scale_range": (0.02, 0.1),
}


def _images(x):
    return x["images"] if "images" in x else [x["input_image_data"]]


def _model_input(images, model_resolution):
    """The trainer's per-image preprocessing (SMILImageRegressor.preprocess_image)."""
    model = types.SimpleNamespace(input_resolution=model_resolution)
    return [SMILImageRegressor.preprocess_image(model, image) for image in images]


@pytest.mark.parametrize("single_view", [False, True])
@pytest.mark.parametrize("augment", [False, True])
@pytest.mark.parametrize("model_resolution", [28, 96])  # smaller than the stored 64 px (downscale), larger (upscale)
def test_resizing_in_the_dataset_leaves_the_model_input_unchanged(tmp_path, augment, single_view, model_resolution):
    path = tmp_path / "multiview.h5"
    write_multiview_h5(path, views_per_sample=[5, 3, 4], image_size=STORED, textured=True)
    kw = dict(augment=augment, augmentation_config=ALL_OPS, return_single_view=single_view)
    stored = SLEAPMultiViewDataset(str(path), **kw)
    resized = SLEAPMultiViewDataset(str(path), image_resolution=model_resolution, **kw)
    returned = min(model_resolution, STORED)  # the workers only downscale; upscaling stays in the trainer
    for i in range(len(stored)):
        images = []
        for ds in (stored, resized):
            np.random.seed(i)  # augmentation parameters
            torch.manual_seed(i)  # noise
            images.append(_images(ds[i][0]))
        assert len(images[0]) == len(images[1]) > 0
        assert all(im.shape == (STORED, STORED, 3) for im in images[0])
        assert all(im.shape == (returned, returned, 3) for im in images[1])
        for a, b in zip(*(_model_input(ims, model_resolution) for ims in images)):
            assert a.shape == (1, 3, model_resolution, model_resolution)
            assert torch.equal(a, b)


def _reference_photometric(image, p, noise):
    """The steps as they were before the in-place rewrite: a new array per step, gray = mean over channels."""
    if p["brightness_delta"] is not None:
        image = image + p["brightness_delta"]
    if p["contrast_factor"] is not None:
        mean = image.mean()
        image = (image - mean) * p["contrast_factor"] + mean
    if p["saturation_factor"] is not None:
        gray = image.mean(axis=2, keepdims=True)
        image = gray + (image - gray) * p["saturation_factor"]
    image = image + noise
    if p["blur_kernel"] is not None:
        image = cv2.GaussianBlur(image, (p["blur_kernel"], p["blur_kernel"]), 0)
    e = p["erase"]
    if e is not None:
        h, w = image.shape[:2]
        area = int(h * w * e["area_frac"])
        eh, ew = min(int(np.sqrt(area * e["aspect"])), h), min(int(np.sqrt(area / e["aspect"])), w)
        y0, x0 = int(e["y0_frac"] * max(0, h - eh)), int(e["x0_frac"] * max(0, w - ew))
        image[y0 : y0 + eh, x0 : x0 + ew] = image.mean()
    return np.clip(image, 0.0, 1.0)


def test_photometric_augmentation_matches_the_step_by_step_reference(tmp_path):
    path = tmp_path / "multiview.h5"
    write_multiview_h5(path, views_per_sample=[1], image_size=STORED)
    ds = SLEAPMultiViewDataset(str(path), augment=True, augmentation_config=ALL_OPS)
    rng = np.random.default_rng(0)
    np.random.seed(0)  # augmentation parameters, independent of the tests that ran before
    for trial in range(5):
        image = (rng.random((STORED, STORED, 3)) * 1.2 - 0.1).clip(0, 1).astype(np.float32)  # with saturated pixels
        before = image.copy()
        params = ds._sample_photometric_params()
        torch.manual_seed(trial)
        augmented = ds._apply_photometric_augmentation(image, shared_params=params)
        torch.manual_seed(trial)
        noise = torch.randn(image.shape).numpy() * np.float32(ALL_OPS["gaussian_noise_std"])
        assert np.array_equal(augmented, _reference_photometric(image, params, noise))
        assert np.array_equal(image, before)  # the caller's image is left alone
