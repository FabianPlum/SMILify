"""Raw SLEAP 2D labels: extraction chain, dataset switch, and appending to an existing HDF5."""

import cv2
import h5py
import numpy as np
import pytest

from sleap_data_loader import SLEAPDataLoader
from smal_fitter.sleap_data.add_raw_keypoints import append_sidecar
from smal_fitter.sleap_data.preprocess_sleap_multiview_dataset import (
    RAW_KEYPOINT_DATASETS,
    SLEAPMultiViewPreprocessor,
)
from smal_fitter.sleap_data.sleap_multiview_dataset import SLEAPMultiViewDataset


def _stub_loader(tracks, scores, joint_mapping):
    """A session_dirs SLEAPDataLoader without files: real extraction and SMAL mapping."""
    loader = object.__new__(SLEAPDataLoader)
    loader.data_structure_type = "session_dirs"
    loader.confidence_threshold = 0.5
    loader.joint_mapping = joint_mapping
    loader.smal_n_joints = len(joint_mapping)
    return loader, {"tracks": tracks, "point_scores": scores}


def test_raw_view_keypoints_match_the_ideal_projection_in_the_crop():
    """Distorted SLEAP pixels -> raw labels == pinhole projection with the crop-adjusted K."""
    rng = np.random.default_rng(0)
    width, height, res = 1280, 1024, 512
    K = np.array([[1100.0, 0, 655.0], [0, 1090.0, 500.0], [0, 0, 1]])
    dist = np.array([-0.21, 0.08, 0.001, -0.0005, 0.0])
    pts3d = np.column_stack([rng.uniform(-0.15, 0.15, (6, 2)), rng.uniform(1.0, 1.4, 6)])
    distorted = cv2.projectPoints(pts3d, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    distorted[4] = np.nan  # lost track
    scores = np.array([0.9, 0.8, 0.3, 0.95, 0.9, 0.7], dtype=np.float32)  # joint 2 below threshold
    loader, camera_data = _stub_loader(
        distorted.T[None, :, :, None], scores[None, :, None], {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: -1}
    )

    pre = SLEAPMultiViewPreprocessor(target_resolution=res, crop_mode="centred", confidence_threshold=0.5)
    info = pre._size_transform_info(height, width)
    kp, vis, conf = pre.raw_view_keypoints(loader, camera_data, 0, info, K, dist)

    K_adj = pre._adjust_intrinsics_for_transform(K, info)
    ideal = (pts3d @ K_adj.T)[:, :2] / pts3d[:, 2:]
    np.testing.assert_allclose(kp[:4], ideal[:4, ::-1] / res, atol=2e-4)  # [y, x], normalised
    np.testing.assert_array_equal(vis, [1, 1, 0, 1, 0, 0])
    np.testing.assert_allclose(conf, [0.9, 0.8, 0.3, 0.95, 0.0, 0.0])  # lost track / unmapped -> 0
    np.testing.assert_array_equal(kp[4:], 0.0)


def _write_labels(path, n=6, views=3, joints=4):
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as f:
        for name in ("multiview_images", "parameters", "auxiliary", "metadata"):
            f.create_group(name)
        f.create_dataset("multiview_images/view_mask", data=np.ones((n, views), bool))
        f.create_dataset("auxiliary/session_name", data=["s"] * n, dtype=h5py.string_dtype())
        f.create_dataset("auxiliary/frame_idx", data=np.arange(n))
        f.create_dataset("auxiliary/camera_names", data=["A,B,C"] * n, dtype=h5py.string_dtype())
        f.create_dataset("multiview_keypoints/keypoints_2d", data=rng.random((n, views, joints, 2), np.float32))
        f.create_dataset("multiview_keypoints/keypoint_visibility", data=np.ones((n, views, joints), np.float32))
    return {
        "keypoints_2d_raw": rng.random((n, views, joints, 2), dtype=np.float32),
        "keypoint_visibility_raw": (rng.random((n, views, joints)) > 0.3).astype(np.float32),
        "keypoint_confidence_raw": rng.random((n, views, joints), dtype=np.float32),
    }


def _write_sidecar(path, dataset_path, raw):
    with h5py.File(path, "w") as s, h5py.File(dataset_path, "r") as f:
        for name, data in raw.items():
            s.create_dataset(f"multiview_keypoints/{name}", data=data)
        for name in ("auxiliary/session_name", "auxiliary/frame_idx", "auxiliary/camera_names"):
            s.create_dataset(name, data=f[name][:])
        s.create_dataset("multiview_images/view_mask", data=f["multiview_images/view_mask"][:])
        s.attrs["confidence_threshold"] = 0.5
        s.attrs["provenance"] = "{}"


def test_append_sidecar_round_trips_and_refuses_mismatches(tmp_path):
    dataset, sidecar = tmp_path / "ds.h5", tmp_path / "side.h5"
    raw = _write_labels(dataset)
    _write_sidecar(sidecar, dataset, raw)

    append_sidecar(str(dataset), str(sidecar))
    with h5py.File(dataset, "r") as f:
        for name in RAW_KEYPOINT_DATASETS:
            np.testing.assert_array_equal(f[f"multiview_keypoints/{name}"][:], raw[name])
        assert f["metadata"].attrs["has_raw_keypoints"]
        assert f["metadata"].attrs["raw_keypoint_confidence_threshold"] == 0.5
    with pytest.raises(ValueError, match="--overwrite"):
        append_sidecar(str(dataset), str(sidecar))

    # A sidecar built from other rows is refused before anything is written.
    other = tmp_path / "other.h5"
    _write_sidecar(other, dataset, raw)
    with h5py.File(other, "a") as s:
        s["auxiliary/frame_idx"][0] = 99
    with pytest.raises(ValueError, match="auxiliary/frame_idx"):
        append_sidecar(str(dataset), str(other), overwrite=True)


def test_dataset_keypoint_source_selects_and_rethresholds_labels(tmp_path):
    path = tmp_path / "ds.h5"
    raw = _write_labels(path)
    with pytest.raises(ValueError, match="keypoint_source"):
        SLEAPMultiViewDataset(str(path), keypoint_source="triangulated")

    ds = object.__new__(SLEAPMultiViewDataset)
    ds.hdf5_path, ds.keypoint_source, ds.raw_keypoint_min_confidence = str(path), "raw", None
    with pytest.raises(ValueError, match="add_raw_keypoints"):
        ds._validate_dataset()

    sidecar = tmp_path / "side.h5"
    _write_sidecar(sidecar, path, raw)
    append_sidecar(str(path), str(sidecar))
    ds._validate_dataset()
    with h5py.File(path, "r") as f:
        kp, vis = ds._read_2d_labels(f, (2, 1))
        np.testing.assert_array_equal(kp, raw["keypoints_2d_raw"][2, 1])
        np.testing.assert_array_equal(vis, raw["keypoint_visibility_raw"][2, 1])
        ds.raw_keypoint_min_confidence = 0.8
        _, vis = ds._read_2d_labels(f, (2, 1))
        np.testing.assert_array_equal(vis, raw["keypoint_confidence_raw"][2, 1] >= 0.8)
        ds.keypoint_source, ds.raw_keypoint_min_confidence = "reprojected", None
        kp, _ = ds._read_2d_labels(f, 3)
        np.testing.assert_array_equal(kp, f["multiview_keypoints/keypoints_2d"][3])
