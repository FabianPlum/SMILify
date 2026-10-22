"""replace_extrinsics: swap a multi-view HDF5's extrinsics and re-project its 2D labels."""

import cv2
import h5py
import numpy as np
import pytest
import toml

from smal_fitter.multiview_common.replace_extrinsics import replace_extrinsics

W, H, RES = 640, 480, 256  # centred crop: 480 px square from x = 80, scaled to 256


def _camera(rvec, tvec):
    K = [[500.0, 0.0, 330.0], [0.0, 505.0, 238.0], [0.0, 0.0, 1.0]]  # floats only: TOML arrays are homogeneous
    return {"name": "A", "size": [W, H], "matrix": K, "distortions": [0.0] * 5, "rotation": rvec, "translation": tvec}


def _project(X, cam):
    """Independent oracle: OpenCV pinhole projection into the 256 px crop, as normalised [y, x]."""
    K = np.array(cam["matrix"])
    K_adj = K.copy()
    K_adj[0, 2] -= 80
    K_adj[:2] *= RES / H
    xy = cv2.projectPoints(X, np.array(cam["rotation"]), np.array(cam["translation"]), K_adj, None)[0][:, 0]
    return xy[:, ::-1] / RES, K_adj


def _dataset(tmp_path, base, n=4, joints=5):
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-0.1, 0.1, (n * joints, 2)), rng.uniform(1.0, 1.3, n * joints)])
    X = X.reshape(n, joints, 3)
    X[1, 2] = 0.0  # one joint without 3D: zeros, invisible
    kp, K_adj = _project(X.reshape(-1, 3), base)
    kp = kp.reshape(n, joints, 2)
    kp[1, 2] = 0.0
    vis = np.ones((n, joints), np.float32)
    vis[1, 2] = 0.0
    R0 = cv2.Rodrigues(np.array(base["rotation"]))[0]
    path = tmp_path / "ds.h5"
    with h5py.File(path, "w") as f:
        f.create_group("metadata").attrs.update({"target_resolution": RES, "crop_mode": "centred"})
        f.create_dataset("auxiliary/session_name", data=["s"] * n, dtype=h5py.string_dtype())
        f.create_dataset("auxiliary/camera_names", data=["A"] * n, dtype=h5py.string_dtype())
        f.create_dataset("multiview_images/view_mask", data=np.ones((n, 1), bool))
        g = f.create_group("multiview_keypoints")
        g["camera_intrinsics"] = np.tile(K_adj, (n, 1, 1, 1)).astype(np.float32)
        g["camera_extrinsics_R"] = np.tile(R0, (n, 1, 1, 1)).astype(np.float32)
        g["camera_extrinsics_t"] = np.tile(base["translation"], (n, 1, 1)).astype(np.float32)
        g["keypoints_3d"] = X.astype(np.float32)
        g["keypoints_2d"] = kp[:, None].astype(np.float32)
        g["keypoint_visibility"] = vis[:, None]
    return path, X


def test_replace_extrinsics_reprojects_labels_and_refuses_foreign_extrinsics(tmp_path):
    base = _camera([0.01, -0.02, 0.005], [0.01, -0.02, 0.05])
    new = _camera([0.02, -0.015, 0.0], [0.012, -0.018, 0.049])
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "calibration.toml").write_text(toml.dumps({"cam_0": base}))
    (tmp_path / "s" / "refined.toml").write_text(toml.dumps({"cam_0": new}))
    path, X = _dataset(tmp_path, base)

    replace_extrinsics(str(path), str(tmp_path), "refined.toml")
    expected, _ = _project(X.reshape(-1, 3).astype(np.float32), new)
    with h5py.File(path, "r") as f:
        g = f["multiview_keypoints"]
        kp, vis = g["keypoints_2d"][:, 0], g["keypoint_visibility"][:, 0]
        np.testing.assert_allclose(
            g["camera_extrinsics_R"][0, 0], cv2.Rodrigues(np.array(new["rotation"]))[0], atol=1e-6
        )
        assert f["metadata"].attrs["extrinsics_calibration"] == "refined.toml"
    has3d = vis > 0
    np.testing.assert_allclose(kp[has3d], expected.reshape(kp.shape)[has3d], atol=1e-5)
    assert vis[1, 2] == 0 and has3d.sum() == has3d.size - 1

    # The file now carries refined.toml extrinsics, so calibration.toml no longer describes it.
    with pytest.raises(ValueError, match="not those of calibration.toml"):
        replace_extrinsics(str(path), str(tmp_path), "refined.toml")
