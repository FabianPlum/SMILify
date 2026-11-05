"""Minimal multi-view HDF5 files for tests, in the layout SLEAPMultiViewDataset reads."""

import cv2
import h5py
import numpy as np


def write_multiview_h5(path, views_per_sample, max_views=5, joints=3, image_size=8, textured=False):
    """The first n view slots of each sample valid, camera index = slot, no cameras or 3D.

    Images are blank JPEGs of image_size px, or with textured=True a different random texture per view and
    sample (fixed seed), so that image operations have something to act on.
    """
    n = len(views_per_sample)
    rng = np.random.default_rng(0)
    blank = np.zeros((image_size, image_size, 3), np.uint8)
    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata").attrs
        meta.update(num_samples=n, max_views=max_views, n_joints=joints, n_pose=joints - 1, n_betas=1)
        meta.update(target_resolution=image_size, backbone_name="vit_large_patch16_224", dataset_type="sleap_multiview")
        f.create_dataset("multiview_images/view_mask", data=np.arange(max_views) < np.array(views_per_sample)[:, None])
        for v in range(max_views):
            images = f.create_dataset(f"multiview_images/image_jpeg_view_{v}", (n,), dtype=h5py.vlen_dtype(np.uint8))
            for i in range(n):
                image = rng.integers(0, 256, blank.shape, dtype=np.uint8) if textured else blank
                images[i] = np.frombuffer(cv2.imencode(".jpg", image)[1].tobytes(), np.uint8)
        f.create_dataset("multiview_keypoints/keypoints_2d", data=np.zeros((n, max_views, joints, 2), np.float32))
        f.create_dataset("multiview_keypoints/keypoint_visibility", data=np.ones((n, max_views, joints), np.float32))
        f.create_dataset("multiview_keypoints/camera_indices", data=np.tile(np.arange(max_views), (n, 1)))
        f.create_dataset("parameters/betas", data=np.zeros((n, 1), np.float32))
        f.create_dataset("auxiliary/session_name", data=["s"] * n, dtype=h5py.string_dtype())
        f.create_dataset("auxiliary/frame_idx", data=np.arange(n))
        f.create_dataset("auxiliary/has_ground_truth_betas", data=np.zeros(n, bool))
        f.create_dataset("auxiliary/camera_names", data=["A,B,C,D,E"] * n, dtype=h5py.string_dtype())
