"""Multi-view training samples num_views_to_use views per sample; validation sees every view."""

import cv2
import h5py
import numpy as np
from torch.utils.data import DataLoader

from smal_fitter.neuralSMIL.train_multiview_regressor import training_sampling
from smal_fitter.sleap_data.sleap_multiview_dataset import SLEAPMultiViewDataset, multiview_collate_fn


def _write_dataset(path, views_per_sample, max_views=5, joints=3):
    """A minimal multi-view HDF5: blank 8 px JPEGs, the first n view slots of a sample valid, camera index = slot."""
    n = len(views_per_sample)
    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata").attrs
        meta.update(num_samples=n, max_views=max_views, n_joints=joints, n_pose=joints - 1, n_betas=1)
        meta.update(target_resolution=8, backbone_name="vit_large_patch16_224", dataset_type="sleap_multiview")
        f.create_dataset("multiview_images/view_mask", data=np.arange(max_views) < np.array(views_per_sample)[:, None])
        jpeg = np.frombuffer(cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes(), np.uint8)
        for v in range(max_views):
            images = f.create_dataset(f"multiview_images/image_jpeg_view_{v}", (n,), dtype=h5py.vlen_dtype(np.uint8))
            for i in range(n):
                images[i] = jpeg
        f.create_dataset("multiview_keypoints/keypoints_2d", data=np.zeros((n, max_views, joints, 2), np.float32))
        f.create_dataset("multiview_keypoints/keypoint_visibility", data=np.ones((n, max_views, joints), np.float32))
        f.create_dataset("multiview_keypoints/camera_indices", data=np.tile(np.arange(max_views), (n, 1)))
        f.create_dataset("parameters/betas", data=np.zeros((n, 1), np.float32))
        f.create_dataset("auxiliary/session_name", data=["s"] * n, dtype=h5py.string_dtype())
        f.create_dataset("auxiliary/frame_idx", data=np.arange(n))
        f.create_dataset("auxiliary/has_ground_truth_betas", data=np.zeros(n, bool))
        f.create_dataset("auxiliary/camera_names", data=["A,B,C,D,E"] * n, dtype=h5py.string_dtype())


def test_training_samples_num_views_to_use_and_validation_every_view(tmp_path):
    path = tmp_path / "multiview.h5"
    _write_dataset(path, views_per_sample=[5, 5, 3, 5])
    dataset = SLEAPMultiViewDataset(str(path), num_views_to_use=None)  # as the trainer builds it
    # One worker each: the workers copy the dataset when iteration starts, as in training
    train_loader, val_loader = (
        DataLoader(dataset, batch_size=2, num_workers=1, collate_fn=multiview_collate_fn) for _ in range(2)
    )

    def camera_indices(loader):
        return [x["camera_indices"].tolist() for x_batch, _ in loader for x in x_batch]

    with training_sampling(dataset, augment=False, num_views_to_use=3):
        train_views = camera_indices(train_loader)
    val_views = camera_indices(val_loader)

    assert [len(v) for v in train_views] == [3, 3, 3, 3]
    assert val_views == [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2], [0, 1, 2, 3, 4]]
