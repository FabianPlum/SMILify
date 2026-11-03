"""Multi-view silhouette loss: masked soft IoU, render vs keypoint-projection convention, the dataset's silhouette
sidecar check, and the SAM 2 generator's box growth and start-up checks."""

import json

import h5py
import numpy as np
import pytest
import torch
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Meshes

from smal_fitter.neuralSMIL.multiview_smil_regressor import MultiViewSMILImageRegressor as MV
from smal_fitter.sleap_data.sleap_multiview_dataset import SLEAPMultiViewDataset


def test_soft_iou_ignores_masked_pixels_and_handles_empty_views():
    target = torch.zeros(3, 8, 8)
    target[:, 2:6, 2:6] = 1
    valid = torch.ones(3, 8, 8)
    pred = target.clone()
    pred[1] = 0  # misses the animal entirely
    pred[2, 0, 0] = 1  # one false-foreground pixel ...
    valid[2, 0, 0] = 0  # ... that lies in an ignore region
    loss = MV._soft_iou_loss(pred, target, valid)
    torch.testing.assert_close(loss, torch.tensor([0.0, 1.0, 0.0]))
    empty = torch.zeros(1, 4, 4)
    assert MV._soft_iou_loss(empty, empty, torch.ones(1, 4, 4)).item() == 0.0


def test_silhouette_render_lands_where_the_keypoint_projection_puts_the_vertices():
    """The silhouette is compared with masks in image (row = y, col = x) layout, the 2D keypoint loss with
    transform_points_screen(...)[..., [1, 0]] / size. Both must agree, or the silhouette pulls the mesh to a
    mirrored position. An off-centre square catches a flip in either axis. Rendered under autocast, as with
    use_mixed_precision: the rasterizer has no half-precision kernel."""
    size = 64
    verts = torch.tensor([[0.25, 0.05, 0.0], [0.55, 0.05, 0.0], [0.55, 0.35, 0.0], [0.25, 0.35, 0.0]])[None]
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]])[None]  # both windings: no culling issue
    cams = FoVPerspectiveCameras(R=torch.eye(3)[None], T=torch.tensor([[0.0, 0.0, 3.0]]), fov=torch.tensor([40.0]))
    yx = cams.transform_points_screen(verts, image_size=((size, size),))[0, :, [1, 0]] / size  # as the 2D loss

    cfg = {"render_size": size, "sigma": 1e-5, "faces_per_pixel": 8, "cull_backfaces": False}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        alpha = MV._render_silhouette_alpha(MV.__new__(MV), Meshes(verts=verts, faces=faces), cams, cfg)[0]
    assert alpha.dtype == torch.float32
    rows, cols = torch.nonzero(alpha > 0.5, as_tuple=True)
    centroid = torch.stack([rows.float().mean(), cols.float().mean()]) / size
    torch.testing.assert_close(centroid, yx.mean(0), atol=1.5 / size, rtol=0)
    assert abs(yx.mean(0)[1] - 0.5) > 0.1  # the square really is off-centre in x


def _write_pair(tmp_path, n=4, views=2, res=8):
    main, side = tmp_path / "ds.h5", tmp_path / "sil.h5"
    for path in (main, side):
        with h5py.File(path, "w") as f:
            f.create_dataset("multiview_images/view_mask", data=np.ones((n, views), bool))
            f.create_dataset("auxiliary/session_name", data=["s"] * n, dtype=h5py.string_dtype())
            f.create_dataset("auxiliary/frame_idx", data=np.arange(n))
            f.create_dataset("auxiliary/camera_names", data=["A,B"] * n, dtype=h5py.string_dtype())
            f.create_dataset("auxiliary/video_frame_idx", data=np.arange(n * views).reshape(n, views))
    with h5py.File(main, "a") as f:
        f.create_group("metadata").attrs["crop_mode"] = "centred"
    with h5py.File(side, "a") as s:
        g = s.create_group("multiview_silhouettes")
        g.create_dataset("target", data=np.zeros((n, views, res, res), np.uint8))
        g.create_dataset("has_mask", data=np.ones((n, views), bool))
        g.create_dataset("label_flag", data=np.eye(n, views, dtype=bool))  # one flagged view in each of 2 rows
        g.create_dataset("score", data=np.ones((n, views), np.float32))
        g.attrs["source_metadata"] = '{"crop_mode": "centred"}'
    return main, side


def _bare_dataset(main, side, geometric=False, raw=False):
    ds = object.__new__(SLEAPMultiViewDataset)
    ds.hdf5_path, ds.silhouette_path = str(main), str(side)
    ds.return_single_view, ds.aug_geometric_enabled = False, geometric
    ds.silhouette_exclude_label_flag, ds.silhouette_min_score = True, 0.0
    ds.has_raw_keypoints, ds.target_resolution = raw, 512
    return ds


def test_dataset_checks_the_silhouette_sidecar(tmp_path):
    main, side = _write_pair(tmp_path)
    ds = _bare_dataset(main, side)
    ds._init_silhouettes()
    assert ds.silhouette_resolution == 8 and ds.has_silhouettes
    assert ds.silhouette_view_counts == (6, 8, 8)  # used, with a mask, all views
    with pytest.raises(ValueError, match="geometric"):
        _bare_dataset(main, side, geometric=True)._init_silhouettes()
    with h5py.File(main, "a") as f:
        f["metadata"].attrs["crop_mode"] = "default"
    with pytest.raises(ValueError, match="crop_mode"):
        _bare_dataset(main, side)._init_silhouettes()
    with h5py.File(main, "a") as f:
        f["metadata"].attrs["crop_mode"] = "centred"
        f["auxiliary/video_frame_idx"][0, 0] = 99  # same rows, other images (e.g. a rebuilt frame map)
    with pytest.raises(ValueError, match="video_frame_idx"):
        _bare_dataset(main, side)._init_silhouettes()
    with h5py.File(main, "a") as f:
        f["auxiliary/video_frame_idx"][0, 0] = 0  # restore: only frame_idx differs below
    with h5py.File(side, "a") as s:
        s["auxiliary/frame_idx"][0] = 99
    with pytest.raises(ValueError, match="in auxiliary/frame_idx"):
        _bare_dataset(main, side)._init_silhouettes()


def test_label_flag_is_recomputed_from_the_dataset_labels(tmp_path):
    """A recalibration moves the reprojected labels, so a sidecar's stored label_flag can be stale. With raw labels in
    the dataset, the flag is recomputed at load with the generator's definition; without, the stored one is used."""
    from smal_fitter.sleap_data.generate_silhouettes_sam2 import segment_view

    main, side = _write_pair(tmp_path)  # 4 rows x 2 views; stored flags on views (0, 0) and (1, 1)
    rep = np.full((4, 2, 4, 2), 0.5, np.float32)  # rows, views, joints, normalised [y, x]
    dx = np.zeros((4, 2, 4), np.float32)  # raw minus reprojected, x px at 512
    dx[0, 1] = 40  # flagged
    dx[1, 0] = [10, 20, 30, 60]  # median exactly 25 px: not above label_flag_px
    dx[2, 0] = [40, 40, 40, 2]  # the three far joints are unusable below; the one left is 2 px off
    dx[3, 1] = 40  # no joint visible in both: no flag
    raw, vis, vis_raw = rep.copy(), np.ones((4, 2, 4), np.float32), np.ones((4, 2, 4), np.float32)
    raw[..., 1] += dx / 512
    raw[2, 0, 0], raw[2, 0, 1, 0], vis_raw[2, 0, 2], vis_raw[3, 1] = np.nan, 1.5, 0, 0  # NaN, outside, not visible
    with h5py.File(main, "a") as f:
        for name, data in (("", rep), ("_raw", raw)):
            f[f"multiview_keypoints/keypoints_2d{name}"] = data
        f["multiview_keypoints/keypoint_visibility"], f["multiview_keypoints/keypoint_visibility_raw"] = vis, vis_raw
    with h5py.File(side, "a") as s:
        s["multiview_silhouettes"].attrs["prompts"] = json.dumps({"label_flag_px": 25.0})
    P, img = {"label_flag_px": 25.0, "box_exclude": []}, np.zeros((512, 512, 3), np.uint8)  # no box joints: no SAM
    generator = np.array(
        [
            segment_view(None, img, rep[r, v], vis[r, v], raw[r, v], vis_raw[r, v], "c", P, {}, None)[1]["label_flag"]
            for r in range(4)
            for v in range(2)
        ]
    ).reshape(4, 2)
    assert np.argwhere(generator).tolist() == [[0, 1]]  # view (0, 1) only

    ds = _bare_dataset(main, side, raw=True)
    ds._init_silhouettes()
    with h5py.File(side, "r") as s:
        g = s["multiview_silhouettes"]
        np.testing.assert_array_equal(ds._usable_silhouettes(g, ()), ~generator)  # has_mask is all True
        np.testing.assert_array_equal(ds._usable_silhouettes(g, 0), [True, False])  # per sample, as __getitem__
    assert "recomputed" in ds.silhouette_label_flag_source and "3 differ" in ds.silhouette_label_flag_source

    ds = _bare_dataset(main, side)  # no raw labels: the stored flags
    ds._init_silhouettes()
    assert ds.silhouette_view_counts == (6, 8, 8) and "stored" in ds.silhouette_label_flag_source


class _BoxBoundSAM:
    """Segments a fixed object, but only inside the prompt box (as SAM mostly does)."""

    def __init__(self, obj):
        self.obj = obj

    def set_image(self, image):
        pass

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False):
        x0, y0, x1, y1 = (int(round(float(c))) for c in box)
        mask = np.zeros_like(self.obj)
        mask[y0 : y1 + 1, x0 : x1 + 1] = self.obj[y0 : y1 + 1, x0 : x1 + 1]
        return mask[None].astype(np.float32), np.array([0.9]), None


@pytest.mark.parametrize("half", [(100, 60), (40, 18)])  # object larger / smaller than the label box
def test_generator_grows_the_box_until_the_mask_is_not_clipped(half):
    from smal_fitter.sleap_data.generate_silhouettes_sam2 import segment_view

    obj = np.zeros((512, 512), bool)
    (hx, hy), c = half, 256
    obj[c - hy : c + hy, c - hx : c + hx] = True
    J = {name: i for i, name in enumerate(["a", "b", "c", "d", "t0", "t1"])}
    # normalised [y, x]: labels span a 100 x 60 px region around the centre (the last two, tail, are not visible)
    kp = np.array([[226, 206], [226, 306], [286, 206], [286, 306], [0, 0], [0, 0]], np.float32) / 512
    vis = np.array([1, 1, 1, 1, 0, 0], np.float32)
    P = {
        "label_flag_px": 25, "box_exclude": [], "box_pad": 0.12, "positive": ["a", "d"], "positive_midpoint": ["a", "d"],
        "tail": ["t0", "t1"], "tail_negative": ["a"], "tail_band_px": 14, "tail_band_fallback_px": 30,
        "tail_dilate_px": 4, "min_tail_score": 0.3, "max_tail_frac": 0.05, "min_body_px": 10,
    }  # fmt: skip
    target, qc = segment_view(
        _BoxBoundSAM(obj), np.zeros((512, 512, 3), np.uint8), kp, vis, kp * 0, vis * 0, "cam", P, J, None
    )
    assert ((target == 1) == obj).all()  # the whole object, nothing cut at the box
    assert qc["box_expansions"] == (2 if hx == 100 else 0)


def test_generator_checks_raw_labels_and_resolution_before_sam_loads(tmp_path):
    from smal_fitter.sleap_data.generate_silhouettes_sam2 import _check_prompts

    P = {k: [] for k in ("box_exclude", "positive", "positive_midpoint", "tail", "tail_negative")}
    P["joint_names"] = ["a"]
    with h5py.File(tmp_path / "ds.h5", "w") as f:
        f.create_dataset("multiview_keypoints/keypoints_2d", data=np.zeros((1, 1, 1, 2), np.float32))
        f.create_group("metadata").attrs.update({"canonical_camera_order": '["A"]', "target_resolution": 224})
        with pytest.raises(ValueError, match="keypoints_2d_raw"):  # e.g. a merged dataset
            _check_prompts(P, f, {}, 112)
        f.create_dataset("multiview_keypoints/keypoints_2d_raw", data=np.zeros((1, 1, 1, 2), np.float32))
        f.create_dataset("multiview_keypoints/keypoint_visibility_raw", data=np.zeros((1, 1, 1), np.float32))
        with pytest.raises(ValueError, match="must divide"):  # the default --resolution 256 on a 224 px dataset
            _check_prompts(P, f, {}, 256)
        _check_prompts(P, f, {}, 112)
