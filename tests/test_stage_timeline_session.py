"""Timeline staging: mapping a shared timeline with inserted frames back to each camera's video frames."""

import json

import cv2
import h5py
import numpy as np
import pytest

import config
from smal_fitter.sleap_data.preprocess_sleap_multiview_dataset import SLEAPMultiViewPreprocessor
from smal_fitter.sleap_data.stage_timeline_session import align_timeline_to_video, stage_session


def test_align_timeline_to_video_with_inserted_and_skipped_frames():
    """Inserted (interpolated) timeline frames map to -1, frames after skipped video frames take the new offset."""
    rng = np.random.default_rng(0)
    video_xy = rng.uniform(0, 1000, size=(40, 5, 2))
    video_xy[10] = np.nan  # no detection: resolved from its neighbours, which share the offset
    video_xy[20:23] = np.nan  # no detections around the skip: video 22's neighbours disagree, so it stays unresolved
    # Timeline: video 0-11, two interpolated frames, video 12-19, (video 20-21 skipped), video 22-39
    truth = np.array(list(range(12)) + [-1, -1] + list(range(12, 20)) + list(range(22, 40)))
    timeline_xy = video_xy[np.maximum(truth, 0)]
    timeline_xy[12] = (2 * video_xy[11] + video_xy[12]) / 3
    timeline_xy[13] = (video_xy[11] + 2 * video_xy[12]) / 3
    timeline_xy[5] = np.nan  # 2 coords only: its neighbours agree on offset 0, but video 5 holds other values
    timeline_xy[5, 0] = video_xy[5, 0] + 1

    fmap = align_timeline_to_video(timeline_xy, video_xy)

    expected = truth.copy()
    expected[truth == 22] = -1
    expected[5] = -1
    np.testing.assert_array_equal(fmap, expected)


def _write_session(root, xy, points3d, truth=None, video_dir="C:\\videos\\"):
    """Two-camera session: .slp (xy per video frame), analysis.h5, empty video, calibration, 3D. Timeline row t of
    a camera is its video frame truth[cam][t] (-1 = interpolated); without truth the timeline is the video frames."""
    nodes = [f"n{k}" for k in range(xy.shape[1])]
    n, k = xy.shape[:2]
    cal = ""
    for c, cam in enumerate(["cam_a", "cam_b"]):
        tmap = np.arange(n) if truth is None else truth[cam]
        timeline_xy = xy[np.maximum(tmap, 0)] + 0.5 * (tmap < 0)[:, None, None]  # +0.5: matches no video frame
        cam_dir = root / "s_pose_estimation" / cam
        cam_dir.mkdir(parents=True)
        with h5py.File(cam_dir / "s.predictions.slp", "w") as f:
            meta = {"nodes": [{"name": m} for m in nodes], "skeletons": [{"nodes": [{"id": i} for i in range(k)]}]}
            f.create_group("metadata").attrs["json"] = json.dumps(meta)
            frames = np.zeros(n, dtype=[("frame_idx", "<i8"), ("instance_id_start", "<u8"), ("instance_id_end", "<u8")])
            frames["frame_idx"] = frames["instance_id_start"] = np.arange(n)
            frames["instance_id_end"] = np.arange(1, n + 1)
            f["frames"] = frames
            inst = np.zeros(n, dtype=[("frame_id", "<i8"), ("point_id_start", "<u8"), ("point_id_end", "<u8")])
            inst["frame_id"] = np.arange(n)
            inst["point_id_start"], inst["point_id_end"] = np.arange(n) * k, np.arange(1, n + 1) * k
            f["instances"] = inst
            pts = np.zeros(n * k, dtype=[("x", "<f8"), ("y", "<f8"), ("visible", "?")])
            pts["x"], pts["y"], pts["visible"] = xy[..., 0].ravel(), xy[..., 1].ravel(), True
            f["pred_points"] = pts
            f["videos_json"] = [json.dumps({"backend": {"filename": f"{video_dir}{cam}.avi"}})]
        with h5py.File(cam_dir / "s.analysis.h5", "w") as f:
            f["tracks"] = timeline_xy.transpose(2, 1, 0)[None]
        (root / "s_cameras").mkdir(exist_ok=True)
        (root / "s_cameras" / f"{cam}.avi").touch()
        cal += f'[cam_{c}]\nname = "{cam}"\nsize = [640, 480]\nmatrix = [[500, 0, 320], [0, 500, 240], [0, 0, 1]]\n'
        cal += f"distortions = [0, 0, 0, 0, 0]\nrotation = [0, 0, 0]\ntranslation = [{c * 10}, 0, 0]\n"
    (root / "calibration.toml").write_text(cal)
    with h5py.File(root / "points3d.h5", "w") as f:
        f["points3d"] = points3d


def test_stage_session_appends_static_keypoints_after_the_skeleton_nodes(tmp_path):
    """Static points become nodes K.. of points3d/reprojections (NaN in unlisted cameras); a 3D whose node count
    differs from the .slp skeleton, or a static name that is a node name, is refused instead of mislabelling."""
    try:
        (tmp_path / "link").symlink_to(tmp_path)
    except OSError:
        pytest.skip("symlinks need Developer Mode or admin rights on Windows")
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 400, size=(12, 3, 2))
    points3d = np.concatenate([rng.uniform(-50, 50, (12, 1, 3, 2)), np.full((12, 1, 3, 1), 1000.0)], axis=-1)
    _write_session(tmp_path / "raw", xy, points3d)
    static = {"names": ["head"], "points": {"head": [0.0, 0.0, 1000.0]}, "visible_cameras": {"head": ["cam_a"]}}
    (tmp_path / "static.json").write_text(json.dumps(static))
    args = [str(tmp_path / "raw"), str(tmp_path / "raw" / "calibration.toml"), str(tmp_path / "raw" / "points3d.h5")]

    stage_session(*args, str(tmp_path / "staged"), str(tmp_path / "static.json"))

    with np.load(tmp_path / "staged" / "frame_map.npz") as z:
        np.testing.assert_array_equal(z["cam_a"], np.arange(12))
    with h5py.File(tmp_path / "staged" / "points3d.h5", "r") as f:
        np.testing.assert_array_equal(f["tracks"][:, 0, 3], np.tile([0.0, 0.0, 1000.0], (12, 1)))
    with h5py.File(tmp_path / "staged" / "reprojections.h5", "r") as f:
        np.testing.assert_allclose(f["cam_a"][:, 0, 3], np.tile([320.0, 240.0], (12, 1)))
        assert np.isnan(f["cam_b"][:, 0, 3]).all() and np.isfinite(f["cam_b"][:, 0, :3]).all()

    with h5py.File(tmp_path / "raw" / "points3d.h5", "w") as f:  # an extra node in the pipeline's 3D export
        f["points3d"] = np.concatenate([points3d, points3d[:, :, :1]], axis=2)
    with pytest.raises(ValueError, match="skeleton has 3 nodes"):
        stage_session(*args, str(tmp_path / "staged2"), str(tmp_path / "static.json"))
    with h5py.File(tmp_path / "raw" / "points3d.h5", "w") as f:
        f["points3d"] = points3d
    (tmp_path / "static.json").write_text(json.dumps({**static, "names": ["n1"], "points": {"n1": [0, 0, 1000]}}))
    with pytest.raises(ValueError, match="n1"):
        stage_session(*args, str(tmp_path / "staged3"), str(tmp_path / "static.json"))


def test_preprocessor_reads_each_view_at_its_mapped_video_frame(tmp_path):
    """With frame_map.npz a view's image and raw SLEAP labels come from its camera's mapped video frame, 3D and
    reprojections from the timeline frame; timeline frames without a video frame are never discovered, and
    auxiliary/video_frame_idx holds the map."""
    try:
        (tmp_path / "link").symlink_to(tmp_path)
    except OSError:
        pytest.skip("symlinks need Developer Mode or admin rights on Windows")
    rng = np.random.default_rng(1)
    xy = rng.uniform(100, 400, size=(8, 3, 2))
    # timeline 3 is interpolated; cam_a drops video frame 6, cam_b starts at video frame 1
    truth = {"cam_a": np.array([0, 1, 2, -1, 3, 4, 5, 7]), "cam_b": np.array([1, 2, 3, -1, 4, 5, 6, 7])}
    points3d = np.concatenate([rng.uniform(-50, 50, (8, 1, 3, 2)), np.full((8, 1, 3, 1), 1000.0)], axis=-1)
    raw, staged = tmp_path / "raw", tmp_path / "sessions" / "s"
    _write_session(raw, xy, points3d, truth, video_dir="/data/raw/")  # a path the preprocessor resolves on any OS
    for cam in truth:  # video frame f is uniformly gray 20 + 25 f
        video = cv2.VideoWriter(str(raw / "s_cameras" / f"{cam}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
        if not video.isOpened():
            pytest.skip("cv2 cannot write MJPG video")
        for frame in range(len(xy)):
            video.write(np.full((480, 640, 3), 20 + 25 * frame, np.uint8))
        video.release()
    stage_session(str(raw), str(raw / "calibration.toml"), str(raw / "points3d.h5"), str(staged))
    joints = config.dd["J_names"][:3]  # nodes n0-n2 -> the model's first three joints
    (tmp_path / "lookup.csv").write_text("model,data\n" + "".join(f"{j},n{i}\n" for i, j in enumerate(joints)))
    pre = SLEAPMultiViewPreprocessor(
        str(tmp_path / "lookup.csv"), target_resolution=32, crop_mode="centred", use_reprojections=True
    )

    stats = pre.process_dataset(str(tmp_path / "sessions"), str(tmp_path / "out.h5"), verbose=False)

    def in_crop(px):  # pixels -> stored [y, x] in [0, 1] of the centred 480 x 480 crop
        return np.stack([px[:, 1], px[:, 0] - 80], axis=-1) / 480

    assert stats["total_views_attempted"] == stats["total_views_included"] == 14  # no view tried at timeline 3
    with h5py.File(tmp_path / "out.h5", "r") as f, h5py.File(staged / "reprojections.h5", "r") as reproj:
        kp = f["multiview_keypoints"]
        np.testing.assert_array_equal(f["auxiliary/frame_idx"][:], [0, 1, 2, 4, 5, 6, 7])
        for s, t in enumerate(f["auxiliary/frame_idx"][:]):
            np.testing.assert_allclose(kp["keypoints_3d"][s, :3], points3d[t, 0], rtol=1e-6)
            for v, cam in enumerate(f["auxiliary/camera_names"][s].decode().split(",")):
                video_f = truth[cam][t]
                assert f["auxiliary/video_frame_idx"][s, v] == video_f
                image = cv2.imdecode(f[f"multiview_images/image_jpeg_view_{v}"][s], cv2.IMREAD_GRAYSCALE)
                assert abs(image.mean() - (20 + 25 * video_f)) < 5, (t, cam)
                np.testing.assert_allclose(kp["keypoints_2d_raw"][s, v, :3], in_crop(xy[video_f]), atol=1e-5)
                np.testing.assert_allclose(kp["keypoints_2d"][s, v, :3], in_crop(reproj[cam][t, 0]), atol=1e-5)
