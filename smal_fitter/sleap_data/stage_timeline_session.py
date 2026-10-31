#!/usr/bin/env python3
"""
Stage a SLEAP/anipose session whose 3D lives on a common timeline that differs from the video frame indices.

Some pipelines (e.g. one for a head-fixed rig) export per-camera ``*.analysis.h5`` and a combined
``points3d`` on a shared timeline that has interpolated frames inserted, while each camera's video and
``*.predictions.slp`` keep their own frame numbering. This script stages such a session into the camera_dirs
layout the multiview preprocessor reads, with symlinks to the original files plus generated:

- ``frame_map.npz``      per camera: timeline index -> video frame index (-1 = no real video frame). Built by
                         matching each timeline row of analysis.h5 to the .slp prediction with exactly the same
                         values. Read by ``preprocess_sleap_multiview_dataset.py`` when present.
- ``points3d.h5``        anipose-style ``tracks`` (T, 1, K, 3); optional static keypoints appended as nodes.
- ``reprojections.h5``   ideal-pinhole reprojections of points3d (as ``generate_reprojections``); a static keypoint
                         is NaN in cameras not listed in its ``visible_cameras``.
- ``extra_keypoints.json``  names of the appended static keypoints, read by ``SLEAPDataLoader`` when present.

The camera folders hold only the ``.predictions.slp`` and the video: an ``analysis.h5`` without ``node_names``
would make the loader fall back to generic keypoint names.

Usage:
    python -m smal_fitter.sleap_data.stage_timeline_session \\
        --session_dir /data/rig/session_01 \\
        --calibration /data/rig/session_01/calibration.toml \\
        --points3d /data/rig/session_01/points3d_timeline.h5 \\
        --static_keypoints /data/rig/static_keypoints.json \\
        --out_dir /data/rig/staged/session_01
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np

from smal_fitter.sleap_data.triangulate_3d_points import generate_reprojections_h5, load_calibration

_NAN_SENTINEL = -1.0e9


def _row_keys(xy: np.ndarray) -> np.ndarray:
    """(N, K, 2) float -> (N,) void keys for exact row comparison (NaN == NaN)."""
    a = np.where(np.isnan(xy), _NAN_SENTINEL, xy).astype(np.float64).reshape(len(xy), -1)
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def align_timeline_to_video(
    timeline_xy: np.ndarray, video_xy: np.ndarray, min_finite: int = 6, max_offset: int = 1000
) -> np.ndarray:
    """Map each timeline frame to the video frame with identical 2D predictions.

    Args:
        timeline_xy: (T, K, 2) per-timeline-frame keypoints (NaN = missing), e.g. analysis.h5 tracks.
        video_xy:    (N, K, 2) per-video-frame keypoints from the .slp predictions (NaN = missing).
        min_finite:  rows with fewer finite coordinates are ambiguous and only resolved by offset continuity.
        max_offset:  largest |t - f| accepted for a match.

    Returns:
        (T,) int64 video frame index per timeline frame, -1 where the timeline frame has no video frame
        (interpolated) or could not be resolved unambiguously. Strictly increasing over the matched frames.
    """
    T = len(timeline_xy)
    fmap = np.full(T, -1, dtype=np.int64)
    vkeys = _row_keys(video_xy)
    index: Dict[bytes, list] = {}
    for f, k in enumerate(vkeys):
        index.setdefault(k.tobytes(), []).append(f)
    tkeys = _row_keys(timeline_xy)
    n_finite = np.isfinite(timeline_xy).reshape(T, -1).sum(1)

    # pass 1: informative rows, unique exact match nearest the running offset
    offset = 0
    last_f = -1
    for t in range(T):
        if n_finite[t] < min_finite:
            continue
        cands = [f for f in index.get(tkeys[t].tobytes(), ()) if abs(t - f) <= max_offset and f > last_f]
        if not cands:
            continue
        f = min(cands, key=lambda c: abs((t - c) - offset))
        fmap[t] = f
        offset, last_f = t - f, f

    # pass 2: ambiguous rows (few/no detections) inherit the offset only if both neighbours agree and the rows match
    matched = np.flatnonzero(fmap >= 0)
    if len(matched):
        for t in np.flatnonzero((fmap < 0) & (n_finite < min_finite)):
            j = np.searchsorted(matched, t)
            if j == 0 or j == len(matched):
                continue
            t0, t1 = matched[j - 1], matched[j]
            off0, off1 = t0 - fmap[t0], t1 - fmap[t1]
            f = t - off0
            if off0 == off1 and 0 <= f < len(video_xy) and vkeys[f].tobytes() == tkeys[t].tobytes():
                fmap[t] = f
    return fmap


def _load_slp_video_xy(slp_path: Path) -> Tuple[np.ndarray, str]:
    """Per-video-frame predicted keypoints (N, K, 2) (NaN = invisible / no instance) and the video basename."""
    with h5py.File(slp_path, "r") as f:
        md = json.loads(f["metadata"].attrs["json"])
        n_nodes = len(md["skeletons"][0]["nodes"])
        frames, instances, pts = f["frames"][:], f["instances"][:], f["pred_points"][:]
        videos = json.loads(f["videos_json"][0])
    n = int(frames["frame_idx"].max()) + 1
    xy = np.full((n, n_nodes, 2), np.nan)
    has = frames["instance_id_end"] > frames["instance_id_start"]
    first = instances[frames["instance_id_start"][has].astype(np.int64)]  # first instance per frame
    starts = first["point_id_start"].astype(np.int64)
    if np.all(first["point_id_end"].astype(np.int64) - starts == n_nodes):  # fast path: full skeleton per instance
        p = pts[starts[:, None] + np.arange(n_nodes)]  # (F, n_nodes)
        v = np.stack([p["x"], p["y"]], -1).astype(np.float64)
        v[~p["visible"].astype(bool)] = np.nan
        xy[frames["frame_idx"][has].astype(np.int64)] = v
    else:
        for fi, inst in zip(frames["frame_idx"][has], first):
            p = pts[int(inst["point_id_start"]) : int(inst["point_id_end"])]
            v = np.stack([p["x"], p["y"]], 1).astype(np.float64)
            v[~p["visible"].astype(bool)] = np.nan
            xy[int(fi), : len(v)] = v
    video = videos["backend"]["filename"] if isinstance(videos, dict) else videos[0]["backend"]["filename"]
    return xy, Path(video.replace("\\", "/")).name


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True).strip()
    except Exception:
        return "unknown"


def _symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def stage_session(
    session_dir: str,
    calibration: str,
    points3d: str,
    out_dir: str,
    static_keypoints: Optional[str] = None,
    points3d_dataset: str = "points3d",
) -> Dict:
    session_dir, out = Path(session_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pose_dir = next(session_dir.glob("*_pose_estimation"))
    cams_dir = next(session_dir.glob("*_cameras"))
    cameras = load_calibration(calibration)
    _symlink(Path(calibration), out / "calibration.toml")

    with h5py.File(points3d, "r") as f:
        tracks = f[points3d_dataset][:]  # (T, 1, K, 3)
    T = tracks.shape[0]
    report = {"timeline_frames": T, "cameras": {}}

    frame_maps = {}
    for cam_dir in sorted(p for p in pose_dir.iterdir() if p.is_dir() and p.name in cameras):
        cam = cam_dir.name
        slp = next(cam_dir.glob("*.predictions.slp"))
        video_xy, video_name = _load_slp_video_xy(slp)
        with h5py.File(next(cam_dir.glob("*.analysis.h5")), "r") as f:
            timeline_xy = f["tracks"][0].transpose(2, 1, 0)  # (T, K, 2)
        assert len(timeline_xy) == T, f"{cam}: analysis.h5 has {len(timeline_xy)} frames, points3d {T}"
        fmap = align_timeline_to_video(timeline_xy, video_xy)
        m = fmap >= 0
        assert np.all(np.diff(fmap[m]) > 0), f"{cam}: frame map not strictly increasing"
        frame_maps[cam] = fmap.astype(np.int32)
        video = cams_dir / video_name
        assert video.exists(), f"{cam}: video {video} not found"
        (out / cam).mkdir(exist_ok=True)
        _symlink(slp, out / cam / slp.name)
        _symlink(video, out / cam / video_name)
        off = np.arange(T)[m] - fmap[m]
        report["cameras"][cam] = {
            "video": video_name,
            "video_frames": len(video_xy),
            "timeline_frames_mapped": int(m.sum()),
            "timeline_frames_unmapped": int((~m).sum()),
            "video_frames_unused": int(len(video_xy) - m.sum()),
            "offset_t_minus_f_range": [int(off.min()), int(off.max())],
        }
        print(f"{cam}: {report['cameras'][cam]}", flush=True)
    np.savez(out / "frame_map.npz", **frame_maps)

    names = []
    visible = {}
    if static_keypoints:
        sk = json.load(open(static_keypoints))
        names = list(sk["names"])
        static = np.array([sk["points"][n] for n in names], dtype=tracks.dtype)  # (S, 3)
        tracks = np.concatenate([tracks, np.broadcast_to(static, (T, tracks.shape[1], len(names), 3))], axis=2)
        visible = sk["visible_cameras"]
        json.dump(
            {"names": names, "source": str(Path(static_keypoints).resolve())},
            open(out / "extra_keypoints.json", "w"),
            indent=1,
        )
    with h5py.File(out / "points3d.h5", "w") as f:
        f.create_dataset("tracks", data=tracks)
        f["tracks"].attrs["Description"] = (
            "Shape: (n_frames, n_tracks, n_nodes, 3). Timeline frames (see frame_map.npz)."
        )
        f.attrs["extra_keypoint_names"] = json.dumps(names)

    generate_reprojections_h5(tracks, cameras, str(out / "reprojections.h5"), verbose=False)
    n_base = tracks.shape[2] - len(names)
    with h5py.File(out / "reprojections.h5", "r+") as f:
        for i, n in enumerate(names):
            for cam in f.keys():
                if cam not in visible.get(n, []):
                    f[cam][:, :, n_base + i, :] = np.nan

    report.update(
        static_keypoints=names,
        static_visible_cameras=visible,
        inputs={
            "session_dir": str(session_dir),
            "calibration": calibration,
            "points3d": points3d,
            "static_keypoints": static_keypoints,
        },
        git_commit=_git_commit(),
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    json.dump(report, open(out / "staging_provenance.json", "w"), indent=1)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session_dir", required=True, help="session folder with *_cameras and *_pose_estimation")
    p.add_argument("--calibration", required=True, help="anipose calibration toml")
    p.add_argument("--points3d", required=True, help="HDF5 with the timeline 3D (T, 1, K, 3)")
    p.add_argument("--points3d_dataset", default="points3d", help="dataset name in --points3d (default: points3d)")
    p.add_argument("--static_keypoints", default=None, help="optional JSON: names, points (world xyz), visible_cameras")
    p.add_argument("--out_dir", required=True, help="staged session folder (its name is the session name)")
    a = p.parse_args()
    stage_session(a.session_dir, a.calibration, a.points3d, a.out_dir, a.static_keypoints, a.points3d_dataset)


if __name__ == "__main__":
    main()
