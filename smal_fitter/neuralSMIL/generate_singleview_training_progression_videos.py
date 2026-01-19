#!/usr/bin/env python3
"""
Generate per-sample, per-view training progression videos from single-view renders.

Scans a visualization root directory with per-epoch subfolders like:
  multiview_singleview_renders/epoch_XXX/sample_000_view_00_epoch_XXX.png

For each (sample, view) pair found across epochs, compiles a video in the root dir.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


EPOCH_DIR_RE = re.compile(r"^epoch_(\d+)$")
SAMPLE_VIEW_FILE_RE = re.compile(r"^sample_(\d+)_view_(\d+)_epoch_(\d+)\.png$")
SAMPLE_3D_FILE_RE = re.compile(r"^sample_(\d+)_epoch_(\d+)_3d_keypoints\.png$")


def _collect_view_samples(vis_root: Path) -> Dict[Tuple[int, int], List[Tuple[int, Path]]]:
    sample_frames: Dict[Tuple[int, int], List[Tuple[int, Path]]] = {}
    for epoch_dir in sorted(vis_root.iterdir()):
        if not epoch_dir.is_dir():
            continue
        match = EPOCH_DIR_RE.match(epoch_dir.name)
        if not match:
            continue
        for image_path in epoch_dir.iterdir():
            if not image_path.is_file():
                continue
            match = SAMPLE_VIEW_FILE_RE.match(image_path.name)
            if not match:
                continue
            sample_idx = int(match.group(1))
            view_idx = int(match.group(2))
            epoch_idx = int(match.group(3))
            key = (sample_idx, view_idx)
            sample_frames.setdefault(key, []).append((epoch_idx, image_path))
    return sample_frames


def _collect_3d_samples(vis_root: Path) -> Dict[int, List[Tuple[int, Path]]]:
    sample_frames: Dict[int, List[Tuple[int, Path]]] = {}
    for epoch_dir in sorted(vis_root.iterdir()):
        if not epoch_dir.is_dir():
            continue
        match = EPOCH_DIR_RE.match(epoch_dir.name)
        if not match:
            continue
        for image_path in epoch_dir.iterdir():
            if not image_path.is_file():
                continue
            match = SAMPLE_3D_FILE_RE.match(image_path.name)
            if not match:
                continue
            sample_idx = int(match.group(1))
            epoch_idx = int(match.group(2))
            sample_frames.setdefault(sample_idx, []).append((epoch_idx, image_path))
    return sample_frames


def _write_video(output_path: Path, frames: List[Path], fps: int) -> None:
    if not frames:
        return
    first = cv2.imread(str(frames[0]))
    if first is None:
        return
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate per-sample training progression videos from single-view renders."
    )
    parser.add_argument(
        "--vis-root",
        type=str,
        default="/home/fabi/dev/SMILify/multiview_singleview_renders",
        help="Root directory with epoch subfolders (default: /home/fabi/dev/SMILify/multiview_singleview_renders)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Output video FPS (default: 10)",
    )
    args = parser.parse_args()

    vis_root = Path(args.vis_root)
    if not vis_root.exists():
        print(f"Visualization root not found: {vis_root}")
        return 1

    view_frames = _collect_view_samples(vis_root)
    if not view_frames:
        print(f"No view samples found in {vis_root}")
        return 1

    for (sample_idx, view_idx), frames in sorted(view_frames.items()):
        frames_sorted = [p for _, p in sorted(frames, key=lambda x: x[0])]
        output_path = vis_root / f"sample_{sample_idx:03d}_view_{view_idx:02d}_training_progression.mp4"
        _write_video(output_path, frames_sorted, args.fps)
        print(f"Wrote {output_path}")

    keypoint_frames = _collect_3d_samples(vis_root)
    for sample_idx, frames in sorted(keypoint_frames.items()):
        frames_sorted = [p for _, p in sorted(frames, key=lambda x: x[0])]
        output_path = vis_root / f"sample_{sample_idx:03d}_3d_keypoints_training_progression.mp4"
        _write_video(output_path, frames_sorted, args.fps)
        print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
