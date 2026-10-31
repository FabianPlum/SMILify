#!/usr/bin/env python3
"""
Generate per-view mouse silhouette targets for a multi-view SLEAP HDF5 with SAM 2.1, as an aligned sidecar HDF5.

Runs in an environment with ``sam2`` (torch >= 2.5.1). Training never imports this file, and the training HDF5 is
only read. The stored views (already undistorted and cropped) are segmented, so the targets align with the training
inputs by construction.

Per view, prompts come from the stored 2D labels: the raw SLEAP detection of that view where available (the image
evidence), else the reprojected label. A padded box around the body/limb/head keypoints, positive points on chosen
body joints, negative points on the static rig hardware. Optional fixed positive points per camera cover body parts
no keypoint reaches (e.g. the back above a head bar). The tail is excluded from the target: SAM segments it
separately (positive points on the tail keypoints, negatives on the body) and the dilated tail mask plus a band along
the tail keypoints is marked "ignore", so no real tail pixel is labelled background. Static per-camera occluder masks
(head-fixation hardware) are also "ignore".

SAM rarely segments beyond its box, and the labels stop short of the outline (spine joints lie inside the body), so a
box side the mask runs along is grown and SAM predicts again (the box_* keys below).

Prompt file (--prompts, JSON). Joints are names from joint_names, points are [x, y] in pixels of the stored views,
and cameras are names from the dataset's canonical_camera_order. Required keys:
    joint_names            list[str]        the dataset's joints, in keypoint order
    box_exclude            list[str]        joints left out of the body box (e.g. the tail)
    box_pad                float            box margin per side, as a fraction of the keypoint box size, plus 8 px
    positive               list[str]        joints that become positive points
    positive_midpoint      [str, str]       one more positive point, midway between these two joints
    tail                   list[str]        tail joints, base to tip (the ignore band follows them)
    tail_negative          list[str]        joints that become negative points of the tail prompt
    tail_band_px           int              width of the ignore band along the tail when the tail mask passes QC,
                                            and the radius around the tail base that stays supervised
    tail_band_fallback_px  int              band width when the tail mask fails QC
    tail_dilate_px         int              dilation radius of a passing tail mask before it is marked ignore
    min_tail_score         float            tail QC: lowest SAM score
    max_tail_frac          float            tail QC: largest fraction of the image
    label_flag_px          float            label_flag is set when label_disagree_px is larger
Optional keys (default):
    static_positive        {cam: [[x, y]]}  fixed positive points per camera; the box's left, top and right sides
                                            are widened to include them
    static_box_margin      [int, int]       x and y margin of the box around those points ([60, 40])
    hardware_points        {cam: [[x, y]]}  negative points on static rig hardware, for the body and the tail
    ignore_masks           str              .npz beside the prompt file: one bool mask per camera, at the dataset's
                                            target_resolution, marked ignore (static occluders)
    min_box_keypoints      int              fewer usable box keypoints give no target (3)
    min_body_px            int              smaller body masks give no target (500)
    min_tail_px            int              smaller tail masks fail the tail QC (50)
    box_clip_frac          float            fraction of a box side the mask must run along to grow it (0.1)
    box_grow               float            growth of such a side, as a fraction of the box size (0.2) ...
    box_grow_min_px        int              ... and at least this many px (24)
    box_max_expansions     int              most growth steps (2)
    box_border_px          int              sides this close to the image border are not grown (8)

Sidecar layout (N samples, V view slots in the order of multiview_images, R = --resolution):
    multiview_silhouettes/target           (N, V, R, R) uint8   0 = background, 1 = mouse, 255 = ignore
    multiview_silhouettes/has_mask         (N, V)       bool    False: no target for this slot (all 255)
    multiview_silhouettes/score            (N, V)       float32 SAM score of the body mask
    multiview_silhouettes/kp_inside        (N, V)       float32 fraction of box keypoints inside the mask
    multiview_silhouettes/label_disagree_px(N, V)       float32 median raw-vs-reprojected label distance (crop px)
    multiview_silhouettes/label_flag       (N, V)       bool    label_disagree_px > label_flag_px
    multiview_silhouettes/tail_used        (N, V)       bool    SAM tail mask passed QC and widened the ignore
    multiview_silhouettes/box              (N, V, 4)    float32 final SAM box (x0, y0, x1, y1, crop px)
    multiview_silhouettes/box_expansions   (N, V)       float32 times the box was grown because the mask ran along it
    multiview_silhouettes/processed        (N,)         bool    row was segmented (False only in a partial merge)
plus copies of the alignment datasets of add_raw_keypoints.py (session, frame, cameras, view mask) and of the
datasets that identify the segmented images (video frame per view, camera intrinsics), and the source crop settings
in the ``source_metadata`` attribute; the training loader compares all of them with the dataset.

Usage (one shard per GPU, then merge):
    CUDA_VISIBLE_DEVICES=0 python smal_fitter/sleap_data/generate_silhouettes_sam2.py run DATA.h5 \\
        --prompts PROMPTS.json --sam_ckpt sam2.1_hiera_large.pt --shard 0/2 --out DATA_silhouettes.shard0.h5
    CUDA_VISIBLE_DEVICES=1 python ... --shard 1/2 --out DATA_silhouettes.shard1.h5
    python smal_fitter/sleap_data/generate_silhouettes_sam2.py merge DATA.h5 DATA_silhouettes.shard*.h5 \\
        --out DATA_silhouettes_256.h5
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

# Same rows-alignment datasets as smal_fitter/sleap_data/add_raw_keypoints.py (kept in sync by hand: this tool runs
# in a different environment and must not import the training package).
ALIGNMENT = ("auxiliary/session_name", "auxiliary/frame_idx", "auxiliary/camera_names", "multiview_images/view_mask")
# Which images were segmented (frame_idx alone is a timeline index when a frame map was used) and how they were cropped;
# compared by SLEAPMultiViewDataset when present in both files.
IMAGE_IDENTITY = ("auxiliary/video_frame_idx", "multiview_keypoints/camera_intrinsics")
SOURCE_ATTRS = ("crop_mode", "target_resolution", "undistort_images")
# Shard attributes that must agree before shards are merged.
SAME_ACROSS_SHARDS = (
    "value_convention",
    "resolution",
    "source_h5",
    "sam_config",
    "sam_ckpt_sha256",
    "prompts",
    "source_metadata",
    "generator_sha256",
)
GROUP = "multiview_silhouettes"
QC_FLOAT = {"score": (), "kp_inside": (), "label_disagree_px": (), "box_expansions": (), "box": (4,)}  # name: shape
QC_BOOL = ("has_mask", "label_flag", "tail_used")
MOUSE, BACKGROUND, IGNORE = 1, 0, 255


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    """HEAD, with "-dirty" if tracked files differ in content (as add_raw_keypoints.py). An untracked or edited copy
    of this script is identified by generator_sha256 instead."""
    try:
        repo = os.path.dirname(os.path.abspath(__file__))
        head = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
        clean = subprocess.run(["git", "-C", repo, "diff", "--quiet", "--ignore-cr-at-eol", "HEAD"]).returncode
        return head if clean == 0 else f"{head}-dirty"
    except Exception:
        return "unknown"


def _copy_source_datasets(f: h5py.File, o: h5py.File) -> None:
    """Row-alignment and image-identity datasets of the source HDF5, so the loader can check the pairing."""
    for name in ALIGNMENT + IMAGE_IDENTITY:
        if name in f:
            o.create_dataset(name, data=f[name][()], dtype=f[name].dtype)  # keeps vlen-string types


def _source_metadata(f: h5py.File) -> str:
    md = f["metadata"].attrs
    return json.dumps({k: (md[k].item() if hasattr(md[k], "item") else md[k]) for k in SOURCE_ATTRS if k in md})


def _check_prompts(P: dict, f: h5py.File, ignore_masks: dict) -> None:
    """Fail before SAM loads when the prompt file does not fit the dataset."""
    n_joints = f["multiview_keypoints/keypoints_2d"].shape[2]
    if len(P["joint_names"]) != n_joints:
        raise ValueError(f"prompts list {len(P['joint_names'])} joint names, the dataset has {n_joints} joints")
    named = list(P["box_exclude"]) + list(P["positive"]) + list(P["positive_midpoint"]) + list(P["tail"])
    unknown = sorted(set(named + list(P["tail_negative"])) - set(P["joint_names"]))
    if unknown:
        raise ValueError(f"prompts name joints that are not in joint_names: {unknown}")
    cams = set(json.loads(f["metadata"].attrs["canonical_camera_order"]))
    res = int(f["metadata"].attrs["target_resolution"])
    for key in ("static_positive", "hardware_points"):
        if set(P.get(key, {})) - cams:
            raise ValueError(f"prompts {key} has unknown cameras {sorted(set(P[key]) - cams)}")
    for cam, mask in ignore_masks.items():
        if cam not in cams or mask.shape != (res, res):
            raise ValueError(f"ignore mask {cam} {mask.shape}: expected one of {sorted(cams)} at ({res}, {res})")


def _stretch(gray: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(gray, 1), np.percentile(gray, 99.5)
    return np.clip((gray.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)


def _box_contact(mask: np.ndarray, box, border_px: float) -> np.ndarray:
    """Per box side (left, top, right, bottom): the fraction of its length along which the mask reaches it (within
    2 px). NaN for a side within border_px of the image border, where the object may simply leave the image."""
    H, W = mask.shape
    x0, y0, x1, y1 = (int(round(float(c))) for c in box)
    out = np.full(4, np.nan)
    for k, c, at_border in (
        (0, x0, x0 <= border_px),
        (1, y0, y0 <= border_px),
        (2, x1, x1 >= W - 1 - border_px),
        (3, y1, y1 >= H - 1 - border_px),
    ):
        if not at_border:
            band = (
                mask[y0 : y1 + 1, max(c - 2, 0) : c + 3].any(1)
                if k in (0, 2)
                else mask[max(c - 2, 0) : c + 3, x0 : x1 + 1].any(0)
            )
            out[k] = band.mean() if band.size else np.nan
    return out


def _downsample(target: np.ndarray, res: int) -> np.ndarray:
    """512 -> res: a block is ignore if any pixel is ignore, else mouse if at least half is mouse."""
    k = target.shape[0] // res
    assert target.shape == (res * k, res * k), (target.shape, res)
    blocks = target.reshape(res, k, res, k)
    ign = (blocks == IGNORE).any(axis=(1, 3))
    mouse = (blocks == MOUSE).mean(axis=(1, 3)) >= 0.5
    return np.where(ign, IGNORE, mouse.astype(np.uint8)).astype(np.uint8)


def segment_view(predictor, img_rgb, kp_rep, vis_rep, kp_raw, vis_raw, cam, P, J, ignore_static):
    """One view: returns (target_512, qc dict) or (None, qc) when there are too few keypoints."""
    H, W = img_rgb.shape[:2]
    inside = lambda k: np.all((k >= 0) & (k <= 1), -1) & np.isfinite(k).all(-1)  # noqa: E731
    ok_rep, ok_raw = (vis_rep > 0) & inside(kp_rep), (vis_raw > 0) & inside(kp_raw)
    both = ok_rep & ok_raw
    disagree = (
        float(np.median(np.linalg.norm((kp_rep[both] - kp_raw[both]) * [H, W], axis=-1))) if both.any() else np.nan
    )
    finite = lambda k: np.isfinite(k).all(-1)  # noqa: E731
    use_raw = (vis_raw > 0) & finite(kp_raw)
    kp = np.where(use_raw[:, None], kp_raw, kp_rep)  # raw detection where available, else the reprojection
    visible = use_raw | ((vis_rep > 0) & finite(kp_rep))  # anywhere, also outside the crop
    ok = visible & inside(kp)  # visible and inside the crop: usable as a prompt
    px = np.stack([kp[:, 1] * W, kp[:, 0] * H], -1)  # stored labels are normalised [y, x]
    qc = {"label_disagree_px": disagree, "label_flag": bool(np.isfinite(disagree) and disagree > P["label_flag_px"])}

    box_idx = [i for n, i in J.items() if n not in set(P["box_exclude"]) and ok[i]]
    if len(box_idx) < P.get("min_box_keypoints", 3):
        return None, qc
    (x0, y0), (x1, y1) = px[box_idx].min(0), px[box_idx].max(0)
    pw, ph = (x1 - x0) * P["box_pad"] + 8, (y1 - y0) * P["box_pad"] + 8
    box = np.array([max(0, x0 - pw), max(0, y0 - ph), min(W - 1, x1 + pw), min(H - 1, y1 + ph)], np.float32)
    pos = [px[J[n]] for n in P["positive"] if ok[J[n]]]
    a, b = (J[n] for n in P["positive_midpoint"])
    if ok[a] and ok[b]:
        pos.append((px[a] + px[b]) / 2)
    static = np.array(P.get("static_positive", {}).get(cam, []), np.float32).reshape(-1, 2)
    if len(static):
        pos += list(static)
        mx, my = P.get("static_box_margin", [60, 40])
        box = np.array(
            [
                max(0, min(box[0], static[:, 0].min() - mx)),
                max(0, min(box[1], static[:, 1].min() - my)),
                min(W - 1, max(box[2], static[:, 0].max() + mx)),
                box[3],
            ],
            np.float32,
        )
    neg_hw = np.array(P.get("hardware_points", {}).get(cam, []), np.float32).reshape(-1, 2)
    pos = np.array(pos, np.float32).reshape(-1, 2)
    pts = np.concatenate([pos, neg_hw], 0)
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg_hw))].astype(np.int32)

    predictor.set_image(img_rgb)
    # SAM rarely segments beyond its box, and the labels stop short of the outline (the spine joints lie inside the
    # body, so a box around them cuts through the back): grow every side the mask runs along and predict again.
    expansions = 0
    while True:
        m, sc, _ = predictor.predict(
            point_coords=pts if len(pts) else None,
            point_labels=lab if len(pts) else None,
            box=box,
            multimask_output=False,
        )
        body = m[0] > 0
        contact = np.nan_to_num(_box_contact(body, box, P.get("box_border_px", 8)))
        clipped = contact >= P.get("box_clip_frac", 0.1)
        if not clipped.any() or expansions >= P.get("box_max_expansions", 2):
            break
        size = np.array([box[2] - box[0], box[3] - box[1]] * 2)
        grow = np.maximum(P.get("box_grow_min_px", 24), P.get("box_grow", 0.2) * size)
        box = (box + grow * clipped * np.array([-1, -1, 1, 1])).clip(0, [W - 1, H - 1, W - 1, H - 1]).astype(np.float32)
        expansions += 1
    qc["box"], qc["box_expansions"] = box.copy(), float(expansions)
    dil = cv2.dilate(body.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    qc["score"] = float(sc[0])
    qc["kp_inside"] = float(
        np.mean([dil[int(np.clip(px[i, 1], 0, H - 1)), int(np.clip(px[i, 0], 0, W - 1))] for i in box_idx])
    )
    if body.sum() < P.get("min_body_px", 500):  # SAM found (next to) nothing: no target rather than "all background"
        return None, qc

    ignore = ignore_static.copy() if ignore_static is not None else np.zeros((H, W), bool)
    # Band: all visible tail keypoints, also outside the crop (drawing clips at the border), so a tail leaving the
    # crop is still covered from its base to the edge. SAM prompts: only the in-crop ones.
    tail = np.array([px[J[n]] for n in P["tail"] if visible[J[n]]], np.float32).reshape(-1, 2)
    tail_in = np.array([px[J[n]] for n in P["tail"] if ok[J[n]]], np.float32).reshape(-1, 2)
    qc["tail_used"] = False
    if len(tail) >= 2:
        base = px[J[P["tail"][0]]] if ok[J[P["tail"][0]]] else None
        tneg = np.array([px[J[n]] for n in P["tail_negative"] if ok[J[n]]], np.float32).reshape(-1, 2)
        tneg = np.concatenate([tneg, neg_hw], 0)
        tbox = (
            np.concatenate([tail.min(0) - 20, tail.max(0) + 20])
            .clip(0, [W - 1, H - 1, W - 1, H - 1])
            .astype(np.float32)
        )
        good = False
        if len(tail_in) >= 1 and tbox[2] > tbox[0] and tbox[3] > tbox[1]:
            tm, tsc, _ = predictor.predict(
                point_coords=np.concatenate([tail_in, tneg]),
                point_labels=np.r_[np.ones(len(tail_in)), np.zeros(len(tneg))].astype(np.int32),
                box=tbox,
                multimask_output=False,
            )
            tail_m = (tm[0] > 0) & ~body
            # An empty tail mask must not pass: it would narrow the band to tail_band_px and widen nothing
            good = (
                float(tsc[0]) >= P["min_tail_score"]
                and tail_m.sum() >= P.get("min_tail_px", 50)
                and tail_m.mean() <= P["max_tail_frac"]
            )
        width = P["tail_band_px"] if good else P["tail_band_fallback_px"]
        band = np.zeros((H, W), np.uint8)
        cv2.polylines(band, [np.round(tail).astype(np.int32)], False, 1, thickness=width)
        tig = band > 0
        if good:
            r = P["tail_dilate_px"]
            tig |= cv2.dilate(tail_m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
            qc["tail_used"] = True
        if base is not None:  # keep the tail base supervised by the body mask
            keep = np.zeros((H, W), np.uint8)
            cv2.circle(keep, (int(base[0]), int(base[1])), P["tail_band_px"], 1, -1)
            tig &= keep == 0
        ignore |= tig
    target = np.where(ignore, IGNORE, np.where(body, MOUSE, BACKGROUND)).astype(np.uint8)
    return target, qc


def _raise_on_stop_signals() -> None:
    """SIGTERM and SIGINT raise KeyboardInterrupt, so a stopped run closes its HDF5 output. Killed without this, the
    metadata is never written and the whole file is unreadable; a shell's background jobs also ignore SIGINT."""

    def _stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop)


def run(a):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    P = json.load(open(a.prompts))
    J = {n: i for i, n in enumerate(P["joint_names"])}
    ignore_masks = dict(np.load(Path(a.prompts).parent / P["ignore_masks"])) if P.get("ignore_masks") else {}
    shard, nshards = (int(v) for v in a.shard.split("/"))
    with h5py.File(a.h5, "r") as f:
        _check_prompts(P, f, ignore_masks)
    predictor = SAM2ImagePredictor(build_sam2(a.sam_config, a.sam_ckpt, device="cuda"))
    with h5py.File(a.h5, "r") as f:
        N, V = f["multiview_images/view_mask"].shape
        rows = (
            [s for s in range(N) if s % nshards == shard][: a.limit]
            if a.limit
            else [s for s in range(N) if s % nshards == shard]
        )
        cams = json.loads(f["metadata"].attrs["canonical_camera_order"])
        res = a.resolution
        with h5py.File(a.out, "w") as o:
            g = o.create_group(GROUP)
            g.create_dataset(
                "target",
                (N, V, res, res),
                np.uint8,
                chunks=(1, V, res, res),
                compression="gzip",
                compression_opts=4,
                fillvalue=IGNORE,
            )
            for k, shape in QC_FLOAT.items():
                g.create_dataset(k, (N, V) + shape, np.float32, fillvalue=np.nan)
            for k in QC_BOOL:
                g.create_dataset(k, (N, V), bool, fillvalue=False)
            g.create_dataset("processed", (N,), bool, fillvalue=False)
            kp_rep, vis_rep = f["multiview_keypoints/keypoints_2d"], f["multiview_keypoints/keypoint_visibility"]
            kp_raw, vis_raw = (
                f["multiview_keypoints/keypoints_2d_raw"],
                f["multiview_keypoints/keypoint_visibility_raw"],
            )
            ci, vm = f["multiview_keypoints/camera_indices"], f["multiview_images/view_mask"]
            # Written first, so a stopped shard is still a self-describing partial file ('processed' marks its rows).
            _copy_source_datasets(f, o)
            g.attrs.update(_provenance(a, P, N))
            g.attrs["source_metadata"] = _source_metadata(f)
            _raise_on_stop_signals()
            t0, done = time.time(), 0
            try:
                for n, s in enumerate(rows):
                    tgt = np.full((V, res, res), IGNORE, np.uint8)
                    qcs = {k: np.full((V,) + shape, np.nan, np.float32) for k, shape in QC_FLOAT.items()}
                    qcs.update({k: np.zeros(V, bool) for k in QC_BOOL})
                    kr, vr, kw, vw = kp_rep[s], vis_rep[s], kp_raw[s], vis_raw[s]
                    for v in np.flatnonzero(vm[s]):
                        cam = cams[int(ci[s, v])]
                        bgr = cv2.imdecode(
                            np.frombuffer(f[f"multiview_images/image_jpeg_view_{v}"][s].tobytes(), np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        img = cv2.cvtColor(_stretch(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2RGB)
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                            t512, qc = segment_view(
                                predictor, img, kr[v], vr[v], kw[v], vw[v], cam, P, J, ignore_masks.get(cam)
                            )
                        for k, val in qc.items():
                            qcs[k][v] = val
                        if t512 is not None:
                            tgt[v] = _downsample(t512, res)
                            qcs["has_mask"][v] = True
                    g["target"][s] = tgt
                    for k, arr in qcs.items():
                        g[k][s] = arr
                    g["processed"][s] = True
                    if n % 500 == 0:
                        el = time.time() - t0
                        print(
                            f"[shard {shard}/{nshards}] {n}/{len(rows)} samples, {el / 60:.1f} min, eta {el / max(n, 1) * (len(rows) - n) / 60:.1f} min",
                            flush=True,
                        )
                    done = n + 1
            except KeyboardInterrupt:  # SIGTERM / SIGINT: leave the with-blocks, so the output file is closed
                print(f"[shard {shard}/{nshards}] stopped after {done}/{len(rows)} samples; {a.out} closed", flush=True)
                raise SystemExit(1)
    print(f"[shard {shard}/{nshards}] done: {len(rows)} samples -> {a.out}")


def _provenance(a, P, N):
    return {
        "value_convention": json.dumps({"background": BACKGROUND, "mouse": MOUSE, "ignore": IGNORE}),
        "resolution": int(a.resolution),
        "source_h5": os.path.basename(a.h5),
        "source_num_samples": int(N),
        "sam_config": a.sam_config,
        "sam_ckpt": os.path.basename(a.sam_ckpt),
        "sam_ckpt_sha256": _sha256(a.sam_ckpt),
        "prompts": json.dumps(P),
        "git_commit": _git_commit(),
        "generator_sha256": _sha256(os.path.abspath(__file__)),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def merge(a):
    with h5py.File(a.h5, "r") as f:
        N, V = f["multiview_images/view_mask"].shape
        shards = [h5py.File(p, "r") for p in a.shards]
        done = np.zeros(N, int)
        for sh in shards:
            done += sh[f"{GROUP}/processed"][()].astype(int)
            for name in ALIGNMENT + IMAGE_IDENTITY:
                if name in f and name in sh and not np.array_equal(sh[name][()], f[name][()]):
                    raise ValueError(f"{name} of shard {sh.filename} does not match {a.h5}")
        for key in SAME_ACROSS_SHARDS:
            values = {str(sh[GROUP].attrs.get(key)) for sh in shards}
            if len(values) > 1:
                raise ValueError(f"shards disagree in {key}: {sorted(values)}")
        if np.any(done > 1) or (not a.allow_partial and not np.all(done == 1)):
            raise ValueError(
                f"rows not covered exactly once: {int((done == 0).sum())} missing, {int((done > 1).sum())} duplicated"
            )
        res = shards[0][f"{GROUP}/target"].shape[-1]
        with h5py.File(a.out, "w") as o:
            g = o.create_group(GROUP)
            t = g.create_dataset(
                "target",
                (N, V, res, res),
                np.uint8,
                chunks=(1, V, res, res),
                compression="gzip",
                compression_opts=4,
                fillvalue=IGNORE,
            )
            for k, shape in QC_FLOAT.items():
                g.create_dataset(k, (N, V) + shape, np.float32, fillvalue=np.nan)
            for k in QC_BOOL:
                g.create_dataset(k, (N, V), bool, fillvalue=False)
            g.create_dataset("processed", data=done == 1)
            for sh in shards:
                rows = np.flatnonzero(sh[f"{GROUP}/processed"][()])
                for start in range(0, len(rows), 256):
                    r = rows[start : start + 256]
                    t[r] = sh[f"{GROUP}/target"][r]
                for k in [*QC_FLOAT, *QC_BOOL]:
                    if k not in sh[GROUP]:  # shards from before the field existed
                        continue
                    tmp = g[k][()]
                    tmp[rows] = sh[f"{GROUP}/{k}"][()][rows]
                    g[k][...] = tmp
            _copy_source_datasets(f, o)
            g.attrs.update({k: v for k, v in shards[0][GROUP].attrs.items()})
            g.attrs["merged_from"] = json.dumps([os.path.basename(p) for p in a.shards])
            g.attrs["rows_processed"] = int((done == 1).sum())
            g.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        for sh in shards:
            sh.close()
    print(f"merged {len(a.shards)} shards -> {a.out}: {N} samples x {V} views at {res} px")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="segment one shard of the dataset")
    r.add_argument("h5")
    r.add_argument(
        "--prompts", required=True, help="JSON prompt config (joint names, prompt joints, per-camera hardware)"
    )
    r.add_argument("--sam_config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    r.add_argument("--sam_ckpt", required=True)
    r.add_argument("--resolution", type=int, default=256)
    r.add_argument("--shard", default="0/1", help="i/n: process samples with index %% n == i")
    r.add_argument("--limit", type=int, default=None, help="process only the first N samples of the shard (testing)")
    r.add_argument("--out", required=True)
    m = sub.add_parser("merge", help="merge shard files into one sidecar")
    m.add_argument("h5")
    m.add_argument("shards", nargs="+")
    m.add_argument("--out", required=True)
    m.add_argument("--allow_partial", action="store_true", help="allow unprocessed rows (they stay has_mask=False)")
    a = p.parse_args()
    run(a) if a.cmd == "run" else merge(a)


if __name__ == "__main__":
    main()
