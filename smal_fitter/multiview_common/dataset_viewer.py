"""Streamlit viewer for multi-view SMIL HDF5 datasets.

Launch:

    streamlit run smal_fitter/multiview_common/dataset_viewer.py

Or with a default path:

    streamlit run smal_fitter/multiview_common/dataset_viewer.py -- --hdf5 path.h5

Supports any HDF5 with the SLEAPMultiViewDataset schema — that includes
SLEAP-preprocessor output, replicAnt-preprocessor output, and the merger's
output. The per-sample inspector shows decoded JPEGs with 2D-keypoint and
reprojection overlays, a 3D Plotly viewer for kp3d + camera centres, and a
metadata + per-view-parameters card. The dataset-stats tab shows num_views
distribution, has_3d_data ratio, origin_dataset distribution (for merged
files), and an on-demand reprojection-error histogram across all samples.

Directory-based formats (raw replicAnt flat dir, single-view JSON) and
SMAL mesh rendering land in follow-up phases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import plotly.graph_objects as go
import streamlit as st


# Make multiview_common importable when streamlit launches the module directly.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # smal_fitter/ on path
from multiview_common.canonical_frame import (  # noqa: E402
    cam_center_world,
    project_world_to_pixel,
)


# ---------------------------------------------------------------------------
# Cached I/O.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _open_hdf5(path: str) -> h5py.File:
    """One file handle per path, cached across reruns."""
    return h5py.File(path, "r")


@st.cache_data(show_spinner=False)
def _read_metadata(path: str) -> Dict[str, object]:
    """All /metadata attrs in a plain dict, with bytes decoded to str."""
    out: Dict[str, object] = {}
    with h5py.File(path, "r") as f:
        if "metadata" not in f:
            return out
        for k, v in f["metadata"].attrs.items():
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            elif isinstance(v, np.ndarray) and v.dtype.kind in ("S", "O"):
                v = [
                    e.decode("utf-8", errors="replace") if isinstance(e, bytes) else e
                    for e in v
                ]
            out[k] = v
    return out


@st.cache_data(show_spinner=False)
def _dataset_summary(path: str) -> Dict[str, object]:
    """Cheap per-sample arrays loaded once: view_mask sums, has_3d_data,
    origin_dataset (if present). Used for filter widgets and stats."""
    with h5py.File(path, "r") as f:
        n = int(f["metadata"].attrs["num_samples"])
        mv = int(f["metadata"].attrs["max_views"])
        out = {
            "num_samples": n,
            "max_views": mv,
            "num_views": (
                f["auxiliary/num_views"][:]
                if "auxiliary/num_views" in f
                else f["multiview_images/view_mask"][:].sum(axis=1)
            ).astype(int),
            "has_3d_data": (
                f["auxiliary/has_3d_data"][:].astype(bool)
                if "auxiliary/has_3d_data" in f
                else np.zeros(n, dtype=bool)
            ),
        }
        if "auxiliary/origin_dataset" in f:
            origins = f["auxiliary/origin_dataset"][:]
            out["origin_dataset"] = np.array([
                s.decode("utf-8", errors="replace") if isinstance(s, bytes) else str(s)
                for s in origins
            ])
        else:
            out["origin_dataset"] = None
        if "auxiliary/origin_source_file" in f:
            srcs = f["auxiliary/origin_source_file"][:]
            out["origin_source_file"] = np.array([
                s.decode("utf-8", errors="replace") if isinstance(s, bytes) else str(s)
                for s in srcs
            ])
        else:
            out["origin_source_file"] = None
        if "auxiliary/session_name" in f:
            sns = f["auxiliary/session_name"][:]
            out["session_name"] = np.array([
                s.decode("utf-8", errors="replace") if isinstance(s, bytes) else str(s)
                for s in sns
            ])
        else:
            out["session_name"] = None
    return out


@st.cache_data(show_spinner=False)
def _read_sample(path: str, sample_idx: int) -> Dict[str, object]:
    """Read everything we need for one sample in one shot. Cached per idx."""
    with h5py.File(path, "r") as f:
        max_views = int(f["metadata"].attrs["max_views"])
        view_mask = f["multiview_images/view_mask"][sample_idx].astype(bool)
        kp2d = f["multiview_keypoints/keypoints_2d"][sample_idx].astype(np.float64)
        kp_vis = f["multiview_keypoints/keypoint_visibility"][sample_idx].astype(np.float64)
        K = f["multiview_keypoints/camera_intrinsics"][sample_idx].astype(np.float64)
        R = f["multiview_keypoints/camera_extrinsics_R"][sample_idx].astype(np.float64)
        t = f["multiview_keypoints/camera_extrinsics_t"][sample_idx].astype(np.float64)
        image_sizes = f["multiview_keypoints/image_sizes"][sample_idx].astype(np.int64)
        kp3d = f["multiview_keypoints/keypoints_3d"][sample_idx].astype(np.float64)
        camera_indices = f["multiview_keypoints/camera_indices"][sample_idx].astype(np.int64)

        images = []
        for v in range(max_views):
            try:
                blob = f[f"multiview_images/image_jpeg_view_{v}"][sample_idx]
            except KeyError:
                blob = None
            images.append(_decode_jpeg(blob))

        has_3d = (
            bool(f["auxiliary/has_3d_data"][sample_idx])
            if "auxiliary/has_3d_data" in f
            else not bool(np.all(kp3d == 0))
        )
        frame_idx = (
            int(f["auxiliary/frame_idx"][sample_idx])
            if "auxiliary/frame_idx" in f
            else int(sample_idx)
        )
        origin_dataset = None
        if "auxiliary/origin_dataset" in f:
            v = f["auxiliary/origin_dataset"][sample_idx]
            origin_dataset = v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
        session = None
        if "auxiliary/session_name" in f:
            v = f["auxiliary/session_name"][sample_idx]
            session = v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)

    return {
        "sample_idx": int(sample_idx),
        "max_views": max_views,
        "view_mask": view_mask,
        "kp2d_norm_yx": kp2d,
        "kp_vis": kp_vis,
        "K": K,
        "R": R,
        "t": t,
        "image_sizes_wh": image_sizes,
        "kp3d": kp3d,
        "images_rgb": images,
        "camera_indices": camera_indices,
        "has_3d": has_3d,
        "frame_idx": frame_idx,
        "origin_dataset": origin_dataset,
        "session_name": session,
    }


def _decode_jpeg(blob) -> Optional[np.ndarray]:
    if blob is None or len(blob) == 0:
        return None
    bgr = cv2.imdecode(np.asarray(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Per-view overlay rendering.
# ---------------------------------------------------------------------------


def _draw_overlay(
    img_rgb: np.ndarray,
    kp2d_norm_yx: np.ndarray,        # (J, 2) — [y/H_calib, x/W_calib]
    kp_vis: np.ndarray,              # (J,)
    R: np.ndarray, t: np.ndarray, K: np.ndarray,
    kp3d: np.ndarray,                # (J, 3) world frame
    image_sizes_wh: Tuple[int, int],  # (W_calib, H_calib) — K is calibrated for this
    show_gt: bool,
    show_reproj: bool,
) -> np.ndarray:
    """Stretch the JPEG to the calibration frame's aspect, then overlay GT
    and projected-3D keypoints. Visibility codes opacity / fill.

    SLEAP HDF5s have JPEG dims != image_sizes; replicAnt are self-consistent.
    Drawing in the calibration frame and then resampling to a fixed display
    size makes both formats render identically.
    """
    W_calib, H_calib = int(image_sizes_wh[0]), int(image_sizes_wh[1])
    H_jpeg, W_jpeg = img_rgb.shape[:2]

    # Resize the decoded JPEG into the calibration frame so K-based projection
    # lands on the right anatomy. Cheap interpolation - this is just for
    # display.
    if (H_jpeg, W_jpeg) != (H_calib, W_calib):
        canvas = cv2.resize(img_rgb, (W_calib, H_calib), interpolation=cv2.INTER_AREA)
    else:
        canvas = img_rgb.copy()

    has_gt_3d = ~np.all(kp3d == 0, axis=1)
    n_joints = kp2d_norm_yx.shape[0]

    if show_gt:
        for j in range(n_joints):
            if kp_vis[j] <= 0:
                continue
            px = int(round(float(kp2d_norm_yx[j, 1]) * W_calib))
            py = int(round(float(kp2d_norm_yx[j, 0]) * H_calib))
            if not (0 <= px < W_calib and 0 <= py < H_calib):
                continue
            cv2.circle(canvas, (px, py), max(3, W_calib // 200), (60, 220, 60), 2)

    if show_reproj and has_gt_3d.any():
        proj = project_world_to_pixel(kp3d, R, t, K)  # pixels in calibration frame
        for j in range(n_joints):
            if not has_gt_3d[j]:
                continue
            xy = proj[j]
            if np.any(np.isnan(xy)):
                continue
            px = int(round(float(xy[0])))
            py = int(round(float(xy[1])))
            if not (0 <= px < W_calib and 0 <= py < H_calib):
                continue
            r = max(2, W_calib // 250)
            cv2.line(canvas, (px - r, py - r), (px + r, py + r), (230, 60, 60), 2)
            cv2.line(canvas, (px - r, py + r), (px + r, py - r), (230, 60, 60), 2)

    return canvas


def _per_view_reproj_max(sample: Dict[str, object]) -> Dict[int, float]:
    """Per-view max reprojection error in calibration pixels, for the
    visible+has_GT joints. Used for the sample-card summary."""
    kp3d = sample["kp3d"]
    view_mask = sample["view_mask"]
    has_gt_3d = ~np.all(kp3d == 0, axis=1)
    out: Dict[int, float] = {}
    for v in range(len(sample["R"])):
        if not view_mask[v]:
            continue
        W, H = int(sample["image_sizes_wh"][v, 0]), int(sample["image_sizes_wh"][v, 1])
        gt = np.stack([
            sample["kp2d_norm_yx"][v, :, 1] * W,
            sample["kp2d_norm_yx"][v, :, 0] * H,
        ], axis=1)
        pr = project_world_to_pixel(kp3d, sample["R"][v], sample["t"][v], sample["K"][v])
        mask = has_gt_3d & (sample["kp_vis"][v] > 0)
        if not mask.any():
            out[v] = float("nan")
            continue
        err = np.linalg.norm(pr[mask] - gt[mask], axis=1)
        out[v] = float(np.nanmax(err))
    return out


# ---------------------------------------------------------------------------
# 3D viewer (Plotly).
# ---------------------------------------------------------------------------


def _build_3d_figure(sample: Dict[str, object]) -> go.Figure:
    R = sample["R"]
    t = sample["t"]
    kp3d = sample["kp3d"]
    view_mask = sample["view_mask"]
    has_gt_3d = ~np.all(kp3d == 0, axis=1)

    fig = go.Figure()
    if has_gt_3d.any():
        fig.add_trace(go.Scatter3d(
            x=kp3d[has_gt_3d, 0], y=kp3d[has_gt_3d, 1], z=kp3d[has_gt_3d, 2],
            mode="markers",
            marker=dict(size=4, color="black"),
            name=f"kp3d (n={int(has_gt_3d.sum())})",
        ))
    cam_xyz, cam_labels = [], []
    for v in range(len(R)):
        if not view_mask[v]:
            continue
        c = cam_center_world(R[v], t[v])
        cam_xyz.append(c)
        cam_labels.append(f"cam {v}")
    if cam_xyz:
        cam_xyz = np.stack(cam_xyz, axis=0)
        fig.add_trace(go.Scatter3d(
            x=cam_xyz[:, 0], y=cam_xyz[:, 1], z=cam_xyz[:, 2],
            mode="markers+text",
            marker=dict(size=6, color="firebrick", symbol="diamond"),
            text=cam_labels,
            textposition="top center",
            name="cameras",
        ))
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=20, b=0),
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        ),
        legend=dict(orientation="h", y=-0.05),
    )
    return fig


# ---------------------------------------------------------------------------
# Dataset stats (cheap to derive; reprojection histogram is opt-in).
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Computing per-sample reprojection error...")
def _all_sample_reproj_errors(path: str, max_samples: int) -> Dict[str, np.ndarray]:
    """Pass once over up to `max_samples` and compute per-sample max
    reprojection error (over all valid views, visible+has_GT joints).
    Returns dict with 'idx' and 'max_err' arrays."""
    with h5py.File(path, "r") as f:
        n_total = int(f["metadata"].attrs["num_samples"])
        n = min(n_total, max_samples)
        idx_arr = np.arange(n)
        max_err = np.full(n, np.nan, dtype=np.float64)
        vm = f["multiview_images/view_mask"]
        kp2d_d = f["multiview_keypoints/keypoints_2d"]
        K_d = f["multiview_keypoints/camera_intrinsics"]
        R_d = f["multiview_keypoints/camera_extrinsics_R"]
        t_d = f["multiview_keypoints/camera_extrinsics_t"]
        sz_d = f["multiview_keypoints/image_sizes"]
        vis_d = f["multiview_keypoints/keypoint_visibility"]
        kp3d_d = f["multiview_keypoints/keypoints_3d"]
        for i in range(n):
            mask = vm[i].astype(bool)
            kp3d = kp3d_d[i].astype(np.float64)
            has_gt = ~np.all(kp3d == 0, axis=1)
            best = 0.0
            counted = False
            for v in range(len(mask)):
                if not mask[v]:
                    continue
                use = has_gt & (vis_d[i, v].astype(np.float64) > 0)
                if not use.any():
                    continue
                W, H = int(sz_d[i, v, 0]), int(sz_d[i, v, 1])
                gt = np.stack([kp2d_d[i, v, :, 1] * W, kp2d_d[i, v, :, 0] * H], axis=1)
                pr = project_world_to_pixel(kp3d, R_d[i, v], t_d[i, v], K_d[i, v])
                err = float(np.nanmax(np.linalg.norm(pr[use] - gt[use], axis=1)))
                if err > best:
                    best = err
                counted = True
            max_err[i] = best if counted else np.nan
    return {"idx": idx_arr, "max_err": max_err}


# ---------------------------------------------------------------------------
# UI sections.
# ---------------------------------------------------------------------------


def _ui_metadata(meta: Dict[str, object]) -> None:
    st.subheader("Metadata")
    headline_keys = [
        "dataset_type", "num_samples", "max_views", "n_joints",
        "target_resolution", "world_scale", "min_views_per_sample",
        "is_multiview", "has_camera_parameters", "has_3d_keypoints",
    ]
    headline = {k: meta.get(k, "—") for k in headline_keys}
    cols = st.columns(5)
    chip_items = list(headline.items())
    for i, (k, v) in enumerate(chip_items):
        cols[i % 5].metric(label=k, value=str(v))

    with st.expander("All metadata attrs (raw)", expanded=False):
        # Try to JSON-decode nested manifests; otherwise show as-is.
        rendered = {}
        for k, v in meta.items():
            if isinstance(v, str) and v.startswith(("[", "{")):
                try:
                    rendered[k] = json.loads(v)
                    continue
                except json.JSONDecodeError:
                    pass
            rendered[k] = v if not isinstance(v, np.ndarray) else v.tolist()
        st.json(rendered)


def _ui_sample(path: str, sample_idx: int, show_2d: bool, show_reproj: bool, show_3d: bool) -> None:
    sample = _read_sample(path, sample_idx)
    valid_v = [int(v) for v in np.where(sample["view_mask"])[0]]

    st.subheader(f"Sample {sample['sample_idx']}")
    info_cols = st.columns(4)
    info_cols[0].metric("source frame", sample["frame_idx"])
    info_cols[1].metric("valid views", f"{len(valid_v)} / {sample['max_views']}")
    info_cols[2].metric("has_3d_data", str(sample["has_3d"]))
    info_cols[3].metric("origin", sample["origin_dataset"] or "—")
    if sample["session_name"]:
        st.caption(f"session: `{sample['session_name']}`")

    if not valid_v:
        st.warning("This sample has no valid views (view_mask is all False).")
        return

    per_view_err = _per_view_reproj_max(sample) if sample["has_3d"] else {}

    # Per-view grid.
    st.markdown("**Per-view images** — green ○ = stored 2D GT, red ✕ = projected stored 3D")
    n_cols = min(len(valid_v), 4)
    rows = (len(valid_v) + n_cols - 1) // n_cols
    for r in range(rows):
        cols = st.columns(n_cols)
        for c in range(n_cols):
            i = r * n_cols + c
            if i >= len(valid_v):
                break
            v = valid_v[i]
            img = sample["images_rgb"][v]
            with cols[c]:
                if img is None:
                    st.warning(f"view {v}: image missing")
                    continue
                drawn = _draw_overlay(
                    img,
                    sample["kp2d_norm_yx"][v],
                    sample["kp_vis"][v],
                    sample["R"][v],
                    sample["t"][v],
                    sample["K"][v],
                    sample["kp3d"],
                    sample["image_sizes_wh"][v],
                    show_gt=show_2d,
                    show_reproj=show_reproj,
                )
                err_str = ""
                if v in per_view_err and not np.isnan(per_view_err[v]):
                    err_str = f"  |  max reproj = {per_view_err[v]:.2f} px"
                st.image(
                    drawn,
                    caption=f"view {v} ({int(sample['image_sizes_wh'][v, 0])}×{int(sample['image_sizes_wh'][v, 1])} calib){err_str}",
                    width="stretch",
                )

    if show_3d:
        st.markdown("**3D scene** — black ● = kp3d, red ◆ = camera centres (drag to rotate)")
        fig = _build_3d_figure(sample)
        st.plotly_chart(fig, width="stretch")

    with st.expander("Per-view camera parameters", expanded=False):
        for v in valid_v:
            st.markdown(f"**view {v}**  (camera_indices = {int(sample['camera_indices'][v])})")
            K, R, t = sample["K"][v], sample["R"][v], sample["t"][v]
            c1, c2, c3 = st.columns(3)
            c1.code(f"K =\n{np.array2string(K, precision=3, suppress_small=True)}")
            c2.code(f"R =\n{np.array2string(R, precision=4, suppress_small=True)}")
            c3.code(f"t = {np.array2string(t, precision=4, suppress_small=True)}\n"
                    f"||t|| = {float(np.linalg.norm(t)):.4g}")


def _ui_stats(path: str, summary: Dict[str, object]) -> None:
    st.subheader("Dataset stats")
    n = int(summary["num_samples"])
    mv = int(summary["max_views"])

    # num_views distribution.
    nv = summary["num_views"]
    nv_counts = np.bincount(nv, minlength=mv + 1)
    fig_nv = go.Figure(go.Bar(x=list(range(len(nv_counts))), y=nv_counts.tolist()))
    fig_nv.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=24, b=0),
        title="Valid views per sample",
        xaxis_title="num_views", yaxis_title="count",
    )

    # has_3d_data ratio.
    has3d = summary["has_3d_data"]
    pct = float(has3d.mean()) if has3d.size else 0.0
    fig_h3 = go.Figure(go.Pie(
        labels=["has_3d_data", "no 3D"],
        values=[int(has3d.sum()), int((~has3d).sum())],
        hole=0.45,
    ))
    fig_h3.update_layout(
        height=260, margin=dict(l=0, r=0, t=24, b=0),
        title=f"has_3d_data ({pct:.0%})",
    )

    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_nv, width="stretch")
    c2.plotly_chart(fig_h3, width="stretch")

    origin = summary.get("origin_dataset")
    if origin is not None and len(origin) > 0:
        uniq, counts = np.unique(origin, return_counts=True)
        fig_o = go.Figure(go.Bar(x=uniq.tolist(), y=counts.tolist()))
        fig_o.update_layout(
            height=260, margin=dict(l=0, r=0, t=24, b=0),
            title="origin_dataset distribution (merged file)",
            xaxis_title="origin", yaxis_title="count",
        )
        st.plotly_chart(fig_o, width="stretch")

    src = summary.get("origin_source_file")
    if src is not None and len(src) > 0:
        uniq, counts = np.unique(src, return_counts=True)
        if len(uniq) > 1:
            fig_s = go.Figure(go.Bar(x=uniq.tolist(), y=counts.tolist()))
            fig_s.update_layout(
                height=260, margin=dict(l=0, r=0, t=24, b=0),
                title="origin_source_file (merged file)",
                xaxis_title="source", yaxis_title="count",
            )
            st.plotly_chart(fig_s, width="stretch")

    sess = summary.get("session_name")
    if sess is not None and len(sess) > 0:
        uniq, counts = np.unique(sess, return_counts=True)
        if len(uniq) > 1 and len(uniq) <= 50:
            fig_se = go.Figure(go.Bar(x=uniq.tolist(), y=counts.tolist()))
            fig_se.update_layout(
                height=260, margin=dict(l=0, r=0, t=24, b=0),
                title="session distribution",
                xaxis_title="session", yaxis_title="count",
            )
            st.plotly_chart(fig_se, width="stretch")

    st.markdown("---")
    st.markdown("**Reprojection error histogram** (one pass over the dataset)")
    # number_input handles n==1 cleanly (slider needs min < max).
    default_n = min(n, 1000)
    max_n = int(st.number_input(
        "Max samples to scan",
        min_value=1, max_value=max(n, 1), value=default_n, step=max(1, n // 100),
        help="Scanning thousands of samples is expensive; limit for speed.",
    ))
    if st.button("Compute reprojection error histogram"):
        data = _all_sample_reproj_errors(path, int(max_n))
        errs = data["max_err"]
        valid_errs = errs[~np.isnan(errs)]
        if valid_errs.size == 0:
            st.warning("No samples with computable reprojection error in this range.")
            return
        fig_err = go.Figure(go.Histogram(x=valid_errs.tolist(), nbinsx=60))
        fig_err.update_layout(
            height=320, margin=dict(l=0, r=0, t=24, b=0),
            title=f"Per-sample max reprojection error (n={valid_errs.size}, "
                  f"median={float(np.median(valid_errs)):.2f}px, "
                  f"p95={float(np.percentile(valid_errs, 95)):.2f}px, "
                  f"max={float(valid_errs.max()):.2f}px)",
            xaxis_title="px (calibration frame)", yaxis_title="count",
        )
        st.plotly_chart(fig_err, width="stretch")


# ---------------------------------------------------------------------------
# App entry.
# ---------------------------------------------------------------------------


def _parse_cli_default_path() -> Optional[str]:
    """`streamlit run viewer.py -- --hdf5 path.h5` plumbing."""
    if "--hdf5" in sys.argv:
        i = sys.argv.index("--hdf5")
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def main() -> None:
    st.set_page_config(
        page_title="SMILify multi-view dataset viewer",
        layout="wide",
    )
    st.title("Multi-view dataset viewer")

    default_path = _parse_cli_default_path() or ""
    with st.sidebar:
        st.header("Dataset")
        path_str = st.text_input("HDF5 path", value=default_path,
                                 help="Path to a SLEAPMultiViewDataset-format HDF5 "
                                      "(SLEAP, replicAnt, or merger output).")
        if not path_str:
            st.info("Enter an HDF5 path to begin.")
            return
        path = Path(path_str)
        if not path.is_file():
            st.error(f"File not found: {path}")
            return

        meta = _read_metadata(str(path))
        if not meta:
            st.error("No /metadata group in this HDF5 — not a recognised multi-view file.")
            return
        summary = _dataset_summary(str(path))
        n = int(summary["num_samples"])

        st.divider()
        st.header("Sample")
        # Use session_state-keyed widget so prev/next buttons update the
        # number_input cleanly without re-running into the default value.
        if "sample_idx" not in st.session_state:
            st.session_state.sample_idx = 0

        def _clamp(value: int) -> int:
            return max(0, min(int(value), max(n - 1, 0)))

        st.session_state.sample_idx = _clamp(st.session_state.sample_idx)

        st.number_input(
            "index", min_value=0, max_value=max(n - 1, 0), step=1,
            key="sample_idx",
            help=f"0 .. {max(n - 1, 0)}",
        )
        sample_idx = int(st.session_state.sample_idx)

        prev_col, next_col = st.columns(2)
        prev_col.button(
            "◀ prev", width="stretch", disabled=sample_idx <= 0,
            on_click=lambda: st.session_state.update(sample_idx=_clamp(st.session_state.sample_idx - 1)),
        )
        next_col.button(
            "next ▶", width="stretch", disabled=sample_idx >= n - 1,
            on_click=lambda: st.session_state.update(sample_idx=_clamp(st.session_state.sample_idx + 1)),
        )

        st.divider()
        st.header("Display")
        show_2d = st.checkbox("Show 2D keypoints", value=True)
        show_reproj = st.checkbox("Show reprojection overlay", value=True)
        show_3d = st.checkbox("Show 3D viewer", value=True)

    tab_sample, tab_stats = st.tabs(["Sample inspector", "Dataset stats"])
    with tab_sample:
        _ui_metadata(meta)
        st.divider()
        _ui_sample(str(path), int(sample_idx), show_2d, show_reproj, show_3d)
    with tab_stats:
        _ui_stats(str(path), summary)


if __name__ == "__main__":
    main()
