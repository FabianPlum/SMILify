#!/usr/bin/env python3
"""Filter a multi-view SMIL HDF5 dataset by supervision-quality criteria.

Why
---

Synthetic multi-view datasets (replicAnt / merged) intentionally contain far
more varied camera placements and occlusion levels than real SLEAP rigs. A
quality sweep of `replicant_sticks_merged.h5` vs `SMILySTICKS_centred_reprojected.h5`
showed the synthetic file contains supervision pathologies the real data never
has, which starve or destabilise training:

- samples whose mean 2D keypoint visibility is near zero (empty loss masks);
- individual views with almost no visible keypoints (nearly blind views);
- views whose subject centroid sits at/behind the camera plane (perspective
  projection blows up -> non-finite losses -> all-NaN collapse).

This script writes a filtered copy of an existing HDF5 (schema produced by
`preprocess_sleap_multiview_dataset.py`, `preprocess_replicant_multiview_dataset.py`,
or `merge_multiview_datasets.py`):

1. **View-level masking** (the view slot is kept on disk but `view_mask` is
   cleared, its visibility zeroed and its `camera_indices` slot set to -1):
   - per-view mean keypoint visibility < `--min-view-visibility`;
   - keypoints_3d centroid depth in that camera <= 0 (OpenCV convention,
     `z = (R @ c + t)[2]`), unless `--no-filter-behind-camera`.
2. **Sample-level dropping** (the row is not copied):
   - fewer than `--min-views` valid views remain;
   - no visible keypoint remains in any valid view;
   - mean visibility across remaining valid views < `--min-sample-visibility`.

Everything else is copied verbatim (same dtypes/chunking/compression, all
attrs preserved), so the output is read by `SLEAPMultiViewDataset` unchanged.
Use `--dry-run` to preview the cost on a given file without writing anything.

NOTE: when the source lives on a network share (CIFS/SMB), write the output
to a *local* path first and copy it to the share afterwards — large direct
writes to flaky mounts can be interrupted mid-write.

Usage
-----

    python smal_fitter/multiview_common/filter_multiview_dataset.py \\
        --input replicant_sticks_merged.h5 \\
        --output /local/scratch/replicant_sticks_filtered.h5 \\
        [--min-sample-visibility 0.3] [--min-view-visibility 0.1] \\
        [--min-views 2] [--no-filter-behind-camera] \\
        [--dry-run] [--chunk-size 256] [--overwrite]
"""

from __future__ import annotations

import argparse
import datetime
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Analysis pass.
# ---------------------------------------------------------------------------


@dataclass
class FilterResult:
    """Outcome of the analysis pass."""

    num_samples: int
    new_view_mask: np.ndarray  # (N, max_views) bool — after view-level masking
    keep: np.ndarray  # (N,) bool — sample-level keep decision
    views_masked_low_vis: int = 0
    views_masked_behind_cam: int = 0
    dropped_min_views: int = 0
    dropped_zero_visible: int = 0
    dropped_low_sample_vis: int = 0
    mean_vis_before: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mean_vis_after: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_keep(self) -> int:
        return int(self.keep.sum())


def _sample_mean_visibility(visibility: np.ndarray, view_mask: np.ndarray) -> np.ndarray:
    """Mean keypoint visibility per sample over valid views (0 when no valid view).

    visibility: (N, V, J) float; view_mask: (N, V) bool.
    """
    n_joints = visibility.shape[2]
    masked = visibility * view_mask[:, :, None]
    denom = view_mask.sum(axis=1) * n_joints
    return np.divide(masked.sum(axis=(1, 2)), denom, out=np.zeros(visibility.shape[0]), where=denom > 0)


def analyze(
    hf: h5py.File, min_sample_visibility: float, min_view_visibility: float, min_views: int, filter_behind_camera: bool
) -> FilterResult:
    """Compute view masks and sample keep/drop decisions from the small
    per-sample arrays (visibility, view_mask, keypoints_3d, extrinsics)."""
    num_samples = int(hf["metadata"].attrs["num_samples"])

    view_mask = hf["multiview_images/view_mask"][:].astype(bool)  # (N, V)
    visibility = hf["multiview_keypoints/keypoint_visibility"][:]  # (N, V, J)

    mean_vis_before = _sample_mean_visibility(visibility, view_mask)

    new_mask = view_mask.copy()

    # --- View criterion 1: nearly blind views -----------------------------
    per_view_vis = visibility.mean(axis=2)  # (N, V)
    blind = view_mask & (per_view_vis < min_view_visibility)
    new_mask &= ~blind

    # --- View criterion 2: subject centroid at/behind the camera plane ----
    behind_count = 0
    kps = hf["multiview_keypoints"]
    if filter_behind_camera and "keypoints_3d" in kps and "camera_extrinsics_R" in kps:
        kp3d = kps["keypoints_3d"][:]  # (N, J, 3)
        R = kps["camera_extrinsics_R"][:]  # (N, V, 3, 3)
        t = kps["camera_extrinsics_t"][:]  # (N, V, 3)

        valid_kp = np.isfinite(kp3d).all(axis=2) & (np.abs(kp3d).sum(axis=2) > 1e-9)
        n_valid = valid_kp.sum(axis=1)  # (N,)
        centroid = np.divide(
            (kp3d * valid_kp[:, :, None]).sum(axis=1),
            n_valid[:, None],
            out=np.zeros((num_samples, 3), dtype=kp3d.dtype),
            where=n_valid[:, None] > 0,
        )
        # Depth of the centroid in each camera frame (OpenCV: +z is forward).
        z = np.einsum("nvj,nj->nv", R[:, :, 2, :], centroid) + t[:, :, 2]  # (N, V)
        behind = new_mask & (z <= 0.0) & (n_valid > 0)[:, None]
        behind_count = int(behind.sum())
        new_mask &= ~behind

    # --- Sample criteria ---------------------------------------------------
    remaining = new_mask.sum(axis=1)
    mean_vis_after = _sample_mean_visibility(visibility, new_mask)
    visible_left = (visibility * new_mask[:, :, None]).sum(axis=(1, 2))

    drop_min_views = remaining < min_views
    drop_zero_visible = ~drop_min_views & (visible_left <= 0)
    drop_low_vis = ~drop_min_views & ~drop_zero_visible & (mean_vis_after < min_sample_visibility)

    keep = ~(drop_min_views | drop_zero_visible | drop_low_vis)

    return FilterResult(
        num_samples=num_samples,
        new_view_mask=new_mask,
        keep=keep,
        views_masked_low_vis=int(blind.sum()),
        views_masked_behind_cam=behind_count,
        dropped_min_views=int(drop_min_views.sum()),
        dropped_zero_visible=int(drop_zero_visible.sum()),
        dropped_low_sample_vis=int(drop_low_vis.sum()),
        mean_vis_before=mean_vis_before,
        mean_vis_after=mean_vis_after,
    )


def print_report(result: FilterResult, args: argparse.Namespace) -> None:
    n = result.num_samples
    kept = result.n_keep
    print("=" * 70)
    print(f"Filter report for: {args.input}")
    print(
        f"  thresholds: min_sample_visibility={args.min_sample_visibility}, "
        f"min_view_visibility={args.min_view_visibility}, min_views={args.min_views}, "
        f"filter_behind_camera={not args.no_filter_behind_camera}"
    )
    print("-" * 70)
    print(f"  views masked (mean visibility < {args.min_view_visibility:.2f}): {result.views_masked_low_vis}")
    print(f"  views masked (subject behind camera plane):  {result.views_masked_behind_cam}")
    print(f"  samples dropped (< {args.min_views} valid views left):   {result.dropped_min_views}")
    print(f"  samples dropped (zero visible keypoints):    {result.dropped_zero_visible}")
    print(f"  samples dropped (mean visibility < {args.min_sample_visibility:.2f}):    {result.dropped_low_sample_vis}")
    print("-" * 70)
    print(f"  samples kept: {kept}/{n} ({100.0 * kept / max(n, 1):.1f}%)")

    edges = np.round(np.linspace(0.0, 1.0, 11), 12)
    hist_before, _ = np.histogram(result.mean_vis_before, bins=edges)
    hist_after, _ = np.histogram(result.mean_vis_after[result.keep], bins=edges)
    print("  mean-visibility histogram (per sample, valid views):")
    print("    bin        before     after(kept)")
    for i in range(10):
        print(f"    {edges[i]:.1f}-{edges[i + 1]:.1f}   {hist_before[i]:8d}   {hist_after[i]:8d}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Copy pass.
# ---------------------------------------------------------------------------

# Datasets rewritten from the analysis result rather than copied from source.
_VIEW_MASK_DS = "multiview_images/view_mask"
_VISIBILITY_DS = "multiview_keypoints/keypoint_visibility"
_CAMERA_INDICES_DS = "multiview_keypoints/camera_indices"
_NUM_VIEWS_DS = "auxiliary/num_views"


def _create_like(
    dst_parent: h5py.Group, name: str, src_ds: h5py.Dataset, n_keep: int, row_subset: bool
) -> h5py.Dataset:
    """Create a dataset in dst mirroring src's dtype/chunks/compression, with
    the first dimension shrunk to n_keep when row_subset."""
    shape = (n_keep,) + src_ds.shape[1:] if row_subset else src_ds.shape
    chunks = src_ds.chunks
    if chunks is not None:
        # Chunk shape must not exceed the (possibly smaller) data shape.
        chunks = tuple(min(c, s) if s > 0 else c for c, s in zip(chunks, shape))
    out = dst_parent.create_dataset(
        name,
        shape=shape,
        dtype=src_ds.dtype,
        chunks=chunks,
        compression=src_ds.compression,
        compression_opts=src_ds.compression_opts,
    )
    for k, v in src_ds.attrs.items():
        out.attrs[k] = v
    return out


def write_filtered(src: h5py.File, output_path: Path, result: FilterResult, args: argparse.Namespace) -> None:
    num_samples = result.num_samples
    keep = result.keep
    n_keep = result.n_keep
    keep_view_mask = result.new_view_mask[keep]  # (n_keep, V)
    block = max(1, int(args.chunk_size))

    with h5py.File(output_path, "w") as dst:
        # Root attrs (usually none, but preserve whatever exists).
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        for group_name in src.keys():
            src_group = src[group_name]
            dst_group = dst.create_group(group_name)
            for k, v in src_group.attrs.items():
                dst_group.attrs[k] = v
            if not isinstance(src_group, h5py.Group):
                raise ValueError(f"Unexpected non-group at root: {group_name}")

            for ds_name in src_group.keys():
                src_ds = src_group[ds_name]
                if not isinstance(src_ds, h5py.Dataset):
                    raise ValueError(f"Nested groups are not supported: {group_name}/{ds_name}")
                full_name = f"{group_name}/{ds_name}"
                row_subset = len(src_ds.shape) >= 1 and src_ds.shape[0] == num_samples
                dst_ds = _create_like(dst_group, ds_name, src_ds, n_keep, row_subset)

                if not row_subset:
                    if src_ds.shape != ():
                        dst_ds[...] = src_ds[...]
                    else:
                        dst_ds[()] = src_ds[()]
                    continue

                # Chunked sequential copy (CIFS-friendly: contiguous reads).
                is_vlen = h5py.check_vlen_dtype(src_ds.dtype) is not None
                pos = 0
                desc = f"copy {full_name}"
                for start in tqdm(range(0, num_samples, block), desc=desc, leave=False):
                    end = min(start + block, num_samples)
                    sel = keep[start:end]
                    if not sel.any():
                        continue
                    data = src_ds[start:end][sel]
                    k = len(data)
                    out_mask = keep_view_mask[pos : pos + k]

                    if full_name == _VIEW_MASK_DS:
                        data = out_mask
                    elif full_name == _VISIBILITY_DS:
                        data = data * out_mask[:, :, None]
                    elif full_name == _CAMERA_INDICES_DS:
                        data = np.where(out_mask, data, -1)
                    elif full_name == _NUM_VIEWS_DS:
                        data = out_mask.sum(axis=1).astype(data.dtype)

                    if is_vlen:
                        # Block-assigning an object array whose elements share a
                        # length lets numpy collapse it to a 2D array, which h5py
                        # rejects for vlen datasets — write row by row instead.
                        for j in range(k):
                            dst_ds[pos + j] = data[j]
                    else:
                        dst_ds[pos : pos + k] = data
                    pos += k
                assert pos == n_keep, f"{full_name}: wrote {pos} rows, expected {n_keep}"

        # Metadata attrs: copy, then update counts + provenance.
        md = dst["metadata"].attrs
        md["num_samples"] = n_keep
        md["filter_source_file"] = str(Path(args.input).resolve())
        md["filter_original_num_samples"] = num_samples
        md["filter_min_sample_visibility"] = float(args.min_sample_visibility)
        md["filter_min_view_visibility"] = float(args.min_view_visibility)
        md["filter_min_views"] = int(args.min_views)
        md["filter_behind_camera"] = not args.no_filter_behind_camera
        md["filter_views_masked"] = int(result.views_masked_low_vis + result.views_masked_behind_cam)
        md["filter_timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")

    print(f"Wrote {n_keep} samples -> {output_path}")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Source multiview HDF5 file")
    p.add_argument(
        "--output",
        default=None,
        help="Output HDF5 path (required unless --dry-run). Prefer a local path when the source is on a network share.",
    )
    p.add_argument(
        "--min-sample-visibility",
        type=float,
        default=0.3,
        help="Drop samples whose mean keypoint visibility over valid views is below this",
    )
    p.add_argument(
        "--min-view-visibility", type=float, default=0.1, help="Mask views whose mean keypoint visibility is below this"
    )
    p.add_argument(
        "--min-views", type=int, default=2, help="Drop samples with fewer valid views than this after masking"
    )
    p.add_argument(
        "--no-filter-behind-camera",
        action="store_true",
        help="Disable masking of views with the subject centroid at/behind the camera plane",
    )
    p.add_argument("--dry-run", action="store_true", help="Analyze and print the report only; write nothing")
    p.add_argument("--chunk-size", type=int, default=256, help="Rows per read/write block during the copy pass")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output file")
    args = p.parse_args(argv)

    if not args.dry_run:
        if args.output is None:
            p.error("--output is required unless --dry-run is given")
        if Path(args.output).resolve() == Path(args.input).resolve():
            p.error("--output must differ from --input")
        if Path(args.output).exists() and not args.overwrite:
            p.error(f"{args.output} exists; pass --overwrite to replace it")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    with h5py.File(input_path, "r") as src:
        if not bool(src["metadata"].attrs.get("is_multiview", True)):
            sys.exit("Input is not a multiview dataset (metadata.is_multiview is False)")
        print("Analyzing (reading visibility/view_mask/keypoints_3d/extrinsics)...")
        result = analyze(
            src,
            min_sample_visibility=args.min_sample_visibility,
            min_view_visibility=args.min_view_visibility,
            min_views=args.min_views,
            filter_behind_camera=not args.no_filter_behind_camera,
        )
        print_report(result, args)

        if args.dry_run:
            return
        if result.n_keep == 0:
            sys.exit("All samples would be dropped; refusing to write an empty dataset")
        if result.n_keep == result.num_samples and result.views_masked_low_vis + result.views_masked_behind_cam == 0:
            print("Note: no samples dropped and no views masked — output will be an identical copy.")
        write_filtered(src, Path(args.output), result, args)


if __name__ == "__main__":
    main()
