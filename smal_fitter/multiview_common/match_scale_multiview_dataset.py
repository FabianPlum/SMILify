#!/usr/bin/env python3
"""Rescale a multi-view SMIL HDF5 dataset's 3D world so its physical scale
matches a reference dataset.

Why
---

Real SLEAP rigs and replicAnt/Unreal synthetic scenes store their 3D worlds
in different physical units (anipose mm via the reader's world_scale=0.001
heuristic vs replicAnt's translation_factor=0.1 Unreal units). Measured on
the stick datasets, the same physical subject ends up ~27x larger in reader
units in synthetic files than in real ones. Trained jointly (or evaluated
across domains), the trans / mesh-scale / 3D-keypoint heads must then output
population-specific magnitudes, which turns the units mismatch into a
domain-leakage channel and de-calibrates loss weights tuned on real data.

Uniform world scaling is projection-invariant (verified numerically: scaling
{keypoints_3d, parameters/trans, camera_extrinsics_t} by the same factor
changes reprojections by <1e-3 px), so matching scales is lossless w.r.t.
all pixel-space supervision. Stored 2D keypoints, visibility, intrinsics,
image_sizes and JPEGs are untouched.

What it does
------------

1. Computes the median subject extent (bbox diagonal of valid keypoints_3d,
   in reader units = stored units x effective world_scale) for the input and
   the reference dataset.
2. Derives a single global factor s = ref_extent / input_extent (or takes an
   explicit --scale-factor).
3. Multiplies these datasets by s, leaving the world_scale attr unchanged:
       multiview_keypoints/keypoints_3d
       multiview_keypoints/camera_extrinsics_t
       parameters/trans
       auxiliary/canonical_to_world_t   (kept consistent with the scaled world)
4. Records provenance attrs (factor, reference, before/after extents) and
   refuses to double-apply unless --force is given.

The factor is global, not per-sample: genuine specimen-size variation within
the dataset is preserved; only the units convention changes.

Usage
-----

    python smal_fitter/multiview_common/match_scale_multiview_dataset.py \\
        --input replicant_sticks_filtered.h5 \\
        --reference SMILySTICKS_centred_reprojected.h5 \\
        --in-place                      # or: --output rescaled.h5
        [--scale-factor 0.037]          # bypass reference measurement
        [--sample-size 400] [--dry-run] [--force]

In-place mode rewrites only the four small 3D datasets (tens of MB) — safe
even on network shares. Copy mode duplicates the whole file first (prefer a
local output path on CIFS).
"""

from __future__ import annotations

import argparse
import datetime
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

SCALED_DATASETS = [
    "multiview_keypoints/keypoints_3d",
    "multiview_keypoints/camera_extrinsics_t",
    "parameters/trans",
    "auxiliary/canonical_to_world_t",  # optional, replicAnt provenance
]


def effective_world_scale(f: h5py.File) -> float:
    """Reader-effective world_scale: explicit attr if present, else the
    SLEAPMultiViewDataset heuristic (translations >> 50 are mm -> 0.001)."""
    md = f["metadata"].attrs
    if "world_scale" in md:
        return float(md["world_scale"])
    try:
        if "camera_extrinsics_t" in f["multiview_keypoints"]:
            t0 = f["multiview_keypoints/camera_extrinsics_t"][0, 0]
            if float(np.linalg.norm(t0)) > 50.0:
                return 0.001
        kp0 = f["multiview_keypoints/keypoints_3d"][0]
        if float(np.nanmax(np.abs(kp0))) > 50.0:
            return 0.001
    except Exception:
        pass
    return 1.0


def median_subject_extent(f: h5py.File, sample_size: int) -> float:
    """Median bbox diagonal of valid keypoints_3d across a strided sample,
    in reader units (stored units x effective world_scale)."""
    ws = effective_world_scale(f)
    n = int(f["metadata"].attrs["num_samples"])
    idx = np.linspace(0, n - 1, min(n, sample_size)).astype(int)
    kp3d = f["multiview_keypoints/keypoints_3d"][idx] * ws
    extents = []
    for kp in kp3d:
        ok = np.isfinite(kp).all(axis=1) & (np.abs(kp).sum(axis=1) > 1e-9)
        if ok.sum() >= 2:
            bb = kp[ok].max(0) - kp[ok].min(0)
            extents.append(np.linalg.norm(bb))
    if not extents:
        raise ValueError("No valid keypoints_3d found to measure subject extent")
    return float(np.median(extents))


def apply_scale(f: h5py.File, s: float) -> list:
    """Multiply the 3D-world datasets by s in place. Returns the list of
    dataset names actually scaled."""
    scaled = []
    for name in SCALED_DATASETS:
        if name not in f:
            continue
        ds = f[name]
        # Whole-array rewrite: all four datasets are small (tens of MB max).
        ds[...] = ds[...] * s
        scaled.append(name)
    return scaled


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Multiview HDF5 to rescale")
    p.add_argument("--reference", default=None, help="Reference multiview HDF5 whose physical scale to match")
    p.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        help="Explicit factor for the stored 3D values (skips reference measurement)",
    )
    p.add_argument("--output", default=None, help="Write a rescaled copy here (whole-file duplicate)")
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Rescale the input file directly (rewrites only the four small 3D datasets)",
    )
    p.add_argument("--sample-size", type=int, default=400, help="Samples used to measure median subject extent")
    p.add_argument("--dry-run", action="store_true", help="Measure and report the factor; change nothing")
    p.add_argument("--force", action="store_true", help="Allow rescaling a file that was already match-scaled")
    args = p.parse_args(argv)

    if args.scale_factor is None and args.reference is None:
        p.error("Provide --reference or --scale-factor")
    if not args.dry_run and bool(args.output) == bool(args.in_place):
        p.error("Choose exactly one of --output / --in-place (or use --dry-run)")

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    # --- measure -----------------------------------------------------------
    with h5py.File(input_path, "r") as f:
        md = dict(f["metadata"].attrs)
        if "match_scale_factor" in md and not args.force:
            sys.exit(
                f"{input_path} already match-scaled "
                f"(factor {md['match_scale_factor']:g} vs "
                f"{md.get('match_scale_reference', '?')}); use --force to re-apply"
            )
        ws_in = effective_world_scale(f)
        ext_in = median_subject_extent(f, args.sample_size)

    if args.scale_factor is not None:
        s = float(args.scale_factor)
        ext_ref = ext_in * s * 1.0  # implied
        ref_label = f"<explicit factor {s:g}>"
    else:
        with h5py.File(args.reference, "r") as fr:
            ext_ref = median_subject_extent(fr, args.sample_size)
        # extent_reader = ws_in * extent_stored; scaling stored values by s
        # scales reader extent by s too, so:
        s = ext_ref / ext_in
        ref_label = str(Path(args.reference).resolve())

    print(f"input:     {input_path}  (world_scale={ws_in:g}, median subject extent={ext_in:.4f} reader units)")
    print(f"reference: {ref_label}  (median subject extent={ext_ref:.4f})")
    print(f"--> scale factor s = {s:.6f}")
    if args.dry_run:
        print("Dry run: nothing written.")
        return

    # --- apply -------------------------------------------------------------
    if args.output:
        out_path = Path(args.output)
        if out_path.resolve() == input_path.resolve():
            sys.exit("--output must differ from --input (use --in-place instead)")
        if out_path.exists():
            sys.exit(f"{out_path} exists; refusing to overwrite")
        print(f"Copying {input_path} -> {out_path} ...")
        shutil.copyfile(input_path, out_path)
        target = out_path
    else:
        target = input_path

    with h5py.File(target, "r+") as f:
        scaled = apply_scale(f, s)
        md = f["metadata"].attrs
        md["match_scale_factor"] = s
        md["match_scale_reference"] = ref_label
        md["match_scale_extent_before"] = ext_in
        md["match_scale_extent_after"] = ext_in * s
        md["match_scale_timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")

    # --- verify ------------------------------------------------------------
    with h5py.File(target, "r") as f:
        ext_check = median_subject_extent(f, args.sample_size)
    print(f"scaled datasets: {scaled}")
    print(
        f"verification: median subject extent now {ext_check:.4f} "
        f"(target {ext_ref:.4f}) -> {'OK' if abs(ext_check - ext_ref) / ext_ref < 0.02 else 'MISMATCH'}"
    )
    print(f"Done: {target}")


if __name__ == "__main__":
    main()
