"""PROBE 17 — which worker scans are fittable, decided BEFORE any fitting?

THE GOAL THIS SERVES
Everything so far has been evaluated on a random 50 of the 757 workers, and every arm has
been judged by its mean over that sample. That mean mixes together specimens the pipeline
handles well with specimens no parametric fit could recover, and a method that helps the
first group can easily look neutral because it cannot help the second. The new objective is
narrower and more useful: fit the top half as well as possible, and be explicit about what
is excluded.

To exclude honestly, the exclusion rule must be computable from the SCAN ALONE. A rule that
needs a fit to decide whether a fit will work is useless for triage, and it also leaks: it
would select the specimens the current pipeline happens to like, and then "prove" the
pipeline is good on them.

WHAT IS MEASURED (all intrinsic, no template, no fit)
  connectivity   n_components, fraction of area outside the largest component. CT of an ant
                 in ethanol produces detached debris and floating fragments.
  integrity      boundary-edge fraction (holes), non-manifold edge fraction, degenerate
                 triangle fraction, duplicated-vertex fraction.
  alignment      deviation of PCA-1 from +x (yaw) and from the x-y plane (pitch). The
                 dataset is hand-canonicalised, but probe-01 found one 62 deg outlier, and
                 the template's shape space is expressed in the canonical frame.
  form           PCA eigenvalue ratios (a squashed or bloated specimen), surface-area to
                 volume^(2/3) (a proxy for how much thin structure survived), and the
                 fraction of surface area more than k radii from the principal axis (how
                 much of the animal is limb rather than body).
  clutter        fraction of sampled surface points whose k-NN radius is anomalously large
                 (isolated speckle) or small (dense clumps of mounting medium).

Features are computed for the 757 workers AND for the 81 ALL_ANTS_CLEAN meshes, so the
"clean" corpus acts as a reference distribution: a worker scan that looks like a clean scan
on these axes should be fittable, and the distance to that distribution is itself a
candidate score that requires no fit labels at all.

Outputs: out/probe17_features_worker.csv, out/probe17_features_clean.csv, out/probe17_*.png
"""

import argparse
import glob
import os
import sys

import numpy as np
import trimesh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
N_SAMP = 20000


def features(path, rng):
    """Intrinsic mesh-quality features. Returns a dict, or None if the mesh cannot load."""
    try:
        m = trimesh.load(path, process=False, force="mesh")
    except Exception:
        return None
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int64)
    if len(V) < 100 or len(F) < 100:
        return None
    d = dict(name=os.path.basename(path), n_verts=len(V), n_faces=len(F))

    # ---- connectivity: how much of the surface is NOT in the main body
    try:
        comp = trimesh.graph.connected_components(m.face_adjacency, min_len=1, nodes=np.arange(len(F)))
        areas = m.area_faces
        ca = np.array([areas[c].sum() for c in comp]) if len(comp) else np.array([areas.sum()])
        d["n_components"] = len(comp)
        d["frac_area_offmain"] = float(1.0 - ca.max() / max(ca.sum(), 1e-12))
    except Exception:
        d["n_components"] = -1
        d["frac_area_offmain"] = np.nan

    # ---- integrity
    try:
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0), axis=1)
        _, cnt = np.unique(e, axis=0, return_counts=True)
        d["frac_boundary_edges"] = float((cnt == 1).mean())
        d["frac_nonmanifold_edges"] = float((cnt > 2).mean())
    except Exception:
        d["frac_boundary_edges"] = d["frac_nonmanifold_edges"] = np.nan
    tri = V[F]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a = 0.5 * np.linalg.norm(cr, axis=1)
    d["frac_degenerate_tri"] = float((a <= 1e-14).mean())
    d["frac_dup_verts"] = float(1.0 - len(np.unique(np.round(V, 9), axis=0)) / len(V))

    # ---- normalise exactly as the fitter does (load_meshes: centre, divide by max|coord|)
    c = V.mean(0)
    P = V - c
    s = np.abs(P).max()
    P = P / max(s, 1e-12)

    # ---- alignment
    sub = P[rng.choice(len(P), min(6000, len(P)), replace=False)]
    _, S, Vt = np.linalg.svd(sub - sub.mean(0), full_matrices=False)
    ax = Vt[0] / np.linalg.norm(Vt[0])
    d["yaw_deg"] = float(np.degrees(np.arctan2(abs(ax[1]), abs(ax[0]))))
    d["pitch_deg"] = float(np.degrees(np.arcsin(np.clip(abs(ax[2]), 0, 1))))
    ev = (S**2) / max((S**2).sum(), 1e-12)
    d["pca_ev1"], d["pca_ev2"], d["pca_ev3"] = float(ev[0]), float(ev[1]), float(ev[2])
    d["elongation"] = float(ev[0] / max(ev[1], 1e-9))
    d["flatness"] = float(ev[1] / max(ev[2], 1e-9))

    # ---- form: area-weighted surface sample in the normalised frame
    try:
        pts, fid = trimesh.sample.sample_surface(m, N_SAMP, seed=int(rng.integers(1 << 30)))
        pts = (np.asarray(pts) - c) / max(s, 1e-12)
    except Exception:
        return d
    area = float(a.sum()) / max(s, 1e-12) ** 2
    vol = abs(float(m.volume)) / max(s, 1e-12) ** 3 if m.is_volume else np.nan
    d["area_norm"] = area
    d["area_over_vol23"] = float(area / max(vol ** (2 / 3), 1e-9)) if np.isfinite(vol) else np.nan

    q = pts - pts.mean(0)
    perp = q - (q @ ax)[:, None] * ax[None, :]
    r = np.linalg.norm(perp, axis=1)
    med = np.median(r)
    d["limb_area_frac"] = float((r > 2.0 * med).mean())  # how much lies far off the axis
    d["radial_p95_over_med"] = float(np.percentile(r, 95) / max(med, 1e-9))

    # ---- clutter: local point spacing anomalies
    kd = cKDTree(pts)
    dk = kd.query(pts, k=9)[0][:, -1]
    lm = np.median(dk)
    d["speckle_frac"] = float((dk > 3.0 * lm).mean())
    d["clump_frac"] = float((dk < 0.25 * lm).mean())
    d["spacing_cv"] = float(dk.std() / max(dk.mean(), 1e-12))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker_dir", default="/media/fabi/Data/SMILify_LEGACY/custom_processing/antscan_proofread_castes/worker")
    ap.add_argument("--clean_dir", default="/media/fabi/Data/SMILify_DATASETS_BACKUP/ALL_ANTS_CLEAN")
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    import pandas as pd

    for tag, d_ in (("clean", args.clean_dir), ("worker", args.worker_dir)):
        fs = sorted(glob.glob(os.path.join(d_, "*.obj")))
        if args.limit > 0:
            fs = fs[: args.limit]
        print(f"[probe17] {tag}: {len(fs)} meshes", flush=True)
        rng = np.random.default_rng(0)
        rows = []
        for i, f in enumerate(fs):
            r = features(f, rng)
            if r:
                rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{len(fs)}", flush=True)
        df = pd.DataFrame(rows)
        p = os.path.join(OUT, f"probe17_features_{tag}.csv")
        df.to_csv(p, index=False)
        print(f"[probe17] wrote {p}  ({len(df)} rows, {df.shape[1]} features)")


if __name__ == "__main__":
    main()
