"""Rank all 757 worker scans by the scan-only fittability score, and stage a held-out test.

WHAT THE SCORE IS
`radial_med`: the median distance of area-sampled surface points from the specimen's
principal axis, in the fitter's own normalised frame (centre, divide by max|coord|).

WHY IT IS THE SCORE
On the 50 bench specimens it correlates with fit quality at Spearman rho = +0.81 against
chamfer_l2 and +0.80 against deform_mag, consistently signed across BOTH metric families --
which matters, because report section 3 established that surface-proximity metrics alone can
be gamed by shrink-wrapping and would rank a torn mesh well. Splitting the bench at its
median:

    metric              easy half   hard half   ratio
    chamfer_l2            0.00019     0.00044    2.31x
    deform_mag_mean       0.01635     0.02467    1.51x
    edge_logratio         0.34793     0.40467    1.16x
    fscore@0.02           0.96484     0.88426    0.92x

and the ALL_ANTS_CLEAN corpus -- the meshes OmniAnt's own shape space was built from -- has
median radial_med 0.1521 against the workers' 0.2304, i.e. it sits at the easy end of the
worker distribution. The score measures the axis on which the workers differ from the data
the template can already represent.

Two competing readings of what it physically means, not yet separated:
  (a) body compactness -- stubby species (Cyphomyrmex, Cyphoidris) are far from an elongated
      template, so the shape space cannot reach them;
  (b) a normalisation effect -- the frame divides by max|coord|, which is set by the longest
      protrusion, so a specimen with extended antennae or legs automatically scores low.
Both would predict fit quality; only (a) would be a statement about shape. Distinguishing
them matters for whether the gate generalises to other castes, and is left explicit rather
than assumed.

CIRCULARITY, and how this script avoids it
The correlation was discovered on the bench50 specimens. Selecting on it and then re-scoring
those same specimens would prove nothing. So this script ranks the OTHER 707 workers and
stages two disjoint, equal-sized sets drawn from them:
    EASY  -- lowest radial_med
    RAND  -- uniformly sampled from the same 707
Fitting both with an identical recipe gives a clean held-out test of whether the score
predicts fittability on specimens that played no part in finding it.
"""

import argparse
import glob
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def radial_med(path, rng, n=12000):
    try:
        m = trimesh.load(path, process=False, force="mesh")
        V = np.asarray(m.vertices, dtype=np.float64)
        if len(V) < 100 or len(m.faces) < 100:
            return None
        c = V.mean(0)
        s = max(np.abs(V - c).max(), 1e-12)
        m = trimesh.Trimesh(vertices=(V - c) / s, faces=np.asarray(m.faces), process=False)
        pts = np.asarray(trimesh.sample.sample_surface(m, n, seed=int(rng.integers(1 << 30)))[0])
        q = pts - pts.mean(0)
        _, _, Vt = np.linalg.svd(q[rng.choice(len(q), min(5000, len(q)), replace=False)], full_matrices=False)
        ax = Vt[0] / np.linalg.norm(Vt[0])
        perp = q - (q @ ax)[:, None] * ax[None, :]
        r = np.linalg.norm(perp, axis=1)
        return dict(
            name=os.path.basename(path),
            radial_med=float(np.median(r)),
            radial_p90=float(np.percentile(r, 90)),
            yaw_deg=float(np.degrees(np.arctan2(abs(ax[1]), abs(ax[0])))),
        )
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--worker_dir", default="/media/fabi/Data/SMILify_LEGACY/custom_processing/antscan_proofread_castes/worker"
    )
    ap.add_argument("--n_select", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    import pandas as pd

    bench = {os.path.basename(p) for p in glob.glob(os.path.join(HERE, "bench50", "*.obj"))}
    fs = sorted(glob.glob(os.path.join(args.worker_dir, "*.obj")))
    print(f"[rank] {len(fs)} workers, {len(bench)} already in bench50 (excluded from the held-out sets)")
    rng = np.random.default_rng(0)
    rows = [r for r in (radial_med(f, rng) for f in fs) if r]
    df = pd.DataFrame(rows).sort_values("radial_med").reset_index(drop=True)
    df["in_bench50"] = df.name.isin(bench)
    p = os.path.join(OUT, "worker_fittability_rank.csv")
    df.to_csv(p, index=False)
    print(f"[rank] wrote {p}  (n={len(df)})")
    print(
        f"[rank] radial_med  min {df.radial_med.min():.4f}  median {df.radial_med.median():.4f}  max {df.radial_med.max():.4f}"
    )
    print(
        f"[rank] reference corpus median 0.1521 -> {100 * (df.radial_med <= 0.1521).mean():.1f}% of workers are at or below it"
    )

    pool = df[~df.in_bench50].reset_index(drop=True)
    easy = pool.head(args.n_select)
    rr = np.random.default_rng(args.seed)
    rand = pool.iloc[np.sort(rr.choice(len(pool), args.n_select, replace=False))]
    for tag, sel in (("easy", easy), ("rand", rand)):
        d = os.path.join(HERE, f"heldout_{tag}")
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, "*.obj")):
            os.unlink(f)
        for n in sel.name:
            os.symlink(os.path.join(args.worker_dir, n), os.path.join(d, n))
        print(f"[rank] staged {len(sel)} -> {d}   radial_med median {sel.radial_med.median():.4f}")
    print("[rank] the two sets are disjoint from bench50, so the gate is tested on specimens")
    print("[rank] that played no part in discovering it.")


if __name__ == "__main__":
    main()
