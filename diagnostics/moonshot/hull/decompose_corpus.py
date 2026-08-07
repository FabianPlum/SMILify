"""Decompose a corpus once, cache to disk, so metrics can be iterated without recomputing.

A decomposition costs 10-60 s per specimen per criterion per seed. Recomputing that every
time a metric changes burns the whole budget on work already done, so this script does the
geometry once and writes a compact npz per (corpus, specimen, criterion, seed):

    points        (P, 3)  the decomposition's own area-weighted surface samples
    atom          (P,)    CoACD atom id per point, after the merge tree's renumbering
    linkage       (M, 4)  the merge tree, scipy-compatible
    n_atoms       int
    n_hulls       int
    seconds       float

`load_tree()` rebuilds a `Hierarchy` from that, so every downstream metric -- cuts at any k,
reproducibility, purity, the fitter integration -- reads the cache instead of CoACD.

Usage:
    python decompose_corpus.py --corpus synth  --criteria volume concavity hybrid visibility
    python decompose_corpus.py --corpus worker --n 20 --criteria volume
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MOON = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

from fitter_3d.hull_decomposition import (  # noqa: E402
    Hierarchy,
    assign_to_hulls,
    build_merge_tree,
    coacd_hulls,
    normalise,
    sample_surface,
)

CACHE = os.path.join(HERE, "cache")

CORPORA = {
    "synth": os.path.join(MOON, "synth_clean", "*.obj"),
    "synth_noisy": os.path.join(MOON, "synth_noisy", "*.obj"),
    "clean": os.path.join(MOON, "clean81", "*.obj"),
    "worker": os.path.join(MOON, "bench50", "*.obj"),
}

# specimens where probe 14's geodesic sweep collapsed to <= 2 branches -- the touching-leg
# cases that are the entire reason a convexity criterion was tried (G3)
GEODESIC_COLLAPSED = [
    "Cephalotes_minutus_CASENT0709253",
    "Cephalotes_simillimus_CASENT0744408",
    "Dilobocondyla_fouqueti_CASENT0745576",
    "Dorylus_fulvus_CASENT0745678",
]


def corpus_files(corpus, n=-1):
    if corpus == "collapsed":
        files = [p for p in sorted(glob.glob(CORPORA["worker"])) if any(c in p for c in GEODESIC_COLLAPSED)]
    else:
        files = sorted(glob.glob(CORPORA[corpus]))
    return files if n < 0 else files[:n]


def cache_path(corpus, name, criterion, seed, thr, npts):
    d = os.path.join(CACHE, f"{corpus}_thr{thr}_n{npts}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{name}__{criterion}__s{seed}.npz")


def load_tree(path):
    """Rebuild a `Hierarchy` from a cache file."""
    d = np.load(path)
    return Hierarchy(d["linkage"], d["atom"], d["points"], int(d["n_atoms"])), d


def decompose_one(path, criteria, seed, thr, npts, force=False, corpus="?"):
    """Decompose one mesh at one seed under SEVERAL merge criteria.

    CoACD plus the point-to-hull assignment cost ~15-20 s and are criterion-independent,
    while the merge trees cost 0.3-19 s. Sharing the front half across criteria is a ~4x
    saving on the dominant cost, so the loop is specimen -> seed -> criteria, not the other
    way round.
    """
    name = os.path.basename(path)[:-4]
    todo = [c for c in criteria if force or not os.path.isfile(cache_path(corpus, name, c, seed, thr, npts))]
    if not todo:
        return {c: (cache_path(corpus, name, c, seed, thr, npts), None) for c in criteria}

    m = trimesh.load(path, process=False, force="mesh")
    v, _, _ = normalise(np.asarray(m.vertices))
    f = np.asarray(m.faces)
    nm = trimesh.Trimesh(vertices=v, faces=f, process=False)

    t0 = time.time()
    pts, _ = sample_surface(nm, npts, seed=seed)
    hulls = coacd_hulls(v, f, threshold=thr, seed=seed)
    atom, n_atoms = assign_to_hulls(pts, hulls)
    t_shared = time.time() - t0

    out = {}
    for crit in todo:
        t1 = time.time()
        tree = build_merge_tree(pts, atom, n_atoms, criterion=crit, mesh=nm, seed=seed)
        sec = t_shared + (time.time() - t1)
        p = cache_path(corpus, name, crit, seed, thr, npts)
        np.savez_compressed(
            p,
            points=tree.points.astype(np.float32),
            atom=tree.atom_of_point.astype(np.int32),
            linkage=tree.linkage.astype(np.float32),
            n_atoms=tree.n_atoms,
            n_hulls=len(hulls),
            seconds=sec,
        )
        out[crit] = (p, sec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="synth", choices=list(CORPORA) + ["collapsed"])
    ap.add_argument("--criteria", nargs="+", default=["volume", "concavity", "hybrid", "visibility"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--n_points", type=int, default=8000)
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    files = corpus_files(args.corpus, args.n)
    total = len(files) * len(args.criteria) * len(args.seeds)
    print(f"decomposing {len(files)} meshes x {len(args.criteria)} criteria x {len(args.seeds)} seeds = {total}")
    print(f"  corpus={args.corpus}  threshold={args.threshold}  n_points={args.n_points}\n", flush=True)

    done = 0
    t_start = time.time()
    for p in files:
        for seed in args.seeds:
            try:
                res = decompose_one(p, args.criteria, seed, args.threshold, args.n_points, args.force, args.corpus)
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"  FAIL {os.path.basename(p)[:40]} s{seed}: {type(e).__name__} {e}", flush=True)
                continue
            done += len(res)
            secs = " ".join(f"{c}={'cached' if s is None else f'{s:.0f}s'}" for c, (_, s) in res.items())
            print(
                f"  [{done:4d}/{total}] s{seed} {os.path.basename(p)[:42]:<42} {secs}"
                f"   elapsed {(time.time() - t_start) / 60:.1f}m",
                flush=True,
            )
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min -> {CACHE}")


if __name__ == "__main__":
    main()
