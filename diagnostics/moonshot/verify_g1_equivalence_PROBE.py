"""ADVERSARIAL CHECK: is probe 14's "20%" a G1-equivalent?

probe_15_partfield_gates.py:225 prints "the geodesic level-set method (probe 14) scored 20%
on the G1-equivalent", and fitter_3d/partfield.py:41 says G1 "is the gate that killed the
geodesic method (20%)".

G1 (probe 15) = per-point LABEL agreement over 30000 points, two samplings, NN-matched.
probe 14's 0.20 = fraction of specimens whose integer BRANCH COUNT was exactly equal across
two samplings.

This script computes the geodesic method's ACTUAL per-point label agreement, using probe 14's
own decomposition and probe 14's own stated naming rule (sign(y) for L/R, x-order for
pro/meso/meta), plus an ORACLE upper bound (Hungarian matching of component ids, i.e. the
best any naming scheme could possibly do).

CPU only.
"""

import glob
import json
import os
import sys
import time

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench50", "*.obj")
NPTS = 20000
K = 10
THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MIN_BRANCH = 40

# ---- verbatim from probe_14_topology_gate.py -------------------------------------------


def sample_points(mesh, n, rng):
    p, _ = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(1 << 30)))
    p = np.asarray(p, dtype=np.float64)
    p = p - p.mean(0)
    return p / np.abs(p).max()


def graph_from_points(P, k=K):
    tree = cKDTree(P)
    d, idx = tree.query(P, k=k + 1)
    n = len(P)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    vals = d[:, 1:].ravel()
    A = coo_matrix((vals, (rows, cols)), shape=(n, n))
    return A.maximum(A.T).tocsr()


def geodesic(P):
    """probe 14's graph + bridging + seed + normalised geodesic field."""
    A = graph_from_points(P)
    ncomp, lab = connected_components(A, directed=False)
    if ncomp > 1:
        big = np.bincount(lab).argmax()
        tree = cKDTree(P[lab == big])
        extra_r, extra_c, extra_v = [], [], []
        big_idx = np.nonzero(lab == big)[0]
        for c in range(ncomp):
            if c == big:
                continue
            ci = np.nonzero(lab == c)[0]
            dd, jj = tree.query(P[ci], k=1)
            j = int(np.argmin(dd))
            extra_r.append(ci[j])
            extra_c.append(big_idx[jj[j]])
            extra_v.append(float(dd[j]))
        if extra_r:
            n = len(P)
            E = coo_matrix((extra_v, (extra_r, extra_c)), shape=(n, n))
            A = (A + E.maximum(E.T)).tocsr()
    tree = cKDTree(P)
    dens = tree.query(P, k=16)[0][:, -1]
    seed = int(np.argmin(dens))
    g = dijkstra(A, indices=seed, directed=False)
    g = g / np.nanmax(g[np.isfinite(g)])
    return A, g


def branch_counts(A, g):
    counts = {}
    for t in THRESHOLDS:
        keep = np.nonzero(g > t)[0]
        if len(keep) < MIN_BRANCH:
            counts[t] = 0
            continue
        sub = A[keep][:, keep]
        nc, sl = connected_components(sub, directed=False)
        sizes = np.bincount(sl)
        counts[t] = int((sizes >= MIN_BRANCH).sum())
    return counts


# ---- labelling: probe 14's own stated naming rule ---------------------------------------
# "sign(y) gives left/right and attachment order in x gives pro-/meso-/metathoracic"
# label 0 = body (below threshold), label 1 = debris (sub-MIN_BRANCH component),
# labels 2.. = named branches by (side, x-rank).


def label_at(A, g, P, t):
    n = len(P)
    lab = np.zeros(n, dtype=np.int64)  # body
    keep = np.nonzero(g > t)[0]
    if len(keep) < MIN_BRANCH:
        return lab, 0
    sub = A[keep][:, keep]
    nc, sl = connected_components(sub, directed=False)
    sizes = np.bincount(sl, minlength=nc)
    big = [c for c in range(nc) if sizes[c] >= MIN_BRANCH]
    # debris first
    for c in range(nc):
        if sizes[c] < MIN_BRANCH:
            lab[keep[sl == c]] = 1
    # name the real branches: side by mean sign(y), rank by mean x within side
    info = []
    for c in big:
        pts = P[keep[sl == c]]
        info.append((c, float(pts[:, 1].mean()), float(pts[:, 0].mean())))
    out = {}
    for side in (0, 1):  # 0 = y<0, 1 = y>=0
        grp = [i for i in info if (i[1] >= 0) == bool(side)]
        grp.sort(key=lambda z: z[2])  # ascending x = attachment order
        for rank, z in enumerate(grp):
            out[z[0]] = 2 + side * 16 + rank
    for c in big:
        lab[keep[sl == c]] = out[c]
    return lab, len(big)


def oracle_agreement(la, lb_matched):
    """Best achievable agreement under ANY bijective relabelling of component ids."""
    ua = np.unique(la)
    ub = np.unique(lb_matched)
    M = np.zeros((len(ua), len(ub)))
    ia = {v: i for i, v in enumerate(ua)}
    ib = {v: i for i, v in enumerate(ub)}
    for x, y in zip(la, lb_matched):
        M[ia[x], ib[y]] += 1
    r, c = linear_sum_assignment(-M)
    return float(M[r, c].sum() / len(la))


def main():
    files = sorted(glob.glob(BENCH))
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    files = files[:nmax]
    print(f"geodesic G1-equivalent on {len(files)} specimens, {NPTS} pts, k={K}\n")
    rows = []
    t0 = time.time()
    for i, f in enumerate(files):
        name = os.path.basename(f).replace("_processed.obj", "")
        try:
            mesh = trimesh.load(f, process=False, force="mesh")
        except Exception as e:
            print(f"  {name[:36]:<36} LOAD FAIL {e}")
            continue
        Ps, As, gs, cs = [], [], [], []
        for s in (0, 1):
            rng = np.random.default_rng(1000 + s)
            P = sample_points(mesh, NPTS, rng)
            A, g = geodesic(P)
            Ps.append(P)
            As.append(A)
            gs.append(g)
            cs.append(branch_counts(A, g))
        best = [max(c.values()) for c in cs]
        # probe 14's per-specimen BEST threshold (its own generous choice), per seed
        tstar = [max(c, key=lambda k: c[k]) for c in cs]
        # NN match sampling 1 onto sampling 0, exactly as probe 15's G1 does
        j = cKDTree(Ps[1]).query(Ps[0], k=1)[1]
        # (i) each seed at its own best threshold
        la, na = label_at(As[0], gs[0], Ps[0], tstar[0])
        lb, nb = label_at(As[1], gs[1], Ps[1], tstar[1])
        g1_named = float((la == lb[j]).mean())
        g1_oracle = oracle_agreement(la, lb[j])
        # (ii) a FIXED shared threshold, which is what a deployed pipeline would use
        fixed = 0.30
        laf, _ = label_at(As[0], gs[0], Ps[0], fixed)
        lbf, _ = label_at(As[1], gs[1], Ps[1], fixed)
        g1_named_fix = float((laf == lbf[j]).mean())
        g1_oracle_fix = oracle_agreement(laf, lbf[j])
        rows.append(
            dict(
                name=name,
                best0=best[0],
                best1=best[1],
                agree=best[0] == best[1],
                g1_named=g1_named,
                g1_oracle=g1_oracle,
                g1_named_fix=g1_named_fix,
                g1_oracle_fix=g1_oracle_fix,
            )
        )
        print(
            f"  {name[:34]:<34} b={best[0]},{best[1]} eq={int(best[0] == best[1])} | "
            f"G1named {100 * g1_named:5.1f}% G1oracle {100 * g1_oracle:5.1f}% | "
            f"@0.30 named {100 * g1_named_fix:5.1f}% oracle {100 * g1_oracle_fix:5.1f}%",
            flush=True,
        )

    a = np.array([r["agree"] for r in rows], dtype=float)
    print("\n" + "=" * 78)
    print(f"n = {len(rows)}   ({time.time() - t0:.0f}s)")
    print(f"  probe 14's published statistic (branch-count exact equality): {100 * a.mean():.1f}%")
    for k, lbl in (
        ("g1_named", "G1-equivalent, probe14's own naming rule, per-seed best t"),
        ("g1_oracle", "G1-equivalent, ORACLE relabelling (upper bound), best t"),
        ("g1_named_fix", "G1-equivalent, naming rule, fixed t=0.30"),
        ("g1_oracle_fix", "G1-equivalent, ORACLE relabelling, fixed t=0.30"),
    ):
        v = np.array([r[k] for r in rows])
        print(f"  {lbl:<58} {100 * v.mean():5.1f}%  (min {100 * v.min():.1f})")
    json.dump(rows, open(os.path.join(HERE, "out", "verify_g1_equivalence_PROBE.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
