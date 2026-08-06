"""REFUTATION PROBE 3 — the two confounds that the published "matched control" does not hold
constant: (A) OPTIMISATION-PATH NOISE FLOOR, and (B) SAMPLING DENSITY of the corpus.

(A) NOISE FLOOR, matched across corpora.
REFUTE2 measured that on the worker bench50 three independent optimiser seeds of the same
recipe over the SAME 50 meshes disagree by 85.4% of the across-specimen spread of
deform_verts, so gen@10 = 0.9870 x spread is only 1.16x the seed disagreement — the metric is
nearly saturated there. There are no extra seeds for CLEAN81, but each corpus has a
structurally MATCHED pair of independently-optimised deform solutions, both started from that
corpus's own H2_joint init, same template, same code:

    worker  : runs/M1_sym/H3_deform.npz     vs runs/M7_handoff_midline/Stage_3_deform_fine.npz
    worker  : runs/LIM_0_hier/H3_deform.npz vs runs/LIM_0/Stage_3_deform_fine.npz
    clean   : runs/CLEAN_hier/H3_deform.npz vs runs/CLEAN_M7/Stage_3_deform_fine.npz

(the logs state M7 was initialised from M1_sym/H2_joint, LIM_0 from LIM_0_hier/H2_joint and
CLEAN_M7 from CLEAN_hier/H2_joint.) disagreement/spread is then an apples-to-apples estimate
of how much of each corpus's deform field is optimisation-path-dependent.

(B) SAMPLING DENSITY.
Leave-one-out PCA generalisation is a function of how densely the corpus samples shape space,
independently of correspondence quality: if a held-out specimen has a near-twin in the
training set, ANY reasonable basis reconstructs it. Measured in REFUTE1:
    nearest-neighbour distance / population spread
      worker LIM_0 1.124, BPX 1.134, M7 1.154, baseline 1.175   (no clusters at all)
      CLEAN81_M7 0.977,  ALL_ANTS_CLEAN 0.723                   (clustered; and it contains
      'ectatomma-tuberculatum' AND 'ectatomma-tuberculatum (1)' at 0.24-0.40 x spread)
This probe density-matches by greedy farthest-point subsampling (FPS) of the clean corpus,
which thins clusters while keeping n fixed, and reports gen/spread as a function of the
resulting NN/spread. If the clean advantage tracks NN/spread, the published contrast is a
corpus-composition statistic, not a registration-quality one.

Output -> shape_decomp_REFUTE3_PROBE_out.txt
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
KS = [1, 5, 10, 20, 40]

PAIRS = [
    (
        "worker bench50 (M1_sym H3 vs M7 Stage3)",
        f"{HERE}/runs/M1_sym/H3_deform.npz",
        f"{HERE}/runs/M7_handoff_midline/Stage_3_deform_fine.npz",
    ),
    (
        "worker bench50 (LIM_0_hier H3 vs LIM_0 Stage3)",
        f"{HERE}/runs/LIM_0_hier/H3_deform.npz",
        f"{HERE}/runs/LIM_0/Stage_3_deform_fine.npz",
    ),
    (
        "clean81 (CLEAN_hier H3 vs CLEAN_M7 Stage3)",
        f"{HERE}/runs/CLEAN_hier/H3_deform.npz",
        f"{HERE}/runs/CLEAN_M7/Stage_3_deform_fine.npz",
    ),
]

DENSITY = [
    ("worker M7 (n=50, all)", f"{HERE}/runs/M7_handoff_midline/Stage_3_deform_fine.npz", None),
    ("worker LIM_0 (n=50, all)", f"{HERE}/runs/LIM_0/Stage_3_deform_fine.npz", None),
    ("CLEAN81_M7", f"{HERE}/runs/CLEAN_M7/Stage_3_deform_fine.npz", [81, 60, 50, 40, 30]),
    (
        "ALL_ANTS_CLEAN",
        "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/"
        "Stage_3_deform_fine.npz",
        [81, 60, 50, 40, 30],
    ),
]


def pop_spread(X):
    return float(np.linalg.norm(X - X.mean(0), axis=-1).mean())


def loo(X, ks=KS):
    N = X.shape[0]
    Xf = X.reshape(N, -1)
    kmax = min(max(ks), N - 2)
    e0 = np.zeros(N)
    ek = np.zeros((N, kmax + 1))
    for i in range(N):
        m = np.ones(N, bool)
        m[i] = False
        Y = Xf[m]
        mu = Y.mean(0)
        _, _, Vt = np.linalg.svd(Y - mu, full_matrices=False)
        r = Xf[i] - mu
        e0[i] = np.linalg.norm(r.reshape(-1, 3), axis=-1).mean()
        for k in range(1, kmax + 1):
            B = Vt[:k]
            ek[i, k] = np.linalg.norm((r - (r @ B.T) @ B).reshape(-1, 3), axis=-1).mean()
    return float(e0.mean()), {k: (float(ek[:, k].mean()) if k <= kmax else np.nan) for k in ks}


def pdist_mesh(X):
    """mean per-vertex euclidean distance between every pair of specimens."""
    N = X.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        D[i] = np.linalg.norm(X[i][None] - X, axis=-1).mean(1)
    return D


def nn_over_spread(X, D=None):
    D = pdist_mesh(X) if D is None else D
    Dc = D.copy()
    np.fill_diagonal(Dc, np.inf)
    return float(Dc.min(1).mean() / pop_spread(X))


def fps(D, m, seed=0):
    """greedy farthest-point subsample of m indices — thins clusters, keeps extremes."""
    N = D.shape[0]
    rng = np.random.default_rng(seed)
    sel = [int(rng.integers(N))]
    d = D[sel[0]].copy()
    while len(sel) < m:
        d[sel] = -1
        j = int(np.argmax(d))
        sel.append(j)
        d = np.minimum(d, D[j])
    return np.array(sorted(sel))


def main():
    os.makedirs(OUT, exist_ok=True)
    fh = open(os.path.join(HERE, "shape_decomp_REFUTE3_PROBE_out.txt"), "w")

    def out(s=""):
        print(s, flush=True)
        fh.write(s + "\n")
        fh.flush()

    res = {}
    out("=" * 104)
    out("A. MATCHED NOISE FLOOR — two independently-optimised deform solutions per corpus,")
    out("   both started from that corpus's own H2_joint, same template, same code.")
    out("=" * 104)
    for name, pa, pb in PAIRS:
        if not (os.path.exists(pa) and os.path.exists(pb)):
            out(f"\n  [skip] {name}")
            continue
        da, db = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
        assert [str(x) for x in da["labels"]] == [str(x) for x in db["labels"]]
        A = np.asarray(da["deform_verts"], dtype=np.float64)
        B = np.asarray(db["deform_verts"], dtype=np.float64)
        spA, spB = pop_spread(A), pop_spread(B)
        dis = float(np.linalg.norm(A - B, axis=-1).mean())
        Ac = (A - A.mean(0)).reshape(A.shape[0], -1)
        Bc = (B - B.mean(0)).reshape(B.shape[0], -1)
        corr = float((Ac * Bc).sum() / np.sqrt((Ac * Ac).sum() * (Bc * Bc).sum()))
        _, gA = loo(A, [10])
        _, gB = loo(B, [10])
        out(f"\n  {name}   n={A.shape[0]}")
        out(f"    A spread {spA:.6f} |F| {np.linalg.norm(A, axis=-1).mean():.6f}  gen@10/spread {gA[10] / spA:.4f}")
        out(f"    B spread {spB:.6f} |F| {np.linalg.norm(B, axis=-1).mean():.6f}  gen@10/spread {gB[10] / spB:.4f}")
        out(f"    path disagreement {dis:.6f} = {100 * dis / spB:.1f}% of spread(B)   corr(centred) {corr:+.4f}")
        out(
            f"    -> path-noise floor {dis / spB:.4f}   measured gen@10/spread {gB[10] / spB:.4f}"
            f"   HEADROOM {gB[10] / spB - dis / spB:+.4f}"
        )
        res.setdefault("floor", {})[name] = dict(
            n=int(A.shape[0]),
            spreadA=spA,
            spreadB=spB,
            dis=dis,
            floor=dis / spB,
            corr=corr,
            genA10=gA[10] / spA,
            genB10=gB[10] / spB,
            headroom=gB[10] / spB - dis / spB,
        )

    out()
    out("=" * 104)
    out("B. SAMPLING DENSITY — gen/spread vs nearest-neighbour distance / spread")
    out("   FPS = greedy farthest-point subsample (thins clusters). rand = random subsample.")
    out("=" * 104)
    out(f"\n  {'corpus':<34}{'n':>5}{'NN/spread':>11}{'gen@10/sp':>11}{'gen@20/sp':>11}{'gen@40/sp':>11}")
    for name, npz, sizes in DENSITY:
        if not os.path.exists(npz):
            out(f"  [skip] {name}")
            continue
        d = np.load(npz, allow_pickle=True)
        F = np.asarray(d["deform_verts"], dtype=np.float64)
        D = pdist_mesh(F)
        N = F.shape[0]
        rows = []
        if sizes is None:
            sizes_ = [N]
        else:
            sizes_ = sizes
        for m in sizes_:
            if m > N:
                continue
            variants = (
                [("all", np.arange(N))]
                if m == N
                else [("FPS", fps(D, m)), ("rand", np.random.default_rng(0).choice(N, m, replace=False))]
            )
            for tag, idx in variants:
                X = F[idx]
                sp = pop_spread(X)
                _, g = loo(X)
                nn = nn_over_spread(X, D[np.ix_(idx, idx)])
                lab = f"{name} [{tag}]"
                out(
                    f"  {lab:<34}{m:>5}{nn:>11.4f}{g[10] / sp:>11.4f}"
                    f"{g[20] / sp:>11.4f}"
                    f"{(g[40] / sp if np.isfinite(g[40]) else float('nan')):>11.4f}"
                )
                rows.append(
                    dict(
                        tag=tag,
                        m=int(m),
                        nn=nn,
                        r10=g[10] / sp,
                        r20=g[20] / sp,
                        r40=(g[40] / sp if np.isfinite(g[40]) else None),
                    )
                )
        res.setdefault("density", {})[name] = rows

    json.dump(res, open(os.path.join(OUT, "shape_decomp_REFUTE3.json"), "w"), indent=1)
    out(f"\nwrote {os.path.join(OUT, 'shape_decomp_REFUTE3.json')}")
    fh.close()


if __name__ == "__main__":
    main()
