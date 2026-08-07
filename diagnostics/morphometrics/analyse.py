"""M3 — is there taxonomic signal in the automated measurements, and is it morphology?

FOUR QUESTIONS, in the order that makes each one worth asking.

1. SIGNAL. Do specimens of the same genus land near each other in shape space? Measured by
   leave-one-out 1-NN genus accuracy against a label-permutation null. 1-NN is deliberately
   the weakest possible classifier: it has no capacity to launder a weak signal into an
   impressive number, and its null is exact by construction.

2. CONFOUND. Is that signal morphology, or is it the pipeline? Every candidate nuisance variable
   is run through the SAME classifier on its own. If scan quality alone predicts genus as well
   as shape does, the shape result means nothing. Shape is then re-tested after linear
   residualisation on all nuisance variables, which is the strict version of the question.

3. REPEATABILITY. Split the variance of each measurement into between-species and within-species
   parts (a one-way ICC). A measurement whose within-species scatter swamps its between-species
   scatter cannot support any comparative claim no matter how significant the pooled test is.

4. CROSS-CORPUS TRANSFER -- the strongest test available, and the one that needs no ground truth.
   ALL_ANTS_CLEAN and the worker scans are different specimens, different preparation, different
   scanners, different mesh processing. 38 genera occur in both. So: take each ALL_ANTS_CLEAN
   specimen, find the nearest WORKER genus centroid in shape space, and ask whether it is that
   specimen's own genus. Nothing about the two corpora was matched, so a hit rate above chance
   cannot come from a shared artefact -- it can only come from the measurements tracking real,
   reproducible morphology. This is a genuine external replication.

WHY 1-NN AND NOT SOMETHING STRONGER. With ~60-150 specimens spread over dozens of genera, most
genera have 1-10 members. Any discriminative model would be fitting more parameters than it has
examples per class, and cross-validated accuracy would be dominated by the few large genera.
1-NN with a permutation null is honest at this sample size.
"""

import argparse
import csv
import json
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
from calib_features import block_of  # noqa: E402

try:
    from taxonomy import SUBFAMILY, ECOLOGY
except ImportError:  # the map is optional; everything still runs at genus level
    SUBFAMILY, ECOLOGY = {}, {}

OUT = os.path.join(HERE, "out")


# ------------------------------------------------------------------ helpers
def pcs(A, k):
    """Standardise, then project onto the leading k principal components."""
    A = np.nan_to_num(np.asarray(A, dtype=np.float64))
    A = (A - A.mean(0)) / np.maximum(A.std(0), 1e-9)
    U, S, _ = np.linalg.svd(A - A.mean(0), full_matrices=False)
    return (U * S)[:, : min(k, A.shape[1])]


def accession_lot(specimen):
    """Collection lot from the accession code: same prefix, same 100-block.

    Museum specimens of one species are accessioned together -- Eciton burchellii is
    CASENT0744556/7/8, Dorylus fulvus is 745670-745678 -- which means same lot implies, in all
    likelihood, same colony, same fixation, same mounting and the same scan session. Grouping by
    this is what makes leave-one-LOT-out possible.
    """
    m = re.match(r"([A-Za-z]+)0*(\d+)", specimen or "")
    return f"{m.group(1)}{int(m.group(2)) // 100}" if m else "?"


def loo_1nn(F, y, groups=None):
    """Leave-one-out 1-NN. With `groups`, leaves out the whole GROUP, not just the specimen.

    Lot-blind validation is the honest protocol here and it is not optional. Specimens sharing an
    accession lot are near-replicates, so plain leave-one-specimen-out lets a specimen be
    classified by a labmate from its own collection and scan session. Measured: at SPECIES level
    that inflates everything to the point of meaninglessness -- accession number alone predicts
    species at 95.7% against shape's 31.9%. At genus level lots are not pure (26 of 29 hold more
    than one genus) and the signal survives, 5.5x lift -> 4.6x, still p < 0.0001.
    """
    D = ((F[:, None, :] - F[None, :, :]) ** 2).sum(-1)
    if groups is None:
        np.fill_diagonal(D, np.inf)
    else:
        groups = np.asarray(groups)
        for g in np.unique(groups):
            k = np.where(groups == g)[0]
            D[np.ix_(k, k)] = np.inf
    ok = np.isfinite(D).any(1)
    if not ok.any():
        return float("nan")
    return float((y[D[ok].argmin(1)] == y[ok]).mean())


def perm_test(F, y, n=400, seed=0, groups=None):
    rng = np.random.default_rng(seed)
    a = loo_1nn(F, y, groups)
    null = np.array([loo_1nn(F, rng.permutation(y), groups) for _ in range(n)])
    return a, float(null.mean()), float(null.std()), float((null >= a).mean())


def icc_oneway(x, groups):
    """One-way intraclass correlation: between-group variance as a fraction of the total.

    0 means the measurement cannot tell two groups apart; 1 means within-group scatter is nil.
    Uses the standard unbiased ANOVA decomposition with an effective group size, so unequal
    group sizes do not bias it.
    """
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    x, groups = x[ok], np.asarray(groups)[ok]
    gs = [x[groups == g] for g in np.unique(groups)]
    gs = [g for g in gs if len(g) >= 2]
    if len(gs) < 2:
        return np.nan
    k, N = len(gs), sum(len(g) for g in gs)
    gm = np.concatenate(gs).mean()
    msb = sum(len(g) * (g.mean() - gm) ** 2 for g in gs) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in gs) / max(N - k, 1)
    n0 = (N - sum(len(g) ** 2 for g in gs) / N) / (k - 1)
    v = (msb - msw) / n0
    return float(v / (v + msw)) if (v + msw) > 0 else np.nan


def build_table(runs, M, bones, tpa):
    """runs: [(run_name, corpus)] -> (rows, length_cols, dev_cols)"""
    allrows, cols, dev = [], None, None
    for run, corpus in runs:
        d = os.path.join(MOON, "runs", run)
        if not os.path.isdir(d):
            print(f"[skip] {run}: not on disk")
            continue
        r, c, v = ms.measure_run(run, corpus, M, bones, tpa)
        if not r:
            print(f"[skip] {run}: no npz")
            continue
        allrows += r
        cols, dev = c, v
        print(f"  {run:<16} {corpus:<7} {len(r):>4} specimens")
    return allrows, cols, dev


def features(rows, cols, dev):
    """Mosimann log-shape-ratios of the LENGTHS, with the already-scale-free devs appended."""
    X = np.array([[r[c] for c in cols] for r in rows], dtype=np.float64)
    Z, size = ms.log_shape_ratios(X)
    D = np.array([[r[c] for c in dev] for r in rows], dtype=np.float64)
    D = (D - np.nanmean(D, 0)) / np.maximum(np.nanstd(D, 0), 1e-9)
    return np.hstack([Z, np.nan_to_num(D)]), Z, size, X


def select_features(cols, which):
    """Restrict to blocks that passed the ground-truth reliability test (see measure.CORE_BLOCKS).

    `all` keeps everything, which is reported alongside `core` everywhere rather than instead of
    it -- the point of having a selection rule is to show what it changes.
    """
    if which == "all":
        return list(cols)
    return [c for c in cols if block_of(c) in ms.CORE_BLOCKS]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker_runs", nargs="+", default=None)
    ap.add_argument("--clean_run", default="MORPH_CLEAN")
    ap.add_argument("--min_n", type=int, default=3, help="min specimens per genus for the signal test")
    ap.add_argument("--npc", type=int, default=10)
    ap.add_argument(
        "--feature_set",
        choices=["core", "all"],
        default="core",
        help="core = blocks that passed ground-truth reliability (measure.CORE_BLOCKS)",
    )
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    M = ms.load_model()
    bones = ms.bone_table(M)
    tpa = ms.template_part_axes(M)

    wr = args.worker_runs
    if wr is None:
        wr = sorted(
            (
                d
                for d in os.listdir(os.path.join(MOON, "runs"))
                if d.startswith("MORPH_W") and not d.endswith("_hier") and os.path.isdir(os.path.join(MOON, "runs", d))
            ),
            key=lambda s: int(s[7:]) if s[7:].isdigit() else 0,
        )
        if not wr:
            wr = ["D1_low_MERGED"]
    print("assembling:")
    runs = [(r, "worker") for r in wr] + [(args.clean_run, "clean")]
    rows, cols, dev = build_table(runs, M, bones, tpa)
    if not rows:
        raise SystemExit("no runs on disk")
    sc_all, asym = ms.symmetrise(rows, cols)
    sdev_all, _ = ms.symmetrise(rows, dev, key="asym_shapedev")
    sc = select_features(sc_all, args.feature_set)
    sdev = select_features(sdev_all, args.feature_set)
    F, Z, size, X = features(rows, sc, sdev)
    print(
        f"feature set '{args.feature_set}': {len(sc)} length + {len(sdev)} shape-dev "
        f"(of {len(sc_all)} + {len(sdev_all)} available)"
    )
    src = np.array([r["source"] for r in rows])
    gen = np.array([r["genus"] if r["genus"] else "?" for r in rows])
    spec = np.array([r["species"] if r["species"] else "?" for r in rows])
    sub = np.array([SUBFAMILY.get(g, "?") for g in gen])
    lot = np.array([accession_lot(r["specimen"]) for r in rows])
    print(
        f"\n{len(rows)} specimens ({(src == 'worker').sum()} worker, {(src == 'clean').sum()} clean), "
        f"{len(sc)} length + {len(sdev)} shape-dev features, {len(set(gen)) - ('?' in gen)} genera"
    )

    # ---------------- export THE deliverable: one tidy morphometric table
    csv_p = os.path.join(OUT, "morphometrics.csv")
    with open(csv_p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["label", "source", "genus", "species", "subfamily", "ecology", "asym_median", "log_size"] + sc + sdev
        )
        for i, r in enumerate(rows):
            w.writerow(
                [
                    r["label"],
                    r["source"],
                    r["genus"],
                    r["species"],
                    SUBFAMILY.get(gen[i], ""),
                    ECOLOGY.get(gen[i], ""),
                    # `asym[i]`, NOT r["asym_median"]. symmetrise() writes that field into every
                    # row, so the second call -- the one for the shape-dev columns -- overwrites
                    # it with mandible-only asymmetry. Reading the field here shipped 0.81 in the
                    # deliverable where the true value is ~0.05.
                    f"{asym[i]:.5f}",
                    f"{size[i]:.5f}",
                ]
                + [f"{r[c]:.6f}" for c in sc]
                + [f"{r[c]:.6f}" for c in sdev]
            )
    print(f"wrote {csv_p}")

    res = {}

    # ---------------- 1. signal, on WORKERS only (the corpus with replication)
    wmask = src == "worker"
    cnt = {g: int((gen[wmask] == g).sum()) for g in set(gen[wmask])}
    keep = wmask & np.array([cnt.get(g, 0) >= args.min_n and g != "?" for g in gen])
    Fk, gk = pcs(F[keep], args.npc), gen[keep]
    print(f"\n=== 1. TAXONOMIC SIGNAL  ({keep.sum()} workers, {len(set(gk))} genera with n>={args.min_n})")
    # Accuracy alone is NOT comparable across corpora: it falls as classes are added even when
    # the signal strengthens. The permutation null already encodes the class structure, so the
    # LIFT (accuracy / null) is the number to track between runs.
    a, m, s, p = perm_test(Fk, gk)
    print(
        f"  {'shape, leave-one-SPECIMEN-out':<40}{100 * a:>7.1f}%   null {100 * m:>5.1f}+-{100 * s:.1f}%"
        f"  p={p:.4f}   LIFT {a / max(m, 1e-9):.1f}x"
    )
    lk = lot[keep]
    a, m, s, p = perm_test(Fk, gk, groups=lk)
    print(
        f"  {'shape, leave-one-LOT-out (defensible)':<40}{100 * a:>7.1f}%   null {100 * m:>5.1f}+-{100 * s:.1f}%"
        f"  p={p:.4f}   LIFT {a / max(m, 1e-9):.1f}x"
    )
    n_mixed = sum(1 for x in set(lk) if len(set(gk[lk == x])) > 1)
    print(f"  ({len(set(lk))} accession lots, {n_mixed} containing >1 genus)")
    res["signal_genus"] = dict(acc=a, null=m, p=p, n=int(keep.sum()), n_genus=len(set(gk)), protocol="lot-blind")
    subk = sub[keep]
    if (subk != "?").sum() > 10:
        ok = subk != "?"
        a2, m2, _, p2 = perm_test(pcs(F[keep][ok], args.npc), subk[ok], groups=lot[keep][ok])
        print(
            f"  {'SUBFAMILY level (lot-blind)':<40}{100 * a2:>7.1f}%   null {100 * m2:>5.1f}%          "
            f"p={p2:.4f}   LIFT {a2 / max(m2, 1e-9):.1f}x"
        )
        res["signal_subfamily"] = dict(acc=a2, null=m2, p=p2, n=int(ok.sum()), n_class=len(set(subk[ok])))

    # ---------------- 2. confound controls
    gate = {}
    gp = os.path.join(MOON, "out", "fittability_gate.csv")
    if os.path.isfile(gp):
        gate = {r["name"]: float(r["radial_med"]) for r in csv.DictReader(open(gp))}
    rad = np.array([gate.get(r["label"], np.nan) for r in rows])
    Dv = np.array([[r[c] for c in sdev] for r in rows], dtype=np.float64)
    print("\n=== 2. CONFOUND CONTROLS — can a non-morphological variable do this too?")
    controls = [
        ("scan quality (radial_med) alone", rad[:, None]),
        ("L-R asymmetry alone", asym[:, None]),
        ("log-size alone", size[:, None]),
        ("shape-dev block alone", Dv),
        ("LENGTH log-ratios only (no shape-dev)", Z),
    ]
    for nm, V in controls:
        Vk = np.nan_to_num(V[keep])
        Fv = Vk if Vk.shape[1] == 1 else pcs(Vk, args.npc)
        aa, mm, _, pp = perm_test(Fv, gk)
        print(f"  {nm:<40}{100 * aa:>7.1f}%   null {100 * mm:>5.1f}%          p={pp:.4f}")
        res.setdefault("controls", {})[nm] = dict(acc=aa, null=mm, p=pp)
    N = np.nan_to_num(np.column_stack([asym, size, rad, Dv]))
    Nk = np.c_[np.ones(len(N)), (N - N.mean(0)) / np.maximum(N.std(0), 1e-9)]
    Zr = Z - Nk @ np.linalg.lstsq(Nk, Z, rcond=None)[0]
    aa, mm, _, pp = perm_test(pcs(Zr[keep], args.npc), gk)
    print(
        f"  {'shape RESIDUALISED on all of the above':<40}{100 * aa:>7.1f}%   null {100 * mm:>5.1f}%          p={pp:.4f}"
    )
    res["residualised"] = dict(acc=aa, null=mm, p=pp)

    # ---------------- 3. repeatability, per measurement
    print("\n=== 3. REPEATABILITY — between-species vs within-species variance (ICC)")
    sp_ok = wmask & (spec != "?")
    spc = {s_: int((spec[sp_ok] == s_).sum()) for s_ in set(spec[sp_ok])}
    rep_mask = sp_ok & np.array([spc.get(s_, 0) >= 2 for s_ in spec])
    iccs = {}
    if rep_mask.sum() > 10:
        allc = sc + sdev
        Fall = np.column_stack([Z, Dv])
        for k, c in enumerate(allc):
            iccs[c] = icc_oneway(Fall[rep_mask, k], spec[rep_mask])
        nrep = len(set(spec[rep_mask]))
        print(f"  {int(rep_mask.sum())} specimens in {nrep} species with >=2 individuals")
        order = sorted((c for c in iccs if np.isfinite(iccs[c])), key=lambda c: -iccs[c])
        print(f"  {'BEST 10 measurements':<26}{'ICC':>7}      {'WORST 6':<26}{'ICC':>7}")
        for i in range(10):
            lft = f"  {order[i]:<26}{iccs[order[i]]:>7.3f}"
            rgt = f"      {order[-(i + 1)]:<26}{iccs[order[-(i + 1)]]:>7.3f}" if i < 6 else ""
            print(lft + rgt)
        res["icc"] = iccs
        res["icc_median"] = float(np.nanmedian(list(iccs.values())))
        print(f"  median ICC across all {len(iccs)} measurements: {res['icc_median']:.3f}")

    # ---------------- 4. cross-corpus transfer
    print("\n=== 4. CROSS-CORPUS TRANSFER — ALL_ANTS_CLEAN specimen -> nearest WORKER genus centroid")
    cm = (src == "clean") & (gen != "?")
    shared = sorted(set(gen[cm]) & set(gen[wmask]))
    if shared:
        Fs = pcs(F, args.npc)
        # Mean centroid, with the median reported alongside as a robustness check.
        #
        # The Dolichoderus failure motivated looking at this: all three of its ALL_ANTS_CLEAN
        # specimens ranked last, yet each sits within |z|<1 of the worker corpus on every index.
        # The fault is the WORKER centroid -- built from 2 specimens, one with a scape index of
        # 1.99 against a corpus mean of 0.96, i.e. a bad antenna fit that a 2-point mean cannot
        # survive. The median is more robust in principle but on this data it only trades top-1
        # for top-3, so the plain mean stays primary and the real fix is more specimens per genus.
        C = np.array([Fs[wmask & (gen == g)].mean(0) for g in shared])
        cen_med = np.array([np.median(Fs[wmask & (gen == g)], axis=0) for g in shared])
        d_med = ((Fs[cm & np.array([g in shared for g in gen])][:, None, :] - cen_med[None, :, :]) ** 2).sum(-1)
        tm = cm & np.array([g in shared for g in gen])
        d = ((Fs[tm][:, None, :] - C[None, :, :]) ** 2).sum(-1)
        pred = np.array(shared)[d.argmin(1)]
        truth = gen[tm]
        hit = pred == truth
        rank = np.array([list(np.argsort(d[i])).index(shared.index(truth[i])) + 1 for i in range(len(truth))])
        chance = 1.0 / len(shared)
        print(f"  {len(shared)} genera present in BOTH corpora; {tm.sum()} clean specimens tested")
        print(
            f"  top-1 {100 * hit.mean():.1f}%   top-3 {100 * (rank <= 3).mean():.1f}%   "
            f"top-5 {100 * (rank <= 5).mean():.1f}%   chance {100 * chance:.1f}%"
        )
        print(f"  median rank of the TRUE genus: {int(np.median(rank))} of {len(shared)}")
        hit_med = (np.array(shared)[d_med.argmin(1)] == truth).mean()
        print(f"  (median centroid, robustness check: top-1 {100 * hit_med:.1f}%)")
        rng = np.random.default_rng(0)
        nl = np.array([(np.array(shared)[d.argmin(1)] == rng.permutation(truth)).mean() for _ in range(400)])
        print(f"  permutation null {100 * nl.mean():.1f}%  p={(nl >= hit.mean()).mean():.4f}")
        res["crosscorpus"] = dict(
            n=int(tm.sum()),
            n_genera=len(shared),
            top1=float(hit.mean()),
            top3=float((rank <= 3).mean()),
            chance=chance,
            p=float((nl >= hit.mean()).mean()),
            median_rank=float(np.median(rank)),
        )
        res["crosscorpus_hits"] = {truth[i]: int(rank[i]) for i in range(len(truth))}
        print(f"  {'genus':<18}{'rank of true genus':>20}")
        for i in np.argsort(rank):
            print(f"  {truth[i]:<18}{rank[i]:>14} / {len(shared)}   {'HIT' if hit[i] else ''}")
    else:
        print("  no shared genera yet (clean run missing?)")

    json.dump(res, open(os.path.join(OUT, "analysis.json"), "w"), indent=1, default=float)
    print(f"\nwrote {os.path.join(OUT, 'analysis.json')}")

    # ---------------- figures
    make_figures(F, Z, size, gen, sub, src, spec, sc, sdev, iccs, res, args)


def make_figures(F, Z, size, gen, sub, src, spec, sc, sdev, iccs, res, args):
    P = pcs(F, 6)
    wm = src == "worker"
    cnt = {g: int((gen[wm] == g).sum()) for g in set(gen[wm])}
    top = sorted((g for g in cnt if g != "?"), key=lambda g: -cnt[g])[:10]

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    ax[0].scatter(P[wm, 0], P[wm, 1], s=12, c="#cbd5e0", label="worker (all)", zorder=1)
    cmap = plt.get_cmap("tab10")
    for i, g in enumerate(top):
        m = wm & (gen == g)
        ax[0].scatter(P[m, 0], P[m, 1], s=34, color=cmap(i % 10), label=f"{g} ({cnt[g]})", zorder=3)
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    ax[0].set_title("Shape space, 10 largest genera", fontsize=10)
    ax[0].legend(fontsize=6.5, ncol=2)

    sm = wm & (sub != "?")
    subs = sorted(set(sub[sm]), key=lambda s_: -(sub[sm] == s_).sum())[:8]
    for i, s_ in enumerate(subs):
        m = sm & (sub == s_)
        ax[1].scatter(P[m, 0], P[m, 1], s=22, color=cmap(i % 10), label=f"{s_} ({m.sum()})", alpha=0.85)
    ax[1].set_xlabel("PC1")
    ax[1].set_ylabel("PC2")
    ax[1].set_title("...coloured by subfamily", fontsize=10)
    ax[1].legend(fontsize=6.5)

    if iccs:
        vals = [(c, iccs[c]) for c in sc + sdev if np.isfinite(iccs.get(c, np.nan))]
        vals.sort(key=lambda t: t[1])
        ax[2].barh(
            range(len(vals)),
            [v for _, v in vals],
            color=["#38a169" if v > 0.5 else "#dd6b20" if v > 0.25 else "#a0aec0" for _, v in vals],
        )
        ax[2].set_yticks(range(len(vals)))
        ax[2].set_yticklabels([c for c, _ in vals], fontsize=5.5)
        ax[2].axvline(0.5, ls="--", c="k", lw=0.8)
        ax[2].set_xlabel("ICC (between-species / total variance)")
        ax[2].set_title("Which measurements are repeatable?", fontsize=10)
    for a in ax[:2]:
        a.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT, "fig_shape_space.png")
    fig.savefig(p, dpi=125)
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
