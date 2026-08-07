"""M5 — the morphological findings: classic indices per genus, and the cross-corpus check.

WHY INDICES AND NOT RAW PCs
A principal component is a direction in a 43-dimensional space and means nothing to a
myrmecologist. The indices below are the ratios ant taxonomy actually uses, so every number here
can be checked against a published description -- which is the point. If the pipeline says
Odontomachus has the most elongate head of any genus measured, that is either right or wrong,
and someone can say which.

Each index is a ratio of two measurements taken from the SAME specimen, so the per-scan
normalisation cancels exactly and no absolute scale is needed (there is none available: see
measure.py). Ratios are also robust to the isometric part of fitting error -- if a fit is
uniformly 5% too large, every ratio is unchanged.

THE CROSS-CORPUS TABLE IS THE ONE THAT MATTERS
ALL_ANTS_CLEAN and the worker scans share 38 genera but nothing else -- different specimens,
preparation, scanners and mesh processing. So for a genus present in both, the two corpora give
two independent estimates of the same biological quantity. Agreement cannot come from a shared
pipeline artefact, because the artefacts are not shared. Disagreement localises the problem.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

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

try:
    from taxonomy import SUBFAMILY, ECOLOGY
except ImportError:
    SUBFAMILY, ECOLOGY = {}, {}

OUT = os.path.join(HERE, "out")

# (key, numerator, denominator, standard abbreviation, what a HIGH value means)
#
# NAMING, and why it is not the obvious naming. `seg_X` is the bone from joint X to its child,
# i.e. the segment whose PROXIMAL joint is X -- verified empirically in measure.bone_table().
# So the femur is `seg_l_3_fe`, the scape is `seg_an_1`. `attach_X` is a body-origin-to-X
# distance and locates an appendage rather than measuring one. `leaflen_X` is the extent of a
# LEAF structure's own vertex cloud, which is the only way to get its size: a leaf joint has no
# child, so no bone describes it. The mandible is the case that matters here -- the first
# version of this table used `b_h -> ma_r`, which is where the mandible ATTACHES, and duly
# reported Strumigenys as having the shortest mandibles of any genus measured.
INDICES = [
    ("cephalic_index", "head_wid", "head_len", "CI", "broad head"),
    ("head_flatness", "head_hei", "head_wid", "-", "deep (not flattened) head"),
    ("mandible_index", "leaflen_ma", "head_len", "MI", "long mandibles"),
    ("mandible_slenderness", "leaflen_ma", "mandible_wid", "-", "narrow blade-like mandible"),
    ("scape_index", "seg_an_1", "head_wid", "SI", "long scape"),
    ("funiculus_scape", "seg_an_2", "seg_an_1", "-", "long funiculus vs scape"),
    ("head_mesosoma", "head_len", "mesosoma_len", "-", "large head for body"),
    ("mesosoma_slenderness", "mesosoma_len", "mesosoma_wid", "-", "elongate mesosoma"),
    ("gaster_mesosoma", "gaster_len", "mesosoma_len", "-", "long gaster"),
    ("gaster_slenderness", "gaster_len", "gaster_wid", "-", "elongate gaster"),
    ("petiole_index", "seg_b_a_1", "mesosoma_len", "-", "long petiole node"),
    ("waist_constriction", "seg_b_a_1", "gaster_wid", "-", "narrow waist vs gaster"),
    ("hindfemur_index", "seg_l_3_fe", "mesosoma_len", "-", "long hind femur"),
    ("tibia_femur", "seg_l_3_ti", "seg_l_3_fe", "-", "long tibia vs femur"),
    ("foreleg_hindleg", "seg_l_1_fe", "seg_l_3_fe", "-", "fore and hind legs similar"),
]


def compute_indices(rows):
    for r in rows:
        for key, num, den, _, _ in INDICES:
            a, b = r.get(num), r.get(den)
            r[key] = float(a / b) if (a is not None and b is not None and b > 1e-12) else np.nan
    return [k for k, *_ in INDICES]


def collect(runs, M, bones, tpa):
    rows = []
    for run, corpus in runs:
        if not os.path.isdir(os.path.join(MOON, "runs", run)):
            continue
        r, c, v = ms.measure_run(run, corpus, M, bones, tpa)
        if r:
            ms.symmetrise(r, c)
            ms.symmetrise(r, v)
            rows += r
    return rows


def fmt(v, w=7, p=3):
    return f"{'-':>{w}}" if not np.isfinite(v) else f"{v:>{w}.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker_runs", nargs="+", default=None)
    ap.add_argument("--clean_run", default="MORPH_CLEAN")
    ap.add_argument("--min_n", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    M = ms.load_model()
    bones = ms.bone_table(M)
    tpa = ms.template_part_axes(M)
    wr = args.worker_runs
    if wr is None:
        wr = sorted(
            d for d in os.listdir(os.path.join(MOON, "runs")) if d.startswith("MORPH_W") and not d.endswith("_hier")
        )
        if not wr:
            wr = ["D1_low_MERGED"]
    rows = collect([(r, "worker") for r in wr] + [(args.clean_run, "clean")], M, bones, tpa)
    keys = compute_indices(rows)
    src = np.array([r["source"] for r in rows])
    gen = np.array([r["genus"] or "?" for r in rows])
    print(f"{len(rows)} specimens: {(src == 'worker').sum()} worker, {(src == 'clean').sum()} clean")

    W = src == "worker"
    cnt = Counter(gen[W])
    big = sorted((g for g in cnt if g != "?" and cnt[g] >= args.min_n), key=lambda g: -cnt[g])

    # ---------------- table 1: indices per genus
    print(f"\n=== TABLE 1 — morphometric indices by genus (worker corpus, n>={args.min_n})")
    hdr = f"{'genus':<16}{'n':>3}  " + "".join(f"{k[:9]:>10}" for k, *_ in INDICES[:7])
    print(hdr)
    print("-" * len(hdr))
    tab = {}
    for g in big:
        m = W & (gen == g)
        tab[g] = {k: float(np.nanmean([r[k] for r, mm in zip(rows, m) if mm])) for k in keys}
        print(f"{g:<16}{cnt[g]:>3}  " + "".join(fmt(tab[g][k], 10) for k, *_ in INDICES[:7]))
    print(f"\n{'genus':<16}{'n':>3}  " + "".join(f"{k[:9]:>10}" for k, *_ in INDICES[7:]))
    for g in big:
        print(f"{g:<16}{cnt[g]:>3}  " + "".join(fmt(tab[g][k], 10) for k, *_ in INDICES[7:]))

    # ---------------- table 2: the extremes, which is where the checkable claims live
    print("\n=== TABLE 2 — extremes: which genus is most/least, per index")
    print(f"{'index':<22}{'abbr':>5}  {'HIGHEST':<32}{'LOWEST':<32}{'high means'}")
    ext = {}
    for k, _n, _d, ab, meaning in INDICES:
        vals = [(g, tab[g][k]) for g in big if np.isfinite(tab[g][k])]
        if len(vals) < 3:
            continue
        vals.sort(key=lambda t: -t[1])
        hi, lo = vals[0], vals[-1]
        ext[k] = dict(high=hi[0], high_v=hi[1], low=lo[0], low_v=lo[1])
        print(f"{k:<22}{ab:>5}  {hi[0] + ' ' + f'{hi[1]:.3f}':<32}{lo[0] + ' ' + f'{lo[1]:.3f}':<32}{meaning}")

    # ---------------- table 3: cross-corpus agreement on shared genera
    C = src == "clean"
    shared = sorted(set(gen[C]) & set(gen[W]) - {"?"})
    print(f"\n=== TABLE 3 — CROSS-CORPUS: {len(shared)} genera measured in BOTH corpora")
    cross = {}
    if shared:
        print(f"{'index':<22}{'R':>7}{'n':>4}   {'mean |diff|':>12}   {'worker SD':>10}  {'|diff| / SD':>11}")
        for k in keys:
            xa, xb = [], []
            for g in shared:
                a = np.nanmean([r[k] for r, m in zip(rows, W & (gen == g)) if m])
                b = np.nanmean([r[k] for r, m in zip(rows, C & (gen == g)) if m])
                if np.isfinite(a) and np.isfinite(b):
                    xa.append(a)
                    xb.append(b)
            if len(xa) < 4:
                continue
            xa, xb = np.array(xa), np.array(xb)
            R = float(np.corrcoef(xa, xb)[0, 1])
            sd = float(np.nanstd([r[k] for r, m in zip(rows, W) if m]))
            md = float(np.mean(np.abs(xa - xb)))
            cross[k] = dict(R=R, n=len(xa), mean_abs_diff=md, worker_sd=sd, ratio=md / max(sd, 1e-12))
            print(f"{k:<22}{R:>7.3f}{len(xa):>4}   {md:>12.4f}   {sd:>10.4f}  {md / max(sd, 1e-12):>11.2f}")
        rs = [v["R"] for v in cross.values()]
        print(f"\n  median cross-corpus R over {len(cross)} indices: {np.median(rs):.3f}")
        print("  (R is across GENERA: does the pipeline rank genera the same way in two independent corpora?)")

    json.dump(
        dict(genus=tab, extremes=ext, crosscorpus=cross, n_worker=int(W.sum()), n_clean=int(C.sum()), shared=shared),
        open(os.path.join(OUT, "genus_table.json"), "w"),
        indent=1,
        default=float,
    )
    with open(os.path.join(OUT, "genus_indices.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["genus", "n", "subfamily", "ecology"] + keys)
        for g in big:
            w.writerow([g, cnt[g], SUBFAMILY.get(g, ""), ECOLOGY.get(g, "")] + [f"{tab[g][k]:.5f}" for k in keys])
    print(f"\nwrote {OUT}/genus_table.json and genus_indices.csv")

    make_figure(rows, keys, gen, src, big, cnt, tab, cross, shared)


def make_figure(rows, keys, gen, src, big, cnt, tab, cross, shared):
    W, C = src == "worker", src == "clean"
    ncol = 3
    show = [
        "cephalic_index",
        "mandible_index",
        "gaster_slenderness",
        "hindfemur_index",
        "scape_index",
        "mesosoma_slenderness",
    ]
    show = [k for k in show if k in keys]
    fig, ax = plt.subplots(2, ncol, figsize=(16, 8.5))
    ax = ax.ravel()
    for i, k in enumerate(show[: 2 * ncol]):
        vals = [(g, tab[g][k]) for g in big if np.isfinite(tab[g][k])]
        vals.sort(key=lambda t: t[1])
        y = np.arange(len(vals))
        ax[i].barh(y, [v for _, v in vals], color="#4a7fb5")
        for j, (g, _v) in enumerate(vals):
            pts = [r[k] for r, m in zip(rows, W & (gen == g)) if m]
            ax[i].plot(pts, [j] * len(pts), ".", ms=3, color="#1a202c", alpha=0.7, zorder=3)
        ax[i].set_yticks(y)
        ax[i].set_yticklabels([g for g, _ in vals], fontsize=6.5)
        ax[i].set_title(k.replace("_", " "), fontsize=10)
        ax[i].grid(alpha=0.3, axis="x")
    fig.suptitle("Morphometric indices by genus (bar = genus mean, dots = individual specimens)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OUT, "fig_genus_indices.png")
    fig.savefig(p, dpi=125)
    plt.close(fig)
    print(f"wrote {p}")

    if cross and shared:
        ks = [k for k in keys if k in cross]
        n = min(6, len(ks))
        fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
        ax = ax.ravel()
        for i, k in enumerate(ks[:n]):
            xa, xb, gs = [], [], []
            for g in shared:
                a = np.nanmean([r[k] for r, m in zip(rows, W & (gen == g)) if m])
                b = np.nanmean([r[k] for r, m in zip(rows, C & (gen == g)) if m])
                if np.isfinite(a) and np.isfinite(b):
                    xa.append(a)
                    xb.append(b)
                    gs.append(g)
            ax[i].scatter(xa, xb, s=26, color="#2b6cb0")
            lim = [min(xa + xb), max(xa + xb)]
            ax[i].plot(lim, lim, "--", c="k", lw=0.9)
            for x, y, g in zip(xa, xb, gs):
                ax[i].annotate(g[:9], (x, y), fontsize=5.5, alpha=0.75)
            ax[i].set_xlabel("worker corpus")
            ax[i].set_ylabel("ALL_ANTS_CLEAN")
            ax[i].set_title(f"{k.replace('_', ' ')}   R={cross[k]['R']:.2f}", fontsize=10)
            ax[i].grid(alpha=0.3)
        fig.suptitle(
            "Cross-corpus replication: same genus, two independent corpora (dashed = perfect agreement)", fontsize=12
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        p = os.path.join(OUT, "fig_crosscorpus.png")
        fig.savefig(p, dpi=125)
        plt.close(fig)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
