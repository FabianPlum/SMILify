"""Score cached decompositions. Reads hull/cache, computes nothing geometric.

THE METRIC THAT MATTERS, AND WHY IT IS NOT ARI
----------------------------------------------
The first pass of this evaluation scored adjusted Rand index against the fitter's seven
anatomical groups (body + 6 legs) and the convexity tree lost to a trivial x-tercile rule.
That was the wrong question, for a reason worth stating plainly:

ARI asks "are the discovered parts THE anatomical parts". `HullPartition` does not need
that. It uses chunks as VOTING UNITS -- each chunk is assigned wholesale to whichever
anatomical group its own points currently sit nearest. For that to work a chunk needs only
to be PURE (lie within one anatomical group); it does not need to BE one. Over-segmentation
is harmless, straddling is fatal. ARI punishes exactly the harmless failure and rewards
matching a grouping (`body` = head + thorax + petiole + gaster) that is a skinning
convention, not a convexity one.

So the primary measure here is **purity**: the fraction of points whose ground-truth part
equals the majority ground-truth part of their own chunk. That is exactly the ceiling on
what a chunk-voting partition can achieve if the naming step were an oracle.

THE CONTROL PURITY NEEDS
------------------------
Purity rises monotonically with k for ANY spatial partition -- at k = n_points it is 1.0 by
construction, and that is meaningless. So every purity figure here is reported beside
**k-means on the same points at the same k**, which is the cheapest possible spatial
clustering. If the convexity tree does not beat k-means at equal k, the convexity criterion
is decoration and the atoms (or mere spatial locality) are doing the work.

Two further controls, both in the same table:
  `random`  -- CoACD atoms merged in random adjacent order. Isolates the merge CRITERION
               from the over-decomposition.
  `trivial` -- REPORT §7.8's anatomy-free x-tercile x sign(y) rule, which beat the trained
               part field on every unsupervised gate. Defined only at k=7.

Outputs: hull/out/score_<corpus>.json and a printed table.
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MOON = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

from decompose_corpus import CACHE, load_tree  # noqa: E402

K_GRID = [2, 3, 5, 7, 9, 13, 20, 30, 45, 65]
OUT = os.path.join(HERE, "out")


# ======================================================================================
# ground truth
# ======================================================================================
def template_part_labels(split_distal=False, split_anterior=False):
    from fitter_3d.trainer_hierarchical import anatomical_groups

    import config

    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    jn = [str(x) for x in dd["J_names"]]
    jg = anatomical_groups(jn, split_distal=split_distal, split_anterior=split_anterior)
    names = sorted(set(jg.values()))
    n2i = {nm: i for i, nm in enumerate(names)}
    dom = np.asarray(dd["weights"]).argmax(1)
    return np.array([n2i[jg[int(d)]] for d in dom], dtype=np.int64), names


# ======================================================================================
# metrics
# ======================================================================================
def purity(labels, gt):
    """Fraction of points agreeing with the majority ground-truth part of their own cluster.

    The oracle ceiling for a chunk-voting partition: if every chunk were named correctly,
    this is the fraction of target points that would land in the right anatomical group.
    """
    labels = np.asarray(labels)
    k = labels.max() + 1
    g = gt.max() + 1
    tally = np.zeros((k, g), dtype=np.int64)
    np.add.at(tally, (labels, gt), 1)
    return float(tally.max(1).sum() / len(labels))


def matched_agreement(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(m, (a, b), 1)
    r, c = linear_sum_assignment(-m)
    return float(m[r, c].sum() / len(a))


def adjusted_rand(a, b):
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(a, b))


def legs_separated(labels, gt, leg_ids):
    """Legs whose dominant cluster dominates no other leg. The fitter's real question."""
    dom = {}
    for lid in leg_ids:
        m = gt == lid
        if m.sum() >= 10:
            dom[lid] = int(np.bincount(labels[m]).argmax())
    counts = {}
    for v in dom.values():
        counts[v] = counts.get(v, 0) + 1
    return int(sum(1 for v in dom.values() if counts[v] == 1))


def straddle_frac(labels, gt, thresh=0.9):
    """Fraction of SURFACE held by chunks that are not clean -- majority below `thresh`.

    The failure mode purity averages away: one big chunk spanning a leg and the thorax can
    leave purity high while making that leg unassignable. This reports the mass at risk.
    """
    labels = np.asarray(labels)
    k = labels.max() + 1
    tally = np.zeros((k, gt.max() + 1), dtype=np.int64)
    np.add.at(tally, (labels, gt), 1)
    tot = tally.sum(1)
    frac = tally.max(1) / np.maximum(tot, 1)
    return float(tot[frac < thresh].sum() / len(labels))


# ======================================================================================
# controls
# ======================================================================================
def kmeans_labels(pts, k, seed=0):
    """The control every purity number needs: plain spatial k-means at the same k."""
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(pts).astype(np.int64)


def trivial_rule(pts):
    """REPORT §7.8's anatomy-free control: x-tercile x sign(y), plus a crude body class."""
    x, y = pts[:, 0], pts[:, 1]
    q = np.quantile(x, [1 / 3, 2 / 3])
    lab = np.digitize(x, q) * 2 + (y > 0).astype(np.int64)
    r = np.linalg.norm(pts[:, 1:], axis=1)
    return np.where(r < np.quantile(r, 0.35), 6, lab).astype(np.int64)


# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="synth")
    ap.add_argument("--criteria", nargs="+", default=["volume", "concavity", "hybrid", "visibility"])
    ap.add_argument("--controls", nargs="+", default=["kmeans"])
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--n_points", type=int, default=8000)
    ap.add_argument(
        "--gt_groups",
        default="7",
        choices=["7", "13", "16"],
        help="which ground-truth grouping to score against. 7 = the fitter's body+6legs, which "
        "lumps head, thorax, petiole and gaster into ONE 'body' class -- a skinning convention "
        "that a convexity method has no reason to reproduce, since those four are separately "
        "convex. 16 = split_distal+split_anterior, the anatomically-resolved grouping, which is "
        "the fair target for a convexity criterion. Scoring both is the honest comparison.",
    )
    args = ap.parse_args()

    cdir = os.path.join(CACHE, f"{args.corpus}_thr{args.threshold}_n{args.n_points}")
    if not os.path.isdir(cdir):
        raise SystemExit(f"no cache at {cdir} -- run decompose_corpus.py first")

    has_gt = args.corpus in ("synth", "synth_noisy")
    gt_verts = vert_label = leg_ids = None
    if has_gt:
        sub = "synth_clean" if args.corpus == "synth" else "synth_noisy"
        sd = args.gt_groups in ("13", "16")
        sa = args.gt_groups == "16"
        vert_label, names = template_part_labels(split_distal=sd, split_anterior=sa)
        gt_verts = np.load(os.path.join(MOON, sub, "ground_truth.npz"))["verts"]
        leg_ids = [i for i, nm in enumerate(names) if nm.startswith("l")]
        print(f"ground truth: {len(names)} groups {names}")

    specimens = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(os.path.join(cdir, "*.npz"))})
    print(f"corpus={args.corpus}  {len(specimens)} specimens  cache={cdir}\n")

    rows = []
    for crit in list(args.criteria) + list(args.controls):
        for name in specimens:
            p0 = os.path.join(cdir, f"{name}__{crit if crit not in args.controls else args.criteria[0]}__s0.npz")
            p1 = os.path.join(cdir, f"{name}__{crit if crit not in args.controls else args.criteria[0]}__s1.npz")
            if not os.path.isfile(p0):
                continue
            t0, d0 = load_tree(p0)
            t1, d1 = load_tree(p1) if os.path.isfile(p1) else (None, None)
            pts0 = t0.points

            gt0 = None
            if has_gt:
                idx = int(name.split("_")[1])
                _, j = cKDTree(gt_verts[idx]).query(pts0, k=1)
                gt0 = vert_label[j]

            probe = pts0[:: max(1, len(pts0) // 4000)]

            r = dict(
                name=name,
                criterion=crit,
                n_hulls=int(d0["n_hulls"]),
                n_atoms=int(d0["n_atoms"]),
                seconds=float(d0["seconds"]),
                by_k={},
            )
            # THE CEILING. Every merge tree over these atoms is bounded by how pure the atoms
            # themselves are: a CoACD atom that straddles a leg and the thorax can never be
            # un-straddled by merging, only made worse. If atom purity is already low, the
            # merge criterion is not what is limiting the method -- the over-decomposition is.
            if has_gt and crit != "kmeans":
                r["atom_purity"] = round(purity(t0.atom_of_point, gt0), 4)
                r["atom_straddle"] = round(straddle_frac(t0.atom_of_point, gt0), 4)
                r["n_atoms_used"] = int(t0.atom_of_point.max() + 1)
            for k in K_GRID:
                if crit == "kmeans":
                    lab0 = kmeans_labels(pts0, k, seed=0)
                    lab1 = kmeans_labels(t1.points, k, seed=1) if t1 is not None else None
                else:
                    lab0 = t0.cut(k)
                    lab1 = t1.cut(k) if t1 is not None else None
                e = {}
                if has_gt:
                    e["purity"] = round(purity(lab0, gt0), 4)
                    e["ari"] = round(adjusted_rand(gt0, lab0), 4)
                    e["legs"] = legs_separated(lab0, gt0, leg_ids)
                    e["straddle"] = round(straddle_frac(lab0, gt0), 4)
                if lab1 is not None:
                    if crit == "kmeans":
                        a = lab0[:: max(1, len(pts0) // 4000)]
                        b = lab1[cKDTree(t1.points).query(probe, k=1)[1]]
                    else:
                        a = t0.transfer(probe, labels=lab0)
                        b = t1.transfer(probe, labels=lab1)
                    e["g1"] = round(matched_agreement(a, b), 4)
                # cluster-size balance: 1.0 = perfectly even, ->0 = one blob plus debris
                c = np.bincount(lab0, minlength=k).astype(float)
                c = c[c > 0] / c.sum()
                e["balance"] = round(float(np.exp(-(c * np.log(c)).sum()) / k), 4)
                r["by_k"][str(k)] = e
            rows.append(r)

            if has_gt:
                tr = trivial_rule(pts0)
                rows.append(
                    dict(
                        name=name,
                        criterion="trivial",
                        n_hulls=0,
                        n_atoms=7,
                        seconds=0.0,
                        by_k={
                            "7": dict(
                                purity=round(purity(tr, gt0), 4),
                                ari=round(adjusted_rand(gt0, tr), 4),
                                legs=legs_separated(tr, gt0, leg_ids),
                                straddle=round(straddle_frac(tr, gt0), 4),
                                balance=1.0,
                            )
                        },
                    )
                )
            print(f"  scored {crit:11s} {name[:40]}", flush=True)

    # ------------------------------------------------------------------ tables
    def agg(crit, k, key):
        v = [
            r["by_k"][str(k)][key]
            for r in rows
            if r["criterion"] == crit and str(k) in r["by_k"] and key in r["by_k"][str(k)]
        ]
        return float(np.mean(v)) if v else np.nan

    all_crit = list(dict.fromkeys(r["criterion"] for r in rows))
    print("\n" + "=" * 104)
    print(f"CORPUS {args.corpus}   n={len(specimens)}")
    print("=" * 104)

    for key, title, gate in [
        ("g1", "G1 REPRODUCIBILITY across two independent surface samplings (geodesic scored 0.20; gate 0.90)", 0.90),
        ("purity", "PURITY vs ground-truth parts -- the ceiling for a chunk-voting partition", None),
        ("straddle", "STRADDLE: surface fraction in chunks that span >1 anatomical part (lower is better)", None),
        ("ari", "ADJUSTED RAND vs the fitter's 7 groups (reported, but see the module docstring)", None),
        ("legs", "LEGS SEPARATED of 6", None),
        ("balance", "CLUSTER BALANCE (1.0 = even; ->0 = one blob plus debris)", None),
    ]:
        vals = {c: [agg(c, k, key) for k in K_GRID] for c in all_crit}
        if all(np.all(np.isnan(v)) for v in vals.values()):
            continue
        print(f"\n{title}")
        print(f"{'method':<12}" + "".join(f"{k:>8}" for k in K_GRID))
        for c in all_crit:
            v = vals[c]
            if np.all(np.isnan(v)):
                continue
            print(f"{c:<12}" + "".join("       -" if np.isnan(x) else f"{x:>8.3f}" for x in v))
        if gate:
            for c in all_crit:
                v = [x for x in vals[c] if not np.isnan(x)]
                if v:
                    print(f"  {c:<10} best {max(v):.3f} -> gate {'PASS' if max(v) >= gate else 'FAIL'}")

    # ---- the ceiling set by the atoms themselves
    ap_rows = [
        (r["criterion"], r["atom_purity"], r["atom_straddle"], r["n_atoms_used"]) for r in rows if "atom_purity" in r
    ]
    if ap_rows:
        print("\nATOM-LEVEL CEILING — purity of the CoACD over-decomposition BEFORE any merging")
        print("(a merge tree can only lose purity relative to this; it bounds every row above)")
        seen = {}
        for c, p_, s_, na in ap_rows:
            seen.setdefault(c, []).append((p_, s_, na))
        for c, v in seen.items():
            a = np.array([x[:2] for x in v])
            print(
                f"  {c:<12} atoms={np.mean([x[2] for x in v]):5.0f}  purity={a[:, 0].mean():.4f}  straddle={a[:, 1].mean():.4f}"
            )

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"score_{args.corpus}_gt{args.gt_groups}.json"), "w") as fh:
        json.dump(dict(args=vars(args), rows=rows), fh, indent=1)
    print(f"\nwrote score_{args.corpus}_gt{args.gt_groups}.json")


if __name__ == "__main__":
    main()
