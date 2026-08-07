"""E7 — does a CONVEXITY-based part hierarchy do what the geodesic decomposition could not?

Scores `fitter_3d/hull_decomposition.py` against the gates pre-registered in that module's
docstring, which are deliberately the gates that killed the geodesic method (probe 14).

  G1  REPRODUCIBILITY   two independent surface samplings must agree >= 90% after optimal
                        label matching. Geodesic scored 20%.
  G2  PART RECOVERY     on the synthetic corpus, where ground-truth anatomical labels are
                        exact, ARI >= 0.5 at k = n_true_parts, and >= 6 legs each dominated
                        by a distinct cluster.
  G3  TOUCHING LEGS     on the four bench50 specimens where probe 14 collapsed to <= 2
                        geodesic branches, >= 6 limb clusters must still separate.
  G4  COST              < 60 s/specimen on CPU.

THE CONTROL THAT MATTERS
------------------------
REPORT §7.8: an anatomy-free rule -- `label = x-tercile x sign(y)` -- BEAT the trained
PointNet++ part field on every unsupervised gate (98.65 vs 91.09 on reproducibility, 100.00
vs 87.81 on bilateral consistency). It paints the gaster as legs and still wins, because
those gates do not measure correctness. So the trivial rule is scored here alongside, on
every gate, from the start. A hull hierarchy that does not beat it has demonstrated nothing.

A second control, `random_atoms`, keeps the CoACD atoms but merges them in random adjacent
order. It isolates how much of any result comes from the convexity criterion versus from
CoACD's over-decomposition alone -- the same "is the mechanism doing the work?" question the
report's ablations kept turning on.

Outputs: hull/out/eval_<corpus>.json and hull/out/eval_summary.txt
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MOON = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

from fitter_3d.hull_decomposition import (  # noqa: E402
    assign_to_hulls,
    build_merge_tree,
    coacd_hulls,
    normalise,
    sample_surface,
)

OUT = os.path.join(HERE, "out")

# specimens where probe 14's geodesic sweep collapsed to <= 2 branches (out/probe14_*.json)
GEODESIC_COLLAPSED = [
    "Cephalotes_minutus_CASENT0709253",
    "Cephalotes_simillimus_CASENT0744408",
    "Dilobocondyla_fouqueti_CASENT0745576",
    "Dorylus_fulvus_CASENT0745678",
]


# ======================================================================================
# ground truth (synthetic corpus only)
# ======================================================================================
def template_part_labels(split_distal=False, split_anterior=False):
    """Per-template-vertex anatomical group, from dominant skinning weight. No annotation."""
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


def gt_labels_for_points(pts, gt_verts, vert_label):
    """Transfer exact per-vertex part labels to sampled surface points by nearest vertex.

    Valid because the synthetic targets ARE the model's own geometry: the .obj was verified
    byte-identical to ground_truth.npz, so a sample point's nearest vertex is its true owner.
    """
    _, j = cKDTree(gt_verts).query(pts, k=1)
    return vert_label[j]


# ======================================================================================
# metrics
# ======================================================================================
def matched_agreement(a, b):
    """Fraction of points on which two labellings agree after the best label permutation.

    Hungarian matching on the contingency table. Without it, two decompositions that carve
    identical parts but number them differently score ~0, which would make G1 meaningless.
    """
    a, b = np.asarray(a), np.asarray(b)
    ka, kb = a.max() + 1, b.max() + 1
    m = np.zeros((ka, kb), dtype=np.int64)
    np.add.at(m, (a, b), 1)
    r, c = linear_sum_assignment(-m)
    return float(m[r, c].sum() / len(a))


def adjusted_rand(a, b):
    """Adjusted Rand index -- permutation-invariant and chance-corrected, so it does not
    need the label sets to have the same size, unlike `matched_agreement`."""
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(a, b))


def legs_separated(labels, gt, leg_ids):
    """How many of the 6 legs are dominated by a cluster that dominates no other leg.

    This is the question the fitter actually needs answered -- not "is the partition good"
    but "can leg 2 be told apart from leg 3" -- and it is the claim the geodesic method
    failed. A leg whose majority cluster is shared with another leg is NOT separated.
    """
    dom = {}
    for lid in leg_ids:
        m = gt == lid
        if m.sum() < 10:
            continue
        dom[lid] = int(np.bincount(labels[m]).argmax())
    if not dom:
        return 0
    counts = {}
    for v in dom.values():
        counts[v] = counts.get(v, 0) + 1
    return int(sum(1 for v in dom.values() if counts[v] == 1))


def limb_clusters(labels, pts, min_frac=0.005, max_frac=0.25):
    """Clusters plausibly corresponding to limbs: neither debris nor the body.

    Used for G3, where there is no ground truth. Deliberately crude and stated as such: it
    counts clusters holding between 0.5% and 25% of the surface, which on an ant excludes
    scan debris below and the body above. It is a necessary condition, not a sufficient one.
    """
    n = len(labels)
    c = np.bincount(labels, minlength=labels.max() + 1)
    return int(((c >= min_frac * n) & (c <= max_frac * n)).sum())


# ======================================================================================
# baselines
# ======================================================================================
def trivial_rule(pts, k=7):
    """REPORT §7.8's anatomy-free control: x-tercile x sign(y), plus a body class.

    Paints the gaster as legs and is anatomically meaningless, yet beat the trained part
    field on every unsupervised gate. Any new method must clear it.
    """
    x, y = pts[:, 0], pts[:, 1]
    q = np.quantile(x, [1 / 3, 2 / 3])
    tercile = np.digitize(x, q)
    lab = tercile * 2 + (y > 0).astype(np.int64)
    if k > 6:
        # crude body class: the points nearest the principal axis
        r = np.linalg.norm(pts[:, 1:], axis=1)
        lab = np.where(r < np.quantile(r, 0.35), 6, lab)
    return lab.astype(np.int64)


def random_merge_tree(points, atom, n_atoms, seed=0):
    """Ablation: keep CoACD's atoms, merge them in random ADJACENT order.

    Isolates the convexity criterion from the over-decomposition. If this scores as well as
    the real tree, the atoms were doing all the work and the merge criterion is decoration.
    """
    from fitter_3d.hull_decomposition import Hierarchy, _bridge_components

    rng = np.random.default_rng(seed)
    members = [np.nonzero(atom == a)[0] for a in range(n_atoms)]
    keep = [a for a in range(n_atoms) if len(members[a]) >= 4]
    remap = {a: i for i, a in enumerate(keep)}
    aop = np.full(len(points), -1, dtype=np.int64)
    for a, i in remap.items():
        aop[members[a]] = i
    if (aop < 0).any():
        src = aop >= 0
        _, j = cKDTree(points[src]).query(points[~src], k=1)
        aop[~src] = aop[src][j]
    n = len(keep)
    pts_of = {i: points[aop == i][:800] for i in range(n)}
    d, _ = cKDTree(points).query(points, k=2)
    eps = 4.0 * float(np.median(d[:, 1]))
    trees = {i: cKDTree(pts_of[i]) for i in range(n)}
    nbr = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if trees[i].count_neighbors(trees[j], eps) > 0:
                nbr[i].add(j)
                nbr[j].add(i)
    _bridge_components(list(range(n)), nbr, pts_of)
    active, nxt, linkage, sizes = set(range(n)), n, [], {i: 1 for i in range(n)}
    while len(active) > 1:
        cand = [(a, b) for a in active for b in nbr[a] if a < b and b in active]
        if not cand:
            break
        a, b = cand[rng.integers(len(cand))]
        new = nxt
        nxt += 1
        linkage.append([a, b, float(len(linkage)), sizes[a] + sizes[b]])
        sizes[new] = sizes[a] + sizes[b]
        nbr[new] = (nbr[a] | nbr[b]) - {a, b}
        for o in nbr[new]:
            nbr[o] = (nbr[o] - {a, b}) | {new}
        active.discard(a)
        active.discard(b)
        active.add(new)
    return Hierarchy(np.asarray(linkage or np.zeros((0, 4))), aop, points, n)


# ======================================================================================
# per-specimen evaluation
# ======================================================================================
def decompose(mesh_v, mesh_f, n_points, thr, criterion, seed, mesh_obj=None):
    nm = trimesh.Trimesh(vertices=mesh_v, faces=mesh_f, process=False)
    pts, _ = sample_surface(nm, n_points, seed=seed)
    hulls = coacd_hulls(mesh_v, mesh_f, threshold=thr, seed=seed)
    atom, n_atoms = assign_to_hulls(pts, hulls)
    if criterion == "random":
        tree = random_merge_tree(pts, atom, n_atoms, seed=seed)
    else:
        tree = build_merge_tree(pts, atom, n_atoms, criterion=criterion, mesh=nm, seed=seed)
    return pts, tree, len(hulls)


K_GRID = [2, 3, 5, 7, 9, 11, 13, 16, 20, 25, 30, 40, 55]


def eval_specimen(path, criterion, args, gt_pack=None):
    """Score one specimen at the pre-registered k AND across the whole granularity range.

    Scoring a HIERARCHY at a single k is a category error, and the first run of this probe
    made it: at k=7 the convexity tree scored ARI 0.111 against the trivial rule's 0.269.
    The reason is anatomical, not algorithmic -- an ant's "body" is five separately convex
    blobs (head, mesosoma, petiole, postpetiole, gaster), so a 7-way cut spends its entire
    budget splitting the body and never reaches the legs. The fitter's `body + 6 legs`
    grouping is a SKINNING convention, not a convexity one. So the honest question for a
    method whose whole selling point is a granularity knob is: over all k, how well does it
    ever recover anatomy, and at what k.
    """
    m = trimesh.load(path, process=False, force="mesh")
    v, _, _ = normalise(np.asarray(m.vertices))
    f = np.asarray(m.faces)

    t0 = time.time()
    pts0, tree0, n_hulls = decompose(v, f, args.n_points, args.threshold, criterion, seed=0)
    t_dec = time.time() - t0
    lab0 = tree0.cut(args.k)

    # ---- G1: independent resampling, independently decomposed
    pts1, tree1, _ = decompose(v, f, args.n_points, args.threshold, criterion, seed=1)
    lab1 = tree1.cut(args.k)
    # compare on a common frozen probe set, so the two labellings describe the same points
    probe = pts0[:: max(1, len(pts0) // 4000)]
    g1 = matched_agreement(tree0.transfer(probe, labels=lab0), tree1.transfer(probe, labels=lab1))

    row = dict(
        name=os.path.basename(path)[:-4],
        criterion=criterion,
        n_hulls=n_hulls,
        n_atoms=int(tree0.n_atoms),
        seconds=round(t_dec, 2),
        g1_reproducibility=round(g1, 4),
        limb_clusters=limb_clusters(lab0, pts0),
        cluster_sizes=np.bincount(lab0, minlength=args.k).tolist(),
    )

    # ---- trivial control on the same points
    triv0 = trivial_rule(pts0, args.k)
    triv1 = trivial_rule(pts1, args.k)
    kt = cKDTree(pts1)
    row["g1_trivial"] = round(matched_agreement(triv0[:: max(1, len(pts0) // 4000)], triv1[kt.query(probe, k=1)[1]]), 4)

    # ---- G1 across the granularity range: a hierarchy is only usable if it is stable at
    # every k the fitter might cut it at, not just at one.
    row["g1_by_k"] = {
        str(k): round(
            matched_agreement(tree0.transfer(probe, labels=tree0.cut(k)), tree1.transfer(probe, labels=tree1.cut(k))), 4
        )
        for k in K_GRID
    }

    # ---- G2: ground truth, synthetic corpus only
    if gt_pack is not None:
        gt_verts, vert_label, leg_ids = gt_pack
        gt0 = gt_labels_for_points(pts0, gt_verts, vert_label)
        row["g2_ari"] = round(adjusted_rand(gt0, lab0), 4)
        row["g2_ari_trivial"] = round(adjusted_rand(gt0, triv0), 4)
        row["g2_legs_separated"] = legs_separated(lab0, gt0, leg_ids)
        row["g2_legs_separated_trivial"] = legs_separated(triv0, gt0, leg_ids)
        row["g2_matched_acc"] = round(matched_agreement(gt0, lab0), 4)

        # sweep the granularity knob
        ari_k, legs_k = {}, {}
        for k in K_GRID:
            lk = tree0.cut(k)
            ari_k[str(k)] = round(adjusted_rand(gt0, lk), 4)
            legs_k[str(k)] = legs_separated(lk, gt0, leg_ids)
        row["ari_by_k"] = ari_k
        row["legs_by_k"] = legs_k
        best_k = max(ari_k, key=lambda kk: ari_k[kk])
        row["best_k"] = int(best_k)
        row["best_ari"] = ari_k[best_k]
        row["max_legs"] = max(legs_k.values())
        row["k_all_legs"] = min((int(k) for k, vv in legs_k.items() if vv >= 6), default=-1)
        # the trivial rule has no granularity knob, so give it the same sweep via k-means on
        # its own label space? No -- it is defined at k=7. Compare best-vs-best honestly and
        # say so: the hierarchy is allowed its knob, the control is reported at its only k.
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="synth", choices=["synth", "worker", "clean", "collapsed"])
    ap.add_argument("--criteria", nargs="+", default=["volume", "concavity", "hybrid", "visibility", "random"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--n_points", type=int, default=8000)
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if args.corpus == "synth":
        files = sorted(glob.glob(os.path.join(MOON, "synth_clean", "*.obj")))[: args.n]
        vert_label, names = template_part_labels(split_distal=False, split_anterior=False)
        gt_verts = np.load(os.path.join(MOON, "synth_clean", "ground_truth.npz"))["verts"]
        leg_ids = [i for i, nm in enumerate(names) if nm.startswith("l")]
        print(f"ground-truth groups (k={len(names)}): {names}")
    elif args.corpus == "clean":
        files = sorted(glob.glob(os.path.join(MOON, "clean81", "*.obj")))[: args.n]
    elif args.corpus == "collapsed":
        files = [
            p
            for p in sorted(glob.glob(os.path.join(MOON, "bench50", "*.obj")))
            if any(c in p for c in GEODESIC_COLLAPSED)
        ]
    else:
        files = sorted(glob.glob(os.path.join(MOON, "bench50", "*.obj")))[: args.n]

    print(f"E7 hull decomposition — corpus={args.corpus}  n={len(files)}  k={args.k}  criteria={args.criteria}\n")

    rows = []
    for crit in args.criteria:
        for i, p in enumerate(files):
            gp = None
            if args.corpus == "synth":
                idx = int(os.path.basename(p)[6:-4])
                gp = (gt_verts[idx], vert_label, leg_ids)
            try:
                r = eval_specimen(p, crit, args, gp)
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"  [{crit}] {os.path.basename(p)} FAILED: {e}", flush=True)
                continue
            rows.append(r)
            extra = ""
            if "g2_ari" in r:
                extra = (
                    f"  ARI={r['g2_ari']:+.3f} (triv {r['g2_ari_trivial']:+.3f})"
                    f"  legs={r['g2_legs_separated']}/6 (triv {r['g2_legs_separated_trivial']})"
                )
            print(
                f"  [{crit:10s}] {r['name'][:34]:<34} G1={r['g1_reproducibility']:.3f} "
                f"(triv {r['g1_trivial']:.3f})  limbs={r['limb_clusters']}  {r['seconds']:5.1f}s{extra}",
                flush=True,
            )

    # ---------------- summary ----------------
    print("\n" + "=" * 100)
    print(
        f"{'criterion':<12}{'G1 repro':>10}{'G1 triv':>9}{'G2 ARI':>9}{'ARI triv':>10}{'legs/6':>8}{'legs triv':>11}{'limbs':>7}{'sec':>7}"
    )
    print("=" * 100)
    summary = {}
    for crit in args.criteria:
        rr = [r for r in rows if r["criterion"] == crit]
        if not rr:
            continue

        def mean(key, default=np.nan):
            vals = [r[key] for r in rr if key in r]
            return float(np.mean(vals)) if vals else default

        s = dict(
            g1=mean("g1_reproducibility"),
            g1_triv=mean("g1_trivial"),
            ari=mean("g2_ari"),
            ari_triv=mean("g2_ari_trivial"),
            legs=mean("g2_legs_separated"),
            legs_triv=mean("g2_legs_separated_trivial"),
            limbs=mean("limb_clusters"),
            sec=mean("seconds"),
            best_ari=mean("best_ari"),
            best_k=mean("best_k"),
            max_legs=mean("max_legs"),
            k_all_legs=mean("k_all_legs"),
            n=len(rr),
        )
        # per-k curves, averaged over specimens
        for key in ("ari_by_k", "legs_by_k", "g1_by_k"):
            have = [r[key] for r in rr if key in r]
            if have:
                s[key] = {k: round(float(np.mean([h[k] for h in have])), 4) for k in have[0]}
        summary[crit] = s
        print(
            f"{crit:<12}{s['g1']:>10.3f}{s['g1_triv']:>9.3f}{s['ari']:>9.3f}{s['ari_triv']:>10.3f}"
            f"{s['legs']:>8.2f}{s['legs_triv']:>11.2f}{s['limbs']:>7.1f}{s['sec']:>7.1f}"
        )

    # ---- the granularity sweep, which is the actual question for a hierarchy
    any_gt = any("ari_by_k" in s for s in summary.values())
    if any_gt:
        print("\nGRANULARITY SWEEP — adjusted Rand index against ground-truth parts, by cut size k")
        print(
            f"{'criterion':<12}" + "".join(f"{k:>7}" for k in K_GRID) + f"{'best':>8}{'@k':>5}{'legs':>6}{'k=6legs':>9}"
        )
        for crit, s in summary.items():
            if "ari_by_k" not in s:
                continue
            print(
                f"{crit:<12}"
                + "".join(f"{s['ari_by_k'][str(k)]:>7.3f}" for k in K_GRID)
                + f"{s['best_ari']:>8.3f}{s['best_k']:>5.0f}{s['max_legs']:>6.1f}{s['k_all_legs']:>9.1f}"
            )
        print("\nLEGS SEPARATED (of 6) by cut size k")
        print(f"{'criterion':<12}" + "".join(f"{k:>7}" for k in K_GRID))
        for crit, s in summary.items():
            if "legs_by_k" not in s:
                continue
            print(f"{crit:<12}" + "".join(f"{s['legs_by_k'][str(k)]:>7.2f}" for k in K_GRID))
        tv = [r["g2_ari_trivial"] for r in rows if "g2_ari_trivial" in r]
        tl = [r["g2_legs_separated_trivial"] for r in rows if "g2_legs_separated_trivial" in r]
        if tv:
            print(f"\n  trivial control (defined only at k=7): ARI {np.mean(tv):.3f}, legs {np.mean(tl):.2f}/6")

    print("\nG1 REPRODUCIBILITY by cut size k  (geodesic method scored 0.20; gate is 0.90)")
    print(f"{'criterion':<12}" + "".join(f"{k:>7}" for k in K_GRID))
    for crit, s in summary.items():
        if "g1_by_k" in s:
            print(f"{crit:<12}" + "".join(f"{s['g1_by_k'][str(k)]:>7.3f}" for k in K_GRID))

    print("\nGATES (pre-registered in fitter_3d/hull_decomposition.py)")
    for crit, s in summary.items():
        has_gt = not np.isnan(s["ari"])
        g1 = "PASS" if s["g1"] >= 0.90 else "FAIL"
        g2 = f"{'PASS' if s['ari'] >= 0.5 else 'FAIL'} ({s['ari']:.3f})" if has_gt else "n/a"
        g4 = "PASS" if s["sec"] < 60 else "FAIL"
        beats = (s["g1"] > s["g1_triv"]) and (not has_gt or s["ari"] > s["ari_triv"])
        print(
            f"  {crit:<12} G1 {g1} ({s['g1']:.3f})   G2 {g2}   "
            f"G4 {g4} ({s['sec']:.1f}s)   beats trivial control: {'YES' if beats else 'NO'}"
        )

    tag = args.tag or args.corpus
    with open(os.path.join(OUT, f"eval_{tag}.json"), "w") as fh:
        json.dump(dict(args=vars(args), rows=rows, summary=summary), fh, indent=1)
    print(f"\nwrote {os.path.join(OUT, f'eval_{tag}.json')}")


if __name__ == "__main__":
    main()
