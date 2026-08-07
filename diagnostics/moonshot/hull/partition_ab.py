"""The A/B that decides whether E7 is worth running at all — no GPU training required.

`HullPartition` replaces the fitter's per-point assignment rule with a per-chunk majority
vote. Whether that can possibly help is answerable BEFORE any fitting, by applying both
rules to the SAME fit and scoring both against ground truth:

  INCUMBENT  `TargetPartition`: each target point takes the anatomical group of its nearest
             vertex on the current fitted mesh.
  PROPOSED   `HullPartition`: each target point takes the majority group of its own convex
             chunk, where the per-point groups feeding that vote come from the identical
             nearest-fitted-vertex rule.

Same fit, same target points, same ground truth, one variable: the rule. On the synthetic
corpus the ground-truth anatomical group of every target point is exact (targets ARE the
model's own geometry, verified byte-identical to ground_truth.npz).

WHAT EACH OUTCOME MEANS, fixed before looking:
  * proposed > incumbent   -- chunk voting repairs per-point assignment errors; E7 is worth
                              running and the mechanism is what is claimed.
  * proposed ~ incumbent   -- the chunks add nothing the fit did not already know.
  * proposed < incumbent   -- chunk voting DESTROYS correct per-point assignments by forcing
                              straddling chunks onto one group. Given that CoACD atoms were
                              measured to straddle 27-45% of the surface, this is the
                              outcome to expect, and it would mean E7 should not be run.

Scored at every stage checkpoint of an existing E6 fit, because the incumbent's accuracy
changes as the fit improves and the interesting question is whether chunks help EARLY (when
the fit is worst and a prior is most valuable) even if they hurt late.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MOON = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from decompose_corpus import cache_path, load_tree  # noqa: E402
from fitter_3d.trainer_hierarchical import anatomical_groups  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        return u.load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="SYN_clean_hier")
    ap.add_argument("--corpus", default="synth")
    ap.add_argument("--criteria", nargs="+", default=["volume", "visibility", "spectral"])
    ap.add_argument("--ks", nargs="+", type=int, default=[7, 13, 20, 30, 45])
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--n_points", type=int, default=8000)
    ap.add_argument("--gt_groups", default="7", choices=["7", "13", "16"])
    ap.add_argument("--stages", nargs="+", default=["H0_body", "H1_legs", "H2_joint", "H3_deform"])
    args = ap.parse_args()

    dd = load_model()
    jn = [str(x) for x in dd["J_names"]]
    sd, sa = args.gt_groups in ("13", "16"), args.gt_groups == "16"
    jg = anatomical_groups(jn, split_distal=sd, split_anterior=sa)
    gnames = sorted(set(jg.values()))
    n2i = {nm: i for i, nm in enumerate(gnames)}
    vgroup = np.array([n2i[jg[int(d)]] for d in np.asarray(dd["weights"]).argmax(1)], dtype=np.int64)
    print(f"grouping: {len(gnames)} groups {gnames}\n")

    gt_verts = np.load(os.path.join(MOON, "synth_clean", "ground_truth.npz"))["verts"]

    results = {}
    for stage in args.stages:
        p = os.path.join(MOON, "runs", args.run, f"{stage}.npz")
        if not os.path.isfile(p):
            print(f"[skip] {stage}: not on disk")
            continue
        d = np.load(p)
        fitted = d["verts"]  # (B, V, 3) in the fitter's target frame
        labels = [str(x) for x in d["labels"]]

        for crit in args.criteria:
            for k in args.ks:
                inc_acc, prop_acc, n_tot = [], [], 0
                for b, nm in enumerate(labels):
                    stem = nm[:-4] if nm.endswith(".obj") else nm
                    cp = cache_path(args.corpus, stem, crit, 0, args.threshold, args.n_points)
                    if not os.path.isfile(cp):
                        continue
                    tree, _ = load_tree(cp)
                    pts = tree.points  # decomposition's own samples, already in the fit frame

                    # ---- ground truth for those points
                    idx = int(stem.split("_")[1])
                    gv = gt_verts[idx]
                    # the loader normalises each target; the decomposition used the same
                    # normalisation, so put ground truth through it too
                    c = gv.mean(0)
                    s = np.abs(gv - c).max()
                    gvn = (gv - c) / s
                    gt = vgroup[cKDTree(gvn).query(pts, k=1)[1]]

                    # ---- INCUMBENT: nearest fitted vertex -> its group
                    per_point = vgroup[cKDTree(fitted[b]).query(pts, k=1)[1]]
                    inc_acc.append((per_point == gt).mean())

                    # ---- PROPOSED: majority group of each convex chunk
                    chunk = tree.cut(k)
                    tally = np.zeros((chunk.max() + 1, len(gnames)), dtype=np.int64)
                    np.add.at(tally, (chunk, per_point), 1)
                    voted = tally.argmax(1)[chunk]
                    prop_acc.append((voted == gt).mean())
                    n_tot += len(pts)

                if not inc_acc:
                    continue
                i_m, p_m = float(np.mean(inc_acc)), float(np.mean(prop_acc))
                wins = int(sum(1 for a, b_ in zip(prop_acc, inc_acc) if a > b_))
                results[(stage, crit, k)] = (i_m, p_m, wins, len(inc_acc))

    print(f"{'stage':<11}{'criterion':<11}{'k':>4}{'incumbent':>11}{'chunk-vote':>12}{'delta':>9}{'wins':>8}")
    print("-" * 68)
    for (stage, crit, k), (i_m, p_m, w, n) in results.items():
        print(f"{stage:<11}{crit:<11}{k:>4}{i_m:>11.4f}{p_m:>12.4f}{p_m - i_m:>+9.4f}{f'{w}/{n}':>8}")

    import json

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", f"partition_ab_gt{args.gt_groups}.json"), "w") as fh:
        json.dump(
            dict(
                args=vars(args),
                rows=[
                    dict(stage=s, criterion=c, k=k, incumbent=i_m, chunk_vote=p_m, wins=w, n=n)
                    for (s, c, k), (i_m, p_m, w, n) in results.items()
                ],
            ),
            fh,
            indent=1,
        )

    print("\nREADING")
    best = max(results.items(), key=lambda kv: kv[1][1] - kv[1][0]) if results else None
    if best:
        (stage, crit, k), (i_m, p_m, w, n) = best
        print(f"  best chunk-vote gain anywhere: {p_m - i_m:+.4f} at {stage}/{crit}/k={k} ({w}/{n} specimens)")
        if p_m - i_m <= 0:
            print("  Chunk voting does not improve the assignment ANYWHERE in the schedule.")
            print("  Running the fitter arm (E7) could then only test whether a WORSE partition")
            print("  happens to help for some other reason, which is not the claim being made.")


if __name__ == "__main__":
    main()
