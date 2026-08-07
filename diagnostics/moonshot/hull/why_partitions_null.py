"""Why every partition-shaped intervention has returned null — measured, not argued.

The convexity hierarchy is the seventh intervention on the data term's PARTITION to return
null (after: robust kernels, per-part robust scaling, geodesic branches, the frozen part
field, the soft partition, the anterior split). Seven nulls with one shape is not seven
coincidences, and this probe tests the obvious explanation directly.

THE HYPOTHESIS
A partition can only fix correspondence errors that CROSS a part boundary -- vertex v
landing on a different leg, or on the thorax instead of the head. It is structurally
incapable of fixing an error INSIDE a part, because every candidate landing site carries the
same label and the data term is therefore identical.

So: decompose E6's measured correspondence error into the two kinds.

    BETWEEN-PART   the fitted vertex landed in a different anatomical part from the one it
                   belongs to. A better partition could in principle fix these.
    WITHIN-PART    the fitted vertex landed in the CORRECT anatomical part but the wrong
                   place inside it. No partition can fix these, at any granularity, ever.

If within-part dominates, then the entire family of interventions -- including this one --
was bounded from the start, and the bound is measurable rather than a matter of opinion.
E6 already hints at it: the worst part is the gaster, "a large, smooth, nearly featureless
ellipsoid" where many correspondences give the same surface error. That is a within-part
failure by definition.

Ground truth is exact here: the synthetic targets ARE the model's own geometry, so fitted
vertex i is supposed to land on generated vertex i, and both carry known part labels.
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
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from fitter_3d.trainer_hierarchical import anatomical_groups  # noqa: E402
from pytorch3d.io import load_obj  # noqa: E402
from pytorch3d.ops import knn_points  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="SYN_clean")
    ap.add_argument("--corpus", default="synth_clean")
    ap.add_argument("--gt_groups", default="7", choices=["7", "13", "16"])
    args = ap.parse_args()

    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    jn = [str(x) for x in dd["J_names"]]
    sd, sa = args.gt_groups in ("13", "16"), args.gt_groups == "16"
    jg = anatomical_groups(jn, split_distal=sd, split_anterior=sa)
    gnames = sorted(set(jg.values()))
    n2i = {nm: i for i, nm in enumerate(gnames)}
    vpart = np.array([n2i[jg[int(d)]] for d in np.asarray(dd["weights"]).argmax(1)], dtype=np.int64)

    gt = np.load(os.path.join(MOON, args.corpus, "ground_truth.npz"))
    gtv = torch.tensor(gt["verts"], dtype=torch.float32, device=DEV)
    names = [str(x) for x in gt["names"]]

    p = os.path.join(MOON, "runs", args.run, "Stage_3_deform_fine.npz")
    if not os.path.isfile(p):
        p = os.path.join(MOON, "runs", args.run + "_hier", "H3_deform.npz")
    d = np.load(p)
    fit = torch.tensor(d["verts"], dtype=torch.float32, device=DEV)
    labels = [str(x) for x in d["labels"]]
    print(f"grouping: {len(gnames)} groups\nfit: {os.path.relpath(p, MOON)}  ({len(labels)} specimens)\n")

    vpart_t = torch.tensor(vpart, device=DEV)
    tot_between, tot_within, tot_correct, errs_w, errs_b = 0, 0, 0, [], []
    per_part = {g: [0, 0] for g in range(len(gnames))}

    for i, nm in enumerate(labels):
        stem = nm[:-4] if nm.endswith(".obj") else nm
        j = names.index(stem)
        ov, _, _ = load_obj(os.path.join(MOON, args.corpus, f"{stem}.obj"), load_textures=False)
        ov = ov.to(DEV)
        c = ov.mean(0)
        s = (ov - c).abs().max()
        g = (gtv[j] - c) / s
        f = fit[i]

        # nearest ground-truth vertex to where each fitted vertex landed
        nn = knn_points(f.unsqueeze(0), g.unsqueeze(0), K=1).idx[0, :, 0]
        correct = nn == torch.arange(g.shape[0], device=DEV)
        # the part it LANDED in, vs the part it BELONGS to
        landed_part = vpart_t[nn]
        own_part = vpart_t
        between = (~correct) & (landed_part != own_part)
        within = (~correct) & (landed_part == own_part)

        err = (f - g).norm(dim=-1)
        errs_w.append(err[within].cpu().numpy())
        errs_b.append(err[between].cpu().numpy())
        tot_correct += int(correct.sum())
        tot_between += int(between.sum())
        tot_within += int(within.sum())
        for gi in range(len(gnames)):
            m = own_part == gi
            per_part[gi][0] += int((within & m).sum())
            per_part[gi][1] += int((between & m).sum())

    n = tot_correct + tot_between + tot_within
    print("DECOMPOSITION OF CORRESPONDENCE ERROR")
    print(f"  exactly correct vertex          {tot_correct:>9d}  {100 * tot_correct / n:6.2f}%")
    print(
        f"  WITHIN-part error               {tot_within:>9d}  {100 * tot_within / n:6.2f}%   "
        f"<- no partition can fix these"
    )
    print(
        f"  BETWEEN-part error              {tot_between:>9d}  {100 * tot_between / n:6.2f}%   "
        f"<- the entire addressable set"
    )
    ew = np.concatenate(errs_w) if errs_w else np.zeros(0)
    eb = np.concatenate(errs_b) if errs_b else np.zeros(0)
    if len(ew):
        print(f"\n  median error, within-part       {100 * np.median(ew):6.2f}% of extent")
    if len(eb):
        print(f"  median error, between-part      {100 * np.median(eb):6.2f}% of extent")

    print(f"\n{'part':<12}{'within':>10}{'between':>10}{'% between':>11}")
    for gi, nm in enumerate(gnames):
        w, b = per_part[gi]
        if w + b == 0:
            continue
        print(f"{nm:<12}{w:>10d}{b:>10d}{100 * b / (w + b):>10.1f}%")

    import json

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", f"error_split_gt{args.gt_groups}.json"), "w") as fh:
        json.dump(
            dict(
                run=args.run,
                gt_groups=args.gt_groups,
                n_vertices=int(n),
                correct=int(tot_correct),
                within=int(tot_within),
                between=int(tot_between),
                median_err_within=float(np.median(ew)) if len(ew) else None,
                median_err_between=float(np.median(eb)) if len(eb) else None,
                per_part={gnames[g]: dict(within=v[0], between=v[1]) for g, v in per_part.items() if sum(v)},
            ),
            fh,
            indent=1,
        )

    addressable = tot_between / n
    print("\nCEILING ON EVERY PARTITION-SHAPED INTERVENTION")
    print("  A PERFECT partition -- one that never lets a vertex cross a part boundary --")
    print(f"  could correct at most {100 * addressable:.2f}% of vertices, and only if being in the")
    print("  right part were sufficient to land on the right vertex, which it is not")
    print(f"  (within-part error is already {100 * tot_within / n:.2f}% of vertices).")
    print(f"  Correspondence correctness would move from {100 * tot_correct / n:.2f}% to at most")
    print(f"  {100 * (tot_correct + tot_between) / n:.2f}% -- and that is a ceiling no method reaches.")


if __name__ == "__main__":
    main()
