"""PROBE 16 — WHERE does the part field predict debris, and is excluding it safe?

MOTIVATION (found by looking at renders, not at numbers)
`partfield_vs_fit.png` shows magenta -- the debris class -- speckled through the THORAX, in
the region where the legs meet the body, on most specimens. That is not where debris is.
Debris is detached fragments and mounting medium, which sit outside the animal.

The suspicion is that this is an artefact of my own augmentation design. `augment()` adds
"debris bridges": filaments of points spanning two surface points, put there to imitate the
mounting medium that makes real legs appear fused. Bridges between a leg and the body land
exactly at the leg-body junction, so the network may have learned "clutter near the coxa =
debris" rather than "material that is not the animal".

WHY IT MATTERS RATHER THAN BEING A CURIOSITY
`PartFieldPartition` maps debris to group -1, which removes those points from EVERY data
term. If debris is being over-predicted at the coxae, the fit is being blinded precisely at
the joint whose placement the entire hierarchical schedule is built around (H0 exists to
place the six coxae). A part-field arm could then lose for a reason that has nothing to do
with the quality of the partition.

WHAT THIS MEASURES
  1. predicted debris fraction per specimen, and how it compares to the fit-transferred
     estimate from make_partfield_data (12.4% mean).
  2. for each predicted-debris point, its distance to the nearest predicted BODY point and
     to the nearest predicted LEG point, normalised by specimen size. Genuine debris should
     be FAR from both. Junction artefacts sit close to both.
  3. the "interiority" of debris: fraction of debris points that lie INSIDE the convex hull
     of the non-debris points. Real debris is mostly outside; a junction artefact is inside.
  4. how much of the coxal region (points near the body/leg boundary) is lost to exclusion.

Pre-registered reading, fixed before running:
  * if >50% of predicted debris is interior to the specimen, the class is mislearned and
    debris must NOT be excluded from the data term -- it should be reassigned to its nearest
    non-debris class instead.
  * if <20% is interior, exclusion is safe and the class is doing its job.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from fitter_3d.utils import load_meshes  # noqa: E402
from fitter_3d.partfield import PartFieldNet, predict_field, normalise  # noqa: E402
from pytorch3d.ops import knn_points, sample_points_from_meshes  # noqa: E402

OUT = os.path.join(HERE, "out")
DATA = os.path.join(HERE, "partfield")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=os.path.join(DATA, "net_B_mirror.pt"))
    ap.add_argument("--mesh_dir", default=os.path.join(HERE, "bench50"))
    ap.add_argument("--n_spec", type=int, default=20)
    ap.add_argument("--n_pts", type=int, default=20000)
    ap.add_argument("--tag", default="B_mirror")
    args = ap.parse_args()

    dev = torch.device("cuda")
    ck = torch.load(args.net, map_location=dev)
    names = ck["names"]
    net = PartFieldNet(n_classes=len(names), width=ck.get("width", 1.0)).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    dbg_id = names.index("debris")
    body_id = names.index("body")
    leg_ids = [i for i, n in enumerate(names) if n != "debris" and n != "body"]

    files = sorted(f for f in os.listdir(args.mesh_dir) if f.endswith(".obj"))
    _, meshes = load_meshes(mesh_dir=args.mesh_dir, sorting=sorted, device=str(dev))
    idx = np.linspace(0, len(files) - 1, min(args.n_spec, len(files))).astype(int)

    from scipy.spatial import Delaunay

    rows = []
    for k, i in enumerate(idx):
        p = sample_points_from_meshes(meshes[int(i)], args.n_pts)[0]
        pn, _, _ = normalise(p)
        lab = predict_field(net, pn, seed=0).argmax(-1)
        d = lab == dbg_id
        nd = ~d
        frac = float(d.float().mean())
        row = dict(name=files[i].replace("_processed.obj", ""), debris_frac=frac)

        if int(d.sum()) >= 20 and int(nd.sum()) >= 100:
            dp = pn[d]
            # distance from each debris point to nearest body / nearest leg point
            bmask = lab == body_id
            lmask = torch.isin(lab, torch.tensor(leg_ids, device=dev))
            for tag, m in (("body", bmask), ("leg", lmask)):
                if int(m.sum()) >= 10:
                    dist = knn_points(dp.unsqueeze(0), pn[m].unsqueeze(0), K=1).dists[0, :, 0].sqrt()
                    row[f"d_to_{tag}_median"] = float(dist.median())
                else:
                    row[f"d_to_{tag}_median"] = float("nan")

            # interiority: is the debris point inside the convex hull of the animal?
            # subsample for the hull, it only needs the shape not every point
            nds = pn[nd].cpu().numpy()
            sub = nds[np.random.default_rng(0).choice(len(nds), min(3000, len(nds)), replace=False)]
            try:
                hull = Delaunay(sub)
                inside = hull.find_simplex(dp.cpu().numpy()) >= 0
                row["interior_frac"] = float(inside.mean())
            except Exception as e:  # degenerate hull
                row["interior_frac"] = float("nan")
                row["hull_error"] = str(e)[:60]
        rows.append(row)
        if (k + 1) % 5 == 0:
            print(f"  ...{k + 1}/{len(idx)}", flush=True)

    fr = np.array([r["debris_frac"] for r in rows])
    inter = np.array([r.get("interior_frac", np.nan) for r in rows], dtype=float)
    db = np.array([r.get("d_to_body_median", np.nan) for r in rows], dtype=float)
    dl = np.array([r.get("d_to_leg_median", np.nan) for r in rows], dtype=float)

    print("\n" + "=" * 84)
    print(f"PROBE 16 — where the part field puts debris  (model {args.tag})")
    print("=" * 84)
    print(f"  predicted debris fraction: mean {100 * fr.mean():.2f}%  median {100 * np.median(fr):.2f}%"
          f"  max {100 * fr.max():.2f}%")
    print("  (for reference, the fit-transferred estimate in make_partfield_data was 12.4% mean)")
    print(f"\n  median distance from a debris point to the nearest BODY point: {np.nanmedian(db):.4f}")
    print(f"  median distance from a debris point to the nearest LEG  point: {np.nanmedian(dl):.4f}")
    print("  (normalised units; specimen half-extent is 1.0, so <0.02 means 'touching')")
    print(f"\n  INTERIOR fraction (debris inside the animal's convex hull): "
          f"{100 * np.nanmean(inter):.1f}%  median {100 * np.nanmedian(inter):.1f}%")
    print("\n  pre-registered reading:")
    m = float(np.nanmean(inter))
    if m > 0.50:
        print(f"    >50% interior ({100 * m:.1f}%) -- the debris class is MISLEARNED. It is firing on")
        print("    the leg-body junction, almost certainly taught by the debris-BRIDGE augmentation.")
        print("    Excluding it from the data term blinds the fit at the coxae, which is the one")
        print("    place the hierarchical schedule cannot afford to lose. Debris must be REASSIGNED")
        print("    to its nearest non-debris class, not dropped.")
    elif m < 0.20:
        print(f"    <20% interior ({100 * m:.1f}%) -- debris is genuinely peripheral, exclusion is safe.")
    else:
        print(f"    {100 * m:.1f}% interior -- ambiguous; treat exclusion as a tunable, not a default.")

    worst = np.argsort(-fr)[:6]
    print("\n  highest predicted-debris specimens:")
    for i in worst:
        print(f"    {100 * fr[i]:5.2f}%  interior {100 * rows[i].get('interior_frac', float('nan')):5.1f}%"
              f"  {rows[i]['name'][:44]}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(100 * fr, bins=14, color="#b83280", alpha=0.85, edgecolor="white")
    ax[0].set_xlabel("predicted debris (% of points)")
    ax[0].set_ylabel("n specimens")
    ax[0].set_title("How much of each scan the field discards")
    ax[0].grid(alpha=0.25)
    ok = np.isfinite(inter)
    ax[1].scatter(100 * fr[ok], 100 * inter[ok], c="#b83280", s=26)
    ax[1].axhline(50, c="#c53030", ls="--", lw=1.5, label="mislearned above this")
    ax[1].axhline(20, c="#2f855a", ls="--", lw=1.5, label="safe to exclude below this")
    ax[1].set_xlabel("predicted debris (%)")
    ax[1].set_ylabel("of which INTERIOR to the animal (%)")
    ax[1].set_title("Interior debris is a junction artefact, not debris")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25)
    plt.tight_layout()
    p = os.path.join(OUT, f"probe16_debris_{args.tag}.png")
    fig.savefig(p, dpi=115)
    json.dump(dict(tag=args.tag, mean_frac=float(fr.mean()), mean_interior=m, rows=rows),
              open(os.path.join(OUT, f"probe16_debris_{args.tag}.json"), "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
