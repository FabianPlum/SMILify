"""PAIRED render: does the arm that LOOKS most plausible have the best correspondence?

WHY THIS FIGURE EXISTS. Eyeballing the E8b fits, D2_frozen reads as the most plausible ant of
the three. That impression is real and it is also exactly the confound this investigation keeps
tripping over, so it deserves a figure rather than a paragraph.

With `w_offset = 200` the free-form offsets are pinned at 0.02% of specimen extent, so rest-space
geometry is `v_template + shapedirs . betas` -- confined to the model's 25-dimensional shape
subspace BY CONSTRUCTION. Every D2 fit is therefore a model ant no matter where it lands on the
target. Looking like an ant is *guaranteed* by the ablation, not earned by the fit. It is the
same structural artefact that made D2 win probe-19 (see run_e8b_synth.sh header) showing up in
the visual channel instead of the numeric one.

The direct round trip disagrees with the eye: D2 is the WORST arm on ground truth
(3.36% correct vs the control's 4.81%, median vertex error 4.87% vs 3.48%).

So this renders the SAME specimens across all three arms, twice:
  row A -- plain shaded. This is what the eye judges, and D2 should look fine or best.
  row B -- the identical mesh coloured by distance to its OWN ground-truth vertex. This is what
           is actually being scored, and D2 should be hotter.

Specimens are chosen at the MEDIAN of the control arm's per-specimen error, not the worst, so
the figure is not cherry-picked in either direction.
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MOON = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, MOON)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

from render_3d import DEV, colour_from_scalar, make_renderer, render  # noqa: E402
from score_synth_roundtrip import score  # noqa: E402

ARMS = [
    ("SYN_clean", "D0 control\nw_offset 0.2/0.08"),
    ("SYN_D1", "D1 low\nw_offset 5.0/2.0"),
    ("SYN_D2", "D2 frozen\nw_offset 200/200"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="synth_clean")
    ap.add_argument("--n", type=int, default=3, help="specimens, taken at the control arm's median")
    ap.add_argument("--vmax", type=float, default=5.0, help="colour scale top, % of extent")
    args = ap.parse_args()

    import pickle
    import config

    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    faces = torch.tensor(np.asarray(dd["f"]).astype(np.int64), device=DEV)

    res = {}
    for run, _ in ARMS:
        r = score(run, args.corpus)
        if r is None:
            print(f"[skip] {run}: no fit on disk")
            continue
        res[run] = {x["name"]: x for x in r}
    if len(res) < 2:
        raise SystemExit("need at least two arms on disk")

    # specimens present in EVERY arm, ordered by the control arm's mean error, median slice taken
    common = sorted(set.intersection(*(set(v) for v in res.values())))
    ctrl = res.get("SYN_clean") or res[next(iter(res))]
    common.sort(key=lambda nm: ctrl[nm]["err"].mean())
    lo = max(0, len(common) // 2 - args.n // 2)
    sel = common[lo : lo + args.n]
    print(f"{len(common)} specimens shared by all arms; taking the median {len(sel)}: {sel}")

    renderer = make_renderer(image_size=430, elev=24.0, azim=140.0)
    runs = [a for a, _ in ARMS if a in res]
    ncol = 1 + len(runs)
    fig, axes = plt.subplots(2 * len(sel), ncol, figsize=(3.3 * ncol, 3.3 * 2 * len(sel)))
    axes = np.atleast_2d(axes)

    for k, nm in enumerate(sel):
        ra, rb = 2 * k, 2 * k + 1
        g = torch.tensor(ctrl[nm]["gt"], device=DEV)
        grey = torch.full((g.shape[0], 3), 0.65, device=DEV)
        img_gt = render(g, faces, grey, renderer)
        for row in (ra, rb):
            axes[row, 0].imshow(img_gt)
            axes[row, 0].axis("off")
        axes[ra, 0].set_title(f"{nm}\nground truth", fontsize=8)

        for c, run in enumerate(runs, start=1):
            x = res[run][nm]
            f = torch.tensor(x["fit"], device=DEV)
            axes[ra, c].imshow(render(f, faces, grey, renderer))
            col = colour_from_scalar(x["err"] * 100, 0.0, args.vmax, cmap="inferno")
            axes[rb, c].imshow(render(f, faces, col, renderer))
            lab = dict(ARMS)[run]
            axes[ra, c].set_title(f"{lab}\nshaded — looks like?", fontsize=8)
            axes[rb, c].set_title(
                f"{100 * x['correct'].mean():.1f}% correct · median {100 * np.median(x['err']):.2f}%", fontsize=8
            )
            axes[ra, c].axis("off")
            axes[rb, c].axis("off")

    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, args.vmax))
    fig.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        fraction=0.018,
        pad=0.012,
        label="distance to the vertex's OWN ground-truth vertex (% of specimen extent)",
    )
    fig.suptitle(
        "PLAUSIBLE IS NOT CORRECT — upper row of each pair is what the eye judges, lower row is what is scored",
        fontsize=12,
        y=0.995,
    )
    out = os.path.join(HERE, "out", "fig_plausibility.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"wrote {out}")

    print(f"\n{'arm':<12}{'correct':>10}{'median err':>13}{'on these ' + str(len(sel)):>14}")
    for run in runs:
        c = np.concatenate([res[run][nm]["correct"] for nm in sel])
        e = np.concatenate([res[run][nm]["err"] for nm in sel])
        print(f"{run:<12}{100 * c.mean():>9.2f}%{100 * np.median(e):>12.2f}%")


if __name__ == "__main__":
    main()
