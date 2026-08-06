"""Figure: where the pose-mobile arms put joints outside anatomically plausible range.

The statistic ("37 of 96 constrained axes, median 19.4 deg") is not something a human can
sanity-check. An entomologist looking at a rendered ant can tell in one glance whether a
tarsus is bent somewhere a real animal could not bend it, and that judgement is worth more
than the number -- especially since the limits themselves are authored by hand and could
simply be too tight.

So this renders the same fitted meshes twice: shaded, and coloured by how far each vertex's
dominant joint is outside its authored range. Red is impossible; grey is legal.

Panels, left to right per specimen:
  target        the scan, plain grey
  fit           the fitted mesh, plain grey -- does it LOOK like an ant?
  violation     the fitted mesh, coloured by limit overshoot in degrees

Plus a per-joint summary across all specimens, grouped anatomically, so it is visible whether
violations concentrate in the distal leg (the §4 prediction) or are spread everywhere (which
would instead suggest the authored limits are simply too tight).

Usage:
  python diagnostics/moonshot/viz_limit_violations.py --run runs/BPX_noprior --n 4
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from pytorch3d.io import load_obj  # noqa: E402
from render_3d import DEV, colour_from_scalar, make_renderer, render  # noqa: E402

LIMIT_MODEL = os.path.join(REPO, "3D_model_prep", "OmniAnt_25PCs_joint_limited.pkl")


def load_pkl(p):
    with open(p, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        return u.load()


def anatomical_group(name):
    if name.startswith("ma"):
        return "mandible"
    if name.startswith("a_"):
        return "antenna"
    if not name.startswith("l_"):
        return "body"
    seg = name.split("_")[2] if len(name.split("_")) > 2 else ""
    return "leg proximal" if seg in ("co", "tr", "fe") else "leg distal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="diagnostics/moonshot/runs/BPX_noprior")
    ap.add_argument("--stage", default="Stage_3_deform_fine")
    ap.add_argument("--mesh_dir", default="diagnostics/moonshot/bench50")
    ap.add_argument("--n", type=int, default=4, help="specimens rendered, worst violators first")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "limit_violations.png"))
    ap.add_argument("--vmax", type=float, default=45.0, help="colour ceiling, degrees over the bound")
    args = ap.parse_args()

    lim = load_pkl(LIMIT_MODEL)
    jn = list(lim["J_names"])
    jl = np.asarray(lim["joint_limits"], float)
    lo, hi = jl[1:, :, 0], jl[1:, :, 1]

    d = np.load(os.path.join(REPO, args.run, f"{args.stage}.npz"))
    jr = d["joint_rot"].astype(float)
    verts, faces = d["verts"].astype(np.float32), d["faces"].astype(np.int64)
    labels = [str(x) for x in d["labels"]]

    # Canonicalise to ||theta|| <= pi first. Axis-angle is non-unique past pi (a 200 deg
    # rotation IS -160 deg), so a raw hinge would report a coordinate choice as an anatomical
    # violation. Measured: this affects 0.44% of joints and moves the median overshoot by
    # 0.2 deg, but it cuts the reported MAXIMUM from 216 to 141 deg -- so it matters only for
    # the extreme tail, which is exactly what the panel titles quote.
    theta = np.linalg.norm(jr, axis=-1, keepdims=True)
    wraps = np.maximum(np.round((theta - np.pi) / (2 * np.pi) + 0.5), 0.0)
    jr = jr * ((theta - 2 * np.pi * wraps) / np.clip(theta, 1e-9, None))

    over = np.maximum(jr - hi, 0.0) + np.maximum(lo - jr, 0.0)  # (n, N_POSE, 3)
    per_joint_over = over.max(-1)  # worst axis per joint, (n, N_POSE)

    # vertex -> dominant joint, from the template the FIT used (vertex counts differ between
    # models; J_names are identical, verified, so the joint indexing carries over)
    src = load_pkl(os.path.join(REPO, "3D_model_prep", "SMIL_OmniAnt.pkl"))
    if np.asarray(src["v_template"]).shape[0] != verts.shape[1]:
        src = lim
    dom = np.asarray(src["weights"]).argmax(1)  # (V,) joint index incl. root

    # Worst AND best, so there is something to compare against. A grid of only the worst
    # specimens cannot show whether the violations are specific to bad fits or universal.
    rank = np.argsort(-per_joint_over.sum(1))
    k = max(args.n // 2, 1)
    order = list(rank[:k]) + list(rank[-k:])
    tags = ["WORST"] * k + ["best"] * k
    renderer = make_renderer(image_size=460, elev=24.0, azim=140.0)

    def norm(v):
        """Same normalisation load_meshes applies to targets, so fit and scan are comparable."""
        v = v - v.mean(0)
        return v / v.abs().max()

    fig, axes = plt.subplots(len(order), 3, figsize=(11.5, 3.7 * len(order)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(order):
        name = labels[i]
        stem = name[:-4] if name.endswith(".obj") else name  # labels already carry the extension
        tp = os.path.join(REPO, args.mesh_dir, f"{stem}.obj")
        if os.path.isfile(tp):
            tv, tf, _ = load_obj(tp, load_textures=False)
            tv = norm(tv.to(DEV))
            img_t = render(tv, tf.verts_idx.to(DEV), torch.full((tv.shape[0], 3), 0.65, device=DEV), renderer)
        else:
            img_t = np.ones((460, 460, 3))
            print(f"  WARNING target not found: {tp}")

        v = norm(torch.tensor(verts[i], device=DEV))
        f = torch.tensor(faces[i], device=DEV)
        img_f = render(v, f, torch.full((v.shape[0], 3), 0.65, device=DEV), renderer)

        # per-vertex overshoot in degrees, via the vertex's dominant joint (root -> 0).
        # Legal vertices are painted NEUTRAL GREY, not colormap-zero: inferno's low end is
        # black, which is indistinguishable from shadow and made the first version unreadable.
        vdeg = np.degrees(np.concatenate([[0.0], per_joint_over[i]]))[dom]
        col = colour_from_scalar(vdeg, 0.0, args.vmax, cmap="inferno")
        col[torch.tensor(vdeg <= 0, device=DEV)] = torch.tensor([0.62, 0.62, 0.62], device=DEV)
        img_v = render(v, f, col, renderer)

        nv = int((per_joint_over[i] > 0).sum())
        for c, (img, ttl) in enumerate(
            [
                (img_t, f"[{tags[r]}] target — {stem[:34]}"),
                (img_f, "fit (shaded)"),
                (img_v, f"overshoot — {nv} joints, max {vdeg.max():.0f}°"),
            ]
        ):
            axes[r, c].imshow(img)
            axes[r, c].set_title(ttl, fontsize=9)
            axes[r, c].axis("off")

    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, args.vmax))
    fig.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        fraction=0.02,
        pad=0.015,
        label="degrees outside the authored joint limit  (grey/black = legal)",
    )
    fig.suptitle(f"{os.path.basename(args.run)} — pose freedom spent outside anatomical range", fontsize=12)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=115, bbox_inches="tight")
    print(f"wrote {args.out}")

    # ---- per-joint summary, grouped anatomically ----
    groups = [anatomical_group(n) for n in jn[1:]]
    wide = np.isclose(np.abs(lo), np.pi) & np.isclose(np.abs(hi), np.pi)
    rows = []
    for g in ["body", "mandible", "antenna", "leg proximal", "leg distal"]:
        idx = [k for k, gg in enumerate(groups) if gg == g]
        if not idx:
            continue
        mask = ~wide[idx]  # (len(idx), 3) -- only axes that are actually constrained
        nc = int(mask.sum())
        if nc == 0:
            rows.append((g, 0, 0.0, 0.0))
            continue
        sub = over[:, idx]  # (n, len(idx), 3)
        hit = (sub > 0) & mask[None]
        frac = 100.0 * hit.sum() / (nc * sub.shape[0])
        rows.append((g, nc, frac, float(np.degrees(sub[hit]).mean()) if hit.any() else 0.0))

    fig2, ax = plt.subplots(1, 2, figsize=(12, 4.0))
    gs = [r[0] for r in rows]
    ax[0].barh(gs, [r[2] for r in rows], color="#c53030")
    ax[0].set_xlabel("% of constrained specimen-axes violating")
    ax[0].set_title("Where the violations are", fontsize=10)
    ax[1].barh(gs, [r[3] for r in rows], color="#2b6cb0")
    ax[1].set_xlabel("mean overshoot (degrees)")
    ax[1].set_title("How far outside", fontsize=10)
    for a in ax:
        a.grid(alpha=0.3, axis="x")
    fig2.suptitle(
        f"{os.path.basename(args.run)} — violations by anatomical group "
        f"(§4 predicts the distal leg, which the data term cannot constrain)",
        fontsize=11,
    )
    plt.tight_layout()
    p2 = args.out.replace(".png", "_bygroup.png")
    fig2.savefig(p2, dpi=120)
    print(f"wrote {p2}")
    for g, nc, frac, mo in rows:
        print(f"  {g:<14} {nc:>3} constrained axes   {frac:>5.1f}% violating   mean overshoot {mo:>5.1f}°")


if __name__ == "__main__":
    main()
