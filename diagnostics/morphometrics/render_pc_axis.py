"""What does the dominant shape axis LOOK like? Specimens rendered in order along a PC.

A principal component is a direction in a 38-dimensional space. Correlating it with named indices
says what it is arithmetically; rendering the specimens along it says what it is anatomically,
and the two should agree. PC1 here carries 21% of the variance and correlates +0.82 with the
mandible index, -0.60 with mesosoma slenderness and +0.49 with the hind-femur index -- so the
prediction is that the low end looks long-bodied and the high end looks compact. This is the
figure that checks it.

Specimens are drawn at evenly-spaced PERCENTILES of the score rather than at the extremes, so
the row is a fair traverse of the axis and not a pair of outliers. Rendering uses the same
anatomical frame as render_measurements (head left, dorsal up, mesosoma Kabsch-aligned to the
template), so shape differences between panels are real and not camera differences.
"""

import argparse
import os
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
from render_measurements import view_axes, proj  # noqa: E402

try:
    from taxonomy import ECOLOGY
except ImportError:
    ECOLOGY = {}

OUT = os.path.join(HERE, "out")


def canonical_verts(indices, rows, runs, M):
    """Rebuild the chosen specimens with EVERY joint rotation set to zero.

    The fitted mesh mixes pose and shape, and pose dominates what the eye sees: two ants of
    identical proportions look completely different if one has its gaster curled. Driving the
    model forward with `joint_rot = 0` and `global_rot = 0` while keeping the fitted `betas`,
    `log_beta_scales`, `betas_trans` and `deform_verts` strips the pose out and leaves exactly
    the quantity the measurements describe.
    """
    import torch
    from fitter_3d.trainer import SMAL3DFitter

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    want = {}
    for i in indices:
        want.setdefault(rows[i]["run"], []).append(i)
    out = {}
    for run, idxs in want.items():
        d = np.load(os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz"))
        # position within this run's npz, since `rows` is the concatenation of all runs
        base = min(k for k, r in enumerate(rows) if r["run"] == run)
        loc = [i - base for i in idxs]
        n = len(loc)
        fitter = SMAL3DFitter(batch_size=n, device=dev, shape_family=-1)
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
        with torch.no_grad():
            v = fitter(
                betas=t(d["betas"][loc]),
                global_rot=torch.zeros(n, 3, device=dev),
                joint_rot=torch.zeros(n, d["joint_rot"].shape[1], 3, device=dev),
                trans=torch.zeros(n, 3, device=dev),
                log_beta_scales=t(d["log_beta_scales"][loc]),
                betas_trans=t(d["betas_trans"][loc]),
                deform_verts=t(d["deform_verts"][loc]),
            )
        v = (v[0] if isinstance(v, tuple) else v).cpu().numpy()
        for k, i in enumerate(idxs):
            out[i] = v[k].astype(np.float64)
        del fitter
        torch.cuda.empty_cache()
    return [out[i] for i in indices]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npc", type=int, default=3, help="how many PCs to draw, one row each")
    ap.add_argument("--n", type=int, default=7, help="specimens per row")
    ap.add_argument(
        "--quality_top",
        type=float,
        default=50.0,
        help="keep only the best N%% of specimens by registration quality (deform/edge/normal "
        "composite, see measure.quality_filter and report §7); 100 = no filter",
    )
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from sklearn.decomposition import PCA

    M = ms.load_model()
    bones = ms.bone_table(M)
    tpa = ms.template_part_axes(M)
    runs = sorted(
        d
        for d in os.listdir(os.path.join(MOON, "runs"))
        if d.startswith("MORPH_W") and not d.endswith("_hier") and os.path.isdir(os.path.join(MOON, "runs", d))
    ) or ["D1_low_MERGED"]

    rows, cols, dev, verts = [], None, None, []
    for r in runs:
        rr, c, v = ms.measure_run(r, "worker", M, bones, tpa)
        if not rr:
            continue
        p = os.path.join(MOON, "runs", r, "Stage_3_deform_fine.npz")
        verts.append(np.load(p)["verts"].astype(np.float64))
        rows += rr
        cols, dev = c, v
    V = np.concatenate(verts)
    keep = ms.quality_filter(rows, M, args.quality_top)
    rows = [r for r, k in zip(rows, keep) if k]
    V = V[keep]
    print(f"kept best {args.quality_top:.0f}% by registration quality: {len(rows)} specimens")
    sc_all, _ = ms.symmetrise(rows, cols)
    sd_all, _ = ms.symmetrise(rows, dev)
    sc = [c for c in sc_all if block_of(c) in ms.CORE_BLOCKS]
    sd = [c for c in sd_all if block_of(c) in ms.CORE_BLOCKS]
    X = np.array([[r[c] for c in sc] for r in rows])
    Z, _ = ms.log_shape_ratios(np.maximum(X, 1e-9))
    D = np.array([[r[c] for c in sd] for r in rows])
    D = (D - D.mean(0)) / np.maximum(D.std(0), 1e-9)
    F = np.hstack([Z, np.nan_to_num(D)])
    F = (F - F.mean(0)) / np.maximum(F.std(0), 1e-9)
    pca = PCA(n_components=args.npc, random_state=0)
    P = pca.fit_transform(F)
    print(f"{len(rows)} specimens; variance explained {(100 * pca.explained_variance_ratio_).round(1)}")

    NAMES = {
        0: "PC1 — elongate body, slender mandibles  →  compact body, long femur",
        1: "PC2 — relative head size",
        2: "PC3 — relative gaster length",
    }
    # Pick the specimens first, then rebuild those few in the CANONICAL pose.
    picks_per_pc = []
    for j in range(args.npc):
        s = P[:, j]
        picks_per_pc.append([int(np.argmin(np.abs(s - np.percentile(s, q)))) for q in np.linspace(2, 98, args.n)])
    flat = [i for pk in picks_per_pc for i in pk]
    CV = canonical_verts(flat, rows, runs, M)
    cidx = {i: k for k, i in enumerate(flat)}
    BF = ms.body_frame(M)  # rows: lateral, antero-posterior, dorso-ventral

    # two rows per PC: as fitted (lateral) above, rotations reset (top-down) below
    fig, axes = plt.subplots(2 * args.npc, args.n, figsize=(2.5 * args.n, 2.7 * 2 * args.npc))
    axes = np.atleast_2d(axes)
    for j in range(args.npc):
        s = P[:, j]
        ra, rb = 2 * j, 2 * j + 1
        for c, i in enumerate(picks_per_pc[j]):
            g = rows[i]["genus"] or "?"
            eco = ECOLOGY.get(g, "")

            ax = axes[ra, c]
            cen, A = view_axes(V[i], M, tpa)
            x, y = proj(V[i], cen, A, 0, 1)
            ax.scatter(x, y, s=0.5, c="#7f95ad", linewidths=0, rasterized=True)
            ax.set_title(f"{g}\n{s[i]:+.1f}" + (f"\n{eco}" if eco else ""), fontsize=7)

            # rotations zeroed: pose is gone, so what remains is shape alone. Viewed down the
            # dorso-ventral axis, which is where head width, mesosoma width and gaster width
            # -- the dimensions the indices are built from -- are actually visible.
            ax = axes[rb, c]
            Q = CV[cidx[i]]
            Q = Q - Q.mean(0)
            Q = Q / np.abs(Q).max()  # per-specimen, since no absolute size exists anyway
            ax.scatter(Q @ BF[1], Q @ BF[0], s=0.5, c="#b06a3b", linewidths=0, rasterized=True)
            for a in (axes[ra, c], axes[rb, c]):
                a.set_aspect("equal")
                a.axis("off")
        axes[ra, 0].text(
            -0.05,
            0.5,
            NAMES.get(j, f"PC{j + 1}") + f"\n({100 * pca.explained_variance_ratio_[j]:.0f}% var)",
            transform=axes[ra, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=8.5,
        )
        axes[ra, 0].text(
            -0.30,
            0.5,
            "as fitted",
            transform=axes[ra, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=7,
            color="#4a5568",
        )
        axes[rb, 0].text(
            -0.30,
            0.5,
            "pose reset,\ntop-down",
            transform=axes[rb, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=7,
            color="#b06a3b",
        )
    fig.suptitle(
        "The shape axes, rendered — specimens at evenly spaced percentiles of each PC.\n"
        "Upper row of each pair as fitted (lateral); lower row with all joint rotations set to "
        "zero and viewed top-down, so pose is removed and only shape remains.\n"
        f"Best {args.quality_top:.0f}% of specimens by registration quality (n={len(rows)}); see §7.",
        fontsize=11.5,
    )
    plt.tight_layout(rect=[0.03, 0, 1, 0.95])
    p = os.path.join(OUT, "fig_pc_axes.png")
    fig.savefig(p, dpi=125)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
