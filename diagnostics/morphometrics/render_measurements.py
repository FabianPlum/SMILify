"""What is actually being measured — the landmark set drawn on real fitted specimens.

Every number in this analysis comes from two things: joint positions recovered by the
J_regressor, and part extents along template-aligned axes. Both are abstractions, and an
abstraction that is never looked at is an abstraction nobody has checked. This draws them.

Three panels per specimen:
  1. the fitted mesh, shaded, beside the scan it was fitted to;
  2. the SKELETON -- every measured bone drawn as a segment between the two joints whose
     distance is the measurement, coloured by anatomical group, with low-support joints
     (<20 dominantly-skinned vertices) marked hollow because their positions are unreliable;
  3. the PART EXTENTS -- head, mesosoma and gaster boxes drawn along each part's
     template-aligned axes, which is literally what head_wid/head_len/head_hei measure.

Projection is orthographic in the specimen's own ANATOMICAL frame -- head left, dorsal up --
recovered by rigidly aligning the mesosoma to the template. See view_axes() for why the
specimen's principal axes, the obvious choice, render some ants upside down.
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

GROUP_COL = {
    "mesosoma": "#2b6cb0",
    "gaster": "#805ad5",
    "head": "#e53e3e",
    "mandible": "#dd6b20",
    "antenna": "#38a169",
    "leg_prox": "#319795",
    "leg_distal": "#a0aec0",
}
PART_COL = {"head": "#e53e3e", "mesosoma": "#2b6cb0", "gaster": "#805ad5"}


def view_axes(V, M, tpa):
    """Anatomical viewing frame for one specimen: head LEFT, dorsal UP, lateral into the page.

    The obvious frame -- the specimen's own principal axes -- is wrong, and visibly so: it
    rendered some ants upside down. Singular vectors carry an arbitrary sign, so the vertical
    axis points up or down at random. Worse, the third principal axis is not even a consistent
    anatomical direction: measured across specimens it comes out as +z for some and +/-y for
    others, because an ant's width and height are similar enough that the eigenvalue order
    swaps. That is the same instability that forced anatomical axis labelling in measure.py.

    So the frame comes from anatomy instead. The mesosoma is a single rigid bone, so Kabsch-
    aligning it to the template recovers the rotation between template and specimen, and the
    template's body frame is carried through it. Signs are fixed from the template's own
    geometry: the head centroid sits at negative antero-posterior (so +AP is posterior, and
    plotting +AP rightwards puts the head on the left), and the tarsi sit at positive
    dorso-ventral (so dorsal is -dv).
    """
    BF = ms.body_frame(M)  # rows: lateral, antero-posterior, dorso-ventral
    info = tpa["mesosoma"]
    T = M["v_template"][info["idx"]] - info["centroid"]
    P = V[info["idx"]]
    R = ms.kabsch(P - P.mean(0), T)  # T ~ (P - mean) @ R, so a template dir d is d @ R.T here
    return V.mean(0), np.stack([BF[1] @ R.T, -BF[2] @ R.T])


def proj(P, c, A, a=0, b=1):
    Q = (P - c) @ A.T
    return Q[:, a], Q[:, b]


def draw_mesh(ax, V, c, A, a, b, s=0.7, col="#8fa3b8"):
    x, y = proj(V, c, A, a, b)
    ax.scatter(x, y, s=s, c=col, linewidths=0, rasterized=True)


def box_corners(centre, axes, half):
    """8 corners of the oriented box with the given axes (rows) and half-extents."""
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                out.append(centre + (sx * half[0]) * axes[0] + (sy * half[1]) * axes[1] + (sz * half[2]) * axes[2])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run name; default = first MORPH_W* on disk")
    ap.add_argument("--corpus", default="worker")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--pick", nargs="*", default=None, help="substring match on label")
    ap.add_argument(
        "--quality_top",
        type=float,
        default=10.0,
        help="illustrate with the best N%% of registrations (deform/edge/normal composite, §7). "
        "This figure shows what the pipeline MEASURES, so it should show it working; failure "
        "cases belong in their own figure, not in the definition.",
    )
    args = ap.parse_args()

    M = ms.load_model()
    bones = ms.bone_table(M)
    tpa = ms.template_part_axes(M)

    # Search EVERY fitted chunk, not just one. --pick names genera, and the corpus is split
    # across a dozen chunks, so restricting to one guarantees most requests find nothing.
    if args.run:
        runs = [args.run]
    else:
        runs = sorted(
            d
            for d in os.listdir(os.path.join(MOON, "runs"))
            if d.startswith("MORPH_W") and not d.endswith("_hier") and os.path.isdir(os.path.join(MOON, "runs", d))
        ) or ["D1_low_MERGED"]
    vv, labels, run_of = [], [], []
    for r in runs:
        p = os.path.join(MOON, "runs", r, "Stage_3_deform_fine.npz")
        if not os.path.isfile(p):
            continue
        d = np.load(p)
        vv.append(d["verts"].astype(np.float64))
        ls = [str(x) for x in d["labels"]]
        labels += ls
        run_of += [(r, x) for x in ls]
    verts = np.concatenate(vv)
    run = runs[0] if len(runs) == 1 else f"{len(runs)} chunks"
    rows = [dict(run=r_, label=lab) for r_, lab in run_of]
    keep = ms.quality_filter(rows, M, args.quality_top)
    verts, labels = verts[keep], [x for x, k in zip(labels, keep) if k]
    print(f"best {args.quality_top:.0f}% by registration quality: {len(labels)} specimens")
    J = ms.joints(verts, M["Jr"])

    if args.pick:
        # one specimen PER query, not the first n matches overall -- otherwise a single
        # well-represented genus fills every row and the comparison shows nothing.
        sel = []
        for q in args.pick:
            hit = next((i for i, s in enumerate(labels) if q.lower() in s.lower() and i not in sel), None)
            if hit is not None:
                sel.append(hit)
    else:
        sel = list(range(min(args.n, len(labels))))
    if not sel:
        raise SystemExit("no specimens matched")

    counts = np.bincount(M["dominant"], minlength=len(M["jnames"]))
    fig, axes = plt.subplots(len(sel), 3, figsize=(15, 4.6 * len(sel)))
    axes = np.atleast_2d(axes)

    for row, i in enumerate(sel):
        V, Ji = verts[i], J[i]
        c, A = view_axes(V, M, tpa)

        draw_mesh(axes[row, 0], V, c, A, 0, 1)
        axes[row, 0].set_title(f"{ms.parse_taxonomy(labels[i], args.corpus)['genus']} — fitted mesh", fontsize=9)

        ax = axes[row, 1]
        draw_mesh(ax, V, c, A, 0, 1, s=0.45, col="#dbe3ec")
        for b in bones:
            pq = np.stack([Ji[b["parent"]], Ji[b["child"]]])
            x, y = proj(pq, c, A, 0, 1)
            ax.plot(
                x,
                y,
                "-",
                lw=1.6,
                color=GROUP_COL.get(b["group"], "#000"),
                zorder=3,
                alpha=0.55 if b["low_support"] else 1.0,
            )
        jx, jy = proj(Ji, c, A, 0, 1)
        good = counts >= ms.LOW_SUPPORT
        ax.scatter(jx[good], jy[good], s=13, c="k", zorder=4)
        ax.scatter(jx[~good], jy[~good], s=13, facecolors="none", edgecolors="k", linewidths=0.7, zorder=4)
        ax.set_title(f"{len(bones)} measured bones (hollow = low support)", fontsize=9)

        ax = axes[row, 2]
        draw_mesh(ax, V, c, A, 0, 1, s=0.45, col="#dbe3ec")
        V0 = M["v_template"]
        for part, info in tpa.items():
            if part not in PART_COL:
                continue
            idx, c0, axs = info["idx"], info["centroid"], info["axes"]
            T = V0[idx] - c0
            P = V[idx]
            pc = P.mean(0)
            R = ms.kabsch(P - pc, T)
            Q = (P - pc) @ R
            pr = Q @ axs.T
            half = 0.5 * (np.percentile(pr, 98, axis=0) - np.percentile(pr, 2, axis=0))
            mid = 0.5 * (np.percentile(pr, 98, axis=0) + np.percentile(pr, 2, axis=0))
            world_axes = axs @ R.T  # template part axes, carried into the specimen's frame
            corners = box_corners(pc + mid @ world_axes, world_axes, half)
            x, y = proj(corners, c, A, 0, 1)
            hull_order = [0, 1, 3, 2, 0, 4, 5, 7, 6, 4]
            ax.plot(x[hull_order], y[hull_order], "-", lw=1.3, color=PART_COL[part], zorder=3)
            for e in [(1, 5), (2, 6), (3, 7)]:
                ax.plot(x[list(e)], y[list(e)], "-", lw=1.0, color=PART_COL[part], alpha=0.6, zorder=3)
            ax.plot([], [], color=PART_COL[part], label=part)
        ax.legend(fontsize=7)
        ax.set_title("part extents, along template-aligned axes", fontsize=9)

        for k in range(3):
            axes[row, k].set_aspect("equal")
            axes[row, k].axis("off")

    hs = [plt.Line2D([], [], color=v, lw=2, label=k) for k, v in GROUP_COL.items()]
    fig.legend(handles=hs, loc="lower center", ncol=len(hs), fontsize=8, frameon=False)
    fig.suptitle(
        f"What the morphometric pipeline measures — best {args.quality_top:.0f}% of registrations ({run})", fontsize=12
    )
    plt.tight_layout(rect=[0, 0.035, 1, 0.98])
    out = os.path.join(HERE, "out", "fig_measurements.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
