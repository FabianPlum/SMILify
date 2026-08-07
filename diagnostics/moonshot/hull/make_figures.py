"""Figures and renders for the E7 convexity-hierarchy report.

fig_granularity.png  purity / reproducibility / straddle vs cut size k, all merge criteria
                     against the k-means control at equal k
fig_ceiling.png      the atom-level purity ceiling, and the incumbent-vs-chunk-vote A/B
fig_errorsplit.png   correspondence error decomposed into within-part and between-part,
                     overall and per anatomical part -- the headline
render_chunks.png    a synthetic specimen coloured by chunk, per criterion and per k
render_errortype.png the same specimen coloured by which KIND of error each vertex has
"""

import json
import os
import pickle
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MOON = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, MOON)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

OUT = os.path.join(HERE, "out")
K_GRID = [2, 3, 5, 7, 9, 13, 20, 30, 45, 65]

# one consistent identity per method across every figure
STYLE = {
    "volume": ("#2b6cb0", "-", "o", "convex bridge (volume)"),
    "concavity": ("#38a169", "-", "s", "hull-surface concavity"),
    "hybrid": ("#805ad5", "-", "^", "hybrid (max of both)"),
    "visibility": ("#dd6b20", "-", "D", "weak convexity (line-of-sight)"),
    "spectral": ("#c53030", "-", "v", "spectral n-cut (Asafi-style)"),
    "kmeans": ("#4a5568", "--", "x", "k-means on xyz  [CONTROL]"),
    "trivial": ("#a0aec0", ":", "*", "x-tercile x sign(y)  [CONTROL]"),
}


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.isfile(p) else None


def agg(rows, crit, k, key):
    v = [
        r["by_k"][str(k)][key]
        for r in rows
        if r["criterion"] == crit and str(k) in r["by_k"] and key in r["by_k"][str(k)]
    ]
    return float(np.mean(v)) if v else np.nan


# ======================================================================================
def fig_granularity():
    for gt in ("7", "16"):
        d = load(f"score_synth_gt{gt}.json")
        if not d:
            continue
        rows = d["rows"]
        crits = [c for c in STYLE if any(r["criterion"] == c for r in rows)]
        fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
        panels = [
            (
                "purity",
                "PURITY vs ground-truth parts",
                "fraction of points in a chunk whose\nmajority part is their own part",
            ),
            ("g1", "G1 REPRODUCIBILITY", "agreement between two independent\nsurface samplings"),
            ("straddle", "STRADDLE (lower is better)", "surface fraction held by chunks\nspanning >1 anatomical part"),
        ]
        for a, (key, title, sub) in zip(ax, panels):
            for c in crits:
                if c == "trivial":
                    continue
                y = [agg(rows, c, k, key) for k in K_GRID]
                if np.all(np.isnan(y)):
                    continue
                col, ls, mk, lab = STYLE[c]
                a.plot(K_GRID, y, ls, color=col, marker=mk, ms=5, lw=2.2 if c != "kmeans" else 2.6, label=lab)
            t = [r["by_k"]["7"][key] for r in rows if r["criterion"] == "trivial" and key in r["by_k"]["7"]]
            if t:
                a.plot([7], [np.mean(t)], "*", color=STYLE["trivial"][0], ms=15, label=STYLE["trivial"][3])
            a.set_xscale("log")
            a.set_xticks(K_GRID)
            a.set_xticklabels(K_GRID)
            a.set_xlabel("cut size k (chunks per specimen)")
            a.set_title(title, fontsize=11, weight="bold")
            a.text(0.02, 0.02, sub, transform=a.transAxes, va="bottom", fontsize=8, color="#4a5568")
            a.grid(alpha=0.3)
            if key == "g1":
                a.axhline(0.90, color="#c53030", ls="--", lw=1.2)
                a.text(K_GRID[-1], 0.912, "gate 0.90", fontsize=8, color="#c53030", ha="right")
                a.axhline(0.20, color="#718096", ls=":", lw=1.2)
                a.text(K_GRID[-1], 0.215, "geodesic method: 0.20", fontsize=8, color="#718096", ha="right")
        ax[0].legend(fontsize=8, loc="lower right")
        fig.suptitle(
            f"E7 — convexity-derived part hierarchy vs a k-means control, at equal granularity "
            f"({len(set(r['name'] for r in rows))} synthetic specimens, exact ground truth, "
            f"{gt}-group anatomy)",
            fontsize=12,
        )
        plt.tight_layout()
        p = os.path.join(OUT, f"fig_granularity_gt{gt}.png")
        fig.savefig(p, dpi=125)
        plt.close(fig)
        print("wrote", p)


def fig_ceiling():
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))

    # ---- left: the atom-level ceiling
    for i, gt in enumerate(("7", "16")):
        d = load(f"score_synth_gt{gt}.json")
        if not d:
            continue
        ap = [r["atom_purity"] for r in d["rows"] if "atom_purity" in r]
        st = [r["atom_straddle"] for r in d["rows"] if "atom_straddle" in r]
        best = max(
            (agg(d["rows"], c, k, "purity") for c in STYLE for k in (13, 20, 30) if c != "kmeans"),
            default=np.nan,
        )
        km = max((agg(d["rows"], "kmeans", k, "purity") for k in (13, 20, 30)), default=np.nan)
        x = np.arange(3) + i * 0.4
        ax[0].bar(x[0], np.mean(ap), 0.36, color="#2b6cb0" if i == 0 else "#63b3ed", label=f"{gt}-group anatomy")
        ax[0].bar(x[1], best, 0.36, color="#2b6cb0" if i == 0 else "#63b3ed", alpha=0.65)
        ax[0].bar(x[2], km, 0.36, color="#2b6cb0" if i == 0 else "#63b3ed", alpha=0.4)
        ax[0].text(x[0], np.mean(ap) + 0.01, f"{np.mean(ap):.3f}", ha="center", fontsize=9)
        ax[0].text(x[2], km + 0.01, f"{km:.3f}", ha="center", fontsize=9)
        ax[0].text(x[0], 0.03, f"straddle\n{np.mean(st):.2f}", ha="center", fontsize=8, color="white")
    ax[0].set_xticks(np.arange(3) + 0.2)
    ax[0].set_xticklabels(
        ["CoACD atoms\n(~96 chunks)\nTHE CEILING", "best merge tree\n(k=13..30)", "k-means control\n(k=13..30)"]
    )
    ax[0].set_ylabel("purity")
    ax[0].set_title("The atoms bound everything above them", fontsize=11, weight="bold")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")
    ax[0].set_ylim(0, 1.0)

    # ---- right: incumbent vs chunk vote
    d = load("partition_ab_gt7.json")
    if d:
        rows = [r for r in d["rows"] if r["stage"] == "H1_legs"]
        ks = sorted({r["k"] for r in rows})
        inc = np.mean([r["incumbent"] for r in rows])
        ax[1].axhline(inc, color="#1a202c", lw=2.4, label=f"incumbent per-point rule ({inc:.3f})")
        for c in ("volume", "visibility", "spectral"):
            y = [np.mean([r["chunk_vote"] for r in rows if r["criterion"] == c and r["k"] == k]) for k in ks]
            col, ls, mk, lab = STYLE[c]
            ax[1].plot(ks, y, "-", color=col, marker=mk, ms=6, lw=2.2, label=lab)
        ax[1].set_xlabel("cut size k")
        ax[1].set_ylabel("partition accuracy vs ground truth")
        ax[1].set_title(
            "Chunk voting never beats the rule it replaces\n(same fit, same points, one variable)",
            fontsize=11,
            weight="bold",
        )
        ax[1].legend(fontsize=8, loc="lower right")
        ax[1].grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT, "fig_ceiling.png")
    fig.savefig(p, dpi=125)
    plt.close(fig)
    print("wrote", p)


def fig_errorsplit():
    d7, d16 = load("error_split_gt7.json"), load("error_split_gt16.json")
    if not d7:
        return
    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])
    a0, a1, a2 = (fig.add_subplot(gs[i]) for i in range(3))

    for a, d, ttl in ((a0, d7, "7-group anatomy"), (a1, d16, "16-group anatomy")):
        if not d:
            continue
        n = d["n_vertices"]
        vals = [d["correct"] / n, d["between"] / n, d["within"] / n]
        cols = ["#38a169", "#dd6b20", "#c53030"]
        labs = ["exactly correct", "BETWEEN-part\n(addressable by a partition)", "WITHIN-part\n(no partition can fix)"]
        b = a.bar([0], [vals[0]], 0.6, color=cols[0])
        bot = vals[0]
        for v, c in zip(vals[1:], cols[1:]):
            a.bar([0], [v], 0.6, bottom=bot, color=c)
            bot += v
        _ = b
        run = 0.0
        for v, c, lab in zip(vals, cols, labs):
            a.text(0.42, run + v / 2, f"{100 * v:.1f}%  {lab}", va="center", fontsize=9, color=c, weight="bold")
            run += v
        a.set_xlim(-0.4, 2.4)
        a.set_xticks([])
        a.set_ylim(0, 1)
        a.set_ylabel("fraction of all vertices")
        a.set_title(ttl, fontsize=11, weight="bold")
        ceiling = (d["correct"] + d["between"]) / n
        a.axhline(ceiling, color="#1a202c", ls="--", lw=1.4)
        a.text(
            -0.35,
            ceiling + 0.015,
            f"ceiling for a PERFECT partition: {100 * ceiling:.1f}%",
            fontsize=8.5,
            color="#1a202c",
        )

    if d16:
        pp = d16["per_part"]
        names = sorted(pp, key=lambda k_: -(pp[k_]["between"] / max(pp[k_]["within"] + pp[k_]["between"], 1)))
        frac = [100 * pp[n_]["between"] / max(pp[n_]["within"] + pp[n_]["between"], 1) for n_ in names]
        a2.barh(np.arange(len(names)), frac, color="#dd6b20")
        a2.set_yticks(np.arange(len(names)))
        a2.set_yticklabels(names, fontsize=8)
        a2.invert_yaxis()
        a2.set_xlabel("% of that part's error that crosses a part boundary")
        a2.set_title("Even the worst part is mostly within-part error", fontsize=11, weight="bold")
        a2.grid(alpha=0.3, axis="x")
    fig.suptitle(
        "E7 — why seven partition-shaped interventions have all returned null "
        "(SYN_clean, 12 specimens, exact ground-truth correspondence)",
        fontsize=12,
    )
    plt.tight_layout()
    p = os.path.join(OUT, "fig_errorsplit.png")
    fig.savefig(p, dpi=125)
    plt.close(fig)
    print("wrote", p)


# ======================================================================================
# renders
# ======================================================================================
def _cluster_colours(lab, seed=0):
    rng = np.random.default_rng(seed)
    k = int(lab.max()) + 1
    base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3]
    pal = np.concatenate([base] * (k // 20 + 1), 0)[:k]
    pal = pal[rng.permutation(k)]
    return pal[lab]


def render_chunks(specimen="synth_000", thr=0.03, npts=8000):
    from decompose_corpus import cache_path, load_tree
    from render_3d import DEV, make_renderer, render
    from scipy.spatial import cKDTree
    import trimesh

    obj = os.path.join(MOON, "synth_clean", f"{specimen}.obj")
    m = trimesh.load(obj, process=False, force="mesh")
    v = np.asarray(m.vertices, dtype=np.float64)
    v = (v - v.mean(0)) / np.abs(v - v.mean(0)).max()
    faces = torch.tensor(np.asarray(m.faces).astype(np.int64), device=DEV)
    verts = torch.tensor(v, dtype=torch.float32, device=DEV)

    crits = ["volume", "concavity", "visibility", "spectral"]
    ks = [7, 13, 30]
    renderer = make_renderer(image_size=380, elev=26.0, azim=140.0)

    fig, axes = plt.subplots(len(crits) + 1, len(ks), figsize=(3.5 * len(ks), 3.3 * (len(crits) + 1)))
    for ri, crit in enumerate(crits):
        cp = cache_path("synth", specimen, crit, 0, thr, npts)
        if not os.path.isfile(cp):
            for ci in range(len(ks)):
                axes[ri, ci].axis("off")
            continue
        tree, _ = load_tree(cp)
        kt = cKDTree(tree.points)
        for ci, k in enumerate(ks):
            lab = tree.cut(k)
            vl = lab[kt.query(v, k=1)[1]]
            col = torch.tensor(_cluster_colours(vl), dtype=torch.float32, device=DEV)
            axes[ri, ci].imshow(render(verts, faces, col, renderer))
            axes[ri, ci].axis("off")
            if ri == 0:
                axes[ri, ci].set_title(f"k = {k}", fontsize=12, weight="bold")
        axes[ri, 0].text(
            -0.06,
            0.5,
            STYLE[crit][3],
            transform=axes[ri, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
            color=STYLE[crit][0],
            weight="bold",
        )

    # k-means control row
    from sklearn.cluster import KMeans

    cp = cache_path("synth", specimen, "volume", 0, thr, npts)
    if os.path.isfile(cp):
        tree, _ = load_tree(cp)
        kt = cKDTree(tree.points)
        for ci, k in enumerate(ks):
            lab = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(tree.points)
            vl = lab[kt.query(v, k=1)[1]]
            col = torch.tensor(_cluster_colours(vl), dtype=torch.float32, device=DEV)
            axes[-1, ci].imshow(render(verts, faces, col, renderer))
            axes[-1, ci].axis("off")
        axes[-1, 0].text(
            -0.06,
            0.5,
            STYLE["kmeans"][3],
            transform=axes[-1, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
            color=STYLE["kmeans"][0],
            weight="bold",
        )

    fig.suptitle(
        f"E7 — what the convexity hierarchy actually finds ({specimen}, colours are arbitrary cluster ids)",
        fontsize=13,
    )
    plt.tight_layout()
    p = os.path.join(OUT, "render_chunks.png")
    fig.savefig(p, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def render_errortype(run="SYN_clean", n_show=3):
    """Colour each vertex by WHICH KIND of correspondence error it has."""
    from fitter_3d.trainer_hierarchical import anatomical_groups
    from pytorch3d.io import load_obj
    from pytorch3d.ops import knn_points
    from render_3d import DEV, make_renderer, render

    import config

    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    jn = [str(x) for x in dd["J_names"]]
    jg = anatomical_groups(jn, split_distal=True, split_anterior=True)
    gn = sorted(set(jg.values()))
    n2i = {nm: i for i, nm in enumerate(gn)}
    vpart = torch.tensor(np.array([n2i[jg[int(d)]] for d in np.asarray(dd["weights"]).argmax(1)]), device=DEV)
    faces = torch.tensor(np.asarray(dd["f"]).astype(np.int64), device=DEV)

    gt = np.load(os.path.join(MOON, "synth_clean", "ground_truth.npz"))
    gtv = torch.tensor(gt["verts"], dtype=torch.float32, device=DEV)
    names = [str(x) for x in gt["names"]]
    d = np.load(os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz"))
    fit = torch.tensor(d["verts"], dtype=torch.float32, device=DEV)
    labels = [str(x) for x in d["labels"]][:n_show]

    renderer = make_renderer(image_size=420, elev=26.0, azim=140.0)
    cmap = ListedColormap(["#38a169", "#c53030", "#dd6b20"])  # correct / within / between
    fig, axes = plt.subplots(len(labels), 3, figsize=(11.5, 3.7 * len(labels)))
    axes = np.atleast_2d(axes)
    for row, nm in enumerate(labels):
        stem = nm[:-4] if nm.endswith(".obj") else nm
        j = names.index(stem)
        ov, _, _ = load_obj(os.path.join(MOON, "synth_clean", f"{stem}.obj"), load_textures=False)
        ov = ov.to(DEV)
        c = ov.mean(0)
        s = (ov - c).abs().max()
        g = (gtv[j] - c) / s
        f = fit[row]
        nn = knn_points(f.unsqueeze(0), g.unsqueeze(0), K=1).idx[0, :, 0]
        correct = nn == torch.arange(g.shape[0], device=DEV)
        between = (~correct) & (vpart[nn] != vpart)
        kind = torch.zeros(len(f), dtype=torch.long, device=DEV)
        kind[~correct] = 1
        kind[between] = 2
        col = torch.tensor(cmap(kind.cpu().numpy())[:, :3], dtype=torch.float32, device=DEV)
        grey = torch.full((g.shape[0], 3), 0.66, device=DEV)
        for ci, (vv, cc, t) in enumerate(
            [
                (g, grey, f"ground truth — {stem}"),
                (f, grey, "fit (shaded) — visually near-perfect"),
                (f, col, f"error type — {100 * float(between.float().mean()):.1f}% between-part"),
            ]
        ):
            axes[row, ci].imshow(render(vv, faces, cc, renderer))
            axes[row, ci].set_title(t, fontsize=9)
            axes[row, ci].axis("off")
    handles = [
        plt.Line2D([], [], marker="s", ls="", color="#38a169", label="correct vertex"),
        plt.Line2D([], [], marker="s", ls="", color="#c53030", label="WITHIN-part error — no partition can fix"),
        plt.Line2D([], [], marker="s", ls="", color="#dd6b20", label="BETWEEN-part error — addressable"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9.5, frameon=False)
    fig.suptitle("E7 — the correspondence error is almost entirely INSIDE anatomical parts", fontsize=12)
    plt.tight_layout(rect=[0, 0.035, 1, 1])
    p = os.path.join(OUT, "render_errortype.png")
    fig.savefig(p, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_granularity()
    fig_ceiling()
    fig_errorsplit()
    render_chunks()
    render_errortype()
