"""Figures for E8 (the free-form ablation) and the volumetric investigation.

fig_e8_tradeoff.png   surface accuracy vs mesh integrity vs deform magnitude, D0/D1/D2
fig_e8_probe19.png    the probe-19 result AND why it is an artefact
fig_nullspace.png     IoU vs chamfer sensitivity per parameter direction
fig_volumetric.png    ellipsoid ceiling, volume shares, per-segment IoU, pose recovery
render_e8.png         one specimen under each arm, next to its target
"""

import glob
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MOON = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, MOON)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
OUT = os.path.join(HERE, "out")

ARMS = [
    ("D0_control", "D0 control\n(stock recipe)", "#2b6cb0"),
    ("D1_low", "D1 low-deform\n(25x penalty)", "#dd6b20"),
    ("D2_frozen", "D2 frozen\n(deform pinned ~0)", "#c53030"),
]


def arm_metrics(arm, stage="Stage_3_deform_fine"):
    fs = sorted(glob.glob(os.path.join(MOON, "runs", f"{arm}_c*", "metrics.csv")))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    return d[d.stage == stage]


# ======================================================================================
def fig_tradeoff():
    data = {a: arm_metrics(a) for a, _, _ in ARMS}
    if any(v is None for v in data.values()):
        print("[skip] tradeoff: missing metrics")
        return
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))

    panels = [
        ("deform_mag_mean", "free-form offset magnitude", "the variable being ablated", True),
        ("fscore@0.01", "fscore @ 0.01  (surface accuracy)", "what you give up", False),
        ("edge_logratio_absmean", "edge distortion  (lower = better)", "what you get back", True),
        ("tri_quality_mean", "triangle quality  (higher = better)", "mesh stays valid", False),
    ]
    for a, (key, title, sub, lower_better) in zip(ax, panels):
        vals = [data[k][key].values for k, _, _ in ARMS]
        bp = a.boxplot(vals, patch_artist=True, widths=0.55, showfliers=False, medianprops=dict(color="black", lw=1.6))
        for patch, (_, _, c) in zip(bp["boxes"], ARMS):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
        a.set_xticks([1, 2, 3])
        a.set_xticklabels([lbl for _, lbl, _ in ARMS], fontsize=8)
        a.set_title(title, fontsize=10.5, weight="bold")
        a.text(0.5, 1.11, sub, transform=a.transAxes, ha="center", fontsize=8.5, color="#4a5568")
        a.grid(alpha=0.3, axis="y")
        for i, v in enumerate(vals):
            a.text(i + 1, np.median(v), f"  {np.median(v):.4g}", fontsize=8, va="center", color="#1a202c")
    fig.suptitle(
        "E8 — freezing free-form deformation: 128 gated workers, paired, identical schedule and "
        "iteration budget, one variable",
        fontsize=12.5,
        y=1.04,
    )
    plt.tight_layout()
    p = os.path.join(OUT, "fig_e8_tradeoff.png")
    fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_probe19():
    """The probe-19 numbers, and the reason they cannot be read as a correspondence win."""
    gs = {"D0 control": 0.8687, "D1 low-deform": 0.5786, "D2 frozen": 0.3982}
    g20 = {"D0 control": 0.8415, "D1 low-deform": 0.3710, "D2 frozen": 0.1286}
    dmag = {"D0 control": 0.013720, "D1 low-deform": 0.003286, "D2 frozen": 0.000185}
    refs = [
        (0.9603, "worker arms so far (0.92-0.96)", "#a0aec0"),
        (0.8946, "E6 calibration: ~5% correct correspondence", "#dd6b20"),
        (0.5383, "ALL_ANTS_CLEAN registrations", "#38a169"),
        (0.4200, "synthetic, EXACT correspondence", "#2b6cb0"),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.2))

    x = np.arange(3)
    cols = [c for _, _, c in ARMS]
    ax[0].bar(x - 0.19, [gs[k] for k in gs], 0.36, color=cols, label="gen@10 / spread")
    ax[0].bar(x + 0.19, [g20[k] for k in g20], 0.36, color=cols, alpha=0.5, label="gen@20 / spread")
    for v, lab, c in refs:
        ax[0].axhline(v, color=c, ls="--", lw=1.4)
        ax[0].text(2.62, v, lab, fontsize=7.6, color=c, va="center")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([lbl for _, lbl, _ in ARMS], fontsize=8.5)
    ax[0].set_ylabel("probe-19 generalisation / spread   (lower = more learnable structure)")
    ax[0].set_title("What probe-19 says", fontsize=11, weight="bold")
    ax[0].set_xlim(-0.5, 4.6)
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].grid(alpha=0.3, axis="y")

    # right: the artefact
    ax[1].scatter([dmag[k] for k in dmag], [gs[k] for k in gs], s=190, c=cols, zorder=3)
    for (k, _), c in zip(gs.items(), cols):
        ax[1].annotate(
            k.split()[0],
            (dmag[k], gs[k]),
            textcoords="offset points",
            xytext=(10, 6),
            fontsize=9.5,
            color=c,
            weight="bold",
        )
    ax[1].set_xscale("log")
    ax[1].set_xlabel("free-form deform magnitude (log)")
    ax[1].set_ylabel("gen@10 / spread")
    ax[1].set_title("…and why it cannot be read as a correspondence win", fontsize=11, weight="bold")
    ax[1].grid(alpha=0.3)
    ax[1].text(
        0.03,
        0.05,
        "With deform pinned at 1.8e-4, rest-space geometry is\n"
        "    v_template + shapedirs · betas\n"
        "EXACTLY — a 25-dimensional linear subspace, by construction.\n\n"
        "probe-19 measures how low-dimensional the registrations are.\n"
        "A near-zero gen@20 is therefore GUARANTEED by the ablation,\n"
        "whether or not the correspondence is anatomically right.\n\n"
        "The pre-registered primary was the wrong instrument.\n"
        "E8b re-runs this on the ground-truth corpus instead.",
        transform=ax[1].transAxes,
        fontsize=8.6,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="#fffaf0", ec="#c53030", lw=1.4),
    )
    fig.suptitle("E8 — the headline number, and the reason it is an artefact", fontsize=12.5)
    plt.tight_layout()
    p = os.path.join(OUT, "fig_e8_probe19.png")
    fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_nullspace():
    f = os.path.join(OUT, "nullspace.json")
    if not os.path.isfile(f):
        print("[skip] nullspace")
        return
    d = json.load(open(f))
    keys = [k for k in d if ":" in k]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.0))
    for a, (fld, title, note) in zip(
        ax,
        [
            ("sens_iou", "VOLUME IoU sensitivity", "the proposed objective"),
            ("sens_ch", "CHAMFER sensitivity", "the objective it would replace"),
        ],
    ):
        vals = [d[k][fld] for k in keys]
        cols = [
            "#2b6cb0" if "BEND" in k else "#c53030" if ("SLIDE" in k or "STRETCH" in k) else "#a0aec0" for k in keys
        ]
        y = np.arange(len(keys))
        a.barh(y, vals, color=cols)
        a.set_yticks(y)
        a.set_yticklabels([k.replace(" (", "\n(") for k in keys], fontsize=7.6)
        a.invert_yaxis()
        a.set_xlabel("objective change per unit RMS vertex motion")
        a.set_title(f"{title}\n({note})", fontsize=10.5, weight="bold")
        a.grid(alpha=0.3, axis="x")
        rk = "random_joint_rot"
        if rk in d:
            a.axvline(d[rk][fld], color="black", ls="--", lw=1.3)
            a.text(
                d[rk][fld],
                len(keys) - 0.4,
                " generic pose perturbation",
                fontsize=7.5,
                rotation=90,
                va="top",
                ha="left",
            )
    handles = [
        plt.Line2D([], [], marker="s", ls="", color="#2b6cb0", label="BEND — rotation ⟂ limb axis"),
        plt.Line2D(
            [], [], marker="s", ls="", color="#c53030", label="SLIDE / STRETCH — joint placement along the axis"
        ),
        plt.Line2D([], [], marker="s", ls="", color="#a0aec0", label="TWIST — rotation ‖ axis"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(
        "V3 — does a VOLUMETRIC objective constrain joint placement better than the surface? "
        "No: chamfer favours SLIDE/STRETCH, IoU does not.",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    p = os.path.join(OUT, "fig_nullspace.png")
    fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_volumetric():
    ce = json.load(open(os.path.join(OUT, "ellipsoid_ceiling.json")))
    pc = json.load(open(os.path.join(OUT, "primitive_count.json")))
    pr = json.load(open(os.path.join(OUT, "pose_recovery.json")))
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.5))

    # 1. K sweep
    ks = sorted(int(k) for k in pc["by_k"])
    ax[0].plot(ks, [pc["by_k"][str(k)]["iou"] for k in ks], "-o", color="#2b6cb0", lw=2.4, label="union IoU")
    ax[0].plot(ks, [pc["by_k"][str(k)]["recall"] for k in ks], "--s", color="#dd6b20", lw=1.8, label="recall")
    ax[0].set_xscale("log")
    ax[0].set_xticks(ks)
    ax[0].set_xticklabels(ks)
    ax[0].set_xlabel("ellipsoids per skeleton segment")
    ax[0].set_title("More primitives is WORSE\n(surface clusters make hollow shells)", fontsize=10, weight="bold")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # 2. volume share
    gs_ = ce["group_volume_share"]
    items = sorted(gs_.items(), key=lambda kv: -kv[1])
    ax[1].barh(range(len(items)), [100 * v for _, v in items], color="#2b6cb0")
    ax[1].set_yticks(range(len(items)))
    ax[1].set_yticklabels([k for k, _ in items], fontsize=7.5)
    ax[1].invert_yaxis()
    ax[1].set_xscale("symlog", linthresh=1e-3)
    ax[1].set_xlabel("% of body VOLUME")
    ax[1].set_title("Distal legs are ~0% of volume\n(29 of 55 segments < 0.1%)", fontsize=10, weight="bold")
    ax[1].grid(alpha=0.3, axis="x")

    # 3. per-segment IoU
    pg = pc["per_group_stratified"]
    it = sorted(pg.items(), key=lambda kv: kv[1]["iou"])
    cols = ["#c53030" if v["iou"] < 0.4 else "#dd6b20" if v["iou"] < 0.6 else "#38a169" for _, v in it]
    ax[2].barh(range(len(it)), [v["iou"] for _, v in it], color=cols)
    ax[2].set_yticks(range(len(it)))
    ax[2].set_yticklabels([k for k, _ in it], fontsize=7.5)
    ax[2].invert_yaxis()
    ax[2].set_xlabel("per-segment IoU (stratified)")
    ax[2].set_title("Where ellipsoids fail\n(antenna 0.11, mandible 0.30)", fontsize=10, weight="bold")
    ax[2].grid(alpha=0.3, axis="x")

    # 4. pose recovery
    fin = pr.get("SYN_clean/Stage_3_deform_fine")
    if fin:
        g = fin["per_group"]
        names = sorted(g, key=lambda k: g[k]["median"])
        rot = [g[k]["median"] for k in names]
        pos = [100 * g[k]["jpos_median"] for k in names if g[k].get("jpos_median") is not None]
        y = np.arange(len(names))
        ax[3].barh(y - 0.2, rot, 0.4, color="#a0aec0", label="rotation error (deg)")
        ax[3].barh(y + 0.2, pos, 0.4, color="#2b6cb0", label="joint POSITION error (% extent)")
        ax[3].set_yticks(y)
        ax[3].set_yticklabels(names, fontsize=7.5)
        ax[3].invert_yaxis()
        ax[3].legend(fontsize=8)
        ax[3].set_title(
            "V2 — 27° rotation error is mostly GAUGE\n(positions are off 4% of extent)", fontsize=10, weight="bold"
        )
        ax[3].grid(alpha=0.3, axis="x")
    fig.suptitle("The volumetric proposal, measured before building it — and why it was dropped", fontsize=12.5)
    plt.tight_layout()
    p = os.path.join(OUT, "fig_volumetric.png")
    fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def render_e8(n_show=3):
    """Shaded renders so the SHAPE is judgeable, plus a per-vertex offset panel.

    Rows are chosen at the 20th / 50th / 90th percentile of D0's deform magnitude, so the
    figure shows a typical specimen rather than only the pathological ones.
    """
    import torch
    import trimesh
    from render_3d import DEV, colour_from_scalar, make_renderer, render

    merged = {a: os.path.join(MOON, "runs", f"{a}_MERGED", "Stage_3_deform_fine.npz") for a, _, _ in ARMS}
    if not all(os.path.isfile(p) for p in merged.values()):
        print("[skip] render: merged npz missing")
        return
    D = {a: np.load(p, allow_pickle=True) for a, p in merged.items()}
    labels = [str(x) for x in D["D0_control"]["labels"]]
    faces = torch.tensor(D["D0_control"]["faces"].astype(np.int64), device=DEV)
    if faces.ndim == 3:
        faces = faces[0]
    met = {a: arm_metrics(a) for a, _, _ in ARMS}

    renderer = make_renderer(image_size=430, elev=24.0, azim=140.0)
    dm = np.linalg.norm(D["D0_control"]["deform_verts"], axis=-1).mean(1)
    order = [int(np.argsort(dm)[int(q * (len(dm) - 1))]) for q in (0.20, 0.50, 0.90)]
    qlab = ["20th pct (easy)", "50th pct (typical)", "90th pct (hard)"]

    for tag, shade in (("shape", True), ("offset", False)):
        fig, axes = plt.subplots(len(order), 4, figsize=(15.5, 3.9 * len(order)))
        axes = np.atleast_2d(axes)
        for r, i in enumerate(order):
            nm = labels[i]
            tp = next(
                (
                    os.path.join(MOON, f"e8_c{c}", nm)
                    for c in (0, 1)
                    if os.path.exists(os.path.join(MOON, f"e8_c{c}", nm))
                ),
                None,
            )
            if tp:
                tm = trimesh.load(tp, process=False, force="mesh")
                tv = np.asarray(tm.vertices)
                tv = (tv - tv.mean(0)) / np.abs(tv - tv.mean(0)).max()
                tvt = torch.tensor(tv, dtype=torch.float32, device=DEV)
                tft = torch.tensor(np.asarray(tm.faces).astype(np.int64), device=DEV)
                axes[r, 0].imshow(render(tvt, tft, torch.full((len(tv), 3), 0.62, device=DEV), renderer))
            axes[r, 0].set_title(f"TARGET — {qlab[r]}\n{nm[:32]}", fontsize=8.5)
            axes[r, 0].axis("off")
            for c, (arm, lbl, _) in enumerate(ARMS):
                v = torch.tensor(D[arm]["verts"][i], dtype=torch.float32, device=DEV)
                off = np.linalg.norm(D[arm]["deform_verts"][i], axis=-1)
                if shade:
                    col = torch.full((v.shape[0], 3), 0.62, device=DEV)
                else:
                    col = torch.tensor(
                        colour_from_scalar(off, 0.0, 0.03, cmap="inferno"), dtype=torch.float32, device=DEV
                    )
                axes[r, c + 1].imshow(render(v, faces, col, renderer))
                f1 = met[arm].iloc[i]["fscore@0.01"] if met[arm] is not None and i < len(met[arm]) else float("nan")
                axes[r, c + 1].set_title(
                    f"{lbl.splitlines()[0]}\nfscore@0.01 {f1:.3f}   |offset| {off.mean():.4f}", fontsize=8.5
                )
                axes[r, c + 1].axis("off")
        if shade:
            fig.suptitle(
                "E8 — does freezing free-form deformation still look like an ant? "
                "(neutral shading, so shape is judgeable)",
                fontsize=12.5,
            )
        else:
            sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 0.03))
            fig.colorbar(
                sm,
                ax=axes,
                orientation="horizontal",
                fraction=0.021,
                pad=0.015,
                label="per-vertex free-form offset magnitude (fraction of extent)",
            )
            fig.suptitle("E8 — how much the mesh had to be bent to reach the target", fontsize=12.5)
        p_ = os.path.join(OUT, f"render_e8_{tag}.png")
        fig.savefig(p_, dpi=112, bbox_inches="tight")
        plt.close(fig)
        print("wrote", p_)


def render_best(n_show=4, rank_by="D2_frozen"):
    """The BEST fits, ranked by the hardest arm, so the ceiling is visible rather than the failures."""
    import torch
    import trimesh
    from render_3d import DEV, make_renderer, render

    merged = {a: os.path.join(MOON, "runs", f"{a}_MERGED", "Stage_3_deform_fine.npz") for a, _, _ in ARMS}
    if not all(os.path.isfile(p) for p in merged.values()):
        return
    D = {a: np.load(p, allow_pickle=True) for a, p in merged.items()}
    labels = [str(x) for x in D["D0_control"]["labels"]]
    faces = torch.tensor(D["D0_control"]["faces"].astype(np.int64), device=DEV)
    if faces.ndim == 3:
        faces = faces[0]
    met = {a: arm_metrics(a).reset_index(drop=True) for a, _, _ in ARMS}
    score = met[rank_by]["fscore@0.01"].values
    order = list(np.argsort(-score)[:n_show])

    renderer = make_renderer(image_size=430, elev=24.0, azim=140.0)
    fig, axes = plt.subplots(len(order), 4, figsize=(15.5, 3.9 * len(order)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(order):
        nm = labels[i]
        tp = next(
            (os.path.join(MOON, f"e8_c{c}", nm) for c in (0, 1) if os.path.exists(os.path.join(MOON, f"e8_c{c}", nm))),
            None,
        )
        if tp:
            tm = trimesh.load(tp, process=False, force="mesh")
            tv = np.asarray(tm.vertices)
            tv = (tv - tv.mean(0)) / np.abs(tv - tv.mean(0)).max()
            axes[r, 0].imshow(
                render(
                    torch.tensor(tv, dtype=torch.float32, device=DEV),
                    torch.tensor(np.asarray(tm.faces).astype(np.int64), device=DEV),
                    torch.full((len(tv), 3), 0.62, device=DEV),
                    renderer,
                )
            )
        axes[r, 0].set_title(f"TARGET\n{nm[:34]}", fontsize=8.5)
        axes[r, 0].axis("off")
        for c, (arm, lbl, _) in enumerate(ARMS):
            v = torch.tensor(D[arm]["verts"][i], dtype=torch.float32, device=DEV)
            axes[r, c + 1].imshow(render(v, faces, torch.full((v.shape[0], 3), 0.62, device=DEV), renderer))
            f1 = met[arm].iloc[i]["fscore@0.01"]
            e = met[arm].iloc[i]["edge_logratio_absmean"]
            axes[r, c + 1].set_title(f"{lbl.splitlines()[0]}\nfscore@0.01 {f1:.3f}   edge dist {e:.3f}", fontsize=8.5)
            axes[r, c + 1].axis("off")
    fig.suptitle(
        "E8 — the BEST fits (ranked by the deform-free arm). This is what the parametric model alone can do.",
        fontsize=12.5,
    )
    p_ = os.path.join(OUT, "render_e8_best.png")
    fig.savefig(p_, dpi=112, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p_)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_tradeoff()
    fig_probe19()
    fig_nullspace()
    fig_volumetric()
    render_e8()
    render_best()
