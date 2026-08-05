"""Qualitative check on the predicted part field: what does it actually say about real scans?

Three figures, each answering a question the gate numbers cannot:

  1. partfield_pred.png       Does the field look anatomically sensible on ordinary scans?
                              Two views per specimen so a leg that is right in top-down but
                              wrong in side view cannot hide.
  2. partfield_hard.png       The specimens where the geodesic level-set method (probe 14)
                              collapsed to <= 2 branches because limbs touch. This is the
                              case the whole learned-classifier argument rests on, so it gets
                              its own figure rather than a row in a summary.
  3. partfield_vs_fit.png     Field partition vs the fit-derived partition it replaces, with
                              the disagreement highlighted. If the two agreed everywhere the
                              exercise would be pointless; WHERE they disagree is the result.
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

import config  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
from fitter_3d.partfield import (  # noqa: E402
    PartFieldNet,
    predict_field,
    normalise,
    template_vertex_labels,
)
from pytorch3d.ops import knn_points, sample_points_from_meshes  # noqa: E402

OUT = os.path.join(HERE, "out")
DATA = os.path.join(HERE, "partfield")
COLS = {
    "body": "#8d99a6",
    "l1p": "#0b3d91",
    "l1d": "#4d94ff",
    "l2p": "#8a4b00",
    "l2d": "#ffa64d",
    "l3p": "#0d5c2f",
    "l3d": "#4dd97f",
    "debris": "#ff00aa",
}


def cmap(names):
    return [COLS[n] if n in COLS else COLS[n[:3]] for n in names]


def draw(ax, p, l, cols, names, title, view="top", hl=None):
    ax0, ax1 = (0, 1) if view == "top" else ((0, 2) if view == "side" else (1, 2))
    is_r = [names[i].endswith("_r") for i in range(len(names))]
    for c in range(len(names)):
        m = l == c
        if not m.any():
            continue
        ax.scatter(p[m, ax0], p[m, ax1], s=0.6, c=cols[c], alpha=0.28 if is_r[c] else 0.95, linewidths=0)
    if hl is not None and hl.any():
        ax.scatter(p[hl, ax0], p[hl, ax1], s=2.2, facecolors="none", edgecolors="#e10600", linewidths=0.35, alpha=0.8)
    ax.set_title(title, fontsize=7)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=os.path.join(DATA, "net_B_mirror.pt"))
    ap.add_argument("--mesh_dir", default=os.path.join(HERE, "bench50"))
    ap.add_argument("--fit_npz", default=os.path.join(HERE, "runs", "M7_handoff_midline", "Stage_3_deform_fine.npz"))
    # 25k points x 14 classes x ~40 axes is minutes of matplotlib per figure and adds
    # nothing visible at print size; 12k reads identically and halves the wall clock.
    ap.add_argument("--n_pts", type=int, default=12000)
    args = ap.parse_args()

    dev = torch.device("cuda")
    ck = torch.load(args.net, map_location=dev)
    names = ck["names"]
    net = PartFieldNet(n_classes=len(names), width=ck.get("width", 1.0)).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    cols = cmap(names)

    files = sorted(f for f in os.listdir(args.mesh_dir) if f.endswith(".obj"))
    _, meshes = load_meshes(mesh_dir=args.mesh_dir, sorting=sorted, device=str(dev))
    short = [f.split("_CASENT")[0].split("_OKENT")[0][:24] for f in files]

    def field(i, n=None):
        # int() is load-bearing: pytorch3d's Meshes.__getitem__ rejects numpy int64, which
        # is what rng.choice and files.index return.
        p = sample_points_from_meshes(meshes[int(i)], n or args.n_pts)[0]
        pn, c, s = normalise(p)
        return pn.cpu().numpy(), predict_field(net, pn, seed=0).argmax(-1).cpu().numpy(), p, c, s

    # ---------------------------------------------------------------- fig 1: ordinary scans
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(files), 8, replace=False))
    fig, axes = plt.subplots(2, 8, figsize=(23, 6.2))
    for k, i in enumerate(pick):
        p, l, *_ = field(i)
        draw(axes[0][k], p, l, cols, names, f"{short[i]}\ntop (x-y)", "top")
        draw(axes[1][k], p, l, cols, names, "side (x-z)", "side")
    fig.legend(
        handles=[plt.Line2D([], [], marker="o", ls="", color=v, label=k, markersize=5) for k, v in COLS.items()],
        loc="lower center",
        ncol=8,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        "Predicted part field on real scans (right-side parts drawn faint). Nothing here was ever hand-labelled.",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(os.path.join(OUT, "partfield_pred.png"), dpi=110)
    plt.close(fig)
    print("wrote partfield_pred.png")

    # ---------------------------------------------------------------- fig 2: the hard cases
    p14 = json.load(open(os.path.join(OUT, "probe14_topology_gate.json")))
    hard = [r["name"] for r in sorted(p14["rows"], key=lambda r: r["best_seed0"])][:8]
    hidx = [files.index(h + "_processed.obj") for h in hard if h + "_processed.obj" in files]
    nb = {r["name"]: r["best_seed0"] for r in p14["rows"]}
    fig, axes = plt.subplots(2, len(hidx), figsize=(2.9 * len(hidx), 6.2))
    for k, i in enumerate(hidx):
        p, l, *_ = field(i)
        nm = files[i].replace("_processed.obj", "")
        legs = sum(
            int((np.isin(l, [j for j, n in enumerate(names) if n.startswith(g) and n.endswith("_" + s)])).sum() >= 50)
            for g in ("l1", "l2", "l3")
            for s in ("l", "r")
        )
        draw(
            axes[0][k],
            p,
            l,
            cols,
            names,
            f"{short[i]}\ngeodesic found {nb[nm]} branches\nPART FIELD: {legs}/6 legs",
            "top",
        )
        draw(axes[1][k], p, l, cols, names, "side (x-z)", "side")
    fig.suptitle(
        "The cases that defeated the geodesic level-set method (probe 14): limbs "
        "touching, so no bottleneck exists to cut.",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(os.path.join(OUT, "partfield_hard.png"), dpi=110)
    plt.close(fig)
    print("wrote partfield_hard.png")

    # ---------------------------------------------------------------- fig 3: field vs fit
    import pickle

    with open(config.SMAL_FILE, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    jn = [n.decode() if isinstance(n, bytes) else n for n in dd["J_names"]]
    vl = torch.tensor(template_vertex_labels(dd["weights"], jn), device=dev)
    fit = np.load(args.fit_npz)
    fv = torch.tensor(fit["verts"], dtype=torch.float32, device=dev)

    fig, axes = plt.subplots(3, 6, figsize=(18, 9.4))
    dis = []
    for k, i in enumerate(pick[:6]):
        pn, l, p_raw, c, s = field(i)
        j = knn_points(p_raw.unsqueeze(0), fv[i].unsqueeze(0), K=1).idx[0, :, 0]
        lfit = vl[j].cpu().numpy()
        d = l != lfit
        dis.append(float(d.mean()))
        draw(axes[0][k], pn, lfit, cols, names, f"{short[i]}\nFIT-derived partition", "top")
        draw(axes[1][k], pn, l, cols, names, "PART FIELD (target-derived)", "top")
        draw(axes[2][k], pn, l, cols, names, f"disagreement {100 * d.mean():.1f}%", "top", hl=d)
    fig.suptitle(
        "Fit-derived partition (top) vs learned part field (middle); where they "
        "disagree, circled (bottom). The field is computed from the scan alone and "
        "never moves during fitting.",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(os.path.join(OUT, "partfield_vs_fit.png"), dpi=110)
    plt.close(fig)
    print(f"wrote partfield_vs_fit.png  (mean disagreement {100 * np.mean(dis):.1f}%)")


if __name__ == "__main__":
    main()
