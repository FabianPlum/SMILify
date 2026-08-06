"""Render fitted meshes against their targets, per arm, so the metrics can be checked
against what the geometry actually looks like.

Two views matter for this failure mode:
  * lateral (XZ) -- shows whether legs reached the target or stayed near the rest pose
  * anterior (YZ) -- shows bilateral symmetry, and whether the template got dragged
    sideways onto one side of the scan

Also renders a WIREFRAME-DENSITY view: colouring the fitted mesh by per-vertex
|deform_verts| makes visible exactly where free-form deformation is doing the work that
pose should have done. That is the difference between a registration and a shrink-wrap,
and no scalar metric communicates it as directly.
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from pytorch3d.io import load_obj  # noqa: E402
from pytorch3d.structures import Meshes  # noqa: E402
from pytorch3d.ops import sample_points_from_meshes  # noqa: E402

DEV = "cuda:0"


def norm(v):
    v = v - v.mean(0)
    return v / v.abs().max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs to compare")
    ap.add_argument("--mesh_dir", required=True)
    ap.add_argument("--n_specimens", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.mesh_dir, "*.obj")))
    sel = np.linspace(0, len(files) - 1, args.n_specimens).astype(int)

    # target point clouds
    tgt = {}
    for i in sel:
        v, f, _ = load_obj(files[i], load_textures=False)
        v = norm(v.to(DEV))
        m = Meshes(verts=[v], faces=[f.verts_idx.to(DEV)])
        tgt[i] = sample_points_from_meshes(m, 9000)[0].cpu().numpy()

    # final-stage fits per run
    fits = {}
    for rd in args.runs:
        npzs = sorted([p for p in glob.glob(os.path.join(rd, "*.npz")) if "_batch_" not in p])
        if not npzs:
            print(f"  skip {rd}: no npz")
            continue
        d = np.load(npzs[-1], allow_pickle=True)
        fits[os.path.basename(rd)] = (d["verts"], d["faces"][0], d["deform_verts"], os.path.basename(npzs[-1]))

    names = list(fits.keys())
    for view, (a, b), vname in [((0, 2), (0, 2), "lateral_XZ"), ((1, 2), (1, 2), "anterior_YZ")]:
        ncol = len(names) + 1
        fig, axes = plt.subplots(len(sel), ncol, figsize=(3.0 * ncol, 2.8 * len(sel)))
        axes = np.atleast_2d(axes)
        for r, i in enumerate(sel):
            axes[r, 0].scatter(tgt[i][:, a], tgt[i][:, b], s=0.4, alpha=0.4, c="#666", linewidths=0)
            axes[r, 0].set_aspect("equal")
            axes[r, 0].set_title("TARGET" if r == 0 else "", fontsize=9)
            axes[r, 0].set_ylabel(os.path.basename(files[i])[:18], fontsize=6)
            axes[r, 0].tick_params(labelsize=5)
            for c, nm in enumerate(names):
                verts, faces, dv, _ = fits[nm]
                V = verts[i]
                mag = np.linalg.norm(dv[i], axis=-1)
                ax = axes[r, c + 1]
                ax.scatter(tgt[i][:, a], tgt[i][:, b], s=0.4, alpha=0.15, c="#bbb", linewidths=0)
                ax.scatter(V[:, a], V[:, b], s=0.5, alpha=0.7, c=mag, cmap="inferno", vmin=0, vmax=0.06, linewidths=0)
                ax.set_aspect("equal")
                ax.tick_params(labelsize=5)
                if r == 0:
                    ax.set_title(nm, fontsize=8)
        fig.suptitle(
            f"{vname}: fitted mesh coloured by |deform_verts| (black=pose/shape did the work, "
            f"bright=free-form offset did it). Grey = target.",
            fontsize=10,
        )
        plt.tight_layout()
        p = os.path.join(args.out, f"fits_{vname}.png")
        fig.savefig(p, dpi=105)
        plt.close(fig)
        print("wrote", p)


if __name__ == "__main__":
    main()
