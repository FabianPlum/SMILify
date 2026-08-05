"""Proper shaded 3D renders of fitted meshes, for qualitative inspection.

Point-cloud scatter plots hide exactly the failure this project is about: a mesh can be
covered in inverted, stretched and self-intersecting triangles while its point cloud looks
perfect. Shaded surface renders make mesh integrity visible.

Three render modes per specimen:
  surface   Phong-shaded fitted mesh. Broken geometry shows as speckle/creasing.
  deform    Fitted mesh coloured by per-vertex |deform_verts|. Bright regions are where
            free-form deformation did the work that pose/shape should have done -- i.e.
            where correspondence was traded away for surface accuracy.
  error     Fitted mesh coloured by distance to the nearest target surface point.

Uses pytorch3d's rasterizer (Ravi et al., "Accelerating 3D Deep Learning with PyTorch3D",
arXiv:2007.08501) so there is no external renderer dependency.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from pytorch3d.io import load_obj  # noqa: E402
from pytorch3d.structures import Meshes  # noqa: E402
from pytorch3d.ops import sample_points_from_meshes, knn_points  # noqa: E402
from pytorch3d.renderer import (  # noqa: E402
    FoVOrthographicCameras,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
    look_at_view_transform,
)

DEV = "cuda:0"


def make_renderer(image_size=420, dist=2.6, elev=22.0, azim=140.0):
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVOrthographicCameras(device=DEV, R=R, T=T, scale_xyz=((1.15, 1.15, 1.15),))
    raster = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
    lights = PointLights(device=DEV, location=[[2.0, 2.0, 2.0]])
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
        shader=SoftPhongShader(device=DEV, cameras=cameras, lights=lights),
    )


def colour_from_scalar(vals, vmin, vmax, cmap="inferno"):
    x = np.clip((vals - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    return torch.tensor(matplotlib.colormaps[cmap](x)[..., :3], dtype=torch.float32, device=DEV)


def render(mesh_verts, faces, colours, renderer):
    m = Meshes(verts=[mesh_verts], faces=[faces], textures=TexturesVertex(colours.unsqueeze(0)))
    img = renderer(m)[0, ..., :3].detach().cpu().numpy()
    return np.clip(img, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--mesh_dir", required=True)
    ap.add_argument("--n_specimens", type=int, default=4)
    ap.add_argument("--mode", default="deform", choices=["surface", "deform", "error"])
    ap.add_argument("--azim", type=float, default=140.0)
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    outdir = args.out or os.path.join(os.path.dirname(__file__), "out", "renders3d")
    os.makedirs(outdir, exist_ok=True)
    renderer = make_renderer(elev=args.elev, azim=args.azim)

    files = sorted(glob.glob(os.path.join(args.mesh_dir, "*.obj")))
    sel = np.linspace(0, len(files) - 1, args.n_specimens).astype(int)

    # targets, rendered plain grey for reference
    tgt_imgs, tgt_pts = {}, {}
    for i in sel:
        v, f, _ = load_obj(files[i], load_textures=False)
        v = v.to(DEV)
        v = v - v.mean(0)
        v = v / v.abs().max()
        fi = f.verts_idx.to(DEV)
        col = torch.full((v.shape[0], 3), 0.65, device=DEV)
        tgt_imgs[i] = render(v, fi, col, renderer)
        tgt_pts[i] = sample_points_from_meshes(Meshes(verts=[v], faces=[fi]), 30000)

    cols = []
    for rd in args.runs:
        npzs = sorted(
            [p for p in glob.glob(os.path.join(rd, "*.npz")) if "_batch_" not in p and "start_selection" not in p]
        )
        if not npzs:
            print(f"  skip {rd}")
            continue
        d = np.load(npzs[-1], allow_pickle=True)
        cols.append((os.path.basename(rd), d))

    ncol = len(cols) + 1
    fig, axes = plt.subplots(len(sel), ncol, figsize=(2.9 * ncol, 2.9 * len(sel)))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")

    for r, i in enumerate(sel):
        axes[r, 0].imshow(tgt_imgs[i])
        axes[r, 0].set_title("TARGET scan" if r == 0 else "", fontsize=9)
        axes[r, 0].text(
            -0.06,
            0.5,
            os.path.basename(files[i])[:20],
            rotation=90,
            va="center",
            ha="center",
            transform=axes[r, 0].transAxes,
            fontsize=6,
        )
        for c, (nm, d) in enumerate(cols):
            V = torch.tensor(d["verts"][i], dtype=torch.float32, device=DEV)
            F = torch.tensor(d["faces"][0].astype(np.int64), dtype=torch.int64, device=DEV)
            if args.mode == "surface":
                col = torch.full((V.shape[0], 3), 0.55, device=DEV)
                col[:, 2] = 0.75
            elif args.mode == "deform":
                mag = np.linalg.norm(d["deform_verts"][i], axis=-1)
                col = colour_from_scalar(mag, 0.0, 0.06)
            else:
                with torch.no_grad():
                    dist = knn_points(V.unsqueeze(0), tgt_pts[i], K=1).dists[0, :, 0].sqrt()
                col = colour_from_scalar(dist.cpu().numpy(), 0.0, 0.05, cmap="viridis")
            axes[r, c + 1].imshow(render(V, F, col, renderer))
            if r == 0:
                axes[r, c + 1].set_title(nm, fontsize=8)

    titles = {
        "surface": "Shaded surface — speckle/creasing = broken mesh integrity",
        "deform": "Coloured by |deform_verts|  (black = pose/shape explained it, "
        "bright = free-form offset did, i.e. correspondence traded away).  scale 0–0.06",
        "error": "Coloured by distance to target surface.  scale 0–0.05",
    }
    import textwrap

    fig.suptitle("\n".join(textwrap.wrap(titles[args.mode], 95)), fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    p = os.path.join(outdir, f"render3d_{args.mode}.png")
    fig.savefig(p, dpi=115)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
