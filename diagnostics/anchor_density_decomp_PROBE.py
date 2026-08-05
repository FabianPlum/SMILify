"""PROBE: decompose the source/target anchor mismatch into (a) vertex-vs-area DENSITY and
(b) label-boundary effects.  If area-weighted vertex moments match the area-sampled
moments, density is the whole story and the claim's fix direction is right.  CPU only.
"""

import os
import pickle
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/home/fabi/dev/SMILify")

import numpy as np
import torch
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes

from fitter_3d.trainer_hierarchical import vertex_groups

with open("/home/fabi/dev/SMILify/3D_model_prep/SMIL_OmniAnt.pkl", "rb") as f:
    u = pickle._Unpickler(f)
    u.encoding = "latin1"
    dd = u.load()
v = torch.tensor(np.asarray(dd["v_template"]), dtype=torch.float64)
fc = torch.tensor(np.asarray(dd["f"]), dtype=torch.int64)
vg, gnames = vertex_groups(np.asarray(dd["weights"]), list(dd["J_names"]), split_distal=True)
vg_t = torch.tensor(vg)

# barycentric vertex areas: 1/3 of each incident triangle's area
tv = v[fc]
tri_a = 0.5 * torch.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0], dim=-1).norm(dim=-1)
w = torch.zeros(v.shape[0], dtype=torch.float64)
for k in range(3):
    w.index_add_(0, fc[:, k], tri_a / 3.0)

torch.manual_seed(0)
mesh = Meshes(verts=[v.float()], faces=[fc])
pts = sample_points_from_meshes(mesh, 200000)[0].double()  # dense, low MC noise
lab = vg_t[knn_points(pts[None].float(), v[None].float(), K=1).idx[0, :, 0]]

extent = float((v.amax(0) - v.amin(0)).amax())


def mom(p, wt=None):
    if wt is None:
        wt = torch.ones(p.shape[0], dtype=p.dtype)
    wn = wt / wt.sum()
    c = (wn[:, None] * p).sum(0)
    d = p - c
    return c, (d * wn[:, None]).t() @ d


print(f"{'group':10s} | {'VERTEX vs AREA':>28s} | {'AREAWT-VERT vs AREA':>28s}")
print(f"{'':10s} | {'dc%ext':>9s} {'trM ratio':>9s} {'dM_F%':>7s} | {'dc%ext':>9s} {'trM ratio':>9s} {'dM_F%':>7s}")
for g, name in enumerate(gnames):
    vm = vg_t == g
    pm = lab == g
    if int(vm.sum()) < 4 or int(pm.sum()) < 40:
        continue
    cv, Mv = mom(v[vm])
    cw, Mw = mom(v[vm], w[vm])
    ca, Ma = mom(pts[pm])
    nf = float(Ma.norm())
    print(
        f"{name:10s} | {100 * float((cv - ca).norm()) / extent:9.3f} "
        f"{float(Mv.trace() / Ma.trace()):9.3f} {100 * float((Mv - Ma).norm()) / nf:7.2f} | "
        f"{100 * float((cw - ca).norm()) / extent:9.3f} "
        f"{float(Mw.trace() / Ma.trace()):9.3f} {100 * float((Mw - Ma).norm()) / nf:7.2f}"
    )
