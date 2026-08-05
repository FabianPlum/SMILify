"""PROBE: is the part-anchor init's source (vertex) vs target (area) density mismatch real
and large?  CPU only.
"""

import os
import pickle

import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys

sys.path.insert(0, "/home/fabi/dev/SMILify")

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

from fitter_3d.trainer_hierarchical import vertex_groups

PKL = "/home/fabi/dev/SMILify/3D_model_prep/SMIL_OmniAnt.pkl"
with open(PKL, "rb") as f:
    u = pickle._Unpickler(f)
    u.encoding = "latin1"
    dd = u.load()

print("keys:", sorted(dd.keys()))
v = np.asarray(dd["v_template"], dtype=np.float64)
faces = np.asarray(dd["f"], dtype=np.int64)
W = np.asarray(dd["weights"], dtype=np.float64)
jnames = list(dd["J_names"])
print("verts", v.shape, "faces", faces.shape, "weights", W.shape, "joints", len(jnames))

vg, gnames = vertex_groups(W, jnames, split_distal=True)
print("groups", len(gnames), gnames)
print("counts", {g: int((vg == i).sum()) for i, g in enumerate(gnames)})

vt = torch.tensor(v, dtype=torch.float32)
ft = torch.tensor(faces, dtype=torch.int64)
mesh = Meshes(verts=[vt], faces=[ft])

torch.manual_seed(0)
N = 30000
pts = sample_points_from_meshes(mesh, N)[0]  # (N,3) area-uniform

# PERFECT labels for the sampled points: nearest template vertex's group
from pytorch3d.ops import knn_points

idx = knn_points(pts[None], vt[None], K=1).idx[0, :, 0]
lab = torch.tensor(vg)[idx]

extent = float((vt.amax(0) - vt.amin(0)).amax())
print(f"\nspecimen bbox extent (max axis) = {extent:.6f}")


def moments(p):
    c = p.mean(0)
    d = p - c
    return c, (d.t() @ d) / p.shape[0]


print(
    f"\n{'group':10s} {'nv':>6s} {'npts':>6s} {'|dc|/extent%':>13s} "
    f"{'|dc|/partext%':>14s} {'trMv/trMa':>10s} {'implied scale':>13s}"
)
rows = []
for g, name in enumerate(gnames):
    vm = torch.tensor(vg) == g
    pm = lab == g
    if int(vm.sum()) < 4 or int(pm.sum()) < 40:
        continue
    cv, Mv = moments(vt[vm].double())
    ca, Ma = moments(pts[pm].double())
    dc = float((cv - ca).norm())
    partext = float((vt[vm].amax(0) - vt[vm].amin(0)).amax())
    r = float(Mv.trace() / Ma.trace())
    rows.append(
        (
            name,
            int(vm.sum()),
            int(pm.sum()),
            100 * dc / extent,
            100 * dc / partext,
            r,
            r**0.5,
        )
    )
    print(
        f"{name:10s} {int(vm.sum()):6d} {int(pm.sum()):6d} {100 * dc / extent:13.3f} {100 * dc / partext:14.3f} {r:10.3f} {r**0.5:13.3f}"
    )

# anchor loss at the identity pose (i.e. source geometry == target geometry exactly)
mu = 1.0
s2 = extent**2
loss = 0.0
nt = 0
for g in range(len(gnames)):
    vm = torch.tensor(vg) == g
    pm = lab == g
    if int(vm.sum()) < 4 or int(pm.sum()) < 40:
        continue
    cv, Mv = moments(vt[vm].double())
    ca, Ma = moments(pts[pm].double())
    loss = loss + float((cv - ca).pow(2).sum()) / s2 + mu * float((Mv - Ma).pow(2).sum()) / s2**2
    nt += 1
print(f"\nanchors used: {nt}/{len(gnames)}   (min_pts=40)")
print(f"anchor_loss at the EXACT answer (source==target geometry) = {loss / max(nt, 1):.6e}")

# control: identical density -- target point set = the template's own vertices
loss2 = 0.0
nt2 = 0
for g in range(len(gnames)):
    vm = torch.tensor(vg) == g
    if int(vm.sum()) < 40:
        continue
    cv, Mv = moments(vt[vm].double())
    loss2 = loss2 + float((cv - cv).pow(2).sum()) / s2 + mu * float((Mv - Mv).pow(2).sum()) / s2**2
    nt2 += 1
print(f"control (identical density): anchor_loss = {loss2 / max(nt2, 1):.6e} over {nt2} anchors")
