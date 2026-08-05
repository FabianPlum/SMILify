"""VERIFY PROBE: when the distal anchors are dropped by min_pts=40, how far do the
UNANCHORED distal segments travel during the 400-iteration init, and is that motion
scored by anything?

Ideal case: target == the model's own rest mesh, PERFECT labels. The correct answer is
"change nothing". Arm A = pipeline default (min_pts=40, 6 distal groups dropped).
Arm B = min_pts=5 (all 13 anchored). Per-group displacement + distal joint rotation.

CPU ONLY.
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

import config
from fitter_3d.part_anchor_init import PartAnchorInit
from fitter_3d.trainer import SMAL3DFitter
from fitter_3d.trainer_hierarchical import anatomical_groups, vertex_groups

dev = "cpu"
with open(config.SMAL_FILE, "rb") as f:
    u = pickle._Unpickler(f)
    u.encoding = "latin1"
    dd = u.load()
jnames = list(dd["J_names"])
vg, gnames = vertex_groups(np.asarray(dd["weights"]), jnames, split_distal=True)
vg_t = torch.tensor(vg)
jg = anatomical_groups(jnames, split_distal=True)


class Stub:
    def __init__(self, assign):
        self.assign = assign


def arm(tag, min_pts, n_it=400):
    torch.manual_seed(0)
    np.random.seed(0)
    smal = SMAL3DFitter(batch_size=1, device=dev, shape_family=-1)
    with torch.no_grad():
        rest = smal(deform_verts=torch.zeros_like(smal.deform_verts))[0].clone()
    faces = smal.faces[0]
    pts = sample_points_from_meshes(Meshes(verts=[rest], faces=[faces]), 8000).detach()
    idx = knn_points(pts, rest[None], K=1).idx[0, :, 0]
    assign = vg_t[idx][None]
    extent = float((rest.amax(0) - rest.amin(0)).amax())

    init = PartAnchorInit(smal, Stub(assign), vg, gnames, dev, n_it=n_it, min_pts=min_pts, log_every=200)
    anchored = sorted(init._targets(pts)[0].keys())
    init.run(pts)
    with torch.no_grad():
        out = smal(deform_verts=torch.zeros_like(smal.deform_verts))[0]
        disp = (out - rest).norm(dim=-1)
        jr = smal.joint_rot[0].norm(dim=-1) * 180 / np.pi
    print(f"\n=== ARM {tag} (min_pts={min_pts}) : anchored {[gnames[a] for a in anchored]} ===")
    print(f"{'group':8s} {'anchored':>8s} {'mean disp %ext':>15s} {'max disp %ext':>14s} {'mean |jrot| deg':>16s}")
    for g, name in enumerate(gnames):
        m = vg_t == g
        jrows = [j - 1 for j in range(1, len(jnames)) if jg[j] == name]
        jm = float(jr[jrows].mean()) if jrows else float("nan")
        print(
            f"{name:8s} {str(g in anchored):>8s} {100 * float(disp[m].mean()) / extent:15.2f} "
            f"{100 * float(disp[m].max()) / extent:14.2f} {jm:16.2f}"
        )


arm("A_default", 40)
arm("B_minpts5", 5)
