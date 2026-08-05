"""PROBE: end-to-end. Target == the model's own rest mesh, PERFECT part labels, so the
correct answer is 'change nothing'.  Does PartAnchorInit move it?  CPU only.

Arm A: target points area-sampled (what the pipeline actually does).
Arm B (control): target points ARE the template vertices (identical density).
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/home/fabi/dev/SMILify")

import pickle

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes

import config
from fitter_3d.part_anchor_init import PartAnchorInit
from fitter_3d.trainer import SMAL3DFitter
from fitter_3d.trainer_hierarchical import vertex_groups

torch.manual_seed(0)
np.random.seed(0)
dev = "cpu"

with open(config.SMAL_FILE, "rb") as f:
    u = pickle._Unpickler(f)
    u.encoding = "latin1"
    dd = u.load()
jnames = list(dd["J_names"])
vg, gnames = vertex_groups(np.asarray(dd["weights"]), jnames, split_distal=True)
vg_t = torch.tensor(vg)


class Stub:
    def __init__(self, assign):
        self.assign = assign


def run_arm(tag, area_sampled, min_pts=40, n_it=400):
    smal = SMAL3DFitter(batch_size=1, device=dev, shape_family=-1)
    with torch.no_grad():
        rest = smal(deform_verts=torch.zeros_like(smal.deform_verts))[0].clone()  # (V,3)
    faces = smal.faces[0]
    tgt_mesh = Meshes(verts=[rest], faces=[faces])
    if area_sampled:
        pts = sample_points_from_meshes(tgt_mesh, 8000).detach()  # (1,N,3) AREA-uniform
    else:
        pts = rest[None].clone().detach()  # identical density
    # PERFECT labels: nearest rest vertex's anatomical group
    idx = knn_points(pts, rest[None], K=1).idx[0, :, 0]
    assign = vg_t[idx][None]

    extent = float((rest.amax(0) - rest.amin(0)).amax())
    ch0 = float(chamfer_distance(rest[None], pts)[0])

    init = PartAnchorInit(smal, Stub(assign), vg, gnames, dev, n_it=n_it, min_pts=min_pts, log_every=100)
    hist = init.run(pts)

    with torch.no_grad():
        out = smal(deform_verts=torch.zeros_like(smal.deform_verts))[0]
        ch1 = float(chamfer_distance(out[None], pts)[0])
        disp = (out - rest).norm(dim=-1)
        jr = smal.joint_rot[0].norm(dim=-1) * 180 / np.pi
        sc = smal.get_joint_scales()[0] if hasattr(smal, "get_joint_scales") else smal.log_beta_scales[0].exp()
    top = torch.topk(jr, 5)
    print(f"\n=== ARM {tag} (area_sampled={area_sampled}, min_pts={min_pts}) ===")
    print(f"  anchor_loss  {hist[0]:.6e} -> {hist[-1]:.6e}")
    print(f"  chamfer      {ch0:.6e} -> {ch1:.6e}   ({ch1 / max(ch0, 1e-30):.1f}x)")
    print(
        f"  vertex displacement: mean {100 * float(disp.mean()) / extent:.2f}% of extent, max {100 * float(disp.max()) / extent:.2f}%"
    )
    print(
        "  joint rot top5 (deg): "
        + ", ".join(f"{jnames[int(i) + 1]}={float(v):.1f}" for v, i in zip(top.values, top.indices))
    )
    print(f"  per-joint scale range: {float(sc.min()):.3f}..{float(sc.max()):.3f}")
    return ch0, ch1


run_arm("A_area", True, min_pts=40)
run_arm("B_control_same_density", False, min_pts=40)
run_arm("C_area_minpts10", True, min_pts=10)
