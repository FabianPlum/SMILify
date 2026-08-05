"""How gameable is part_leg_distal_dist_mean really?

Three variants of the same M8a fit, per specimen:
  as-fit      the real fit
  best-rigid  each distal chain rigidly translated to the translation that MINIMISES its
              own mean distance-to-any-target-surface (200 steps Adam). This is the best a
              'give up and go somewhere cheap' strategy can do without deforming.
  snap        every distal vertex moved to its nearest target surface point (metric = 0 by
              construction; the global minimiser). Reachable in principle by H3_deform.
Reports where best-rigid moves the chain (toward the body centroid?) and what the HEADLINE
metrics do in each case.
"""

import os
import sys
import pickle
import numpy as np
import torch

REPO = "/home/fabi/dev/SMILify"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics/moonshot"))
os.chdir(REPO)
import config
import metrics as M
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, knn_points
from fitter_3d.utils import load_meshes

torch.manual_seed(0)
with open(config.SMAL_FILE, "rb") as f:
    u = pickle._Unpickler(f)
    u.encoding = "latin1"
    dd = u.load()
J = [n.decode() if isinstance(n, bytes) else n for n in dd["J_names"]]
W = np.asarray(dd["weights"])
dom = W.argmax(1)
part_ids, part_names = M.template_part_segmentation(W, list(dd["J_names"]))
pn2id = dict(zip(part_names, part_ids))
body_ids = pn2id["body"].numpy()
sym = torch.tensor(np.asarray(dd["sym_verts"]).astype(np.int64))

chains = {}
for j, nm in enumerate(J):
    if nm.startswith("l_") and any(s in nm for s in ["_ti_", "_ta_", "_pt_"]):
        chains.setdefault(nm[2] + nm[-2:], []).append(j)
chain_verts = {k: np.nonzero(np.isin(dom, v))[0] for k, v in chains.items()}

d = np.load("diagnostics/moonshot/runs/M8a_split_only/H3_deform.npz", allow_pickle=True)
verts = torch.tensor(d["verts"], dtype=torch.float32)
faces = torch.tensor(d["faces"][0].astype(np.int64))
dv = torch.tensor(d["deform_verts"], dtype=torch.float32)
labels = list(d["labels"])
sel = [0, 5, 12, 23, 41]
objs = [os.path.join("diagnostics/moonshot/bench50", labels[i]) for i in sel]
_, tgts = load_meshes(mesh_files=objs, device="cpu")


def metrics_of(v, i, tm):
    pm = Meshes(verts=[v], faces=[faces])
    torch.manual_seed(1234)
    p = M.part_metrics(pm, tm, part_ids, part_names, n_points=30000)
    torch.manual_seed(1234)
    s = M.surface_metrics(pm, tm, n_points=30000)
    g = M.deformation_metrics(v, verts[i] - dv[i], faces, v - (verts[i] - dv[i]))
    y = M.symmetry_metrics(v, sym, verts[i] - dv[i])
    return dict(
        part_leg_distal=p["part_leg_distal_dist_mean"],
        within_tau=p["part_leg_distal_within_tau"],
        chamfer_l2=s["chamfer_l2"],
        **{"fscore@0.02": s["fscore@0.02"]},
        comp_mean=s["comp_mean"],
        edge_lr=g["edge_logratio_absmean"],
        tri_q=g["tri_quality_mean"],
        folded=g["folded_face_frac"],
        deform=g["deform_mag_mean"],
        midline=y["midline_dev_mean"],
    )


agg = {}
for n, i in enumerate(sel):
    tm = Meshes(verts=[tgts.verts_list()[n]], faces=[tgts.faces_list()[n]])
    torch.manual_seed(7)
    tp = sample_points_from_meshes(tm, 30000)
    v0 = verts[i]
    bodyc = v0[body_ids].mean(0)

    # ---- best rigid translation per chain
    v_rig = v0.clone()
    info = []
    for k, ids in chain_verts.items():
        base = v0[ids]
        t = torch.zeros(3, requires_grad=True)
        opt = torch.optim.Adam([t], lr=0.02)
        for step in range(250):
            opt.zero_grad()
            loss = knn_points((base + t).unsqueeze(0), tp, K=1).dists[..., 0].sqrt().mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            v_rig[ids] = base + t
            d0 = knn_points(base.unsqueeze(0), tp, K=1).dists[..., 0].sqrt().mean().item()
            d1 = loss.item()
            c0 = base.mean(0)
            c1 = (base + t).mean(0)
            toward = ((c0 - bodyc).norm() - (c1 - bodyc).norm()).item()
            info.append((k, d0, d1, t.detach().norm().item(), toward))

    # ---- oracle snap (global minimiser of the metric)
    v_snap = v0.clone()
    all_d = np.concatenate([v for v in chain_verts.values()])
    idx = knn_points(v0[all_d].unsqueeze(0), tp, K=1).idx[0, :, 0]
    v_snap[all_d] = tp[0][idx]

    for tag, v in (("as-fit", v0), ("best-rigid", v_rig), ("snap", v_snap)):
        r = metrics_of(v, i, tm)
        agg.setdefault(tag, []).append(r)
    print(f"\n{labels[i][:40]}")
    print("   chain  d_meanature->  as-fit   best-rigid   |t|     moved-toward-body(+)")
    for k, d0, d1, tn, tw in info:
        print(f"   {k:<7}            {d0:8.5f}   {d1:8.5f}  {tn:7.4f}   {tw:+7.4f}")

keys = list(agg["as-fit"][0].keys())
inhead = {"chamfer_l2", "fscore@0.02", "edge_lr", "tri_q", "midline", "part_leg_distal", "deform"}
print("\n" + "=" * 92)
print(f"{'metric':<18}{'as-fit':>12}{'best-rigid':>12}{'snap':>12}   in compare_arms HEADLINE?")
for k in keys:
    a = np.mean([r[k] for r in agg["as-fit"]])
    b = np.mean([r[k] for r in agg["best-rigid"]])
    c = np.mean([r[k] for r in agg["snap"]])
    print(f"{k:<18}{a:12.5f}{b:12.5f}{c:12.5f}   {'yes' if k in inhead else 'NO'}")
