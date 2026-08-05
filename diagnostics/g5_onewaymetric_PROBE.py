"""Adversarial check: is part_leg_distal_dist_mean minimised by retracting the distal
chains onto the thorax, and do the HEADLINE metrics catch that?

CPU only. Takes real M8a fits, builds a 'gave up' variant in which each of the six distal
leg chains is rigidly translated so its centroid lands on the nearest body-part vertex of
the same fit (a pose-plausible retraction: rigid, so it barely touches edge/triangle stats).
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
from fitter_3d.utils import load_meshes

DEV = "cpu"
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

# six distal chains, by joint name
chains = {}
for j, nm in enumerate(J):
    if nm.startswith("l_") and any(s in nm for s in ["_ti_", "_ta_", "_pt_"]):
        chains.setdefault(nm[2] + nm[-2:], []).append(j)
chain_verts = {k: np.nonzero(np.isin(dom, v))[0] for k, v in chains.items()}
print("distal chains:", {k: len(v) for k, v in chain_verts.items()})
print(
    "n distal verts total:",
    sum(len(v) for v in chain_verts.values()),
    " part_leg_distal group size:",
    len(pn2id["leg_distal"]),
)

d = np.load("diagnostics/moonshot/runs/M8a_split_only/H3_deform.npz", allow_pickle=True)
verts = torch.tensor(d["verts"], dtype=torch.float32)
faces = torch.tensor(d["faces"][0].astype(np.int64))
dv = torch.tensor(d["deform_verts"], dtype=torch.float32)
labels = list(d["labels"])

sel = [0, 5, 12, 23, 41]
objs = [os.path.join("diagnostics/moonshot/bench50", labels[i]) for i in sel]
_, tgts = load_meshes(mesh_files=objs, device=DEV)


def retract(v):
    v2 = v.clone()
    bpos = v[body_ids]
    for k, ids in chain_verts.items():
        c = v[ids].mean(0)
        nb = bpos[(bpos - c).norm(dim=-1).argmin()]
        v2[ids] = v[ids] + (nb - c)
    return v2


hdr = f"{'specimen':<34}{'variant':<10}{'part_leg_distal':>16}{'within_tau':>11}{'chamfer_l2':>11}{'fscore@.02':>11}{'comp_mean':>10}{'edge_lr':>9}{'tri_q':>7}{'folded':>8}{'midline':>9}"
print("\n" + hdr)
agg = {}
for n, i in enumerate(sel):
    tm = Meshes(verts=[tgts.verts_list()[n]], faces=[tgts.faces_list()[n]])
    for tag, v in (("as-fit", verts[i]), ("retracted", retract(verts[i]))):
        pm = Meshes(verts=[v], faces=[faces])
        torch.manual_seed(1234)
        p = M.part_metrics(pm, tm, part_ids, part_names, n_points=30000)
        torch.manual_seed(1234)
        s = M.surface_metrics(pm, tm, n_points=30000)
        g = M.deformation_metrics(v, verts[i] - dv[i], faces, dv[i])
        y = M.symmetry_metrics(v, torch.tensor(np.asarray(dd["sym_verts"]).astype(np.int64)), verts[i] - dv[i])
        row = (
            p["part_leg_distal_dist_mean"],
            p["part_leg_distal_within_tau"],
            s["chamfer_l2"],
            s["fscore@0.02"],
            s["comp_mean"],
            g["edge_logratio_absmean"],
            g["tri_quality_mean"],
            g["folded_face_frac"],
            y["midline_dev_mean"],
        )
        agg.setdefault(tag, []).append(row)
        print(
            f"{labels[i][:32]:<34}{tag:<10}{row[0]:16.5f}{row[1]:11.4f}{row[2]:11.5f}{row[3]:11.4f}{row[4]:10.5f}{row[5]:9.4f}{row[6]:7.4f}{row[7]:8.4f}{row[8]:9.5f}"
        )

print()
A = np.array(agg["as-fit"]).mean(0)
B = np.array(agg["retracted"]).mean(0)
names = [
    "part_leg_distal",
    "within_tau",
    "chamfer_l2",
    "fscore@0.02",
    "comp_mean",
    "edge_logratio",
    "tri_quality",
    "folded_face_frac",
    "midline_dev",
]
inhead = {"chamfer_l2", "fscore@0.02", "edge_logratio", "tri_quality", "midline_dev", "part_leg_distal"}
print(f"{'metric':<20}{'as-fit':>12}{'retracted':>12}{'rel%':>10}   in HEADLINE?")
for k, a, b in zip(names, A, B):
    print(f"{k:<20}{a:12.5f}{b:12.5f}{100 * (b - a) / abs(a):+9.1f}%   {'yes' if k in inhead else 'NO'}")
