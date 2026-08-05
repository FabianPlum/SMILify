"""CPU-only verification of the '--part_field deletes the thin distal legs' claim.

For a handful of bench50 specimens:
  * run the part field exactly as optimise_hierarchical does (30k pts, seed 0, 4 votes)
  * report the debris fraction
  * classify each TARGET point by the anatomical part of the nearest vertex of the
    ALREADY-FITTED M8a mesh (the control arm's H3 result -- an independent, decent fit),
    then report the debris rate CONDITIONED on that part.
    If the claim is right, debris rate on distal-leg territory >> on body territory.
"""

import os
import sys
import json
import numpy as np
import torch

REPO = "/home/fabi/dev/SMILify"
sys.path.insert(0, REPO)
os.chdir(REPO)

from fitter_3d.utils import load_meshes
from fitter_3d.partfield import PartFieldNet, predict_field, part_names
from fitter_3d.trainer_hierarchical import vertex_groups
from pytorch3d.ops import knn_points, sample_points_from_meshes
import pickle

dev = "cpu"
torch.set_num_threads(8)

ck = torch.load("diagnostics/moonshot/partfield/net_B_mirror.pt", map_location=dev)
names = ck["names"]
net = PartFieldNet(n_classes=len(names), width=ck.get("width", 1.0)).to(dev)
net.load_state_dict(ck["state"])
net.eval()
dbg_id = names.index("debris")
print("classes:", names, "debris_id:", dbg_id)

with open("config.py") as f:
    pass
import config

with open(config.SMAL_FILE, "rb") as f:
    u = pickle._Unpickler(f)
    u.encoding = "latin1"
    dd = u.load()
jn = list(dd["J_names"])
vg, gnames = vertex_groups(np.asarray(dd["weights"]), jn, split_distal=True)
print("fitter groups:", gnames)
assert part_names(jn)[:-1] == list(gnames)
vg_t = torch.tensor(vg)
distal_ids = [i for i, g in enumerate(gnames) if g.endswith("d_l") or g.endswith("d_r")]
prox_ids = [i for i, g in enumerate(gnames) if g.endswith("p_l") or g.endswith("p_r")]
body_id = gnames.index("body")
print("distal groups:", [gnames[i] for i in distal_ids])

# M8a control fit (H3) -- used only to define anatomical territory on the target
z = np.load("diagnostics/moonshot/runs/M8a_split_only/H3_deform.npz", allow_pickle=True)
fit_v = torch.tensor(z["verts"], dtype=torch.float32)
labels = list(z["labels"])
print("M8a verts:", fit_v.shape, "n labels:", len(labels))

MESH = "diagnostics/moonshot/bench50"
files = sorted(f for f in os.listdir(MESH) if f.endswith(".obj"))
assert files == labels, (files[:2], labels[:2])

want = [files[0], files[12], files[25], files[37]]
for f in files:
    if "Cyphomyrmex_cf.minutus" in f:
        want.append(f)
want = list(dict.fromkeys(want))
print("specimens:", want)

rows = []
for nm in want:
    i = files.index(nm)
    _, meshes = load_meshes(mesh_files=[os.path.join(MESH, nm)], device=dev)
    p = sample_points_from_meshes(meshes[0], 30000)[0]
    prob = predict_field(net, p, seed=0, device=dev)
    lab = prob.argmax(-1)
    d = lab == dbg_id
    frac = float(d.float().mean())

    # anatomical territory from the M8a fit
    idx = knn_points(p.unsqueeze(0), fit_v[i].unsqueeze(0), K=1).idx[0, :, 0]
    terr = vg_t[idx]
    is_distal = torch.isin(terr, torch.tensor(distal_ids))
    is_prox = torch.isin(terr, torch.tensor(prox_ids))
    is_body = terr == body_id
    # distance to the fitted surface -- genuine debris should be FAR from the fit
    dist = knn_points(p.unsqueeze(0), fit_v[i].unsqueeze(0), K=1).dists[0, :, 0].sqrt()

    r = dict(
        name=nm[:40],
        debris_frac=frac,
        n_distal=int(is_distal.sum()),
        n_prox=int(is_prox.sum()),
        n_body=int(is_body.sum()),
        dbg_on_distal=float(d[is_distal].float().mean()) if int(is_distal.sum()) else float("nan"),
        dbg_on_prox=float(d[is_prox].float().mean()) if int(is_prox.sum()) else float("nan"),
        dbg_on_body=float(d[is_body].float().mean()) if int(is_body.sum()) else float("nan"),
        med_dist_debris=float(dist[d].median()) if int(d.sum()) else float("nan"),
        med_dist_keep=float(dist[~d].median()),
    )
    rows.append(r)
    print(json.dumps(r), flush=True)

print("\n=== summary ===")
for k in ["debris_frac", "dbg_on_distal", "dbg_on_prox", "dbg_on_body", "med_dist_debris", "med_dist_keep"]:
    v = np.array([r[k] for r in rows], dtype=float)
    print(f"  {k:18s} mean {np.nanmean(v):.4f}   vals " + " ".join(f"{x:.4f}" for x in v))
json.dump(rows, open("/home/fabi/dev/SMILify/diagnostics/moonshot/out/VERIFY_debris_territory.json", "w"), indent=1)
