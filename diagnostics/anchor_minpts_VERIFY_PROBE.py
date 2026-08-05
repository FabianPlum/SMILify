"""VERIFY PROBE: does min_pts=40 at n_sample=8000 drop the distal-leg anchors?

Part 1 (template proxy): area fraction per anatomical group, and empirical P(n >= 40)
over repeated 8000-point area-uniform samples with PERFECT (nearest-vertex) labels.

Part 2 (real pipeline): the actual bench50 targets + the actual trained part field,
reproducing exactly what optimise_hierarchical.py does before PartAnchorInit:
    tgt0 = sample_points_from_meshes(targets, 8000)
    partition.update(None, tgt0)          # nearest-neighbour lookup into 30k ref cloud
    part kept iff (assign == g).sum() >= 40
Reports which groups are dropped, per specimen, and what the 30k ref cloud would give.

CPU ONLY.
"""

import os
import pickle
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/home/fabi/dev/SMILify")

import glob

import numpy as np
import torch
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes

import config
from fitter_3d.trainer_hierarchical import vertex_groups

N_SAMPLE = 8000
MIN_PTS = 40
REF_PTS = 30000

with open(config.SMAL_FILE, "rb") as f:
    u = pickle._Unpickler(f)
    u.encoding = "latin1"
    dd = u.load()
jnames = list(dd["J_names"])
vg, gnames = vertex_groups(np.asarray(dd["weights"]), jnames, split_distal=True)
vg_t = torch.tensor(vg)
print("SMAL_FILE:", config.SMAL_FILE)
print("groups:", gnames)

vt = torch.tensor(np.asarray(dd["v_template"]), dtype=torch.float32)
ft = torch.tensor(np.asarray(dd["f"]), dtype=torch.int64)
tmpl = Meshes(verts=[vt], faces=[ft])

# ---------------------------------------------------------------- Part 1: template proxy
print("\n=== PART 1: template, PERFECT labels, area-uniform sampling ===")
NREP = 200
counts = np.zeros((NREP, len(gnames)), dtype=np.int64)
torch.manual_seed(0)
for r in range(NREP):
    p = sample_points_from_meshes(tmpl, N_SAMPLE)[0]
    idx = knn_points(p[None], vt[None], K=1).idx[0, :, 0]
    lab = vg_t[idx]
    for g in range(len(gnames)):
        counts[r, g] = int((lab == g).sum())

print(f"{'group':8s} {'nverts':>7s} {'mean n@8000':>12s} {'area%':>7s} {'P(n>=40)':>9s} {'n@30000':>9s}")
for g, name in enumerate(gnames):
    m = counts[:, g].mean()
    p40 = float((counts[:, g] >= MIN_PTS).mean())
    print(
        f"{name:8s} {int((vg == g).sum()):7d} {m:12.1f} {100 * m / N_SAMPLE:7.3f} "
        f"{p40:9.3f} {m * REF_PTS / N_SAMPLE:9.1f}"
    )
kept = (counts >= MIN_PTS).sum(axis=1)
print(f"anchors kept per draw: mean {kept.mean():.2f} / {len(gnames)}  (min {kept.min()}, max {kept.max()})")

# ---------------------------------------------------------------- Part 2: real pipeline
NET = "/home/fabi/dev/SMILify/diagnostics/moonshot/partfield/net_B_mirror.pt"
MESH = "/home/fabi/dev/SMILify/diagnostics/moonshot/bench50"
NMESH = int(os.environ.get("NMESH", "50"))
if os.path.exists(NET):
    print(f"\n=== PART 2: real bench targets + real part field ({NMESH} specimens) ===")
    from fitter_3d.partfield import PartFieldNet, PartFieldPartition, part_names, predict_field
    from fitter_3d.utils import load_meshes

    files = sorted(glob.glob(os.path.join(MESH, "*.obj")))[:NMESH]
    _, targets = load_meshes(mesh_files=files, device="cpu")
    pf_names = part_names(jnames)
    assert pf_names[:-1] == list(gnames)
    ck = torch.load(NET, map_location="cpu")
    net = PartFieldNet(n_classes=len(ck["names"]), width=ck.get("width", 1.0)).to("cpu")
    net.load_state_dict(ck["state"])
    net.eval()
    debris_id = pf_names.index("debris")

    torch.manual_seed(0)
    np.random.seed(0)
    ref_p, ref_l, ref_c = [], [], []
    for b in range(len(targets)):
        p = sample_points_from_meshes(targets[b], REF_PTS)[0]
        prob = predict_field(net, p, seed=0)
        conf, lab = prob.max(-1)
        ref_p.append(p)
        ref_l.append(lab)
        ref_c.append(conf)
        print(
            f"  [{b + 1}/{len(targets)}] {os.path.basename(files[b])[:40]:40s} debris {100 * float((lab == debris_id).float().mean()):.1f}%",
            flush=True,
        )
    partition = PartFieldPartition(
        torch.stack(ref_p),
        torch.stack(ref_l),
        len(gnames),
        "cpu",
        debris_id=debris_id,
        ref_conf=torch.stack(ref_c),
        min_conf=0.0,
    )
    tgt0 = sample_points_from_meshes(targets, N_SAMPLE).detach()
    partition.update(None, tgt0)

    B = len(targets)
    c8 = np.zeros((B, len(gnames)), dtype=np.int64)
    c30 = np.zeros((B, len(gnames)), dtype=np.int64)
    for b in range(B):
        for g in range(len(gnames)):
            c8[b, g] = int((partition.assign[b] == g).sum())
            c30[b, g] = int((partition.ref_label[b] == g).sum())
    kept8 = c8 >= MIN_PTS
    kept30 = c30 >= MIN_PTS
    print(
        f"\ntotal anchors kept @8000/min_pts=40: {int(kept8.sum())} over {B} specimens "
        f"({kept8.sum() / B:.1f}/specimen of {len(gnames)})   <-- compare run log"
    )
    print(f"if anchors came from the 30k ref cloud: {int(kept30.sum())} ({kept30.sum() / B:.1f}/specimen)")
    print(f"\n{'group':8s} {'kept@8k':>8s} {'kept@30k':>9s} {'median n@8k':>12s} {'min':>5s} {'max':>5s}")
    for g, name in enumerate(gnames):
        print(
            f"{name:8s} {int(kept8[:, g].sum()):8d} {int(kept30[:, g].sum()):9d} "
            f"{np.median(c8[:, g]):12.1f} {c8[:, g].min():5d} {c8[:, g].max():5d}"
        )
    np.savez(
        "/home/fabi/dev/SMILify/diagnostics/anchor_minpts_VERIFY_counts.npz",
        c8=c8,
        c30=c30,
        names=np.array(gnames),
        files=np.array([os.path.basename(f) for f in files]),
    )
else:
    print("\n(part field checkpoint missing; skipping part 2)")
