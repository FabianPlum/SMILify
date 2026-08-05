"""VERIFY (part 2): does the 1328-pt trailing chunk measurably move probe 15's G1?

Replicates probe_15_partfield_gates.py's G1 exactly (two independent 30k surface samplings,
seed=1 / seed=2, nearest-neighbour matched) for three variants of predict_field:

  cur   the shipped code                      -- trailing 1328-pt chunk is fed to the net
  skip  trailing chunk dropped                -- isolates the degraded chunk, adds nothing
  pad   trailing chunk padded back to 4096    -- the fix the claim proposes

CPU-only.
"""

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)

from fitter_3d.partfield import PartFieldNet, normalise  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
from pytorch3d.ops import knn_points, sample_points_from_meshes  # noqa: E402

DEV = "cpu"
torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
N_SPEC = int(os.environ.get("N_SPEC", 5))
N_REF = 30000

ck = torch.load("diagnostics/moonshot/partfield/net_B_mirror.pt", map_location=DEV)
net = PartFieldNet(n_classes=len(ck["names"]), width=ck.get("width", 1.0)).to(DEV)
net.load_state_dict(ck["state"])
net.eval()


@torch.no_grad()
def pf(pts, seed=0, mode="cur", chunk_n=4096, n_votes=4):
    P = pts.shape[0]
    prob = torch.zeros(P, net.n_classes)
    cnt = torch.zeros(P, 1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    npts, _, _ = normalise(pts)
    for _ in range(n_votes):
        perm = torch.randperm(P, generator=g)
        for i in range(0, P, chunk_n):
            idx = perm[i : i + chunk_n]
            if idx.numel() < 64:
                continue
            if idx.numel() < chunk_n:
                if mode == "skip":
                    continue
                if mode == "pad":
                    idx = torch.cat([idx, perm[: chunk_n - idx.numel()]])
            p = torch.softmax(net(npts[idx].unsqueeze(0))[0], -1)
            prob.index_add_(0, idx, p)
            cnt.index_add_(0, idx, torch.ones(idx.numel(), 1))
    return (prob / cnt.clamp_min(1)).argmax(-1), int((cnt.squeeze(-1) == 0).sum())


mesh_dir = os.path.join(HERE, "moonshot", "bench50")
files = sorted(f for f in os.listdir(mesh_dir) if f.endswith(".obj"))[:N_SPEC]
_, meshes = load_meshes(mesh_files=[os.path.join(mesh_dir, f) for f in files], device=DEV)

print(f"{'specimen':38s} {'G1 cur':>8s} {'G1 skip':>8s} {'G1 pad':>8s} {'d_skip':>8s} {'d_pad':>8s}")
res = []
for i, f in enumerate(files):
    t0 = time.time()
    with torch.no_grad():
        torch.manual_seed(100 + i)
        a = sample_points_from_meshes(meshes[i], N_REF)[0]
        b = sample_points_from_meshes(meshes[i], N_REF)[0]
    an, _, _ = normalise(a)
    bn, _, _ = normalise(b)
    j = knn_points(an.unsqueeze(0), bn.unsqueeze(0), K=1).idx[0, :, 0]
    row = {}
    for mode in ("cur", "skip", "pad"):
        la, u1 = pf(an, seed=1, mode=mode)
        lb, u2 = pf(bn, seed=2, mode=mode)
        row[mode] = float((la == lb[j]).float().mean())
        row[mode + "_unseen"] = u1 + u2
    res.append(row)
    print(
        f"{f.replace('_processed.obj', '')[:38]:38s} {100 * row['cur']:8.3f} {100 * row['skip']:8.3f} "
        f"{100 * row['pad']:8.3f} {100 * (row['skip'] - row['cur']):+8.3f} "
        f"{100 * (row['pad'] - row['cur']):+8.3f}   unseen {row['cur_unseen']}/{row['skip_unseen']}"
        f"   [{time.time() - t0:.0f}s]",
        flush=True,
    )

c = np.array([r["cur"] for r in res])
s = np.array([r["skip"] for r in res])
p = np.array([r["pad"] for r in res])
print(f"\nMEAN over {len(res)} specimens (probe15 gate is 90%):")
print(f"  G1 current  {100 * c.mean():7.3f}%")
print(
    f"  G1 skip     {100 * s.mean():7.3f}%   delta {100 * (s - c).mean():+7.3f}pp   "
    f"wins {int((s > c).sum())}/{len(res)}"
)
print(
    f"  G1 pad      {100 * p.mean():7.3f}%   delta {100 * (p - c).mean():+7.3f}pp   "
    f"wins {int((p > c).sum())}/{len(res)}"
)
