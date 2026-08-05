"""PROBE 4: how many pose hypotheses per specimen can this box afford?

Decides whether population/particle search over pose (CMA-ES, PSO, annealed particle
filter) is affordable, or whether only a handful of restarts is.
Measures batched SMAL forward + chamfer-style nearest-neighbour cost on one RTX 4090.
"""

import os
import sys
import time

sys.path.insert(0, "/home/fabi/dev/SMILify")
os.chdir("/home/fabi/dev/SMILify")
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/SMIL_OmniAnt.pkl")

import torch  # noqa: E402

import config  # noqa: E402
from pytorch3d.ops import knn_points  # noqa: E402
from smal_model.smal_torch import SMAL  # noqa: E402

dev = "cuda:0"
print("SMAL_FILE:", config.SMAL_FILE, "N_POSE:", config.N_POSE, "N_BETAS:", config.N_BETAS)
smal = SMAL(dev)
NJ = config.N_POSE + 1
print("joints:", NJ)

target = torch.randn(6000, 3, device=dev) * 0.3  # stand-in point cloud (drive unmounted)

for B in [1, 32, 128, 512]:
    betas = torch.zeros(B, config.N_BETAS, device=dev)
    pose = torch.zeros(B, NJ, 3, device=dev)
    trans = torch.zeros(B, 3, device=dev)
    torch.cuda.synchronize()
    # warmup
    for _ in range(3):
        v = smal(betas, pose, trans)[0]
    torch.cuda.synchronize()
    t0 = time.time()
    N = 20
    for _ in range(N):
        pose.normal_(0, 0.3)
        v = smal(betas, pose, trans)[0]
        d1 = knn_points(v, target[None].expand(B, -1, -1), K=1).dists
        d2 = knn_points(target[None].expand(B, -1, -1), v, K=1).dists
        loss = d1.mean(1) + d2.mean(1)  # noqa: F841
    torch.cuda.synchronize()
    dt = (time.time() - t0) / N
    print(f"batch {B:4d}: {dt * 1000:8.2f} ms / batch -> {B / dt:9.0f} hypotheses/s  ({B / dt * 60:,.0f} per minute)")
