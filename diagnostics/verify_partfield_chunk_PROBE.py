"""VERIFY: is predict_field's 1328-point trailing chunk a real source of label noise?

CPU-only. Uses the shipped net_B_mirror.pt checkpoint and the 30k reference clouds in
real_weak.npz (the same clouds the fitter/probe15 build).

Stage 1  timing + direct degradation: run the net on the SAME 1328 points (a) alone, as
         predict_field's trailing chunk does, and (b) embedded in a full 4096-point chunk.
Stage 2  neighbour-count check: how many ball-query hits at r=0.08 at each density.
Stage 3  end-to-end: current predict_field vs a padded variant, same seed and across seeds.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fitter_3d.partfield import PartFieldNet, normalise  # noqa: E402
from fitter_3d.pointcloud2smil.pointnet2_utils import (  # noqa: E402
    farthest_point_sample,
    query_ball_point,
    index_points,
)

DEV = "cpu"
torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))

ck = torch.load("diagnostics/moonshot/partfield/net_B_mirror.pt", map_location=DEV)
net = PartFieldNet(n_classes=len(ck["names"]), width=ck.get("width", 1.0)).to(DEV)
net.load_state_dict(ck["state"])
net.eval()
names = list(ck["names"])
print(f"net: {len(names)} classes, width {ck.get('width', 1.0)}, trained args:")
print("   ", {k: v for k, v in ck["args"].items() if k not in ("out",)})

d = np.load("diagnostics/moonshot/partfield/real_weak.npz", allow_pickle=True)
pts_all, files = d["pts"], d["files"]
print(f"clouds: {pts_all.shape}")

N_SPEC = int(os.environ.get("N_SPEC", 4))


# ------------------------------------------------------------------ padded variant
@torch.no_grad()
def predict_field_pad(net, pts, chunk_n=4096, n_votes=4, device=DEV, seed=0, pad=True):
    """Same as fitter_3d.partfield.predict_field, plus an option to pad the last chunk."""
    net.eval()
    P = pts.shape[0]
    prob = torch.zeros(P, net.n_classes, device=device)
    cnt = torch.zeros(P, 1, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    npts, _, _ = normalise(pts.to(device))
    for v in range(n_votes):
        perm = torch.randperm(P, generator=g)
        for i in range(0, P, chunk_n):
            idx = perm[i : i + chunk_n]
            if idx.numel() < 64:
                continue
            if pad and idx.numel() < chunk_n:
                idx = torch.cat([idx, perm[: chunk_n - idx.numel()]])
            idx = idx.to(device)
            sub = npts[idx].unsqueeze(0)
            logit = net(sub)[0]
            p = torch.softmax(logit, dim=-1)
            prob.index_add_(0, idx, p)  # index_add so duplicated rows count twice
            cnt.index_add_(0, idx, torch.ones(idx.numel(), 1, device=device))
    prob = prob / cnt.clamp_min(1)
    return prob, int((cnt.squeeze(-1) == 0).sum())


# ------------------------------------------------------------------ stage 1 + 2
print("\n=== STAGE 1: same points, alone (1328) vs inside a full chunk (4096) ===")
rng_g = torch.Generator().manual_seed(1)
agree_short, conf_drop, kl = [], [], []
for s in range(N_SPEC):
    p = torch.tensor(pts_all[s], dtype=torch.float32)
    npts, _, _ = normalise(p)
    perm = torch.randperm(npts.shape[0], generator=rng_g)
    full_idx = perm[:4096]
    short_idx = perm[:1328]  # the same 1328 points are a prefix of the full chunk
    t0 = time.time()
    with torch.no_grad():
        lf = torch.softmax(net(npts[full_idx].unsqueeze(0))[0], -1)
        t1 = time.time()
        ls = torch.softmax(net(npts[short_idx].unsqueeze(0))[0], -1)
    t2 = time.time()
    a = lf[:1328].argmax(-1)
    b = ls.argmax(-1)
    ag = float((a == b).float().mean())
    agree_short.append(ag)
    conf_drop.append(float(ls.max(-1)[0].mean() - lf[:1328].max(-1)[0].mean()))
    kl.append(float((lf[:1328] * (lf[:1328].clamp_min(1e-9).log() - ls.clamp_min(1e-9).log())).sum(-1).mean()))
    print(
        f"  [{s}] {str(files[s])[:40]:42s} argmax agree {100 * ag:6.2f}%  "
        f"mean-conf delta {conf_drop[-1]:+.4f}  KL(full||short) {kl[-1]:.4f}   "
        f"(fwd 4096 {t1 - t0:.2f}s, 1328 {t2 - t1:.2f}s)"
    )
print(f"  MEAN argmax agreement short-vs-full on the identical points: {100 * np.mean(agree_short):.2f}%")

print("\n=== STAGE 2: ball-query neighbour counts at r=0.08, nsample=32 ===")
p = torch.tensor(pts_all[0], dtype=torch.float32)
npts, _, _ = normalise(p)
perm = torch.randperm(npts.shape[0], generator=torch.Generator().manual_seed(0))
for n_in, npoint in ((4096, 1024), (1328, 1024)):
    xyz = npts[perm[:n_in]].unsqueeze(0)
    npoint_eff = min(npoint, n_in)
    fps = farthest_point_sample(xyz, npoint_eff)
    new_xyz = index_points(xyz, fps)
    # true neighbour count (before pad-by-repeat)
    from fitter_3d.pointcloud2smil.pointnet2_utils import square_distance

    sd = square_distance(new_xyz, xyz)
    nb = (sd <= 0.08**2).sum(-1).float()
    gi = query_ball_point(0.08, 32, xyz, new_xyz)
    uniq = torch.tensor([len(torch.unique(gi[0, k])) for k in range(gi.shape[1])], dtype=torch.float32)
    print(
        f"  N={n_in:5d} npoint={npoint_eff:5d}: true nbrs within r  mean {nb.mean():6.2f}  "
        f"median {nb.median():6.1f}  frac<32 {float((nb < 32).float().mean()):.3f}   "
        f"unique pts per group after pad-by-repeat: mean {uniq.mean():5.2f}/32"
    )

# ------------------------------------------------------------------ stage 3
print("\n=== STAGE 3: end-to-end label effect over the full 30k cloud ===")
rows = []
for s in range(N_SPEC):
    p = torch.tensor(pts_all[s], dtype=torch.float32)
    t0 = time.time()
    cur1, u1 = predict_field_pad(net, p, seed=1, pad=False)
    cur2, _ = predict_field_pad(net, p, seed=2, pad=False)
    fix1, u2 = predict_field_pad(net, p, seed=1, pad=True)
    fix2, _ = predict_field_pad(net, p, seed=2, pad=True)
    a1, a2 = cur1.argmax(-1), cur2.argmax(-1)
    b1, b2 = fix1.argmax(-1), fix2.argmax(-1)
    same_seed = float((a1 == b1).float().mean())
    self_cur = float((a1 == a2).float().mean())
    self_fix = float((b1 == b2).float().mean())
    rows.append((same_seed, self_cur, self_fix))
    print(
        f"  [{s}] {str(files[s])[:36]:38s} current-vs-padded (seed1) {100 * same_seed:6.2f}%  "
        f"| same-cloud seed1-vs-seed2 agreement: current {100 * self_cur:6.2f}%  "
        f"padded {100 * self_fix:6.2f}%  (delta {100 * (self_fix - self_cur):+.2f}pp)  "
        f"unseen {u1}/{u2}  [{time.time() - t0:.0f}s]"
    )
r = np.array(rows)
print(
    f"\n  MEAN current-vs-padded label change: {100 * (1 - r[:, 0].mean()):.3f}% of points\n"
    f"  MEAN same-cloud self-agreement: current {100 * r[:, 1].mean():.3f}%  "
    f"padded {100 * r[:, 2].mean():.3f}%  delta {100 * (r[:, 2] - r[:, 1]).mean():+.3f}pp"
)
