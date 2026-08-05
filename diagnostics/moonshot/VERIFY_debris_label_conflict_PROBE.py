"""Apply the REAL-WEAK labelling rule to SYNTHETIC clouds, using the actual posed mesh
(10229 verts) as the reference -- exactly what make_partfield_data's real-weak branch uses.
CPU only."""

import os
import sys
import pickle
import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = "/home/fabi/dev/SMILify"
HERE = os.path.join(REPO, "diagnostics/moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import config
from fitter_3d.trainer import SMAL3DFitter
from fitter_3d.partfield import part_names, template_vertex_labels, normalise
import make_partfield_data as M

dev = torch.device("cpu")
N_PTS = M.N_PTS
rng = np.random.default_rng(20260805)
gen = torch.Generator(device=dev).manual_seed(20260805)

with open(config.SMAL_FILE, "rb") as fh:
    u = pickle._Unpickler(fh)
    u.encoding = "latin1"
    dd = u.load()
jnames = [n.decode() if isinstance(n, bytes) else n for n in dd["J_names"]]
names = part_names(jnames)
debris_id = names.index("debris")
vlab = torch.tensor(template_vertex_labels(dd["weights"], jnames), device=dev)
fit = np.load(os.path.join(HERE, "runs/M7_handoff_midline/Stage_3_deform_fine.npz"))
smal = SMAL3DFitter(batch_size=1, device=dev)
faces = torch.tensor(np.asarray(dd["f"], dtype=np.int64), device=dev)
P = {
    k: torch.tensor(fit[k], dtype=torch.float32, device=dev)
    for k in ("global_rot", "joint_rot", "betas", "log_beta_scales", "trans", "betas_trans")
}
pool = np.arange(50)


def augment_tracked(pts, lab, debris_id, gen, rng):
    """Byte-for-byte copy of M.augment with the final rotation R returned."""
    dev = pts.device
    n = pts.shape[0]
    sig = float(rng.uniform(0.0008, 0.006))
    pts = pts + torch.randn(pts.shape, device=dev, generator=gen) * sig
    keep = torch.ones(n, dtype=torch.bool, device=dev)
    for _ in range(int(rng.integers(0, 4))):
        c = pts[int(rng.integers(0, n))]
        r = float(rng.uniform(0.03, 0.11))
        keep &= (pts - c).norm(dim=-1) > r
    extra_p, extra_l, src = [], [], []
    for _ in range(int(rng.integers(0, 3))):
        c = pts[int(rng.integers(0, n))] + torch.randn(3, device=dev, generator=gen) * 0.05
        k = int(rng.integers(40, 260))
        extra_p.append(c + torch.randn(k, 3, device=dev, generator=gen) * float(rng.uniform(0.01, 0.05)))
        extra_l.append(torch.full((k,), debris_id, dtype=torch.long, device=dev))
        src.append(torch.zeros(k, dtype=torch.long))
    for _ in range(int(rng.integers(0, 3))):
        a, b = pts[int(rng.integers(0, n))], pts[int(rng.integers(0, n))]
        if float((a - b).norm()) > 0.7:
            continue
        k = int(rng.integers(30, 150))
        t = torch.rand(k, 1, device=dev, generator=gen)
        seg = a * (1 - t) + b * t + torch.randn(k, 3, device=dev, generator=gen) * 0.008
        extra_p.append(seg)
        extra_l.append(torch.full((k,), debris_id, dtype=torch.long, device=dev))
        src.append(torch.ones(k, dtype=torch.long))
    pts, lab = pts[keep], lab[keep]
    kind = torch.full((pts.shape[0],), -1, dtype=torch.long)
    if extra_p:
        pts = torch.cat([pts] + extra_p, 0)
        lab = torch.cat([lab] + extra_l, 0)
        kind = torch.cat([kind] + src, 0)
    axis = torch.randn(3, device=dev, generator=gen)
    axis = axis / axis.norm()
    proj = pts @ axis
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-9)
    p_keep = 1.0 - float(rng.uniform(0.0, 0.5)) * proj
    keep = torch.rand(pts.shape[0], device=dev, generator=gen) < p_keep
    if int(keep.sum()) > N_PTS:
        pts, lab, kind = pts[keep], lab[keep], kind[keep]
    ang = torch.randn(3, device=dev, generator=gen) * float(rng.uniform(0.0, 0.05))
    th = ang.norm().clamp_min(1e-9)
    k_ = (ang / th).unsqueeze(0)
    K = torch.zeros(3, 3, device=dev)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k_[0, 2], k_[0, 1], k_[0, 2], -k_[0, 0], -k_[0, 1], k_[0, 0]
    R = torch.eye(3, device=dev) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)
    pts = pts @ R.T
    if pts.shape[0] < N_PTS:
        idx = torch.randint(0, pts.shape[0], (N_PTS,), device=dev, generator=gen)
    else:
        idx = torch.randperm(pts.shape[0], device=dev, generator=gen)[:N_PTS]
    return pts[idx], lab[idx], kind[idx], R


allmesh, allsamp, allkind = [], [], []
NC = 25
for i in range(NC):
    a, b = int(rng.choice(pool)), int(rng.choice(pool))
    t = float(rng.uniform(0, 1))
    kw = {}
    for k, jit in (
        ("global_rot", 0.03),
        ("joint_rot", 0.06),
        ("betas", 0.25),
        ("log_beta_scales", 0.05),
        ("trans", 0.01),
        ("betas_trans", 0.01),
    ):
        mix = P[k][a] * (1 - t) + P[k][b] * t
        sd = P[k].std(0).clamp_min(1e-6) * jit
        kw[k] = (mix + torch.randn(mix.shape, device=dev, generator=gen) * sd).unsqueeze(0)
    with torch.no_grad():
        v = smal(**kw)
        v = v[0] if isinstance(v, (tuple, list)) else v
    V = v[0]
    p, l = M.sample_with_labels(V, faces, vlab, N_PTS * 3, gen)
    p, c1, s1 = normalise(p)
    Vn = (V - c1.squeeze(0)) / s1.squeeze()
    p, l, kind, R = augment_tracked(p, l, debris_id, gen, rng)
    Vr = Vn @ R.T
    p, c2, s2 = normalise(p)
    Vf = (Vr - c2.squeeze(0)) / s2.squeeze()
    pn = p.numpy()
    ln = l.numpy()
    kn = kind.numpy()
    vfn = Vf.numpy()
    m = ln == debris_id
    if m.sum() < 5:
        continue
    tm = cKDTree(vfn)
    dm, _ = tm.query(pn[m], k=1)  # real-weak rule reference
    tc = cKDTree(pn[~m])
    dc, _ = tc.query(pn[m], k=1)  # sampled-point proxy
    allmesh.append(dm)
    allsamp.append(dc)
    allkind.append(kn[m])
    if (i + 1) % 5 == 0:
        print("  ...", i + 1, flush=True)

dm = np.concatenate(allmesh)
dc = np.concatenate(allsamp)
kk = np.concatenate(allkind)
print("\nclouds=%d  debris pts=%d  (blob=%d, bridge=%d)" % (NC, dm.size, (kk == 0).sum(), (kk == 1).sum()))
print("REAL-WEAK RULE APPLIED TO SYNTHETIC DEBRIS (dist to nearest posed-mesh vertex):")
for th in (0.010, 0.020, 0.035, 0.05):
    print(f"   frac < {th}: {100 * (dm < th).mean():6.2f}%   [sampled-point proxy: {100 * (dc < th).mean():6.2f}%]")
print("   median dist: mesh %.4f  sampled-proxy %.4f" % (np.median(dm), np.median(dc)))
for nm, sel in (("blobs", kk == 0), ("bridges", kk == 1)):
    if sel.sum():
        print(f"   {nm}: n={sel.sum()}  frac<0.035 = {100 * (dm[sel] < 0.035).mean():.2f}%")
