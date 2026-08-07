"""PROBE: is the 'interior' of a non-watertight worker scan recoverable by a generalized
winding number (Jacobson et al., SIGGRAPH 2013)? GPU solid-angle sum, no libigl needed.
Also: what fraction of the interior VOLUME lies in the distal legs?
"""
import glob, time, numpy as np, torch, trimesh, warnings
warnings.filterwarnings("ignore")
dev = "cuda" if torch.cuda.is_available() else "cpu"

def gwn(Vt, Ft, Q, chunk=256):
    """generalized winding number at Q (M,3) for triangle soup (Vt,Ft)."""
    A = Vt[Ft[:, 0]]; B = Vt[Ft[:, 1]]; C = Vt[Ft[:, 2]]
    out = torch.empty(Q.shape[0], device=Q.device)
    for i in range(0, Q.shape[0], chunk):
        q = Q[i:i + chunk][:, None, :]
        a = A[None] - q; b = B[None] - q; c = C[None] - q
        la = a.norm(dim=-1); lb = b.norm(dim=-1); lc = c.norm(dim=-1)
        num = (a * torch.cross(b, c, dim=-1)).sum(-1)
        den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
        out[i:i + chunk] = 2.0 * torch.atan2(num, den).sum(-1)
    return out / (4 * np.pi)

for f in sorted(glob.glob("diagnostics/moonshot/bench50/*.obj"))[:4]:
    m = trimesh.load(f, process=False)
    Vt = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=dev)
    Ft = torch.tensor(np.asarray(m.faces), dtype=torch.long, device=dev)
    lo, hi = Vt.min(0).values, Vt.max(0).values
    pad = 0.02 * (hi - lo)
    Q = torch.rand(30000, 3, device=dev) * (hi - lo + 2 * pad) + lo - pad
    t = time.time(); w = gwn(Vt, Ft, Q); dt = time.time() - t
    wn = w.cpu().numpy()
    # crispness: fraction of samples in the ambiguous band 0.2 < w < 0.8
    amb = ((wn > 0.2) & (wn < 0.8)).mean()
    ins = (wn > 0.5).mean()
    print(f"{f.split('/')[-1][:40]:<42s} F={len(m.faces):7d}  gwn 60k pts in {dt:5.2f}s  "
          f"inside {100*ins:5.2f}%  ambiguous-band {100*amb:5.2f}%  "
          f"[|w|>1.5: {100*(np.abs(wn)>1.5).mean():.2f}%]")
