"""PROBE: throughput of a batched ellipsoid-union soft-occupancy objective on one 4090.
Decides whether 'massively parallel batched restarts' is affordable vs CMA-ES/PSO.
"""
import time, torch
dev = "cuda"
torch.manual_seed(0)
J, NP = 55, 20000          # 55 segments, 20k interior/exterior query points

def bench(K, iters=30, backward=True):
    c = torch.randn(K, J, 3, device=dev, requires_grad=True)      # centres
    logs = torch.randn(K, J, 3, device=dev, requires_grad=True)   # log semi-axes
    q = torch.randn(K, 1, NP, 3, device=dev)                      # query points
    tgt = (torch.rand(K, NP, device=dev) > 0.5).float()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        d = (q - c[:, :, None, :]) / logs.exp()[:, :, None, :]
        r2 = (d * d).sum(-1)                       # (K,J,NP)
        occ_j = torch.sigmoid((1.0 - r2) * 8.0)    # soft indicator, temperature 8
        occ = 1 - torch.prod(1 - occ_j, dim=1)     # soft union (K,NP)
        inter = (occ * tgt).sum(-1); union = (occ + tgt - occ * tgt).sum(-1)
        loss = -(inter / union).sum()
        if backward:
            loss.backward()
            c.grad = None; logs.grad = None
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return dt

for K in [64, 256, 1024]:
    try:
        f = bench(K, backward=False); b = bench(K, backward=True)
        print(f"K={K:5d} restarts x {J} ellipsoids x {NP} pts:  fwd {1000*f:7.2f} ms  fwd+bwd {1000*b:7.2f} ms "
              f"-> {K/b:9.0f} restart-iterations/sec")
    except torch.cuda.OutOfMemoryError:
        print(f"K={K}: OOM (needs chunking)")
        torch.cuda.empty_cache()
