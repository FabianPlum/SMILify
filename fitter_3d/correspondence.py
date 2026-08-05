"""Alternative correspondence operators for the registration data term.

The measured problem (see diagnostics/moonshot/REPORT.md §2): chamfer's hard
nearest-neighbour assignment is massively multimodal on an ant, because six legs are
near-identical and mutually substitutable. Arm E1 showed this is not a step-size issue --
raising the pose learning rate 10x made the fit WORSE than its own initialisation, i.e. it
reached a wrong basin faster.

This module implements three standard alternatives to hard-NN, all of which attack the
assignment rather than the optimizer:

  mutual_nn   Mutual nearest neighbours + Lowe-style ratio test. Keeps only correspondences
              that both sides agree on AND that are unambiguous (nearest is clearly closer
              than second-nearest). Directly discards the ambiguous matches that leg-swap
              failures are made of. Cheapest of the three.
              [Lowe, IJCV 2004, §7.1 ratio test; standard mutual-NN filtering in
               FPFH/RANSAC pipelines, Rusu et al. ICRA 2009]

  sinkhorn    Entropic optimal transport: a doubly-stochastic soft assignment computed by
              Sinkhorn iterations, with the blur annealed from coarse to fine. Soft
              assignment means an ambiguous point spreads its mass over several candidates
              instead of committing to one wrong one, and annealing the blur is a
              continuation method on the assignment problem.
              [Cuturi, NeurIPS 2013 "Sinkhorn Distances"; Feydy et al., AISTATS 2019
               "Interpolating between Optimal Transport and MMD"; the annealing idea is
               the same as deterministic annealing in Chui & Rangarajan's TPS-RPM,
               CVIU 2003]

  softnn      A cheap middle ground: softmin over the K nearest candidates at temperature
              tau, i.e. a local Boltzmann average rather than a global transport plan.
              O(NK) instead of O(N^2).

All operators return a scalar loss and a dict of diagnostics, so the fraction of
correspondences actually being used is observable rather than assumed.
"""

import numpy as np
import torch
from pytorch3d.ops import knn_points


def _sq(x):
    return x.clamp_min(0.0)


def mutual_nn_loss(src, tgt, ratio_thresh=0.9, return_mask=False):
    """Bidirectional distance over MUTUAL, UNAMBIGUOUS nearest-neighbour pairs.

    A correspondence (i -> j) is kept iff
      * j is i's nearest point in tgt, AND i is j's nearest point in src  (mutual), AND
      * d(i, j) / d(i, second-nearest) < ratio_thresh                    (unambiguous)

    The ratio test is Lowe's: if the two best candidates are nearly equidistant the match
    carries no information and, on a hexapod, is exactly the kind of match that assigns a
    leg to its neighbour. Discarding it is better than averaging over it.
    """
    f = knn_points(src, tgt, K=2)
    b = knn_points(tgt, src, K=1)
    d1 = f.dists[..., 0]
    d2 = f.dists[..., 1].clamp_min(1e-12)
    nn_f = f.idx[..., 0]  # (B,N) index into tgt
    nn_b = b.idx[..., 0]  # (B,M) index into src

    B, N = nn_f.shape
    ar = torch.arange(N, device=src.device).unsqueeze(0).expand(B, N)
    # mutual: following i -> j -> back lands on i again
    back = torch.gather(nn_b, 1, nn_f)
    mutual = back == ar
    # ratio test on SQUARED distances -> compare against squared threshold
    unambiguous = (d1 / d2) < (ratio_thresh**2)
    keep = mutual & unambiguous

    n_keep = keep.float().sum(dim=1).clamp_min(1.0)
    loss = (d1 * keep.float()).sum(dim=1) / n_keep
    info = {
        "mutual_frac": float(mutual.float().mean().item()),
        "kept_frac": float(keep.float().mean().item()),
    }
    # keep the reverse direction too, otherwise the template can hide inside the target
    loss = loss.mean() + b.dists[..., 0].mean()
    if return_mask:
        return loss, info, keep
    return loss, info


def soft_nn_loss(src, tgt, K=16, tau=0.02):
    """Softmin over the K nearest candidates at temperature tau (a length).

    As tau -> 0 this becomes hard nearest-neighbour; at larger tau an ambiguous point
    averages over its candidates instead of committing. O(NK).
    """
    f = knn_points(src, tgt, K=K)
    d = f.dists  # (B,N,K) squared
    w = torch.softmax(-d / (tau**2), dim=-1)
    fwd = (w * d).sum(-1).mean()
    b = knn_points(tgt, src, K=K)
    wb = torch.softmax(-b.dists / (tau**2), dim=-1)
    bwd = (wb * b.dists).sum(-1).mean()
    # effective number of candidates each point spreads over -- 1.0 means it committed
    eff = torch.exp(-(w * (w + 1e-12).log()).sum(-1)).mean()
    return fwd + bwd, {"eff_candidates": float(eff.item())}


def sinkhorn_loss(src, tgt, blur=0.05, n_iter=25, max_pts=1200, chunk=8, generator=None):
    """Batched wrapper -- see _sinkhorn_chunk for the algorithm.

    The transport plan is O(B*n*m). At B=50, n=m=3000 that is 4.5e8 entries (1.8GB per
    tensor in fp32) and log-domain Sinkhorn allocates several of them, which OOMs a 24GB
    card. Chunking the batch keeps peak memory at chunk*n*m and makes the point budget an
    independent knob from the batch size.
    """
    outs, infos = [], []
    for s in range(0, src.shape[0], chunk):
        l, i = _sinkhorn_chunk(src[s : s + chunk], tgt[s : s + chunk], blur, n_iter, max_pts, generator)
        outs.append(l)
        infos.append(i["plan_concentration"])
    return torch.stack(outs).mean(), {"plan_concentration": float(np.mean(infos))}


def _sinkhorn_chunk(src, tgt, blur=0.05, n_iter=25, max_pts=1200, generator=None):
    """Entropic-OT (Sinkhorn) divergence between two point sets.

    Subsamples to `max_pts` per side because the transport plan is O(N*M) in memory:
    at 3000x3000 that is 9e6 entries (~36MB in fp32), which is fine; at 8000x8000 it is
    64e6 (~256MB) per batch element, which is not.

    blur is the entropic regularisation expressed as a LENGTH (epsilon = blur^2), so it is
    directly comparable to the robust-kernel scale and can be annealed on the same schedule.
    """
    B, N, _ = src.shape
    M = tgt.shape[1]
    if N > max_pts:
        i = torch.randperm(N, device=src.device, generator=generator)[:max_pts]
        src = src[:, i]
    if M > max_pts:
        j = torch.randperm(M, device=tgt.device, generator=generator)[:max_pts]
        tgt = tgt[:, j]

    C = torch.cdist(src, tgt).pow(2)  # (B,n,m)
    eps = blur**2
    n, m = C.shape[1], C.shape[2]
    mu = torch.full((B, n), 1.0 / n, device=C.device)
    nu = torch.full((B, m), 1.0 / m, device=C.device)
    f = torch.zeros_like(mu)
    g = torch.zeros_like(nu)

    # log-domain Sinkhorn for numerical stability at small blur
    for _ in range(n_iter):
        f = -eps * torch.logsumexp((g.unsqueeze(1) - C) / eps + nu.log().unsqueeze(1), dim=2)
        g = -eps * torch.logsumexp((f.unsqueeze(2) - C) / eps + mu.log().unsqueeze(2), dim=1)

    loss = (f * mu).sum(-1).mean() + (g * nu).sum(-1).mean()
    with torch.no_grad():
        P = torch.exp((f.unsqueeze(2) + g.unsqueeze(1) - C) / eps) * mu.unsqueeze(2) * nu.unsqueeze(1)
        # how concentrated is the plan? 1/n means fully spread, 1.0 means a permutation
        conc = float((P.max(dim=2).values.sum(-1)).mean().item())
    return loss, {"plan_concentration": conc}
