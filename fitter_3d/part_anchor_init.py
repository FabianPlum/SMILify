"""Pose initialisation from part anchors: put the fit in the right basin before any chamfer.

WHY
---
The central finding of this investigation (report section 5) is that pose is *trapped*, not
under-optimised: raising the pose learning rate made the fit worse than its own
initialisation, because the chamfer objective over a hexapod is multimodal -- six
substitutable legs, and gradient descent cannot cross between basins. Every fix so far has
attacked the *objective* (partitioning, robust kernels, staging). None attacked the
*initialisation*, because there was nothing to initialise from: the template rest pose is
the only prior available, and it is the same for every specimen.

A target-derived part field changes that. It says where each of the six legs is on THIS
scan before a single iteration has run. That is exactly the information needed to jump to
the right basin instead of descending into whichever one the rest pose happens to sit in.

HOW
---
For each anatomical part, compare two 3x3-plus-3 summaries -- centroid and second moment --
one from the target points the field assigned to that part, one from the posed template
vertices of that part:

    L = sum_g  w_g [ ||c_s - c_t||^2 / s^2  +  mu * ||M_s - M_t||_F^2 / s^4 ]

Matching centroid AND covariance pins each part's position, orientation and extent. There
is NO correspondence and NO nearest-neighbour search anywhere in it, which is the point:
chamfer's multimodality comes from the freedom to choose which target point a source point
explains, and this objective removes that freedom entirely. Thirteen parts contribute 13x9
numbers; the pose has ~160 degrees of freedom, so it is heavily overdetermined and
well-conditioned.

Deliberately NOT using SVD to extract a principal axis per part. torch.linalg.svd is
differentiable but degenerates when two singular values are close -- which happens
constantly here, since a coxa is nearly isotropic -- and the gradient blows up. The raw
second-moment matrix carries the same information with no decomposition and no failure
mode.

Scale s is the specimen's own extent, so the two terms are dimensionless and mu is
comparable across specimens of very different absolute size.
"""

import torch


def part_moments(pts, mask, w=None, eps=1e-8):
    """Centroid and second moment of the masked points, optionally AREA-WEIGHTED.

    `w` (P,) is a per-point weight. It is not optional in spirit, only in signature: see
    `vertex_areas` for why an unweighted vertex moment is biased against an area-sampled
    target moment, and by how much.

    Returns (c, M, n) with c (3,), M (3,3), n the point count. Empty parts return zeros and
    n=0 so the caller can drop them rather than divide by zero.
    """
    n = int(mask.sum())
    if n < 4:
        return (
            torch.zeros(3, device=pts.device, dtype=pts.dtype),
            torch.zeros(3, 3, device=pts.device, dtype=pts.dtype),
            0,
        )
    p = pts[mask]
    if w is None:
        c = p.mean(0)
        d = p - c
        M = (d.t() @ d) / (n + eps)
    else:
        ww = w[mask].unsqueeze(-1)
        s = ww.sum() + eps
        c = (ww * p).sum(0) / s
        d = p - c
        M = (d * ww).t() @ d / s
    return c, M, n


def vertex_areas(verts, faces):
    """Barycentric (one-third-of-incident-face) area per vertex. Differentiable in `verts`.

    THIS IS A BUG FIX, and the bug was severe enough to invert a conclusion.

    The target moments are computed over AREA-SAMPLED surface points, whose density is
    uniform per unit area. The source moments were computed over template VERTICES, whose
    density follows the tessellation and is wildly non-uniform on this mesh. Vertex density
    is not area density, so the two moment sets DISAGREE EVEN FOR A PERFECT FIT.

    Measured, on the ideal case where the target literally is the model's own mesh so the
    correct answer is "do not move": the unweighted objective reads a loss of 5.45e-04, and
    the body group's centroid is off by 7.92% of the specimen extent. The `M9c_anchorinit`
    arm converged to 3.4e-05 -- SIXTEEN TIMES BELOW the floor a perfect fit gives -- which
    means it was not finding a pose, it was deforming the model to chase a sampling bias.
    An end-to-end check on a perfectly-fitted specimen showed it driving chamfer 105x worse,
    rotating mandibles and antennae by 52-94 deg, and pushing per-joint scales to 0.15..11.7.

    Weighting each vertex by its barycentric area makes the discrete vertex moment a
    consistent estimator of the continuous surface moment, so a perfect fit reads ~0.
    """
    v = verts[faces]  # (F,3,3)
    a = 0.5 * torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1).norm(dim=-1)  # (F,)
    w = torch.zeros(verts.shape[0], device=verts.device, dtype=verts.dtype)
    w = w.index_add(0, faces.reshape(-1), (a / 3.0).repeat_interleave(3))
    return w


class PartAnchorInit:
    """Fit pose to per-part (centroid, second moment) anchors. Correspondence-free.

    Only runs with a target-derived partition. With the fit-derived one the anchors would be
    computed from the fit's own current opinion, so the objective would be trivially
    satisfied at the starting point and the stage would do nothing -- which is a compact way
    of saying why the part field is the enabling piece here.
    """

    def __init__(
        self,
        smal,
        partition,
        vertex_group,
        group_names,
        device,
        faces=None,
        n_it=400,
        lr=0.05,
        joint_lr=0.05,
        mu=1.0,
        min_pts=40,
        log_every=100,
    ):
        self.smal = smal
        self.faces = faces if faces is not None else smal.faces.detach()
        self.part = partition
        self.vg = torch.as_tensor(vertex_group, device=device)
        self.group_names = list(group_names)
        self.device = device
        self.n_it = n_it
        self.mu = mu
        self.min_pts = min_pts
        self.log_every = log_every
        # log_beta_scales is included because the second-moment term encodes each part's
        # EXTENT, not just its position and orientation. A specimen whose femora are longer
        # than the template's cannot match its leg anchors by rotation alone, and per-joint
        # scale is exactly the parameter that expresses that. It moves at half rate so the
        # init prefers to explain an anchor by posing the limb before stretching it.
        # betas are deliberately excluded: 13 anchors driving a 13-dimensional shape space
        # would fit shape to what is really pose error, and shape is what H0 onward is for.
        self.optimizer = torch.optim.Adam(
            [
                {"params": [smal.global_rot], "lr": lr},
                {"params": [smal.trans], "lr": lr},
                {"params": [smal.joint_rot], "lr": joint_lr},
                {"params": [smal.log_beta_scales], "lr": lr * 0.5},
            ]
        )
        self.history = []

    def _targets(self, tgt_pts):
        """Precompute target anchors once -- they never change, the partition is frozen."""
        B = tgt_pts.shape[0]
        out = []
        for b in range(B):
            per = {}
            for g in range(len(self.group_names)):
                c, M, n = part_moments(tgt_pts[b], self.part.assign[b] == g)
                if n >= self.min_pts:
                    per[g] = (c.detach(), M.detach())
            out.append(per)
        return out

    def run(self, tgt_pts, scale=None):
        """tgt_pts (B,P,3) target sample. Returns the loss history."""
        B = tgt_pts.shape[0]
        tgt = self._targets(tgt_pts)
        if scale is None:
            scale = (tgt_pts.amax(dim=1) - tgt_pts.amin(dim=1)).amax(dim=1).clamp_min(1e-6)  # (B,)
        s2 = (scale**2).view(B)
        n_anchor = sum(len(t) for t in tgt)
        print(
            f"    [H_init] correspondence-free part-anchor init: {n_anchor} anchors over "
            f"{B} specimens ({n_anchor / max(B, 1):.1f}/specimen of "
            f"{len(self.group_names)} parts)",
            flush=True,
        )
        for it in range(self.n_it):
            self.optimizer.zero_grad(set_to_none=True)
            fitted = self.smal(deform_verts=torch.zeros_like(self.smal.deform_verts))
            w_area = torch.stack([vertex_areas(fitted[b], self.faces) for b in range(B)])
            loss = 0.0
            nt = 0
            for b in range(B):
                for g, (ct, Mt) in tgt[b].items():
                    vm = self.vg == g
                    if int(vm.sum()) < 4:
                        continue
                    # AREA-WEIGHTED source moments -- see vertex_areas() for why an
                    # unweighted vertex moment is biased against an area-sampled target
                    # moment even at a perfect fit.
                    p = fitted[b][vm]
                    ww = w_area[b][vm].unsqueeze(-1)
                    sw = ww.sum() + 1e-12
                    cs = (ww * p).sum(0) / sw
                    d = p - cs
                    Ms = (d * ww).t() @ d / sw
                    loss = loss + (cs - ct).pow(2).sum() / s2[b] + self.mu * (Ms - Mt).pow(2).sum() / s2[b] ** 2
                    nt += 1
            loss = loss / max(nt, 1)
            loss.backward()
            self.optimizer.step()
            self.history.append(float(loss))
            if it % self.log_every == 0 or it == self.n_it - 1:
                print(f"    [H_init] {it:4d}/{self.n_it} anchor_loss={float(loss):.6f}", flush=True)
        return self.history
