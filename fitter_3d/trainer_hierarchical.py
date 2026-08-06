"""Hierarchical, part-anchored registration — the main structural experiment.

MOTIVATION (measured, not assumed)

The baseline diagnosis says pose is the bottleneck: `joint_rot` moves 0.216 rad mean per
joint in the only stage that can move it, then freezes for 83% of the run, and free-form
`deform_verts` (mean 0.040, max 0.380 of specimen half-extent) papers over the residual at
the cost of 74% average edge distortion.

The obvious fix -- give pose a bigger learning rate -- was TRIED AND FAILED (arm E1):
raising `joint_rot` lr from 0.002 to 0.02 made the pose fit WORSE than its own
initialisation (chamfer 0.0053 -> 0.0083). That is diagnostic. It means pose is not
gradient-starved, it is TRAPPED: chamfer against an ant is massively multimodal because six
legs are near-identical and mutually substitutable, so a bigger step just reaches a wrong
basin faster.

You cannot fix a multimodal objective with a better optimizer. You have to remove the
ambiguity. That is what this file does.

THE IDEA

An ant's legs are not interchangeable in ANATOMY, only in APPEARANCE. Each leg attaches at
a known coxa on the thorax, and the template knows where every coxa is. So:

  1. Fit the BODY only (thorax, gaster, head -- the unambiguous, high-mass part).
     No leg joint may move. The body is a chain of three blobs along one axis and is
     essentially unimodal, so this converges reliably.

  2. With the body placed, PARTITION the target point cloud by anatomical territory:
     assign every target point to the template part whose current surface is nearest,
     restricted so that leg points can only be claimed by legs. This is an EM-style hard
     assignment, recomputed as the fit improves.

  3. Fit each leg chain against ONLY its own assigned target points. A leg can no longer be
     rewarded for matching a different leg's surface, because that surface is not in its
     data term. The multimodality is gone by construction, not by regularization.

  4. Refine everything jointly, with the partition still restricting the data term.

WHAT WOULD FALSIFY THIS
If per-part assignment is unstable across iterations (points flip-flopping between legs),
the data term is non-stationary and the fit will oscillate rather than converge. The
`assignment_churn` diagnostic below measures exactly that and is logged every re-assignment,
so the failure is visible rather than silent.
"""

import os

import numpy as np
import torch
from pytorch3d.ops import knn_points, sample_points_from_meshes

import config
from fitter_3d.joint_limits import joint_limit_tensors, limit_hinge, scale_barrier, trans_barrier
from fitter_3d.trainer import get_meshes
from fitter_3d.trainer_moonshot import (
    rest_edge_loss,
    robust_chamfer,
    build_lr_joint_pairs,
    symmetry_penalty,
    midline_penalty,
)

nn = torch.nn


# --------------------------------------------------------------------------------------
# anatomical grouping
# --------------------------------------------------------------------------------------
# segments counted as DISTAL when split_distal is enabled
DISTAL_SEGMENTS = ("ti", "ta", "pt")


def anatomical_groups(joint_names, split_distal=False, split_anterior=False):
    """Map each joint index to 'body', or to a leg group.

    split_anterior: give the head, mandibles and antennae their own groups instead of folding
    them into 'body'.

    THE FOLD WAS A MEASURED MISTAKE (REPORT §6.10). The paragraph below used to argue that
    splitting the anterior "would create parts too small to carry a stable data term". That
    is false, and backwards for the head. Measured on the template:

        head      19.53% of surface area, 1563 expected samples at n_sample=8000
        mandible   3.11%,  248 samples
        antenna    1.70%,  136 samples
        each leg   5.5-6.6%, 439-531 samples   <- and each leg DOES get its own term

    The head is three times larger than any leg group, and even the antenna clears the code's
    own n_t<10 skip threshold by 13x. Nothing here was ever too small.

    What the fold actually cost: with head, mandible and antenna target points in one
    undifferentiated pool, covering the head with mandible vertices costs the chamfer nothing
    -- the entire anti-multimodality mechanism is simply not applied anteriorly. Measured
    consequence in every pose-mobile arm: the head deforms 24-40% LESS than the thorax while
    the antennae deform 44-82% MORE, i.e. the head is not carried forward and the appendages
    stretch to reach. That is the failure the domain expert reported from renders.

    ORIGINAL NOTE, retained because the distal reasoning below still stands:
    they are anterior and unambiguous relative to the legs, and splitting them further
    would create parts too small to carry a stable data term (pretarsus is 7-15 verts).

    split_distal: give each leg TWO groups, proximal (coxa/trochanter/femur) and distal
    (tibia/tarsus/pretarsus), instead of one.

    WHY THIS EXISTS (probe 13, measured on the template):
      coxa+trochanter+femur hold 94.0% of a leg chain's surface area; tarsus+pretarsus hold
      2.4%. With one area-sampled chamfer over the whole chain, the distal tip is effectively
      unconstrained -- at the fitter's n_sample=8000 the pretarsus expects ~0.3 samples and
      the tarsus ~12.6, against the code's own n_t<10 skip threshold. That is why M7 still
      regressed part_leg_distal (+15.5%) after every other metric improved.
      Splitting the chain gives the distal half its own equally-weighted data term, so a
      2.4%-of-area structure gets a 50%-of-leg vote.

      Note this supersedes the 'couple mirrored leg chains' idea: the template is left/right
      symmetric to <0.3% by area, so a mirror-coupling term has almost nothing to correct,
      and coupling two equally under-constrained tips would not constrain either.
    """
    groups = {}
    for j, nm in enumerate(joint_names):
        if nm.startswith("l_"):
            # 'l_2_fe_r' -> leg 2, segment 'fe', side 'r'
            bits = nm.split("_")
            k, seg, side = bits[1], bits[2], bits[-1]
            if split_distal:
                half = "d" if seg in DISTAL_SEGMENTS else "p"
                groups[j] = f"l{k}{half}_{side}"
            else:
                groups[j] = f"l{k}_{side}"
        elif split_anterior and nm == "b_h":
            groups[j] = "head"
        elif split_anterior and nm.startswith("ma"):
            groups[j] = "mandible"
        elif split_anterior and nm.startswith("an_"):
            groups[j] = "antenna"
        else:
            groups[j] = "body"
    return groups


def vertex_groups(weights, joint_names, split_distal=False, split_anterior=False):
    """Assign each template vertex to an anatomical group by dominant skinning weight."""
    dom = np.asarray(weights).argmax(axis=1)
    jg = anatomical_groups(joint_names, split_distal=split_distal, split_anterior=split_anterior)
    names = sorted(set(jg.values()))
    name2i = {n: i for i, n in enumerate(names)}
    vg = np.array([name2i[jg[d]] for d in dom], dtype=np.int64)
    return vg, names


def joint_mask_for_groups(joint_names, allowed, split_distal=False, split_anterior=False):
    """Boolean mask over joint_rot rows (i.e. joints EXCLUDING the global root) selecting
    the joints belonging to `allowed` groups."""
    jg = anatomical_groups(joint_names, split_distal=split_distal, split_anterior=split_anterior)
    # joint_rot has N_POSE = len(joint_names) - 1 rows; row i corresponds to joint i+1
    m = torch.zeros(len(joint_names) - 1, dtype=torch.bool)
    for j in range(1, len(joint_names)):
        if jg[j] in allowed:
            m[j - 1] = True
    return m


def part_thickness(verts, vertex_group, n_groups):
    """Characteristic 'tube radius' of each anatomical group on the TEMPLATE.

    For each group, fit its principal axis and take the median distance of its vertices to
    that axis. For a limb segment this is the radius; for the body it is the half-width.

    WHY: finding 4 of the report -- robust kernels and correspondence filters delete thin
    structures, because a single global scale that is sensible for the body treats a leg as
    an outlier. Measured: distal legs are read as outliers at 36x the body's rate at c=0.10,
    and retain 5.8x fewer correspondences under a mutual-NN + ratio filter.
    A robust scale must therefore be proportional to the LOCAL structure size, not global.

    Returned in template units; the caller scales it by a multiplier to get the kernel c.
    """
    out = torch.zeros(n_groups, dtype=torch.float32, device=verts.device)
    for g in range(n_groups):
        m = vertex_group == g
        if int(m.sum()) < 8:
            out[g] = float("nan")
            continue
        p = verts[m]
        c = p.mean(0, keepdim=True)
        d = p - c
        # principal axis via SVD of the centred coordinates
        try:
            _, _, V = torch.linalg.svd(d.double(), full_matrices=False)
            axis = V[0].float()
        except Exception:
            axis = torch.tensor([1.0, 0.0, 0.0], device=verts.device)
        # perpendicular distance to that axis
        perp = d - (d @ axis).unsqueeze(-1) * axis.unsqueeze(0)
        out[g] = perp.norm(dim=-1).median()
    # groups too small to measure inherit the median of the rest
    good = ~torch.isnan(out)
    if good.any():
        out[~good] = out[good].median()
    return out


# --------------------------------------------------------------------------------------
# the partition
# --------------------------------------------------------------------------------------
class TargetPartition:
    """Assigns target surface points to anatomical groups, EM-style.

    Recomputed on demand from the CURRENT fitted mesh: each target point is given the group
    of its nearest fitted-template vertex. Because the body is fitted first and frozen-ish,
    the body claims its own territory reliably, and leg territories then fall out of
    proximity to the placed coxae.

    Churn (fraction of target points that changed group since the last recompute) is
    reported so a non-stationary data term is visible instead of silent.
    """

    def __init__(self, vertex_group, n_groups, device):
        self.vg = torch.as_tensor(vertex_group, device=device)
        self.n_groups = n_groups
        self.device = device
        self.assign = None  # (B, P) group id per target point
        # Churn must be measured on a FIXED set of probe points. The working target sample
        # is redrawn each reassignment (deliberately -- it stops the partition from being
        # fitted to one point sample), so comparing successive assignment arrays directly
        # compares different points and reports ~0.65 churn no matter what the fit does.
        # That is a broken diagnostic, so churn is tracked separately on frozen probes.
        self.probe_pts = None
        self.probe_assign = None

    def set_probes(self, probe_pts):
        self.probe_pts = probe_pts
        self.probe_assign = None

    def soft_responsibilities(self, fitted_verts, tgt_pts, temp):
        """E-step: posterior that each target point belongs to each anatomical group.

        The hard assignment in `update` gives every point entirely to its argmin group, so a
        point equidistant between two legs commits fully to one of them and the fit is then
        pulled toward that commitment. Report §6.4 measured the population consequence: the
        resulting registrations carry no learnable shape structure, i.e. correspondence is not
        consistent across specimens. Softening lets an ambiguous point contribute to several
        groups in proportion to how well each explains it, so uncertainty reduces influence
        instead of becoming a wrong commitment -- the E-step of the EM formulation used for
        robust statistical-shape-model fitting.

        Returns (B, P, G) responsibilities summing to 1 over G.
        """
        with torch.no_grad():
            B, P, _ = tgt_pts.shape
            G = self.n_groups
            d2 = torch.full((B, P, G), float("inf"), device=tgt_pts.device)
            for g in range(G):
                m = self.vg == g
                if int(m.sum()) < 4:
                    continue
                for b in range(B):
                    d2[b, :, g] = knn_points(tgt_pts[b : b + 1], fitted_verts[b : b + 1, m], K=1).dists[0, :, 0]
            # temp is a LENGTH; squaring puts it in the same units as the squared distances
            r = torch.softmax(-d2 / (2.0 * temp**2), dim=-1)
            return r

    def update(self, fitted_verts, tgt_pts):
        """fitted_verts (B,V,3), tgt_pts (B,P,3). Returns churn on the fixed probe set."""
        with torch.no_grad():
            idx = knn_points(tgt_pts, fitted_verts, K=1).idx[..., 0]  # (B,P)
            self.assign = self.vg[idx]

            churn = None
            if self.probe_pts is not None:
                pidx = knn_points(self.probe_pts, fitted_verts, K=1).idx[..., 0]
                pnew = self.vg[pidx]
                if self.probe_assign is not None:
                    churn = float((pnew != self.probe_assign).float().mean().item())
                self.probe_assign = pnew
            return churn

    def mask_for(self, group_id):
        return self.assign == group_id


# --------------------------------------------------------------------------------------
# hierarchical stage
# --------------------------------------------------------------------------------------
class HierarchicalStage:
    """A stage that optimises a SUBSET of joints against a PARTITIONED data term.

    Parameters
    ----------
    active_groups : list[str] | None
        anatomical groups whose joints may rotate this stage. None = all.
    partitioned : bool
        if True, each group's template vertices are matched only against target points
        assigned to that group. If False, a plain global chamfer is used.
    """

    def __init__(
        self,
        name,
        nits,
        smal,
        target_meshes,
        vertex_group,
        group_names,
        joint_names,
        active_groups=None,
        partitioned=True,
        split_distal=False,
        split_anterior=False,
        lr=0.01,
        joint_lr=0.01,
        n_sample=8000,
        loss_weights=None,
        reassign_every=50,
        device="cuda",
        out_dir=".",
        log_every=100,
        robust_kernel="l2",
        robust_scale=0.2,
        optimise_deform=False,
        part_scale=None,
        robust_mult=3.0,
        partition=None,
        chamfer_groups=None,
        soft_partition=0.0,
    ):
        self.name = name
        self.n_it = nits
        self.smal = smal
        self.targets = target_meshes
        self.device = device
        self.out_dir = out_dir
        self.n_sample = n_sample
        self.reassign_every = reassign_every
        self.log_every = log_every
        self.partitioned = partitioned
        self.group_names = group_names
        self.robust_kernel = robust_kernel
        self.robust_scale = robust_scale
        # per-part robust scale (item 2). None = disabled, keeping the previous behaviour.
        self.part_scale = part_scale
        self.robust_mult = robust_mult
        self.chamfer_groups = set(chamfer_groups) if chamfer_groups is not None else None
        # soft_partition > 0 is the responsibility temperature, as a LENGTH in target units.
        # 0 keeps the hard argmin assignment every pre-existing arm uses.
        self.soft_partition = float(soft_partition)
        self.resp = None

        self.lw = dict(
            w_chamfer=1.0,
            w_edge=0.05,
            w_normal=0.01,
            w_laplacian=0.005,
            w_beta_prior=0.002,
            w_sym=0.5,
            w_offset=0.0,
            w_midline=0.0,
            w_limit=0.0,
            w_scale=0.0,
            w_trans=0.0,
        )
        if loss_weights:
            self.lw.update(loss_weights)

        # Authored rotation limits, or (None, None) when the model has none. Built once here
        # rather than per-iteration; config.dd is the same dict SMAL loaded from config.SMAL_FILE.
        self.min_limits, self.max_limits = joint_limit_tensors(config.dd, device)

        self.vg = torch.as_tensor(vertex_group, device=device)
        # An injected partition (see fitter_3d.partfield.PartFieldPartition) is TARGET-derived
        # and frozen; the default one is FIT-derived and recomputed from the current mesh.
        # Both satisfy the same three-method interface, so nothing below this line changes.
        self.part = partition if partition is not None else TargetPartition(vertex_group, len(group_names), device)

        # which joint_rot rows may move
        if active_groups is None:
            self.joint_mask = torch.ones(smal.joint_rot.shape[1], dtype=torch.bool, device=device)
        else:
            self.joint_mask = joint_mask_for_groups(
                joint_names, set(active_groups), split_distal=split_distal, split_anterior=split_anterior
            ).to(device)

        params = [
            {"params": [smal.global_rot], "lr": lr},
            {"params": [smal.trans], "lr": lr},
            {"params": [smal.betas], "lr": lr},
            {"params": [smal.log_beta_scales], "lr": lr * 0.25},
            {"params": [smal.betas_trans], "lr": lr * 0.25},
            {"params": [smal.joint_rot], "lr": joint_lr},
        ]
        if optimise_deform:
            params.append({"params": [smal.deform_verts], "lr": lr * 0.1})
        self.optimise_deform = optimise_deform
        self.optimizer = torch.optim.Adam(params)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(nits, 1))

        with torch.no_grad():
            self.src_verts = smal().detach()
            self.rest_verts = smal(deform_verts=torch.zeros_like(smal.deform_verts)).detach()
        self.faces = smal.faces.detach()
        self.src_mesh = get_meshes(self.src_verts, self.faces, device=device)

        self.lr_pairs = build_lr_joint_pairs(joint_names).to(device)
        # midsagittal vertex set, for the planarity constraint (see midline_penalty).
        # Loaded only when the term is active so the default path is unchanged.
        self.sym_verts = None
        if self.lw.get("w_midline", 0.0) > 0:
            import pickle as _p

            with open(config.SMAL_FILE, "rb") as _f:
                _u = _p._Unpickler(_f)
                _u.encoding = "latin1"
                self.sym_verts = torch.tensor(np.asarray(_u.load()["sym_verts"]).astype(np.int64), device=device)
        self.history = []
        self.churn_history = []

    # -------------------------------------------------- data term
    def _partitioned_chamfer(self, fitted, tgt_pts):
        """Sum of per-group bidirectional chamfer, each restricted to its own territory.

        Implementation note: this loops over groups (7 of them: body + 6 legs) rather than
        vectorising, because each group has a different number of assigned target points
        per batch element. 7 small knn calls per iteration is cheap relative to the
        forward pass, and keeping it explicit makes the restriction auditable.
        """
        B = fitted.shape[0]
        total = 0.0
        n_terms = 0
        for g, gname in enumerate(self.group_names):
            # chamfer_groups=None (the default, and what every pre-existing arm uses) sums
            # over ALL groups, so a frozen part is still anchored while an active one moves.
            # Restricting it is only meaningful for the H0 body stage under a target-derived
            # partition, where summing over the leg groups would drag the body toward legs
            # that have not been fitted yet.
            if self.chamfer_groups is not None and gname not in self.chamfer_groups:
                continue
            vmask = self.vg == g
            if vmask.sum() < 10:
                continue
            src_g = fitted[:, vmask, :]  # (B, Vg, 3)
            for b in range(B):
                tmask = self.part.assign[b] == g
                n_t = int(tmask.sum())
                if n_t < 10 and self.resp is None:
                    # This group claimed almost no target surface. Skip rather than force a
                    # match: forcing would drag the part onto whatever is nearest, which is
                    # exactly the leg-swap failure this design exists to prevent.
                    continue
                sg = src_g[b].unsqueeze(0)
                if self.resp is not None:
                    # SOFT: every target point contributes to this group in proportion to its
                    # responsibility. The target->source direction is weighted exactly. The
                    # source->target direction cannot be weighted inside a knn, so it searches
                    # the points this group is at all responsible for (w above uniform), which
                    # is a soft-derived subset rather than an argmin commitment.
                    w = self.resp[b, :, g]
                    sub = w > (1.0 / len(self.group_names))
                    if int(sub.sum()) < 10:
                        continue
                    tg = tgt_pts[b][sub].unsqueeze(0)
                    d_st = knn_points(sg, tg, K=1).dists[..., 0]
                    d_ts_all = knn_points(tgt_pts[b].unsqueeze(0), sg, K=1).dists[..., 0]
                    d_ts = (d_ts_all * w.unsqueeze(0)) / w.sum().clamp_min(1e-8) * w.numel()
                else:
                    tg = tgt_pts[b][tmask].unsqueeze(0)
                    d_st = knn_points(sg, tg, K=1).dists[..., 0]
                    d_ts = knn_points(tg, sg, K=1).dists[..., 0]
                if self.part_scale is not None:
                    # PER-PART robust kernel: c is proportional to this part's own
                    # characteristic thickness, so a leg is not judged by the body's
                    # tolerance. See part_thickness() for why a global scale fails.
                    c2 = (self.robust_mult * self.part_scale[g]) ** 2
                    d_st = d_st * c2 / (d_st + c2)
                    d_ts = d_ts * c2 / (d_ts + c2)
                total = total + d_st.mean() + d_ts.mean()
                n_terms += 1
        return total / max(n_terms, 1)

    def forward(self, mesh, fitted, tgt_pts):
        comp = {}
        loss = 0.0
        if self.lw["w_chamfer"] > 0:
            if self.partitioned:
                l_ch = self._partitioned_chamfer(fitted, tgt_pts)
            else:
                src_pts = sample_points_from_meshes(mesh, self.n_sample)
                l_ch, _, _ = robust_chamfer(src_pts, tgt_pts, kernel=self.robust_kernel, scale=self.robust_scale)
            comp["chamfer"] = l_ch
            loss = loss + self.lw["w_chamfer"] * l_ch

        if self.lw["w_edge"] > 0:
            l_e = rest_edge_loss(mesh.verts_padded(), self.rest_verts, self.faces[0])
            comp["edge"] = l_e
            loss = loss + self.lw["w_edge"] * l_e

        if self.lw["w_laplacian"] > 0:
            from pytorch3d.loss import mesh_laplacian_smoothing

            l_l = mesh_laplacian_smoothing(mesh, method="uniform")
            comp["lap"] = l_l
            loss = loss + self.lw["w_laplacian"] * l_l

        if self.lw["w_beta_prior"] > 0:
            d = self.smal.betas - self.smal.mean_betas.unsqueeze(0)
            l_b = (d @ self.smal.betas_prec).pow(2).sum(-1).mean()
            comp["beta"] = l_b
            loss = loss + self.lw["w_beta_prior"] * l_b

        if self.lw["w_sym"] > 0:
            l_s = symmetry_penalty(self.smal.log_beta_scales, self.smal.betas_trans, self.lr_pairs)
            comp["sym"] = l_s
            loss = loss + self.lw["w_sym"] * l_s

        if self.lw.get("w_jresid", 0.0) > 0:
            # Penalise the FREE per-joint scale/translation residual, so that once
            # scaledirs/transdirs are coupled to the betas (config.COUPLE_JOINT_BLENDSHAPES)
            # the shape space carries the variation and the free parameters only fine-tune.
            # Without this the residual is unregularised and simply out-competes the betas,
            # which is the defect measured in REPORT.md §6.6 (70-400x more inter-specimen
            # variation in log_beta_scales than in betas).
            l_j = self.smal.log_beta_scales.pow(2).mean() + self.smal.betas_trans.pow(2).mean()
            comp["jres"] = l_j
            loss = loss + self.lw["w_jresid"] * l_j

        if self.lw.get("w_limit", 0.0) > 0:
            # Authored per-joint rotation limits (REPORT §6.9). The arms that unfreeze pose
            # drive 37 of 96 constrained axes out of anatomical range on every specimen,
            # concentrated on the distal hind leg -- exactly the joints §4 shows the area-
            # sampled data term cannot constrain. This is the only term here that acts on
            # those joints at all.
            if self.min_limits is None:
                raise ValueError(
                    "w_limit > 0 but the model file authors no 'joint_limits'. "
                    "Point config.SMAL_FILE at a model that does (e.g. OmniAnt_25PCs_joint_limited.pkl)."
                )
            l_lim = limit_hinge(self.smal.joint_rot, self.min_limits, self.max_limits)
            comp["limit"] = l_lim
            loss = loss + self.lw["w_limit"] * l_lim

        if self.lw.get("w_scale", 0.0) > 0:
            # Bound per-joint scale (REPORT §6.10.1): the anterior partition split was a null,
            # leaving unbounded scale as the last untested mechanism for the head collapsing
            # into the thorax while the appendages stretch to cover it.
            l_sc = scale_barrier(self.smal.log_beta_scales)
            comp["scale"] = l_sc
            loss = loss + self.lw["w_scale"] * l_sc

        if self.lw.get("w_trans", 0.0) > 0:
            # Separate from w_scale: per-part translation is banded in absolute model units
            # (a fraction of body length), scale in log-ratio. See trans_barrier.
            l_tr = trans_barrier(self.smal.betas_trans)
            comp["trans"] = l_tr
            loss = loss + self.lw["w_trans"] * l_tr

        if self.lw.get("w_midline", 0.0) > 0 and self.sym_verts is not None:
            l_m = midline_penalty(mesh.verts_padded(), self.sym_verts)
            comp["mid"] = l_m
            loss = loss + self.lw["w_midline"] * l_m

        if self.optimise_deform and self.lw.get("w_offset", 0) > 0:
            l_o = self.smal.deform_verts.pow(2).sum(-1).mean()
            comp["off"] = l_o
            loss = loss + self.lw["w_offset"] * l_o

        return loss, comp

    def run(self):
        tgt_pts = sample_points_from_meshes(self.targets, self.n_sample)
        # frozen probe set for an honest churn measurement (see TargetPartition.set_probes)
        self.part.set_probes(sample_points_from_meshes(self.targets, 3000).detach())
        for i in range(self.n_it):
            self.optimizer.zero_grad()
            fitted = self.smal()
            mesh = self.src_mesh.offset_verts((fitted - self.src_verts).view(-1, 3))

            if self.partitioned and (i % self.reassign_every == 0):
                # resample the target periodically too, so the partition is not fitted to
                # one fixed point sample
                tgt_pts = sample_points_from_meshes(self.targets, self.n_sample)
                churn = self.part.update(fitted.detach(), tgt_pts)
                if churn is not None:
                    self.churn_history.append(churn)
                if self.soft_partition > 0 and hasattr(self.part, "soft_responsibilities"):
                    self.resp = self.part.soft_responsibilities(fitted.detach(), tgt_pts, self.soft_partition)

            loss, comp = self.forward(mesh, fitted, tgt_pts)
            loss.backward()

            # freeze joints outside the active groups by zeroing their gradient. Done on
            # the gradient rather than by excluding the parameter, because joint_rot is a
            # single tensor -- this is the only way to move a SUBSET of its rows.
            if self.smal.joint_rot.grad is not None:
                self.smal.joint_rot.grad[:, ~self.joint_mask, :] = 0.0

            self.optimizer.step()
            self.scheduler.step()
            self.history.append(float(loss.detach()))

            if self.log_every and (i % self.log_every == 0 or i == self.n_it - 1):
                parts = "  ".join(f"{k}={float(v):.5f}" for k, v in comp.items())
                ch = f"  churn={self.churn_history[-1]:.3f}" if self.churn_history else ""
                print(f"    [{self.name}] {i:4d}/{self.n_it} loss={float(loss):.6f}  {parts}{ch}", flush=True)

    def save_npz(self, labels=None):
        out = {}
        for p in ["global_rot", "joint_rot", "betas", "log_beta_scales", "trans", "deform_verts", "betas_trans"]:
            out[p] = getattr(self.smal, p).cpu().detach().numpy()
        out["verts"] = self.smal().cpu().detach().numpy()
        out["faces"] = self.faces.cpu().detach().numpy()
        out["labels"] = labels
        np.savez(os.path.join(self.out_dir, f"{self.name}.npz"), **out)
