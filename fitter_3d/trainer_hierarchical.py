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
from pytorch3d.structures import Meshes

import config
from fitter_3d.trainer import SMAL3DFitter, get_meshes
from fitter_3d.trainer_moonshot import (
    MoonshotStage,
    rest_edge_loss,
    robust_chamfer,
    build_lr_joint_pairs,
    symmetry_penalty,
)

nn = torch.nn


# --------------------------------------------------------------------------------------
# anatomical grouping
# --------------------------------------------------------------------------------------
def anatomical_groups(joint_names):
    """Map each joint index to one of: 'body', or one of the six leg chains 'l{k}_{side}'.

    Head, mandibles and antennae are folded into 'body' for the purpose of the partition:
    they are anterior and unambiguous relative to the legs, and splitting them further
    would create parts too small to carry a stable data term (pretarsus is 7-15 verts).
    """
    groups = {}
    for j, nm in enumerate(joint_names):
        if nm.startswith("l_"):
            # 'l_2_fe_r' -> leg group 'l2_r'
            bits = nm.split("_")
            groups[j] = f"l{bits[1]}_{bits[-1]}"
        else:
            groups[j] = "body"
    return groups


def vertex_groups(weights, joint_names):
    """Assign each template vertex to an anatomical group by dominant skinning weight."""
    dom = np.asarray(weights).argmax(axis=1)
    jg = anatomical_groups(joint_names)
    names = sorted(set(jg.values()))
    name2i = {n: i for i, n in enumerate(names)}
    vg = np.array([name2i[jg[d]] for d in dom], dtype=np.int64)
    return vg, names


def joint_mask_for_groups(joint_names, allowed):
    """Boolean mask over joint_rot rows (i.e. joints EXCLUDING the global root) selecting
    the joints belonging to `allowed` groups."""
    jg = anatomical_groups(joint_names)
    # joint_rot has N_POSE = len(joint_names) - 1 rows; row i corresponds to joint i+1
    m = torch.zeros(len(joint_names) - 1, dtype=torch.bool)
    for j in range(1, len(joint_names)):
        if jg[j] in allowed:
            m[j - 1] = True
    return m


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

        self.lw = dict(
            w_chamfer=1.0, w_edge=0.05, w_normal=0.01, w_laplacian=0.005,
            w_beta_prior=0.002, w_sym=0.5, w_offset=0.0,
        )
        if loss_weights:
            self.lw.update(loss_weights)

        self.vg = torch.as_tensor(vertex_group, device=device)
        self.part = TargetPartition(vertex_group, len(group_names), device)

        # which joint_rot rows may move
        if active_groups is None:
            self.joint_mask = torch.ones(smal.joint_rot.shape[1], dtype=torch.bool, device=device)
        else:
            self.joint_mask = joint_mask_for_groups(joint_names, set(active_groups)).to(device)

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
            vmask = self.vg == g
            if vmask.sum() < 10:
                continue
            src_g = fitted[:, vmask, :]  # (B, Vg, 3)
            for b in range(B):
                tmask = self.part.assign[b] == g
                n_t = int(tmask.sum())
                if n_t < 10:
                    # This group claimed almost no target surface. Skip rather than force a
                    # match: forcing would drag the part onto whatever is nearest, which is
                    # exactly the leg-swap failure this design exists to prevent.
                    continue
                tg = tgt_pts[b][tmask].unsqueeze(0)
                sg = src_g[b].unsqueeze(0)
                d_st = knn_points(sg, tg, K=1).dists[..., 0]
                d_ts = knn_points(tg, sg, K=1).dists[..., 0]
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
                l_ch, _, _ = robust_chamfer(
                    src_pts, tgt_pts, kernel=self.robust_kernel, scale=self.robust_scale
                )
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
