"""Moonshot trainer — an isolated, improved Stage for ant template registration.

DISCIPLINE: this file is a copy-and-extend of fitter_3d/trainer.py. The live training path
(fitter_3d/trainer.py, fitter_3d/utils.py) is NOT modified anywhere. Everything here is
opt-in via fitter_3d/optimise_moonshot.py.

WHY EACH CHANGE EXISTS — every one is tied to a measured defect in the baseline
(see diagnostics/moonshot/probe_0*.py and the numbers quoted below):

1. POSE IS FROZEN AFTER STAGE_1  [probe 07]
   Measured: in Stage_2/Stage_3 the deltas for global_rot, joint_rot, betas and trans are
   exactly 0.00000 -- those stages use scheme 'deform' (deform_verts only). 2000 of 2400
   iterations (83%) cannot change pose.
   Measured: joint_rot moves only 0.216 rad (~12 deg) mean per joint during Stage_1, which
   is 36% of even the theoretical 300*0.002 = 0.6 rad ceiling.
   Measured: the resulting free-form offsets are enormous -- mean |deform_verts| = 0.040,
   p95 = 0.097, max = 0.380, on meshes normalized to max|coord| = 1.
   => FIX: give pose a real budget (higher lr, more iterations, cosine decay), and allow
      late stages to keep refining pose jointly with offsets ('all' scheme) instead of
      freezing it.

2. NOTHING PENALIZES deform_verts MAGNITUDE  [code reading + probe 07]
   Stage.forward() applies chamfer + edge + normal + laplacian. None of these directly
   bound the per-vertex offset, so offsets grow until they dominate the fit. Since these
   offsets are exactly what destroys vertex correspondence (the entire point of doing
   registration), they must be paid for.
   => FIX: explicit w_offset * ||deform_verts||^2 term, so the optimizer prefers to explain
      the target with POSE and SHAPE, and only spends offsets where it must.

3. mesh_edge_loss IS A SHRINKAGE FORCE, NOT A REGULARIZER  [source reading]
   pytorch3d's mesh_edge_loss defaults to target_length = 0.0, i.e. it penalizes
   (edge_length - 0)^2. At w_edge = 0.8 that actively pulls the mesh smaller. The stock
   config even comments 'reduces shrinkage' while still using it.
   => FIX: penalize deviation from the TEMPLATE's OWN rest edge lengths. This regularizes
      distortion without imposing a scale bias -- effectively a stiffness/ARAP-lite term.

4. CHAMFER IS NOT OUTLIER-ROBUST, AND THE TARGETS ARE CT SCANS  [user context + code]
   The targets are CT scans of ants in ethanol: messy geometry, debris, holes. Plain
   chamfer is a least-squares estimator and a single piece of floating debris drags the
   template toward it.
   => FIX: robust kernels (trimmed / Geman-McClure / Welsch) with optional graduated
      non-convexity, so far-away outliers get bounded influence.

5. CHAMFER SAMPLING IS ASYMMETRIC AND DENSITY-BIASED  [code reading]
   Baseline: chamfer_distance(sample_points_from_meshes(target, 3000), src.verts_padded()).
   The target side is area-sampled (3000 pts) but the source side uses ALL ~10229 raw
   template VERTICES. Vertex sets are density-biased toward finely-tessellated regions, so
   the loss silently weights parts of the template by tessellation rather than by area.
   => FIX: area-sample BOTH sides, with a configurable and much larger point budget.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from pytorch3d.ops import sample_points_from_meshes, knn_points
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency

import config
from fitter_3d.joint_limits import joint_limit_tensors, limit_hinge, scale_barrier, trans_barrier
from fitter_3d.trainer import SMALParamGroup, get_meshes  # reuse model, unmodified
from fitter_3d.utils import plot_meshes

nn = torch.nn

default_weights = dict(
    w_chamfer=1.0,
    w_edge=1.0,
    w_normal=0.01,
    w_laplacian=0.1,
    w_offset=0.0,  # NEW: penalty on ||deform_verts||
    w_normal_align=0.0,  # NEW: point-to-plane / normal agreement along correspondence
    w_beta_prior=0.0,  # NEW: Mahalanobis shape prior (see note 6)
    w_sym=0.0,  # NEW: left/right symmetry on per-joint scale & translation (note 7)
    w_midline=0.0,  # NEW: keep midsagittal vertices on y=0 (see midline_penalty)
    w_limit=0.0,  # NEW: authored per-joint rotation limits (see fitter_3d/joint_limits.py)
    w_scale=0.0,  # NEW: barrier on per-joint scale beyond a 2x free band (see scale_barrier)
    w_trans=0.0,  # NEW: barrier on per-joint translation beyond an absolute band (trans_barrier)
)


# --------------------------------------------------------------------------------------
# 6. THE SHAPE PRIOR SHIPPED WITH THE MODEL IS DEAD CODE
#    SMAL3DFitter.__init__ (fitter_3d/trainer.py:80-83) inverts the model's 13x13
#    'shape_cov', takes its Cholesky factor, and stores it as self.betas_prec -- and
#    nothing in the repository ever reads it. Verified by grep across all *.py: the only
#    occurrence of 'betas_prec' is the assignment itself. So `betas` is optimized with NO
#    prior at all, even though SMIL_OmniAnt.pkl ships a covariance estimated over 60+
#    species. Unregularized betas are free to take extreme values that the shape space was
#    never meant to express, which is one way the fit reaches implausible geometry.
#    => FIX: an opt-in Mahalanobis penalty ||L^T (beta - mu)||^2, which is exactly the
#       negative log-likelihood of a Gaussian shape prior up to a constant.
#
# 7. NOTHING ENFORCES LEFT/RIGHT CONSISTENCY ON PER-JOINT SCALE
#    log_beta_scales and betas_trans are per-joint (55x3) and completely independent
#    left vs right. Measured: log_beta_scales moves 12.5 units in Stage_0 alone -- more
#    than any other block -- and the fitted midline deviation grows from 0.0185 (Stage_0)
#    to 0.0354 (Stage_3). An ant is bilaterally symmetric in SHAPE even when its POSE is
#    not, so left/right joint SCALES should agree while joint ROTATIONS stay free.
#    => FIX: an opt-in penalty tying each '*_r' joint's scale/translation to its '*_l'
#       twin, leaving joint_rot untouched.
# --------------------------------------------------------------------------------------


def build_lr_joint_pairs(joint_names):
    """Index pairs (i_right, i_left) for joints whose names differ only by _r/_l suffix.

    Returns a LongTensor (P,2). Joints with no twin (b_t, b_h, b_a_*) are excluded.
    """
    idx = {n: i for i, n in enumerate(joint_names)}
    pairs = []
    for n, i in idx.items():
        if n.endswith("_r"):
            twin = n[:-2] + "_l"
            if twin in idx:
                pairs.append((i, idx[twin]))
    return torch.tensor(pairs, dtype=torch.long)


def midline_penalty(verts, sym_verts):
    """Keep the template's midsagittal vertices on the y = 0 plane.

    Added to fix the one regression of the M5 arm: it improved 8 of 11 metrics against the
    stock pipeline but worsened bilateral midline deviation by 43.8% (13/50, p=9.4e-04).

    `symmetry_penalty` ties left/right joint SCALES, which does not constrain the midline
    itself -- an out-of-plane rotation of a body joint (thorax, gaster, head) moves the
    midsagittal vertices off y=0 while leaving every left/right scale pair identical. That
    is what the hierarchical body stage was doing.

    This is legitimate because the constraint is on SHAPE, not pose: the specimens are
    ethanol-preserved and their limb pose is asymmetric, but their body midline is still a
    plane. `sym_verts` (223 indices, shipped in the model) is exactly that vertex set --
    verified here to have y-mean 0.0 and y-std exactly 0.0 on the template.

    Penalises the VARIANCE of the midline y, not its absolute value, so a legitimate global
    y-translation costs nothing and only non-planarity is charged. (Absolute-y would be
    almost right, since the targets are canonically aligned, but it would fight `trans`.)
    """
    y = verts[:, sym_verts.long(), 1]
    return (y - y.mean(dim=1, keepdim=True)).pow(2).mean()


def symmetry_penalty(log_beta_scales, betas_trans, pairs):
    """Penalise left/right disagreement in per-joint SCALE and TRANSLATION only.

    Mirroring convention: a scale is a magnitude, so left and right should match directly.
    A translation offset should match after negating the y (mediolateral) component,
    because the template's symmetry plane is y = 0.
    """
    if pairs.numel() == 0:
        return log_beta_scales.sum() * 0.0
    r, ll = pairs[:, 0], pairs[:, 1]
    loss = (log_beta_scales[:, r] - log_beta_scales[:, ll]).pow(2).mean()
    if betas_trans is not None:
        bt_r = betas_trans[:, r].clone()
        bt_l = betas_trans[:, ll].clone()
        mirror = torch.tensor([1.0, -1.0, 1.0], device=betas_trans.device)
        loss = loss + (bt_r - bt_l * mirror).pow(2).mean()
    return loss


# --------------------------------------------------------------------------------------
# robust kernels
# --------------------------------------------------------------------------------------
def robust_kernel(sq_dist, kind="l2", scale=1.0):
    """Apply a robust loss to SQUARED distances.

    scale ('c' in the usual parameterisation) sets the distance beyond which a residual is
    treated as an outlier. All kernels are normalised so that they behave like the squared
    distance for residuals much smaller than scale, which keeps the loss magnitude (and
    therefore sensible weight values) comparable to the baseline.

    l2            : plain squared distance (baseline behaviour)
    gm            : Geman-McClure  -- bounded influence, smooth, no hard cutoff
    welsch        : Welsch/Leclerc -- exponentially discounts outliers, strongest rejection
    huber         : quadratic near 0, linear beyond scale -- mildest
    """
    if kind == "l2":
        return sq_dist
    c2 = scale * scale
    if kind == "gm":
        # r^2 * c^2 / (r^2 + c^2)  -> saturates at c^2
        return sq_dist * c2 / (sq_dist + c2)
    if kind == "welsch":
        # c^2 * (1 - exp(-r^2/c^2))
        return c2 * (1.0 - torch.exp(-sq_dist / c2))
    if kind == "huber":
        r = torch.sqrt(sq_dist + 1e-12)
        quad = sq_dist
        lin = c2 + 2.0 * scale * (r - scale)
        return torch.where(r <= scale, quad, lin)
    raise ValueError(f"unknown robust kernel '{kind}'")


def robust_chamfer(
    src_pts,
    tgt_pts,
    src_normals=None,
    tgt_normals=None,
    kernel="l2",
    scale=1.0,
    trim_frac=0.0,
    w_normal_align=0.0,
):
    """Symmetric, area-sampled, optionally robust and trimmed chamfer.

    src_pts, tgt_pts : (B, N, 3) points sampled by AREA from each mesh.
    trim_frac        : fraction of the WORST residuals to discard entirely in each
                       direction (trimmed least squares). This is the cheapest effective
                       defence against scan debris: a fixed fraction of the target may be
                       junk and is simply not explained.
    w_normal_align   : if > 0, add a point-to-plane style term penalising normal
                       disagreement along the established correspondence.

    Returns (loss, info_dict).
    """
    fwd = knn_points(src_pts, tgt_pts, K=1)
    bwd = knn_points(tgt_pts, src_pts, K=1)
    d_fwd = fwd.dists[..., 0]  # squared distances (B,N)
    d_bwd = bwd.dists[..., 0]

    def reduce(d):
        r = robust_kernel(d, kernel, scale)
        if trim_frac > 0.0:
            k = max(1, int(round(r.shape[1] * (1.0 - trim_frac))))
            r, _ = torch.sort(r, dim=1)
            r = r[:, :k]
        return r.mean()

    loss = reduce(d_fwd) + reduce(d_bwd)
    info = {
        "chamfer_raw": float((d_fwd.mean() + d_bwd.mean()).item()),
        "frac_beyond_scale": float((d_fwd.sqrt() > scale).float().mean().item()),
    }

    # The normal-alignment term is returned SEPARATELY rather than folded into the chamfer
    # value. Folding it in makes the logged 'chamfer' incomparable across stages that use
    # different w_normal_align, which silently masks whether the robust kernel is working
    # (a saturating kernel at scale c can never exceed 2*c^2, so a conflated number that
    # exceeds that bound looks like a bug when it is not).
    align_loss = None
    if w_normal_align > 0.0 and src_normals is not None and tgt_normals is not None:
        nn_t = torch.gather(tgt_normals, 1, fwd.idx[..., 0:1].expand(-1, -1, 3))
        cos = torch.nn.functional.cosine_similarity(src_normals, nn_t, dim=-1)
        # 1 - |cos| : penalise misaligned surface orientation, sign-agnostic because
        # scan normal orientation is not reliable on CT data
        align_loss = (1.0 - cos.abs()).mean()
        info["normal_align_cos"] = float(cos.abs().mean().item())

    return loss, align_loss, info


def rest_edge_loss(pred_verts, rest_verts, faces):
    """Penalise edge-length distortion introduced by FREE-FORM OFFSETS only.

    Replaces pytorch3d.mesh_edge_loss(target_length=0), which penalises edge length itself
    and therefore shrinks the mesh.

    IMPORTANT semantics, corrected after a measurement error. `rest_verts` must be the
    model's own posed-and-shaped geometry with deform_verts zeroed, recomputed at the
    CURRENT parameters -- NOT the original template. Comparing against the original template
    penalises legitimate pose and shape change: per-joint scaling alone reaches ~2.6x, which
    drove this term to 3.53 while chamfer sat at 0.005, i.e. it outweighed the data term by
    ~35x and silently turned the run into "stay near the template".

    With the corrected reference an articulated, shaped, but undeformed mesh scores exactly
    zero, so the term measures only what free-form deformation did to the surface -- which
    is the thing that actually destroys correspondence.

    Uses the relative (log-free) form ((l/l0) - 1)^2 so that thin structures -- legs,
    antennae, which have short edges -- are not under-weighted relative to the body.
    """
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    l0 = (rest_verts[:, e[:, 0]] - rest_verts[:, e[:, 1]]).norm(dim=-1)
    l1 = (pred_verts[:, e[:, 0]] - pred_verts[:, e[:, 1]]).norm(dim=-1)
    return ((l1 / (l0 + 1e-9)) - 1.0).pow(2).mean()


# --------------------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------------------
class MoonshotStage:
    """One stage of optimisation, with the fixes documented at the top of this file."""

    def __init__(
        self,
        nits,
        scheme,
        smal_3d_fitter,
        target_meshes,
        mesh_names=(),
        name="optimise",
        loss_weights=None,
        lr=1e-3,
        out_dir="moonshot_output",
        custom_lrs=None,
        device="cuda",
        plot_normals=False,
        n_sample=6000,
        robust_kernel="l2",
        robust_scale=0.05,
        robust_scale_end=None,
        trim_frac=0.0,
        lr_decay=True,
        log_every=50,
        edge_mode="rest",
        n_sample_tgt=None,
        corr_mode="chamfer",
        corr_tau=0.02,
        corr_ratio=0.9,
        corr_blur=0.05,
        corr_blur_end=None,
    ):
        self.n_it = nits
        self.name = name
        self.out_dir = out_dir
        self.target_meshes = target_meshes
        self.mesh_names = list(mesh_names)
        self.smal_3d_fitter = smal_3d_fitter
        self.device = device
        self.plot_normals = plot_normals
        self.n_sample = n_sample
        self.robust_kernel_kind = robust_kernel
        self.robust_scale = robust_scale
        # graduated non-convexity: optionally shrink the robust scale over the stage, so
        # the loss starts near-convex (wide basin) and progressively rejects outliers.
        self.robust_scale_end = robust_scale if robust_scale_end is None else robust_scale_end
        self.trim_frac = trim_frac
        self.lr_decay = lr_decay
        self.log_every = log_every
        # 'rest' = deviation from template rest edge lengths (the fix).
        # 'shrink' = pytorch3d mesh_edge_loss(target_length=0), i.e. baseline behaviour,
        # retained so a control arm can reproduce the stock pipeline inside this harness.
        self.edge_mode = edge_mode
        # correspondence operator for the data term. See fitter_3d/correspondence.py --
        # motivated by arm E1, which showed the failure is a multimodal ASSIGNMENT problem,
        # not an optimizer step-size problem.
        self.corr_mode = corr_mode
        self.corr_tau = corr_tau
        self.corr_ratio = corr_ratio
        self.corr_blur = corr_blur
        self.corr_blur_end = corr_blur if corr_blur_end is None else corr_blur_end
        # allow an asymmetric point budget so the control can mimic the baseline's
        # 3000-target-samples-vs-all-source-verts sampling if needed
        self.n_sample_tgt = n_sample_tgt or n_sample

        self.loss_weights = default_weights.copy()
        if loss_weights:
            self.loss_weights.update(loss_weights)

        if custom_lrs:
            for attr in custom_lrs:
                assert hasattr(smal_3d_fitter, attr), f"attr '{attr}' not in SMAL."

        self.param_group = SMALParamGroup(smal_3d_fitter, scheme, custom_lrs)
        self.optimizer = torch.optim.Adam(self.param_group, lr=lr)
        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(nits, 1)) if lr_decay else None
        )

        with torch.no_grad():
            self.src_verts = smal_3d_fitter().detach()
        self.faces = smal_3d_fitter.faces.detach()
        self.src_mesh = get_meshes(self.src_verts, self.faces, device=device)
        self.n_verts = self.src_verts.shape[1]

        # rest geometry for the stiffness term: the template BEFORE any deformation,
        # i.e. with deform_verts zeroed. Captured once, never updated.
        with torch.no_grad():
            zero_dv = torch.zeros_like(smal_3d_fitter.deform_verts)
            self.rest_verts = smal_3d_fitter(deform_verts=zero_dv).detach()

        # left/right joint twins, for the symmetry penalty. Read from the model file so
        # this stays correct if the template changes.
        self.min_limits, self.max_limits = joint_limit_tensors(config.dd, device)

        self.sym_verts = None
        if self.loss_weights.get("w_midline", 0.0) > 0:
            import pickle as _p

            with open(config.SMAL_FILE, "rb") as _f:
                _u = _p._Unpickler(_f)
                _u.encoding = "latin1"
                self.sym_verts = torch.tensor(np.asarray(_u.load()["sym_verts"]).astype(np.int64), device=device)

        self.lr_pairs = None
        if self.loss_weights.get("w_sym", 0.0) > 0:
            import pickle as _pkl

            with open(config.SMAL_FILE, "rb") as _f:
                _u = _pkl._Unpickler(_f)
                _u.encoding = "latin1"
                _jn = list(_u.load()["J_names"])
            self.lr_pairs = build_lr_joint_pairs(_jn).to(device)

        self.losses_to_plot = []
        self.loss_components_to_plot = {}

        # pre-sample the target once per stage is NOT done on purpose: resampling every
        # iteration acts as a stochastic regulariser and avoids locking onto one sample.

    def _current_scale(self, it):
        if self.n_it <= 1:
            return self.robust_scale
        t = it / (self.n_it - 1)
        # geometric interpolation -- scale is a length, so interpolate multiplicatively
        return float(self.robust_scale * (self.robust_scale_end / self.robust_scale) ** t)

    def forward(self, src_mesh, it=0):
        lw = self.loss_weights
        loss = 0.0
        comp = {}

        need_normals = lw.get("w_normal_align", 0.0) > 0.0
        if need_normals:
            src_pts, src_nrm = sample_points_from_meshes(src_mesh, self.n_sample, return_normals=True)
            tgt_pts, tgt_nrm = sample_points_from_meshes(self.target_meshes, self.n_sample, return_normals=True)
        else:
            src_pts = sample_points_from_meshes(src_mesh, self.n_sample)
            tgt_pts = sample_points_from_meshes(self.target_meshes, self.n_sample)
            src_nrm = tgt_nrm = None

        if lw["w_chamfer"] > 0:
            if self.corr_mode == "chamfer":
                l_ch, l_align, info = robust_chamfer(
                    src_pts,
                    tgt_pts,
                    src_nrm,
                    tgt_nrm,
                    kernel=self.robust_kernel_kind,
                    scale=self._current_scale(it),
                    trim_frac=self.trim_frac,
                    w_normal_align=lw.get("w_normal_align", 0.0),
                )
            else:
                from fitter_3d import correspondence as CORR

                l_align = None
                if self.corr_mode == "mutual":
                    l_ch, info = CORR.mutual_nn_loss(src_pts, tgt_pts, ratio_thresh=self.corr_ratio)
                elif self.corr_mode == "softnn":
                    l_ch, info = CORR.soft_nn_loss(src_pts, tgt_pts, tau=self.corr_tau)
                elif self.corr_mode == "sinkhorn":
                    t = it / max(self.n_it - 1, 1)
                    blur = float(self.corr_blur * (self.corr_blur_end / self.corr_blur) ** t)
                    l_ch, info = CORR.sinkhorn_loss(src_pts, tgt_pts, blur=blur)
                else:
                    raise ValueError(f"unknown corr_mode {self.corr_mode}")
            comp["chamfer"] = l_ch
            loss = loss + lw["w_chamfer"] * l_ch
            self.last_corr_info = info
            if l_align is not None:
                comp["nrm_align"] = l_align
                loss = loss + lw["w_normal_align"] * l_align

        if lw["w_edge"] > 0:
            if self.edge_mode == "shrink":
                # Baseline behaviour, kept ONLY so a control arm can reproduce the stock
                # pipeline exactly inside this harness. target_length=0 means this
                # penalises edge length itself, i.e. it shrinks the mesh.
                from pytorch3d.loss import mesh_edge_loss

                l_e = mesh_edge_loss(src_mesh)
            else:
                # reference = the CURRENT pose/shape with free-form offsets removed, so
                # legitimate articulation and shape change cost nothing (see rest_edge_loss)
                with torch.no_grad():
                    rest_now = self.smal_3d_fitter(
                        deform_verts=torch.zeros_like(self.smal_3d_fitter.deform_verts)
                    ).detach()
                l_e = rest_edge_loss(src_mesh.verts_padded(), rest_now, self.faces[0])
            comp["edge"] = l_e
            loss = loss + lw["w_edge"] * l_e

        if lw["w_normal"] > 0:
            l_n = mesh_normal_consistency(src_mesh)
            comp["normal"] = l_n
            loss = loss + lw["w_normal"] * l_n

        if lw["w_laplacian"] > 0:
            l_l = mesh_laplacian_smoothing(src_mesh, method="uniform")
            comp["laplacian"] = l_l
            loss = loss + lw["w_laplacian"] * l_l

        if lw.get("w_offset", 0.0) > 0:
            l_o = self.smal_3d_fitter.deform_verts.pow(2).sum(-1).mean()
            comp["offset"] = l_o
            loss = loss + lw["w_offset"] * l_o

        if lw.get("w_beta_prior", 0.0) > 0:
            f = self.smal_3d_fitter
            d = f.betas - f.mean_betas.unsqueeze(0)
            # betas_prec is the Cholesky factor of the inverse covariance, so this is the
            # squared Mahalanobis distance of betas under the model's own shape prior.
            l_b = (d @ f.betas_prec).pow(2).sum(-1).mean()
            comp["beta_prior"] = l_b
            loss = loss + lw["w_beta_prior"] * l_b

        if lw.get("w_sym", 0.0) > 0 and self.lr_pairs is not None:
            f = self.smal_3d_fitter
            l_s = symmetry_penalty(f.log_beta_scales, f.betas_trans, self.lr_pairs)
            comp["sym"] = l_s
            loss = loss + lw["w_sym"] * l_s

        if lw.get("w_midline", 0.0) > 0 and self.sym_verts is not None:
            l_m = midline_penalty(src_mesh.verts_padded(), self.sym_verts)
            comp["mid"] = l_m
            loss = loss + lw["w_midline"] * l_m

        if lw.get("w_scale", 0.0) > 0:
            l_sc = scale_barrier(self.smal_3d_fitter.log_beta_scales)
            comp["scale"] = l_sc
            loss = loss + lw["w_scale"] * l_sc

        if lw.get("w_trans", 0.0) > 0:
            l_tr = trans_barrier(self.smal_3d_fitter.betas_trans)
            comp["trans"] = l_tr
            loss = loss + lw["w_trans"] * l_tr

        if lw.get("w_limit", 0.0) > 0:
            # The handoff stages run 1000 iterations each with scheme 'all', so joint_rot keeps
            # moving here -- which is where §6.9's out-of-range distal poses are actually
            # acquired. Applying the limit only in the hierarchical stages would leave the
            # longest pose-mobile phase of the schedule unconstrained.
            if self.min_limits is None:
                raise ValueError(
                    "w_limit > 0 but the model file authors no 'joint_limits'. "
                    "Point config.SMAL_FILE at a model that does (e.g. OmniAnt_25PCs_joint_limited.pkl)."
                )
            l_lim = limit_hinge(self.smal_3d_fitter.joint_rot, self.min_limits, self.max_limits)
            comp["limit"] = l_lim
            loss = loss + lw["w_limit"] * l_lim

        return loss, comp

    def step(self, it):
        new_verts = self.smal_3d_fitter()
        offsets = new_verts - self.src_verts
        new_mesh = self.src_mesh.offset_verts(offsets.view(-1, 3))
        loss, comp = self.forward(new_mesh, it)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss, comp

    def run(self, plot=False, quiet=True):
        it_range = range(self.n_it)
        bar = None
        if not quiet:
            bar = tqdm(it_range)
            it_range = bar
        for i in it_range:
            self.optimizer.zero_grad()
            loss, comp = self.step(i)
            self.losses_to_plot.append(float(loss.detach()))
            for k, v in comp.items():
                self.loss_components_to_plot.setdefault(k, []).append(float(v.detach()))
            if bar is not None:
                bar.set_description(f"{self.name} loss={float(loss):.6f}")
            elif self.log_every and (i % self.log_every == 0 or i == self.n_it - 1):
                parts = "  ".join(f"{k}={float(v):.5f}" for k, v in comp.items())
                print(f"    [{self.name}] it {i:4d}/{self.n_it}  loss={float(loss):.6f}   {parts}", flush=True)
        if plot:
            self.plot()

    def plot(self):
        new_verts = self.smal_3d_fitter()
        offsets = new_verts - self.src_verts
        new_mesh = self.src_mesh.offset_verts(offsets.view(-1, 3))
        plot_meshes(
            self.target_meshes,
            new_mesh,
            self.mesh_names,
            title=self.name,
            figtitle=f"{self.name}, its = {self.n_it}",
            out_dir=os.path.join(self.out_dir, "meshes"),
            plot_normals=self.plot_normals,
        )

    def save_npz(self, labels=None):
        out = {}
        for p in ["global_rot", "joint_rot", "betas", "log_beta_scales", "trans", "deform_verts", "betas_trans"]:
            out[p] = getattr(self.smal_3d_fitter, p).cpu().detach().numpy()
        out["verts"] = self.smal_3d_fitter().cpu().detach().numpy()
        out["faces"] = self.faces.cpu().detach().numpy()
        out["labels"] = labels
        np.savez(os.path.join(self.out_dir, f"{self.name}.npz"), **out)


class MoonshotManager:
    def __init__(self, out_dir="moonshot_output", labels=None):
        self.stages = []
        self.out_dir = out_dir
        self.labels = labels

    def add_stage(self, s):
        self.stages.append(s)

    def run(self, save_every_stage=True):
        for stage in self.stages:
            stage.run(plot=False)
            if save_every_stage:
                stage.save_npz(labels=self.labels)

    def plot_losses(self, out_src="losses"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        start = 0
        for s in self.stages:
            n = len(s.losses_to_plot)
            ax.semilogy(np.arange(start, start + n), s.losses_to_plot, label=s.name)
            start += n
        ax.set_xlabel("iteration")
        ax.set_ylabel("total loss")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        fig.savefig(os.path.join(self.out_dir, out_src + ".png"), dpi=110)
        plt.close(fig)
