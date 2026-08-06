"""Per-joint rotation limits as a hinge prior, for the moonshot/hierarchical trainers.

`smal_fitter/priors/joint_limits_prior.py` already turns a model's authored `joint_limits`
into per-axis ranges, and `smal_fitter/fitter.py` already consumes them for the 2D path.
PR #98 wires the same term into `fitter_3d/trainer.py`. Neither reaches
`trainer_hierarchical.py` or `trainer_moonshot.py`, which carry their own `loss_weights`
dicts and their own `forward()`. This module supplies the tensors for those two.

WHY THIS MATTERS HERE, measured (REPORT §4, and the scan below):
The distal leg segments hold 2.4% of a leg chain's surface area, so at n_sample=8000 the
pretarsus draws ~0.3 target samples. They are effectively unconstrained by the data term, and
the arms that unfreeze pose exploit exactly that: M7 and BPX_noprior drive 37 of 96 constrained
axes out of range on 100% of specimens, median 19.4°, p95 77.5°, concentrated on `l_3_ta_*`
and `l_3_ti_*`. The stock baseline violates 3.8 axes only because §1's freeze stops it moving
pose at all. So this prior constrains precisely the joints the data cannot.

Interpretation caveat inherited from the prior module: bounds are per-axis in AXIS-ANGLE space,
which is non-unique past |theta| = pi. Author within [-pi, +pi]; exactly +/-pi means free.
"""

import numpy as np
import torch

import config

from smal_fitter.priors.joint_limits_prior import _ranges_from_joint_limits


def joint_limit_tensors(dd, device):
    """(min_limits, max_limits) of shape (N_POSE, 3) from a loaded SMAL dict, root dropped.

    Returns (None, None) when the model authors no `joint_limits`, so callers can treat
    "no limits" and "limits off" identically rather than silently applying a +/-pi no-op.
    Malformed limits raise here: unlike the fitter_3d path, nothing constructs these trainers
    unconditionally, so an invalid model file should fail loudly at setup.
    """
    if dd.get("joint_limits", None) is None:
        return None, None
    ranges = _ranges_from_joint_limits(dd)
    names = dd["J_names"]
    lo = np.array([ranges[j][ax][0] for j in names for ax in range(3)])
    hi = np.array([ranges[j][ax][1] for j in names for ax in range(3)])
    n_pose = len(names) - 1
    # the root's three values come first and are dropped, matching joint_rot's (N_POSE, 3)
    return (
        torch.as_tensor(lo[3:], dtype=torch.float32, device=device).view(n_pose, 3),
        torch.as_tensor(hi[3:], dtype=torch.float32, device=device).view(n_pose, 3),
    )


def limit_hinge(joint_rot, min_limits, max_limits):
    """Flat inside the box, linear outside. Mean over every (N_POSE, 3) entry, matching
    `smal_fitter/fitter.py` so a weight tuned in one path means the same in the other."""
    zeros = torch.zeros_like(joint_rot)
    return torch.mean(torch.max(joint_rot - max_limits, zeros) + torch.max(min_limits - joint_rot, zeros))


def violation_report(joint_rot, min_limits, max_limits, joint_names):
    """Per-joint violation counts and overshoot, for diagnostics and figures.

    joint_rot: (B, N_POSE, 3) tensor or array. Returns a dict of numpy arrays.
    """
    jr = joint_rot.detach().cpu().numpy() if torch.is_tensor(joint_rot) else np.asarray(joint_rot)
    lo = min_limits.detach().cpu().numpy() if torch.is_tensor(min_limits) else np.asarray(min_limits)
    hi = max_limits.detach().cpu().numpy() if torch.is_tensor(max_limits) else np.asarray(max_limits)
    over = np.maximum(jr - hi, 0.0) + np.maximum(lo - jr, 0.0)
    viol = over > 0
    wide = np.isclose(np.abs(lo), np.pi) & np.isclose(np.abs(hi), np.pi)
    return dict(
        over=over,
        viol=viol,
        per_specimen=viol.sum((1, 2)),
        per_joint=viol.sum((0, 2)),
        constrained=(~wide),
        joint_names=list(joint_names[1:]),
    )


def scale_barrier(log_beta_scales, free_log=0.6931471805599453, power=2.0):
    """Penalise per-joint scale beyond a free band. Zero inside, (|ln s| - free_log)^power outside.

    WHY THIS EXISTS (REPORT §6.10, and the domain reading of the renders): nothing in this
    pipeline bounds per-joint scale. `joint_limits` constrains rotation only; the sole terms
    touching `log_beta_scales` anywhere are the L/R symmetry tie and the opt-in `w_jresid` L2,
    neither of which is a bound. Measured across 50 specimens in the stock baseline:

        head joint     0.033 - 4.612   (138x range)
        mandible       0.030 - 11.801  (389x)
        antenna        0.024 - 21.227  (889x)

    A joint scaled to 0.024 is a part collapsing into its parent; 21x is a part exploding to
    cover geometry it does not belong to. That is literally "the head shrinks into the thorax
    while the mandibles and antennae explode to form the head".

    The band is set from domain judgement, not fitted: 2x is unremarkable biological variation,
    4x is concerning, 8x is improbable. So `free_log = ln 2` costs nothing up to 2x, and the
    quadratic then gives 0.48 at 4x and 1.92 at 8x -- a 4:1 ratio across the range the expert
    called "concerning" to "improbable". Raise `power` to 4 for a 16:1 escalation if the
    quadratic proves too permissive at the tail.

    Note this is deliberately a SOFT penalty rather than a hard clamp: a clamp would put a
    discontinuity in the gradient exactly where the difficult specimens live, and §5.1 already
    showed that hard rejection of awkward geometry preferentially destroys thin structures.

    Applied to the RAW per-joint value. Scales also accumulate down the kinematic chain when
    `propagate_scaling` is on, so a chain of mild scalings can still compound -- but the
    measured failures above are single joints already far outside the band, which this reaches.
    """
    excess = (log_beta_scales.abs() - free_log).clamp_min(0.0)
    return (excess**power).mean()


def trans_barrier(betas_trans, free_abs=0.02):
    """Penalise per-joint translation beyond a free band, in model units. Companion to
    `scale_barrier`, with its OWN weight so translation can be allowed more freely than scale.

    Scale and translation need separate control because they are different quantities with
    different anatomy. Scale is multiplicative and its plausible range is a ratio (2x fine,
    8x improbable), so `scale_barrier` bands it in log space. Translation is additive and its
    plausible range is a fraction of body length, so it is banded in absolute model units.
    Lumping them together -- as the pre-existing `w_jresid` does, a single L2 over
    log_beta_scales.pow(2) + betas_trans.pow(2) -- makes "allow the parts to shift a little
    while holding their size" inexpressible.

    Default band, derived rather than guessed. The model's own shape space moves joints by
    (OmniAnt_25PCs, at the fitted betas): rms 0.00957, p95 0.0219, max 0.0775 model units,
    against a template extent of 1.79. So the entangled shape space itself uses up to ~4.3% of
    body length of joint translation. A FREE residual should be allowed less than the shape
    space uses, since the point is for betas to carry the bulk: 0.02 is ~1.1% of extent, below
    the shape space's p95.

    For reference, the currently unregularised free parameter reaches max 0.973 -- a single
    joint displaced by more than half the specimen extent.
    """
    return (betas_trans.abs() - free_abs).clamp_min(0.0).pow(2).mean()


def composed_log_scale(fitter):
    """Total per-joint log-scale actually applied: shape-space contribution + free residual.

    The final per-part scale is the PRODUCT of two things, and both are always present:
      * what the betas drive through `scaledirs` -- the shape space -- active only when
        config.COUPLE_JOINT_BLENDSHAPES is on;
      * the free per-specimen `log_beta_scales`, available regardless, so individual parts
        can still be nudged where the shape space cannot reach.
    A product of scales is a SUM of log-scales, which is exactly how smal_torch.py composes
    them before skinning.

    The barrier must judge that product. Otherwise a 1.8x shape-space scale multiplied by a
    1.8x free scale -- 3.24x in total, well past the band -- reads as compliant, because each
    factor on its own is inside 2x.
    """
    free = fitter.log_beta_scales
    if not getattr(config, "COUPLE_JOINT_BLENDSHAPES", False):
        return free
    sd = getattr(fitter.smal_model, "scaledirs", None)
    if sd is None:
        return free
    nb = fitter.betas.shape[1]
    driven = torch.einsum("bk,kjc->bjc", fitter.betas, sd[:nb])
    return free + torch.log(torch.clamp(1.0 + driven, min=config.COUPLE_MIN_SCALE))


def composed_trans(fitter):
    """Total per-joint translation actually applied: shape-space contribution + free residual.

    Translation composes ADDITIVELY, unlike scale which composes multiplicatively and is
    therefore summed in log space above. Matches smal_torch.py. Judged by its own barrier
    with its own weight, because a length and a ratio are not interchangeable.
    """
    free = fitter.betas_trans
    if not getattr(config, "COUPLE_JOINT_BLENDSHAPES", False):
        return free
    td = getattr(fitter.smal_model, "transdirs", None)
    if td is None:
        return free
    nb = fitter.betas.shape[1]
    return free + torch.einsum("bk,kjc->bjc", fitter.betas, td[:nb]) * config.COUPLE_TRANSLATION_FACTOR
