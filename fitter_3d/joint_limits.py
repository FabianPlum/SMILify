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
