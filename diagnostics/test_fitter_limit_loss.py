"""
§5 fitter limit-loss behaviour test (issue #56).

Reproduces the optimisation fitter's limit loss exactly (see
smal_fitter/fitter.py: `self.max_limits`/`self.min_limits` built from
LimitPrior, and the `objs["limit"]` hinge) against the CURRENT model in
config.dd, and checks the three properties §5 cares about:

  1. an in-range pose  -> limit loss == 0
  2. an out-of-range pose -> limit loss > 0
  3. the gradient at the violated entry points back toward the allowed range
     (i.e. gradient descent would pull the joint back inside the limit)

This captures the same guarantee as running the full optimiser, without needing
images / silhouettes / a dataset.

To test against your Blender-exported model (with a real authored limit), set
`SMAL_FILE` in config.py to that .pkl and make sure it's reachable from where
you run. Otherwise it runs against whatever config.py currently points at (a
model with no `joint_limits` is wide-open, so it violates past +/-pi instead).

Run:  python -m diagnostics.test_fitter_limit_loss
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import config
from smal_fitter.priors.joint_limits_prior import LimitPrior

# Build the same (N_POSE, 3) limit tensors the fitter uses (root dropped via [3:]).
lp = LimitPrior()
n = config.N_POSE
max_limits = torch.tensor(np.asarray(lp.max_values[3:], dtype=np.float32).reshape(n, 3))
min_limits = torch.tensor(np.asarray(lp.min_values[3:], dtype=np.float32).reshape(n, 3))
jn = config.dd["J_names"]


def limit_loss(joint_rotations, w_limit=1.0):
    """The fitter's limit loss: flat inside the range, linear past it."""
    zeros = torch.zeros_like(joint_rotations)
    return w_limit * torch.mean(
        torch.max(joint_rotations - max_limits, zeros) + torch.max(min_limits - joint_rotations, zeros)
    )


def main():
    print(f"model: {config.SMAL_FILE}")
    print(f"N_POSE={n}, has authored joint_limits={'joint_limits' in config.dd}")

    # 1. In-range pose (all-zero rotations sit inside every range that brackets 0).
    jr_ok = torch.zeros(n, 3)
    loss_ok = float(limit_loss(jr_ok))
    print(f"\nin-range pose      -> limit loss = {loss_ok:.6f}")

    # 2/3. Violate the tightest (smallest-max) joint/axis by +0.5 rad.
    maxv = max_limits.numpy()
    j, a = map(int, np.unravel_index(np.argmin(maxv), maxv.shape))
    axis = "xyz"[a]
    print(f"tightest max: joint {j} ({jn[j + 1]}) axis {axis}, max={maxv[j, a]:.4f} rad")

    jr_bad = torch.zeros(n, 3, requires_grad=True)
    with torch.no_grad():
        jr_bad[j, a] = max_limits[j, a] + 0.5
    loss = limit_loss(jr_bad)
    loss.backward()
    g = float(jr_bad.grad[j, a])
    print(f"out-of-range pose  -> limit loss = {float(loss):.6f}")
    print(f"gradient at violated entry = {g:+.6f}  (positive => descent lowers the angle back toward the limit)")

    assert loss_ok == 0.0, "in-range pose should give zero limit loss"
    assert float(loss) > 0.0, "out-of-range pose should give positive limit loss"
    assert g > 0.0, "gradient should push the violating angle back down toward the limit"
    print("\nPASS: in-range = 0, violation > 0, gradient pulls back toward the range.")

    print(loss)


if __name__ == "__main__":
    main()
