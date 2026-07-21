"""
§7 neural joint-limit penalty test (issue #56).

The neural regressors gained an optional, OFF-BY-DEFAULT `joint_limit_regularization`
penalty (in multiview_smil_regressor.py and smil_image_regressor.py). Fully training
a regressor needs a dataset + backbone weights (downloads, GPU), so instead this
exercises the penalty's real ingredients:

  - the actual 6D -> axis-angle conversion the regressors use,
  - the real per-joint limits from LimitPrior (read from config.dd),
  - the exact hinge the regressor loss computes,

and verifies the default weight is 0.0 in both regressors (so existing training is
bit-for-bit unchanged unless you opt in).

Run:  python -m diagnostics.test_neural_limit_penalty
"""

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import config
from smal_fitter.priors.joint_limits_prior import LimitPrior
from smal_fitter.neuralSMIL.smil_image_regressor import rotation_6d_to_axis_angle
from smal_fitter.neuralSMIL import smil_image_regressor, multiview_smil_regressor

# Real per-joint limits, built exactly as the regressors build them.
lp = LimitPrior()
N = config.N_POSE
min_lim = torch.tensor(np.asarray(lp.min_values[3:], dtype=np.float32).reshape(N, 3))
max_lim = torch.tensor(np.asarray(lp.max_values[3:], dtype=np.float32).reshape(N, 3))


def neural_limit_penalty(joint_rot_pred, representation):
    """The exact hinge the regressor loss runs (uses the real 6D conversion)."""
    if representation == "6d":
        joint_rot_aa = rotation_6d_to_axis_angle(joint_rot_pred)
    else:
        joint_rot_aa = joint_rot_pred
    zeros = torch.zeros_like(joint_rot_aa)
    return torch.mean(torch.maximum(joint_rot_aa - max_lim, zeros) + torch.maximum(min_lim - joint_rot_aa, zeros))


def test_axis_angle_batch_zero_and_violation():
    B = 4
    # In-range: a batch of all-zero rotations sits inside every bracketing range.
    jr_ok = torch.zeros(B, N, 3)
    assert float(neural_limit_penalty(jr_ok, "axis_angle")) == 0.0

    # Violate the tightest joint/axis; check positive loss + corrective gradient.
    maxv = max_lim.numpy()
    j, a = map(int, np.unravel_index(np.argmin(maxv), maxv.shape))
    jr_bad = torch.zeros(B, N, 3, requires_grad=True)
    with torch.no_grad():
        jr_bad[:, j, a] = max_lim[j, a] + 0.5
    loss = neural_limit_penalty(jr_bad, "axis_angle")
    loss.backward()
    assert float(loss) > 0.0
    assert float(jr_bad.grad[0, j, a]) > 0.0
    print(f"  axis_angle: violated {config.dd['J_names'][j + 1]} axis {'xyz'[a]} -> loss {float(loss):.6f}")


def test_6d_representation_path_runs():
    # Identity 6D (first two rotation-matrix columns) -> zero rotation -> in range.
    B = 2
    six = torch.zeros(B, N, 6)
    six[..., 0] = 1.0
    six[..., 4] = 1.0
    loss = float(neural_limit_penalty(six, "6d"))
    assert loss >= 0.0  # exercises the real rotation_6d_to_axis_angle without error
    print(f"  6d identity pose -> loss {loss:.6f}")


def test_penalty_is_off_by_default():
    # Both regressors must default the weight to 0.0 so existing training is unchanged.
    for mod in (smil_image_regressor, multiview_smil_regressor):
        src = inspect.getsource(mod)
        assert '"joint_limit_regularization": 0.0' in src, f"{mod.__name__} does not default the weight to 0.0"
    print("  default weight 0.0 confirmed in both regressors")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
