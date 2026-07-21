"""
Automated checks for issue #56 (user-defined joint limits) — consumer side.

Covers the parts that do NOT need Blender: the LimitPrior read path, the
wide-open fallback, validation, and the reshape the fitter performs. The Blender
export helper is tested manually (see docs/design/issue56_test_plan.md).

Run from the repo root with the pytorch3d env active:

    python -m diagnostics.test_issue56        # standalone, prints PASS/FAIL
    pytest diagnostics/test_issue56.py        # also works under pytest
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import config


def _fresh_limit_prior(joint_limits=None):
    """Instantiate a LimitPrior with an optional injected joint_limits on config.dd.

    Restores config.dd afterwards so tests don't leak state into each other.
    """
    from smal_fitter.priors.joint_limits_prior import LimitPrior

    had = "joint_limits" in config.dd
    prev = config.dd.get("joint_limits", None)
    try:
        if joint_limits is None:
            config.dd.pop("joint_limits", None)
        else:
            config.dd["joint_limits"] = joint_limits
        return LimitPrior()
    finally:
        if had:
            config.dd["joint_limits"] = prev
        else:
            config.dd.pop("joint_limits", None)


def test_fallback_is_wide_open():
    """No joint_limits -> every non-root joint is wide-open [-pi, pi], root is 0."""
    lp = _fresh_limit_prior(None)
    pairs = set(zip(np.round(lp.min_values, 3), np.round(lp.max_values, 3)))
    assert pairs == {(0.0, 0.0), (round(-np.pi, 3), round(np.pi, 3))}, pairs
    # The old placeholder value must be gone.
    assert (round(-0.01, 3), round(0.01, 3)) not in pairs


def test_authored_limits_flow_to_correct_joint():
    """An injected joint_limits reaches the right (N_POSE, 3) slot the fitter uses."""
    n_joints = len(config.dd["J_names"])
    jl = np.zeros((n_joints, 3, 2), dtype=np.float32)
    jl[..., 0] = -0.5
    jl[..., 1] = 0.5
    jl[0] = 0.0  # root
    jl[8] = [[-0.1, 0.1], [-1.2, 0.3], [-0.05, 0.05]]  # some non-root joint

    lp = _fresh_limit_prior(jl)
    n_pose = config.N_POSE
    max_l = np.asarray(lp.max_values[3:]).reshape(n_pose, 3)
    min_l = np.asarray(lp.min_values[3:]).reshape(n_pose, 3)

    # J index 8 -> pose index 7 (root dropped).
    assert np.allclose(min_l[7], [-0.1, -1.2, -0.05], atol=1e-6), min_l[7]
    assert np.allclose(max_l[7], [0.1, 0.3, 0.05], atol=1e-6), max_l[7]
    # A generic joint keeps the +/-0.5 we set.
    assert np.allclose(min_l[0], [-0.5, -0.5, -0.5], atol=1e-6)


def test_hinge_penalty_matches_fitter_formula():
    """LimitPrior.__call__ = flat inside, linear past the limit (fitter's formula)."""
    lp = _fresh_limit_prior(None)  # wide-open
    x = np.zeros_like(lp.max_values)
    assert np.allclose(lp(x, np), 0.0)  # zero cost inside the range
    # Push one axis past the wide-open max -> positive cost.
    x2 = lp.max_values + 0.1
    assert np.all(lp(x2, np) > 0)


def test_bad_shape_raises():
    from smal_fitter.priors.joint_limits_prior import _ranges_from_joint_limits

    dd = dict(config.dd)
    dd["joint_limits"] = np.zeros((3, 3, 2))  # wrong J
    try:
        _ranges_from_joint_limits(dd)
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong-shaped joint_limits")


def test_min_greater_than_max_raises():
    from smal_fitter.priors.joint_limits_prior import _ranges_from_joint_limits

    n = len(config.dd["J_names"])
    jl = np.zeros((n, 3, 2))
    jl[..., 0] = 1.0  # min
    jl[..., 1] = -1.0  # max < min
    dd = dict(config.dd)
    dd["joint_limits"] = jl
    try:
        _ranges_from_joint_limits(dd)
    except ValueError:
        return
    raise AssertionError("expected ValueError for min > max")


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
