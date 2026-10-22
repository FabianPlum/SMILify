"""Tests for issue #56 (user-defined joint limits) — consumer side.

Covers the parts that do NOT need Blender:

- ``LimitPrior`` read path: authored ``joint_limits`` from the model .pkl reach
  the right ``(N_POSE, 3)`` slots, the old ±0.01 rad freeze is gone, and the
  wide-open fallback applies when no limits are present.
- Validation: wrong shape / min > max raise ``ValueError``.
- The fitter's hinge loss (flat inside the range, linear past it): zero
  in-range, positive on violation, gradient pulls back toward the range.
- The neural regressors' ``joint_limit_regularization`` penalty: same hinge
  through the real 6D → axis-angle conversion, verifiably off by default
  (weight 0.0) in both regressors, and fail-fast validation at construction —
  including the wide-open-authored-set guard, which is a provable no-op in 6D.

The Blender-side export helper is covered by ``tests/test_axis_remap.py`` (the
bpy-free remap math) and manually per ``docs/design/issue56_implementation.md``.
"""

import os
import sys
import warnings

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from smal_fitter.priors.joint_limits_prior import (  # noqa: E402
    LimitPrior,
    _ranges_from_joint_limits,
)


def _fresh_limit_prior(joint_limits=None):
    """Instantiate a LimitPrior with an optional injected joint_limits on config.dd.

    Restores config.dd afterwards so tests don't leak state into each other.
    """
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


def _fitter_limit_tensors(lp):
    """The same (N_POSE, 3) limit tensors the fitter builds (root dropped via [3:])."""
    n = config.N_POSE
    max_limits = torch.tensor(np.asarray(lp.max_values[3:], dtype=np.float32).reshape(n, 3))
    min_limits = torch.tensor(np.asarray(lp.min_values[3:], dtype=np.float32).reshape(n, 3))
    return min_limits, max_limits


def _hinge(joint_rotations, min_limits, max_limits):
    """The fitter's limit loss: flat inside the range, linear past it."""
    zeros = torch.zeros_like(joint_rotations)
    return torch.mean(torch.max(joint_rotations - max_limits, zeros) + torch.max(min_limits - joint_rotations, zeros))


# ---------------------------------------------------------------------------
# LimitPrior read path + fallback
# ---------------------------------------------------------------------------


def test_fallback_is_wide_open():
    """No joint_limits -> every non-root joint is wide-open [-pi, pi], root is 0."""
    lp = _fresh_limit_prior(None)
    pairs = set(zip(np.round(lp.min_values, 3), np.round(lp.max_values, 3)))
    assert pairs == {(0.0, 0.0), (round(-np.pi, 3), round(np.pi, 3))}, pairs
    # The old ±0.01 rad placeholder freeze must be gone.
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
    min_l, max_l = _fitter_limit_tensors(lp)

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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_bad_shape_raises():
    dd = dict(config.dd)
    dd["joint_limits"] = np.zeros((3, 3, 2))  # wrong J
    with pytest.raises(ValueError):
        _ranges_from_joint_limits(dd)


def test_min_greater_than_max_raises():
    n = len(config.dd["J_names"])
    jl = np.zeros((n, 3, 2))
    jl[..., 0] = 1.0  # min
    jl[..., 1] = -1.0  # max < min
    dd = dict(config.dd)
    dd["joint_limits"] = jl
    with pytest.raises(ValueError):
        _ranges_from_joint_limits(dd)


# ---------------------------------------------------------------------------
# Fitter limit-loss behaviour (issue-56 §5 guarantees, without images/datasets)
# ---------------------------------------------------------------------------


def test_fitter_limit_loss_zero_in_range():
    min_l, max_l = _fitter_limit_tensors(LimitPrior())
    jr_ok = torch.zeros(config.N_POSE, 3)
    assert float(_hinge(jr_ok, min_l, max_l)) == 0.0


def test_fitter_limit_loss_violation_and_gradient():
    """Out-of-range pose -> positive loss; gradient pulls the joint back inside."""
    min_l, max_l = _fitter_limit_tensors(LimitPrior())
    maxv = max_l.numpy()
    j, a = map(int, np.unravel_index(np.argmin(maxv), maxv.shape))

    jr_bad = torch.zeros(config.N_POSE, 3, requires_grad=True)
    with torch.no_grad():
        jr_bad[j, a] = max_l[j, a] + 0.5
    loss = _hinge(jr_bad, min_l, max_l)
    loss.backward()

    assert float(loss) > 0.0
    # Positive gradient => descent lowers the angle back toward the limit.
    assert float(jr_bad.grad[j, a]) > 0.0


# ---------------------------------------------------------------------------
# Neural penalty (heavy imports: regressor modules) — marked slow
# ---------------------------------------------------------------------------


def test_neural_penalty_zero_in_range_and_violation():
    from smal_fitter.neuralSMIL.smil_image_regressor import rotation_6d_to_axis_angle

    min_l, max_l = _fitter_limit_tensors(LimitPrior())
    B, n = 4, config.N_POSE

    # In-range batch of all-zero rotations.
    assert float(_hinge(torch.zeros(B, n, 3), min_l, max_l)) == 0.0

    # Violate the tightest joint/axis; positive loss + corrective gradient.
    maxv = max_l.numpy()
    j, a = map(int, np.unravel_index(np.argmin(maxv), maxv.shape))
    jr_bad = torch.zeros(B, n, 3, requires_grad=True)
    with torch.no_grad():
        jr_bad[:, j, a] = max_l[j, a] + 0.5
    loss = _hinge(jr_bad, min_l, max_l)
    loss.backward()
    assert float(loss) > 0.0
    assert float(jr_bad.grad[0, j, a]) > 0.0

    # Identity 6D pose exercises the real 6D -> axis-angle conversion path.
    six = torch.zeros(B, n, 6)
    six[..., 0] = 1.0
    six[..., 4] = 1.0
    assert float(_hinge(rotation_6d_to_axis_angle(six), min_l, max_l)) >= 0.0


def test_neural_penalty_real_method_integration():
    """Call the REAL SMILImageRegressor._joint_limit_penalty (issue #56 review).

    The earlier tests exercised a file-local ``_hinge`` replica, so a regression
    inside the method itself (mean -> sum, wrong reshape, broken 6D path) would
    have passed the suite. Here the actual unbound method runs against a stub
    carrying only the attributes it reads — no backbone/renderer construction.
    """
    import types

    from smal_fitter.neuralSMIL.smil_image_regressor import SMILImageRegressor

    min_l, max_l = _fitter_limit_tensors(_fresh_limit_prior(None))  # wide-open
    B, n = 4, config.N_POSE
    stub = types.SimpleNamespace(rotation_representation="axis_angle", _joint_limit_bounds=(min_l, max_l))

    # In-range -> exactly zero.
    assert float(SMILImageRegressor._joint_limit_penalty(stub, torch.zeros(B, n, 3))) == 0.0

    # One joint/axis 0.5 rad past max in every sample: the method returns the
    # MEAN over batch x joints x axes, so the exact value is 0.5 / (3 n).
    # A mean -> sum regression (or a wrong reshape) changes this number.
    jr_bad = torch.zeros(B, n, 3, requires_grad=True)
    with torch.no_grad():
        jr_bad[:, 2, 1] = max_l[2, 1] + 0.5
    penalty = SMILImageRegressor._joint_limit_penalty(stub, jr_bad)
    assert np.isclose(float(penalty), 0.5 / (3 * n), atol=1e-6), float(penalty)
    penalty.backward()
    assert float(jr_bad.grad[0, 2, 1]) > 0.0  # corrective gradient

    # 6D representation goes through the real conversion path: the identity
    # rotation maps to axis-angle zero, which is in-range -> exactly zero.
    stub6 = types.SimpleNamespace(rotation_representation="6d", _joint_limit_bounds=(min_l, max_l))
    six = torch.zeros(B, n, 6)
    six[..., 0] = 1.0
    six[..., 4] = 1.0
    assert float(SMILImageRegressor._joint_limit_penalty(stub6, six)) == 0.0


def test_neural_penalty_fail_fast_validation():
    """Issue #56 review C1: the real _init_joint_limit_bounds raises RuntimeError
    when the penalty is enabled but unusable (missing 'joint_limits' key), rather
    than silently no-op'ing."""
    import types

    from smal_fitter.neuralSMIL.smil_image_regressor import SMILImageRegressor

    had = "joint_limits" in config.dd
    prev = config.dd.get("joint_limits", None)
    try:
        # No authored limits in the .pkl -> wide-open fallback would be a
        # guaranteed no-op, so it must raise.
        config.dd.pop("joint_limits", None)
        with pytest.raises(RuntimeError, match="joint_limits"):
            SMILImageRegressor._init_joint_limit_bounds(types.SimpleNamespace())

        # With an authored set and parent tensors present, bounds are cached.
        n_joints = len(config.dd["J_names"])
        jl = np.zeros((n_joints, 3, 2), dtype=np.float32)
        jl[..., 0], jl[..., 1] = -0.5, 0.5
        config.dd["joint_limits"] = jl
        min_l, max_l = _fitter_limit_tensors(_fresh_limit_prior(jl))
        stub = types.SimpleNamespace(min_limits=min_l, max_limits=max_l)
        SMILImageRegressor._init_joint_limit_bounds(stub)
        assert stub._joint_limit_bounds == (min_l, max_l)
    finally:
        if had:
            config.dd["joint_limits"] = prev
        else:
            config.dd.pop("joint_limits", None)


def test_neural_penalty_rejects_wide_open_authored_limits():
    """Issue #56 review C1 (third sub-point): an authored 'joint_limits' key is
    not proof the penalty can bite. A set left wide open on EVERY joint/axis is a
    guaranteed no-op in 6D (canonical axis-angle keeps |theta| <= pi), so it must
    raise; in axis-angle mode it can still fire past +/-pi, so it only warns."""
    import types

    from smal_fitter.neuralSMIL.smil_image_regressor import SMILImageRegressor

    n_joints = len(config.dd["J_names"])

    # Authored, but every axis at the full +/-180 deg == "free".
    jl_open = np.zeros((n_joints, 3, 2), dtype=np.float32)
    jl_open[..., 0], jl_open[..., 1] = -np.pi, np.pi
    open_min, open_max = _fitter_limit_tensors(_fresh_limit_prior(jl_open))

    # 6D: provably zero for every possible prediction -> hard error.
    with pytest.raises(RuntimeError, match="wide open"):
        SMILImageRegressor._check_bounds_can_bite(open_min, open_max, "6d")

    # Sanity-check the "provably zero" claim the error message rests on.
    six = torch.zeros(2, config.N_POSE, 6)
    six[..., 0], six[..., 4] = 1.0, 1.0
    stub6 = types.SimpleNamespace(rotation_representation="6d", _joint_limit_bounds=(open_min, open_max))
    assert float(SMILImageRegressor._joint_limit_penalty(stub6, six)) == 0.0

    # Axis-angle: reachable past +/-pi, so warn rather than raise...
    with pytest.warns(RuntimeWarning, match="wide open"):
        SMILImageRegressor._check_bounds_can_bite(open_min, open_max, "axis_angle")

    # ...and it really is reachable there (i.e. warning, not error, is right).
    stub_aa = types.SimpleNamespace(rotation_representation="axis_angle", _joint_limit_bounds=(open_min, open_max))
    jr = torch.zeros(2, config.N_POSE, 3)
    jr[:, 1, 0] = np.pi + 0.5
    assert float(SMILImageRegressor._joint_limit_penalty(stub_aa, jr)) > 0.0

    # A partially authored set (one joint constrained, rest free) is the normal
    # case: no error and no warning, in either representation.
    jl_partial = jl_open.copy()
    jl_partial[1, 0, 0], jl_partial[1, 0, 1] = -0.5, 0.5
    part_min, part_max = _fitter_limit_tensors(_fresh_limit_prior(jl_partial))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        SMILImageRegressor._check_bounds_can_bite(part_min, part_max, "6d")
        SMILImageRegressor._check_bounds_can_bite(part_min, part_max, "axis_angle")


def test_prior_module_import_survives_malformed_joint_limits():
    """Issue #56 review C2: importing the prior module must never validate the
    .pkl. A malformed 'joint_limits' used to crash EVERY entry point at import
    time; validation now happens only when LimitPrior is constructed."""
    import importlib

    import smal_fitter.priors.joint_limits_prior as jlp

    had = "joint_limits" in config.dd
    prev = config.dd.get("joint_limits", None)
    try:
        config.dd["joint_limits"] = np.zeros((2, 2))  # garbage shape
        importlib.reload(jlp)  # would have raised ValueError pre-fix
        with pytest.raises(ValueError):
            jlp.LimitPrior()  # validation still happens at construction
    finally:
        if had:
            config.dd["joint_limits"] = prev
        else:
            config.dd.pop("joint_limits", None)
        importlib.reload(jlp)


def test_neural_penalty_off_by_default():
    """Both regressors default the weight to 0.0 so existing training is unchanged."""
    import inspect

    from smal_fitter.neuralSMIL import multiview_smil_regressor, smil_image_regressor

    for mod in (smil_image_regressor, multiview_smil_regressor):
        src = inspect.getsource(mod)
        assert '"joint_limit_regularization": 0.0' in src, f"{mod.__name__} does not default the weight to 0.0"
