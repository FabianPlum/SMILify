"""Tests for ``joint_twist_regularization`` (``SMILImageRegressor._joint_twist_penalty``).

Toy kinematic chains are swapped into ``config.dd``. The real methods run on a stub that carries only the attributes
they read (no backbone or renderer), as in ``test_joint_limits_prior.py``.
"""

import math
import os
import sys
import types

import numpy as np
import pytest
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


def _toy_model(monkeypatch, J, parents, joint_limits=None):
    """Swap a toy model into config.dd for one test: rest joint positions, parent per joint, optional limits."""
    dd = {"J": np.asarray(J, np.float32), "kintree_table": np.array([parents, range(len(parents))], np.int32)}
    if joint_limits is not None:
        dd["joint_limits"] = joint_limits
    monkeypatch.setattr(config, "dd", dd)


def _twist_stub(rotation_representation):
    from smal_fitter.neuralSMIL.smil_image_regressor import SMILImageRegressor

    stub = types.SimpleNamespace(rotation_representation=rotation_representation, device="cpu")
    stub._TWIST_FREE_DEG = SMILImageRegressor._TWIST_FREE_DEG
    stub._init_twist_axes = types.MethodType(SMILImageRegressor._init_twist_axes, stub)
    stub.penalty = types.MethodType(SMILImageRegressor._joint_twist_penalty, stub)
    return stub


def test_twist_penalty_hinges_on_axial_twist_only(monkeypatch):
    """Twist about the rest bone direction beyond the free angle is penalised; twist within it and pure swing are not.
    A locked proxy leaf does not count as a chain child."""
    # root -> 1 -> 2 -> 3, plus a proxy leaf 4 under joint 1 whose limits are all zero (locked).
    limits = np.zeros((5, 3, 2), np.float32)
    limits[1:4, :, 0], limits[1:4, :, 1] = -np.pi, np.pi
    _toy_model(monkeypatch, [[0, 0, 0], [1, 0, 0], [2, 1, 0], [2, 1, 1], [1, -0.5, 0]], [-1, 0, 1, 2, 1], limits)
    stub = _twist_stub("axis_angle")
    stub._init_twist_axes()

    # Joint 1 twists about its bone to joint 2 (tilted, so no per-axis box can bound it), joint 2 about its bone to
    # joint 3; the leaves get no axis. joint_rot index = joint index - 1.
    u = torch.tensor([1.0, 1.0, 0.0]) / math.sqrt(2)
    z = torch.tensor([0.0, 0.0, 1.0])
    assert stub._twist_idx.tolist() == [0, 1]
    assert torch.allclose(stub._twist_axes, torch.stack([u, z]))

    free = math.radians(stub._TWIST_FREE_DEG)

    def pose(aa_joint1):
        jr = torch.zeros(1, 4, 3)
        jr[0, 0] = aa_joint1
        return jr.requires_grad_(True)

    assert stub.penalty(pose(torch.zeros(3))).item() == 0.0
    assert stub.penalty(pose((free - 0.25) * u)).item() == 0.0  # twist within the free angle
    assert stub.penalty(pose(math.radians(90) * z)).item() == 0.0  # 90 deg swing, perpendicular to the bone

    jr = pose((free + 0.25) * u)
    penalty = stub.penalty(jr)
    assert math.isclose(penalty.item(), 0.25 / 2, abs_tol=1e-5)  # mean over the 2 twist joints
    penalty.backward()
    # The gradient untwists joint 1 about its bone axis and leaves the other joints alone.
    assert torch.allclose(jr.grad[0, 0], u / 2, atol=1e-4)
    assert torch.all(jr.grad[0, 1:] == 0)

    # The same twist after a 70 deg swing, through the 6D path: the swing does not leak into the twist.
    R = axis_angle_to_matrix(math.radians(70) * z) @ axis_angle_to_matrix((free + 0.25) * u)
    six = matrix_to_rotation_6d(torch.eye(3).expand(1, 4, 3, 3).clone())
    six[0, 0] = matrix_to_rotation_6d(R)
    assert math.isclose(_twist_stub("6d").penalty(six).item(), 0.25 / 2, abs_tol=1e-5)

    # Without the limits, the proxy leaf is joint 1's second chain child, so joint 1 gets no axis.
    monkeypatch.delitem(config.dd, "joint_limits")
    no_limits = _twist_stub("axis_angle")
    no_limits._init_twist_axes()
    assert no_limits._twist_idx.tolist() == [1]


def test_twist_penalty_without_twist_joints_is_finite_zero(monkeypatch):
    """A model where no joint has a single chain child (root with three leaves) has no twist axis. The mean over the
    empty selection would be NaN in every batch; the penalty is a +0.0 on the graph instead, with one warning."""
    _toy_model(monkeypatch, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], [-1, 0, 0, 0])
    stub = _twist_stub("axis_angle")
    jr = torch.linspace(-1.0, 0.5, 18).reshape(2, 3, 3).requires_grad_(True)  # negative sum: sum() * 0.0 is -0.0

    with pytest.warns(RuntimeWarning, match="no bone has a twist axis") as record:
        penalties = [stub.penalty(jr) for _ in range(3)]
    assert sum("twist" in str(w.message) for w in record) == 1  # once per model, not per batch

    for penalty in penalties:
        assert torch.isfinite(penalty) and penalty.item() == 0.0
        assert not torch.signbit(penalty)  # 0.0, not the -0.0 of a bare sum() * 0.0
        assert penalty.requires_grad
    penalties[0].backward()
    assert torch.all(jr.grad == 0)
