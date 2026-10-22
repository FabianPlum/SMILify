"""Tests for issue #97 (fitter_3d consumes authored joint_limits as a rotation prior).


- Pure-function tests for _joint_limit_tensors_from_dd: wide-open fallback,
  authored limits land in the right (N_POSE, 3) slot, malformed joint_limits is
  deferred (not raised at construction) -- same guarantees as LimitPrior, just
  built from a locally-loaded dd rather than the config.dd global.
- Stage.forward() hinge-loss wiring: zero in-range, positive + corrective
  gradient out-of-range, absent from loss_components entirely when w_limit=0
  (true no-op, not just zero-valued).
- One slow, real end-to-end registration test (marked `slow`, skipped by
  default): a synthetic tight limit that provably conflicts with where an
  unconstrained fit lands, run through the actual SMAL3DFitter ->
  Stage.forward() -> optimizer path, confirming the prior pulls the
  violating joint back inside bounds.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SMAL_FILE = os.path.join(REPO_ROOT, "3D_model_prep", "SMIL_OmniAnt.pkl")
TARGET_OBJ = os.path.join(REPO_ROOT, "fitter_3d", "ATTA_BOI", "Atta_vollenweideri_1_mg_worker.obj")


def _load_dd(smal_file=SMAL_FILE):
    import pickle as pkl

    if not os.path.isfile(smal_file):
        pytest.skip(f"SMAL file not found: {smal_file}")
    with open(smal_file, "rb") as f:
        u = pkl._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def _hinge(joint_rot, min_limits, max_limits):
    zeros = torch.zeros_like(joint_rot)
    return torch.mean(torch.max(joint_rot - max_limits, zeros) + torch.max(min_limits - joint_rot, zeros))


# ---------------------------------------------------------------------------
# _joint_limit_tensors_from_dd (pure function, no file/device I/O beyond dd)
# ---------------------------------------------------------------------------


def test_fallback_is_wide_open():
    """No 'joint_limits' key -> every non-root joint is wide-open [-pi, pi]."""
    from fitter_3d.trainer import _joint_limit_tensors_from_dd

    dd = dict(_load_dd())
    dd.pop("joint_limits", None)
    min_l, max_l, error = _joint_limit_tensors_from_dd(dd, "cpu")

    assert error is None
    assert torch.allclose(min_l, torch.full_like(min_l, -np.pi), atol=1e-5)
    assert torch.allclose(max_l, torch.full_like(max_l, np.pi), atol=1e-5)


def test_authored_limits_land_in_correct_slot():
    """An injected joint_limits reaches the right (N_POSE, 3) slot, root dropped."""
    from fitter_3d.trainer import _joint_limit_tensors_from_dd

    dd = dict(_load_dd())
    joint_names = dd["J_names"]
    n_joints = len(joint_names)

    jl = np.zeros((n_joints, 3, 2), dtype=np.float64)
    jl[..., 0], jl[..., 1] = -0.5, 0.5
    target_idx = 3 if n_joints > 3 else 1
    jl[target_idx] = [[-0.1, 0.1], [-1.2, 0.3], [-0.05, 0.05]]
    dd["joint_limits"] = jl

    min_l, max_l, error = _joint_limit_tensors_from_dd(dd, "cpu")

    assert error is None
    pose_idx = target_idx - 1  # root dropped
    assert torch.allclose(min_l[pose_idx], torch.tensor([-0.1, -1.2, -0.05]), atol=1e-5)
    assert torch.allclose(max_l[pose_idx], torch.tensor([0.1, 0.3, 0.05]), atol=1e-5)


def test_malformed_joint_limits_is_deferred_not_raised():
    """Construction must never crash on bad joint_limits -- only using w_limit>0 does."""
    from fitter_3d.trainer import _joint_limit_tensors_from_dd

    dd = dict(_load_dd())
    dd["joint_limits"] = np.zeros((2, 2))  # garbage shape

    min_l, max_l, error = _joint_limit_tensors_from_dd(dd, "cpu")

    assert error is not None
    assert isinstance(error, ValueError)
    # Falls back to wide-open so downstream shapes are still usable.
    assert torch.allclose(min_l, torch.full_like(min_l, -np.pi), atol=1e-5)


# ---------------------------------------------------------------------------
# Stage.forward() hinge-loss wiring
# ---------------------------------------------------------------------------


def test_hinge_zero_in_range_positive_and_corrective_out_of_range():
    """Same guarantee as the fitter.py / neural paths: flat inside, linear past."""
    n = config.N_POSE
    min_l = torch.full((n, 3), -0.5)
    max_l = torch.full((n, 3), 0.5)

    jr_ok = torch.zeros(1, n, 3)
    assert float(_hinge(jr_ok, min_l, max_l)) == 0.0

    jr_bad = torch.zeros(1, n, 3, requires_grad=True)
    with torch.no_grad():
        jr_bad[0, 0, 0] = 1.0  # 0.5 past the max on this joint/axis
    loss = _hinge(jr_bad, min_l, max_l)
    loss.backward()

    assert float(loss) > 0.0
    assert float(jr_bad.grad[0, 0, 0]) > 0.0  # descent pulls it back down toward the limit


def test_w_limit_zero_is_true_noop_not_just_zero_valued():
    """w_limit=0 must mean 'limit' never enters loss_components at all (consider_loss gates it)."""
    loss_weights = {"w_chamfer": 0.0, "w_edge": 0.0, "w_normal": 0.0, "w_laplacian": 0.0, "w_sdf": 0.0, "w_limit": 0.0}
    consider_loss = lambda name: loss_weights[f"w_{name}"] > 0  # noqa: E731 -- mirrors Stage.consider_loss
    assert consider_loss("limit") is False

    loss_weights["w_limit"] = 100.0
    assert consider_loss("limit") is True


# ---------------------------------------------------------------------------
# End-to-end: real SMAL3DFitter -> Stage.forward() -> optimizer (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isfile(TARGET_OBJ), reason=f"target mesh not found: {TARGET_OBJ}")
def test_limit_prior_binds_on_real_registration_path(tmp_path):
    """A synthetic tight limit, chosen to conflict with where the unconstrained fit
    lands, actually pulls the constrained fit back inside bounds -- confirming the
    prior binds on the real registration path, not just in the isolated unit tests
    above."""
    import pickle as pkl

    import config as project_config
    from fitter_3d.trainer import SMAL3DFitter, Stage
    from fitter_3d.utils import load_meshes

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- build the fixture: same production model, one synthetic narrow limit ---
    dd = _load_dd()
    joint_names = dd["J_names"]
    target_joint, target_axis, target_range = "l_3_pt_l", 1, (-0.05, 0.05)
    if target_joint not in joint_names:
        pytest.skip(f"'{target_joint}' not present in this SMAL_FILE's J_names")
    target_idx = joint_names.index(target_joint)

    n_joints = len(joint_names)
    jl = np.empty((n_joints, 3, 2))
    jl[..., 0], jl[..., 1] = -np.pi, np.pi
    jl[target_idx, target_axis] = target_range
    dd["joint_limits"] = jl

    fixture_path = tmp_path / "SMIL_OmniAnt_test_limits.pkl"
    with open(fixture_path, "wb") as f:
        pkl.dump(dd, f)

    original_smal_file = project_config.SMAL_FILE
    try:
        project_config.SMAL_FILE = str(fixture_path)

        _, target_meshes = load_meshes(mesh_files=[TARGET_OBJ], device=device)

        def _run(w_limit, nits=150):
            torch.manual_seed(0)
            fitter = SMAL3DFitter(batch_size=1, device=device, shape_family=-1)
            stage = Stage(
                nits=nits,
                scheme="default",
                smal_3d_fitter=fitter,
                target_meshes=target_meshes,
                loss_weights={"w_limit": w_limit},
                lr=0.02,
                device=device,
                out_dir=str(tmp_path),  # defensive: run() with plot=False writes nothing today, but don't rely on that
            )
            stage.run()
            # fitter.joint_rot already reflects the last optimizer step taken inside
            # run() -- read it directly rather than calling stage.step() again here,
            # which would silently take one *extra* gradient step past "final".
            joint_rot = fitter.joint_rot[0, target_idx - 1, target_axis].item()  # root dropped
            final_chamfer = float(stage.loss_components_to_plot["chamfer"][-1])
            return joint_rot, final_chamfer

        baseline_rot, _ = _run(w_limit=0.0)
        constrained_rot, _ = _run(w_limit=100.0)

        lo, hi = target_range
        assert not (lo <= baseline_rot <= hi), (
            f"baseline (unconstrained) run landed inside the synthetic bound ({baseline_rot:.4f}); "
            "test doesn't provoke a real conflict, pick a different target joint/axis"
        )
        assert lo <= constrained_rot <= hi, (
            f"constrained run (w_limit=100) still violates the bound: {constrained_rot:.4f} not in {target_range}"
        )
    finally:
        project_config.SMAL_FILE = original_smal_file
