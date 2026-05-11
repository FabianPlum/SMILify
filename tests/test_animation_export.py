"""Round-trip tests for the Phase 1 animation exporter.

Builds a synthetic `predicted_params` dict (no model, no checkpoint), runs it
through AnimationRecorder, reloads the resulting .npz + .json, and asserts the
schema and tensor values are preserved.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
NEURAL_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "smal_fitter", "neuralSMIL")
sys.path.append(os.path.dirname(CURRENT_DIR))
sys.path.append(NEURAL_DIR)

from smal_fitter.neuralSMIL.animation_export import (  # noqa: E402
    AnimationRecorder,
    SCHEMA_VERSION,
)


N_JOINTS = 8
N_POSE = N_JOINTS - 1
N_BETAS = 20


def _make_params_axis_angle(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "global_rot": torch.randn(1, 3, generator=g),
        "joint_rot": torch.randn(1, N_POSE, 3, generator=g),
        "trans": torch.randn(1, 3, generator=g),
        "betas": torch.randn(1, N_BETAS, generator=g),
        "log_beta_scales": torch.randn(1, N_JOINTS, 3, generator=g),
        "betas_trans": torch.randn(1, N_JOINTS, 3, generator=g),
        "mesh_scale": torch.rand(1, 1, generator=g) + 0.5,
        "cam_rot": torch.eye(3).unsqueeze(0),
        "cam_trans": torch.randn(1, 3, generator=g),
        "fov": torch.tensor([[45.0]]),
    }


def _make_recorder(tmp_path, rotation_representation="axis_angle"):
    return AnimationRecorder(
        output_path=tmp_path / "clip",
        rotation_representation=rotation_representation,
        n_joints=N_JOINTS,
        n_betas=N_BETAS,
        joint_names=[f"J_{i}" for i in range(N_JOINTS)],
        parents=[-1] + list(range(N_JOINTS - 1)),
        fps=30.0,
        static_joint_locs=True,
        ignore_hardcoded_body=False,
        source_checkpoint="ckpt.pth",
        source_input="input.mp4",
        model_id="test_model",
    )


def test_round_trip_axis_angle(tmp_path):
    rec = _make_recorder(tmp_path)
    frames = [_make_params_axis_angle(i) for i in range(5)]
    for f in frames:
        rec.record(f)

    out = rec.write()
    assert out["npz"].exists() and out["json"].exists()

    data = np.load(out["npz"])
    assert data["poses"].shape == (5, N_JOINTS, 3)
    assert data["trans"].shape == (5, 3)
    assert data["betas"].shape == (N_BETAS,)
    assert data["betas_per_frame"].shape == (5, N_BETAS)
    assert data["log_beta_scales"].shape == (5, N_JOINTS, 3)
    assert data["betas_trans"].shape == (5, N_JOINTS, 3)
    assert data["mesh_scale"].shape == (5,)
    assert float(data["fps"]) == pytest.approx(30.0)

    # poses[f, 0] must equal global_rot; poses[f, 1:] must equal joint_rot.
    for f_idx, params in enumerate(frames):
        np.testing.assert_allclose(
            data["poses"][f_idx, 0], params["global_rot"][0].numpy(), atol=1e-6
        )
        np.testing.assert_allclose(
            data["poses"][f_idx, 1:], params["joint_rot"][0].numpy(), atol=1e-6
        )
        np.testing.assert_allclose(
            data["betas_per_frame"][f_idx], params["betas"][0].numpy(), atol=1e-6
        )

    # Clip-averaged betas equal numpy mean of the per-frame betas.
    expected_avg = np.stack([p["betas"][0].numpy() for p in frames]).mean(axis=0)
    np.testing.assert_allclose(data["betas"], expected_avg, atol=1e-6)

    with open(out["json"]) as fh:
        sidecar = json.load(fh)
    assert sidecar["schema_version"] == SCHEMA_VERSION
    assert sidecar["rotation_representation"] == "axis_angle"
    assert sidecar["root_joint_index"] == 0
    assert sidecar["static_joint_locs"] is True
    assert sidecar["n_frames"] == 5
    assert sidecar["n_joints"] == N_JOINTS
    assert sidecar["n_betas"] == N_BETAS
    assert len(sidecar["joint_names"]) == N_JOINTS
    assert len(sidecar["parents"]) == N_JOINTS
    assert sidecar["fps"] == 30.0
    assert len(sidecar["cameras"]) == 1
    assert sidecar["cameras"][0]["view_name"] == "view_0"


def test_6d_rotations_normalise_to_axis_angle(tmp_path):
    rec = _make_recorder(tmp_path, rotation_representation="6d")
    params = {
        "global_rot": torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),  # identity 6D
        "joint_rot": torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]).expand(1, N_POSE, 6).clone(),
        "trans": torch.zeros(1, 3),
        "betas": torch.zeros(1, N_BETAS),
    }
    rec.record(params)
    out = rec.write()
    data = np.load(out["npz"])
    # Identity rotation in axis-angle is the zero vector.
    np.testing.assert_allclose(data["poses"], np.zeros((1, N_JOINTS, 3)), atol=1e-5)
    with open(out["json"]) as fh:
        sidecar = json.load(fh)
    assert sidecar["rotation_representation"] == "axis_angle"


def test_optional_fields_absent_when_not_recorded(tmp_path):
    rec = _make_recorder(tmp_path)
    params = {
        "global_rot": torch.zeros(1, 3),
        "joint_rot": torch.zeros(1, N_POSE, 3),
        "trans": torch.zeros(1, 3),
        "betas": torch.zeros(1, N_BETAS),
    }
    rec.record(params)
    out = rec.write()
    data = np.load(out["npz"])
    assert "log_beta_scales" not in data.files
    assert "betas_trans" not in data.files
    assert "mesh_scale" not in data.files
    with open(out["json"]) as fh:
        sidecar = json.load(fh)
    assert sidecar["cameras"] == []


def test_write_without_frames_raises(tmp_path):
    rec = _make_recorder(tmp_path)
    with pytest.raises(RuntimeError):
        rec.write()


def test_invalid_rotation_representation_raises(tmp_path):
    with pytest.raises(ValueError):
        AnimationRecorder(
            output_path=tmp_path / "clip",
            rotation_representation="quaternion",
            n_joints=N_JOINTS,
            n_betas=N_BETAS,
            joint_names=[f"J_{i}" for i in range(N_JOINTS)],
            parents=[-1] + list(range(N_JOINTS - 1)),
            fps=30.0,
            static_joint_locs=True,
            ignore_hardcoded_body=False,
        )
