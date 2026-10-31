"""Multi-view shape regularisers: left/right joint pairs, scale symmetry and per-session scale consistency."""

import torch

import config
from smal_fitter.neuralSMIL.multiview_smil_regressor import MultiViewSMILImageRegressor as MV


class _Model:
    """Just the state the penalty methods touch."""

    def __init__(self, training=True):
        self.training = training


def test_left_right_pairs_and_symmetry(monkeypatch):
    monkeypatch.setattr(config, "joint_names", ["root", "femur_L", "femur_R", "Foot_L_tip", "Foot_R_tip", "Tail_01"])
    assert MV._lr_joint_pairs(_Model()) == [(1, 2), (3, 4)]


def test_scale_consistency_tracks_a_per_session_mean_in_training_only():
    m = _Model()
    constant = torch.zeros(4, 5, 3)
    assert MV._scale_consistency_loss(m, constant, None, ["a"] * 4).item() == 0.0
    varying = torch.randn(4, 5, 3)
    assert MV._scale_consistency_loss(m, varying, torch.zeros(4, 1), ["a", "a", "b", "b"]).item() > 0
    ema_a = m.__dict__["_scale_ema"]["a"][0].clone()
    m.training = False
    MV._scale_consistency_loss(m, varying + 1, torch.zeros(4, 1), ["a", "a", "new", "new"])
    torch.testing.assert_close(m.__dict__["_scale_ema"]["a"][0], ema_a)  # eval does not move the mean ...
    assert "new" not in m.__dict__["_scale_ema"]  # ... nor create one for a session it sees first


def test_scale_consistency_leaves_out_diverged_frames():
    m = _Model()
    scales = torch.zeros(3, 5, 3)
    scales[1, 0, 0] = float("inf")
    scales[2, 0, 0] = 3.4e38  # what nan_to_num makes of an inf
    loss = MV._scale_consistency_loss(m, scales, torch.tensor([[0.0], [float("nan")], [0.0]]), ["a"] * 3)
    assert loss.item() == 0.0  # only frame 0 is left, and it is the mean
    mu_s, mu_m = m.__dict__["_scale_ema"]["a"]
    assert torch.isfinite(mu_s).all() and torch.isfinite(mu_m).all()
    assert MV._scale_consistency_loss(m, torch.full((2, 5, 3), 0.1), torch.zeros(2, 1), ["a"] * 2).isfinite()
