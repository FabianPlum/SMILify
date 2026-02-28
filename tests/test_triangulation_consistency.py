"""
Tests for the differentiable triangulation consistency loss.

Validates the full round-trip: known 3D joints → project to 2D via synthetic
cameras → triangulate back to 3D via _triangulate_joints_dlt → compare.

Uses the real SMAL model (SMILy_STICK.pkl) with random joint angles to produce
realistic 3D joint configurations, then constructs multiple synthetic cameras
to verify the loss computation.
"""

import os
import sys
import math

import pytest
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — make smal_fitter and neuralSMIL importable
# ---------------------------------------------------------------------------
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_neural_smil = os.path.join(_repo_root, "smal_fitter", "neuralSMIL")
for p in [_repo_root, _neural_smil]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Model / config paths
# ---------------------------------------------------------------------------
SMAL_FILE = os.path.join(_repo_root, "3D_model_prep", "SMILy_STICK.pkl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRenderer:
    """Minimal stub satisfying the attributes that _triangulate_joints_dlt and
    _batch_project_joints_to_views read from self.renderer."""

    DEFAULT_ZNEAR = 0.001
    DEFAULT_ZFAR = 1000.0

    def __init__(self, image_size: int = 512):
        self.image_size = image_size


def _rotation_matrix_y(angle_rad: float, device: torch.device) -> torch.Tensor:
    """Rotation matrix around the Y axis. Returns (3, 3)."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return torch.tensor([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=torch.float32, device=device)


def _rotation_matrix_x(angle_rad: float, device: torch.device) -> torch.Tensor:
    """Rotation matrix around the X axis. Returns (3, 3)."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return torch.tensor([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=torch.float32, device=device)


def _make_cameras_around_origin(
    num_views: int,
    radius: float,
    fov_deg: float,
    device: torch.device,
    batch_size: int = 1,
    elevation_deg: float = 15.0,
):
    """Create cameras evenly spaced in azimuth around the origin.

    Returns:
        fov_per_view:   list of (B, 1) tensors
        R_per_view:     list of (B, 3, 3) rotation matrices
        T_per_view:     list of (B, 3) translation vectors
        aspect_ratios:  list of (B,) tensors
    """
    from pytorch3d.renderer import look_at_view_transform

    azimuths = torch.linspace(0, 360, num_views + 1)[:num_views]
    elevations = torch.full_like(azimuths, elevation_deg)

    R, T = look_at_view_transform(
        dist=radius,
        elev=elevations,
        azim=azimuths,
        device=device,
    )  # R: (V, 3, 3), T: (V, 3)

    B = batch_size
    fov_per_view = [torch.full((B, 1), fov_deg, device=device) for _ in range(num_views)]
    R_per_view = [R[v:v+1].expand(B, -1, -1).clone() for v in range(num_views)]
    T_per_view = [T[v:v+1].expand(B, -1).clone() for v in range(num_views)]
    aspect_ratios = [torch.ones(B, device=device) for _ in range(num_views)]

    return fov_per_view, R_per_view, T_per_view, aspect_ratios


def _project_joints_to_views(
    joints_3d, fov_per_view, R_per_view, T_per_view,
    aspect_ratios, image_size, device,
):
    """Project 3D joints to normalised [0,1] 2D keypoints for each view.

    Mirrors the logic in _batch_project_joints_to_views so we have
    a ground-truth projection that is independent of the regressor class.

    Args:
        joints_3d: (B, J, 3)

    Returns:
        keypoints_2d: (B, V, J, 2) in the (y, x) normalised convention.
    """
    from pytorch3d.renderer import FoVPerspectiveCameras

    B, J, _ = joints_3d.shape
    V = len(fov_per_view)

    # Stack per-view params to (B, V, ...) then flatten to (B*V, ...)
    # This matches the interleaving order of joints_flat below.
    fov_stacked = torch.stack(
        [f.squeeze(-1) if f.dim() > 1 else f for f in fov_per_view], dim=1
    )  # (B, V)
    R_stacked = torch.stack(R_per_view, dim=1)      # (B, V, 3, 3)
    T_stacked = torch.stack(T_per_view, dim=1)      # (B, V, 3)
    aspect_stacked = torch.stack(aspect_ratios, dim=1)  # (B, V)

    cameras = FoVPerspectiveCameras(
        device=device,
        R=R_stacked.reshape(B * V, 3, 3).float(),
        T=T_stacked.reshape(B * V, 3).float(),
        fov=fov_stacked.reshape(B * V).float(),
        aspect_ratio=aspect_stacked.reshape(B * V).float(),
        znear=0.001,
        zfar=1000.0,
    )

    joints_expanded = joints_3d.unsqueeze(1).expand(-1, V, -1, -1)  # (B, V, J, 3)
    joints_flat = joints_expanded.reshape(B * V, J, 3)

    screen_size = torch.ones(B * V, 2, device=device) * image_size
    proj = cameras.transform_points_screen(
        joints_flat.float(), image_size=screen_size
    )[:, :, [1, 0]]  # swap to (y, x)

    kp_2d = proj / image_size
    kp_2d = kp_2d.view(B, V, J, 2)
    return kp_2d


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def smal_model(device):
    """Load the real SMAL model."""
    if not os.path.exists(SMAL_FILE):
        pytest.skip(f"SMAL file not found: {SMAL_FILE}")

    # Temporarily override config.SMAL_FILE so SMAL() picks up the stick model
    import config
    original_smal_file = config.SMAL_FILE
    config.SMAL_FILE = SMAL_FILE

    from smal_model.smal_torch import SMAL
    model = SMAL(device)

    config.SMAL_FILE = original_smal_file
    return model


@pytest.fixture(scope="module")
def stick_joint_info(smal_model):
    """Return (N_JOINTS, N_BETAS) from the loaded model."""
    n_joints = smal_model.J_regressor.shape[1]
    n_betas = smal_model.num_betas
    return n_joints, n_betas


@pytest.fixture(scope="module")
def sample_joints_3d(smal_model, stick_joint_info, device):
    """Generate 3D joints from random (small) joint angles through the SMAL model.

    Returns (joints_3d, n_joints) where joints_3d is (B=2, J, 3).
    """
    n_joints, n_betas = stick_joint_info
    B = 2
    torch.manual_seed(42)

    betas = torch.zeros(B, n_betas, device=device)
    # Small random joint angles to get a realistic but non-trivial pose
    theta = torch.randn(B, n_joints, 3, device=device) * 0.15
    theta[:, 0, :] = 0.0  # zero global rotation for simplicity

    _, joints, _, _ = smal_model(betas, theta)
    return joints, n_joints


# ---------------------------------------------------------------------------
# Triangulator helper — wraps _triangulate_joints_dlt with a fake self
# ---------------------------------------------------------------------------

class _TriangulatorStub:
    """Minimal object that exposes _triangulate_joints_dlt and
    _batch_project_joints_to_views by borrowing the unbound methods
    from MultiViewSMILRegressor and providing just the attributes they read."""

    def __init__(self, image_size: int, device: torch.device):
        self.renderer = _FakeRenderer(image_size)
        self.device = device

        # Import the actual method and bind to this stub
        from multiview_smil_regressor import MultiViewSMILImageRegressor
        import types

        self._triangulate_joints_dlt = types.MethodType(
            MultiViewSMILImageRegressor._triangulate_joints_dlt, self
        )


@pytest.fixture(scope="module")
def triangulator(device):
    return _TriangulatorStub(image_size=512, device=device)


# ===========================================================================
# Tests
# ===========================================================================

class TestTriangulateJointsDLT:
    """Round-trip tests: project 3D → 2D → triangulate → compare to 3D."""

    def test_perfect_cameras_recover_joints(
        self, sample_joints_3d, triangulator, device
    ):
        """With ground-truth cameras and perfect 2D observations, the
        triangulated 3D joints should match the originals within tight
        tolerance."""
        joints_3d, n_joints = sample_joints_3d
        B = joints_3d.shape[0]
        V = 6
        IMAGE_SIZE = triangulator.renderer.image_size

        fov_per_view, R_per_view, T_per_view, aspect_ratios = \
            _make_cameras_around_origin(V, radius=3.0, fov_deg=60.0, device=device, batch_size=B)

        kp_2d = _project_joints_to_views(
            joints_3d, fov_per_view, R_per_view, T_per_view,
            aspect_ratios, IMAGE_SIZE, device,
        )  # (B, V, J, 2)

        visibility = torch.ones(B, V, n_joints, device=device)

        triangulated, valid_mask = triangulator._triangulate_joints_dlt(
            kp_2d, visibility,
            fov_per_view, R_per_view, T_per_view,
            aspect_ratio_per_view=aspect_ratios,
        )

        assert valid_mask.all(), "All joints should be valid with full visibility"

        error = (triangulated - joints_3d).norm(dim=-1)  # (B, J)
        max_error = error.max().item()
        mean_error = error.mean().item()

        assert max_error < 0.05, (
            f"Max triangulation error {max_error:.6f} exceeds 0.05 — "
            f"round-trip should be near-exact with perfect cameras"
        )
        assert mean_error < 0.01, (
            f"Mean triangulation error {mean_error:.6f} exceeds 0.01"
        )

    def test_two_views_sufficient(
        self, sample_joints_3d, triangulator, device
    ):
        """Triangulation should work with just 2 views."""
        joints_3d, n_joints = sample_joints_3d
        B = joints_3d.shape[0]
        V = 2
        IMAGE_SIZE = triangulator.renderer.image_size

        # Use well-separated cameras (90 degrees apart)
        fov_per_view, R_per_view, T_per_view, aspect_ratios = \
            _make_cameras_around_origin(V, radius=3.0, fov_deg=60.0, device=device, batch_size=B)

        kp_2d = _project_joints_to_views(
            joints_3d, fov_per_view, R_per_view, T_per_view,
            aspect_ratios, IMAGE_SIZE, device,
        )
        visibility = torch.ones(B, V, n_joints, device=device)

        triangulated, valid_mask = triangulator._triangulate_joints_dlt(
            kp_2d, visibility,
            fov_per_view, R_per_view, T_per_view,
            aspect_ratio_per_view=aspect_ratios,
        )

        assert valid_mask.all()
        error = (triangulated - joints_3d).norm(dim=-1)
        assert error.max().item() < 0.1, (
            f"Max error {error.max().item():.4f} with 2 views — "
            f"should still be reasonable"
        )

    def test_masked_views_respected(
        self, sample_joints_3d, triangulator, device
    ):
        """Masking all but 1 view should produce invalid triangulations
        (need >= 2 views)."""
        joints_3d, n_joints = sample_joints_3d
        B = joints_3d.shape[0]
        V = 4
        IMAGE_SIZE = triangulator.renderer.image_size

        fov_per_view, R_per_view, T_per_view, aspect_ratios = \
            _make_cameras_around_origin(V, radius=3.0, fov_deg=60.0, device=device, batch_size=B)

        kp_2d = _project_joints_to_views(
            joints_3d, fov_per_view, R_per_view, T_per_view,
            aspect_ratios, IMAGE_SIZE, device,
        )

        # Only view 0 is visible
        visibility = torch.zeros(B, V, n_joints, device=device)
        visibility[:, 0, :] = 1.0

        _, valid_mask = triangulator._triangulate_joints_dlt(
            kp_2d, visibility,
            fov_per_view, R_per_view, T_per_view,
            aspect_ratio_per_view=aspect_ratios,
        )

        assert not valid_mask.any(), (
            "With only 1 visible view, no joints should be valid"
        )

    def test_partial_visibility(
        self, sample_joints_3d, triangulator, device
    ):
        """Some joints visible in 2+ views, others in <2. Valid mask should
        reflect this correctly."""
        joints_3d, n_joints = sample_joints_3d
        B = joints_3d.shape[0]
        V = 4
        IMAGE_SIZE = triangulator.renderer.image_size

        fov_per_view, R_per_view, T_per_view, aspect_ratios = \
            _make_cameras_around_origin(V, radius=3.0, fov_deg=60.0, device=device, batch_size=B)

        kp_2d = _project_joints_to_views(
            joints_3d, fov_per_view, R_per_view, T_per_view,
            aspect_ratios, IMAGE_SIZE, device,
        )

        visibility = torch.ones(B, V, n_joints, device=device)
        # Make joint 0 visible only in view 0 → should be invalid
        visibility[:, 1:, 0] = 0.0

        triangulated, valid_mask = triangulator._triangulate_joints_dlt(
            kp_2d, visibility,
            fov_per_view, R_per_view, T_per_view,
            aspect_ratio_per_view=aspect_ratios,
        )

        assert not valid_mask[:, 0].any(), "Joint 0 with 1 view should be invalid"
        assert valid_mask[:, 1:].all(), "Joints 1+ with 4 views should all be valid"

        # Valid joints should still triangulate accurately
        valid_error = (triangulated[:, 1:] - joints_3d[:, 1:]).norm(dim=-1)
        assert valid_error.max().item() < 0.05

    def test_view_mask(
        self, sample_joints_3d, triangulator, device
    ):
        """view_mask should disable entire views regardless of per-joint
        visibility."""
        joints_3d, n_joints = sample_joints_3d
        B = joints_3d.shape[0]
        V = 4
        IMAGE_SIZE = triangulator.renderer.image_size

        fov_per_view, R_per_view, T_per_view, aspect_ratios = \
            _make_cameras_around_origin(V, radius=3.0, fov_deg=60.0, device=device, batch_size=B)

        kp_2d = _project_joints_to_views(
            joints_3d, fov_per_view, R_per_view, T_per_view,
            aspect_ratios, IMAGE_SIZE, device,
        )

        visibility = torch.ones(B, V, n_joints, device=device)
        # Disable views 1-3 via view_mask, leaving only view 0
        view_mask = torch.zeros(B, V, dtype=torch.bool, device=device)
        view_mask[:, 0] = True

        _, valid_mask = triangulator._triangulate_joints_dlt(
            kp_2d, visibility,
            fov_per_view, R_per_view, T_per_view,
            aspect_ratio_per_view=aspect_ratios,
            view_mask=view_mask,
        )

        assert not valid_mask.any(), (
            "Only 1 view enabled via view_mask — nothing should be valid"
        )


class TestTriangulationGradients:
    """Verify that gradients flow through the triangulation into camera
    parameters (the whole point of using a differentiable solver)."""

    def test_gradients_flow_to_cameras(
        self, sample_joints_3d, triangulator, device
    ):
        """Camera translation should receive gradients from the
        triangulation loss."""
        joints_3d, n_joints = sample_joints_3d
        joints_3d_detached = joints_3d.detach()
        B = joints_3d.shape[0]
        V = 4
        IMAGE_SIZE = triangulator.renderer.image_size

        fov_per_view, R_per_view, T_per_view, aspect_ratios = \
            _make_cameras_around_origin(V, radius=3.0, fov_deg=60.0, device=device, batch_size=B)

        # Make camera translations require grad
        T_per_view_grad = [t.clone().requires_grad_(True) for t in T_per_view]

        kp_2d = _project_joints_to_views(
            joints_3d_detached, fov_per_view, R_per_view, T_per_view,
            aspect_ratios, IMAGE_SIZE, device,
        ).detach()  # GT 2D from original cameras, no grad

        # Triangulate with grad-enabled cameras
        triangulated, valid_mask = triangulator._triangulate_joints_dlt(
            kp_2d, torch.ones(B, V, n_joints, device=device),
            fov_per_view, R_per_view, T_per_view_grad,
            aspect_ratio_per_view=aspect_ratios,
        )

        loss = ((triangulated - joints_3d_detached) ** 2).mean()
        loss.backward()

        for v, t in enumerate(T_per_view_grad):
            assert t.grad is not None, f"View {v}: cam_trans has no gradient"
            assert t.grad.abs().sum() > 0, f"View {v}: cam_trans gradient is zero"

    def test_no_gradient_to_target_joints(
        self, sample_joints_3d, triangulator, device
    ):
        """joints_3d used as target should NOT receive gradients (it's
        detached in the actual loss)."""
        joints_3d, n_joints = sample_joints_3d
        joints_target = joints_3d.detach().requires_grad_(True)
        B = joints_3d.shape[0]
        V = 4
        IMAGE_SIZE = triangulator.renderer.image_size

        fov_per_view, R_per_view, T_per_view, aspect_ratios = \
            _make_cameras_around_origin(V, radius=3.0, fov_deg=60.0, device=device, batch_size=B)

        kp_2d = _project_joints_to_views(
            joints_3d.detach(), fov_per_view, R_per_view, T_per_view,
            aspect_ratios, IMAGE_SIZE, device,
        ).detach()

        # Enable grad on FOV so backward() has a grad path through triangulation
        fov_per_view = [f.detach().requires_grad_(True) for f in fov_per_view]

        triangulated, _ = triangulator._triangulate_joints_dlt(
            kp_2d, torch.ones(B, V, n_joints, device=device),
            fov_per_view, R_per_view, T_per_view,
            aspect_ratio_per_view=aspect_ratios,
        )

        # Mimic the actual loss: detach the target
        loss = ((triangulated - joints_target.detach()) ** 2).mean()
        loss.backward()

        assert joints_target.grad is None or joints_target.grad.abs().sum() == 0, (
            "Target joints_3d should not receive gradients"
        )


class TestTriangulationLossComputation:
    """Test the loss value computation mirrors what _compute_multiview_losses
    would produce."""

    def test_perfect_reconstruction_gives_near_zero_loss(
        self, sample_joints_3d, triangulator, device
    ):
        """When cameras are perfect, loss should be near zero."""
        joints_3d, n_joints = sample_joints_3d
        B = joints_3d.shape[0]
        V = 6
        IMAGE_SIZE = triangulator.renderer.image_size

        fov_per_view, R_per_view, T_per_view, aspect_ratios = \
            _make_cameras_around_origin(V, radius=3.0, fov_deg=60.0, device=device, batch_size=B)

        kp_2d = _project_joints_to_views(
            joints_3d, fov_per_view, R_per_view, T_per_view,
            aspect_ratios, IMAGE_SIZE, device,
        )
        visibility = torch.ones(B, V, n_joints, device=device)

        triangulated, tri_valid = triangulator._triangulate_joints_dlt(
            kp_2d, visibility,
            fov_per_view, R_per_view, T_per_view,
            aspect_ratio_per_view=aspect_ratios,
        )

        # Compute loss exactly as in _compute_multiview_losses
        diff_sq = (triangulated - joints_3d.detach()) ** 2
        mask_weights = tri_valid.float()
        masked_loss = diff_sq * mask_weights.unsqueeze(-1)
        eps = 1e-8
        denom = tri_valid.sum().float() * 3 + eps
        tri_loss = masked_loss.sum() / denom

        assert tri_loss.item() < 1e-3, (
            f"Loss with perfect cameras should be near zero, got {tri_loss.item():.6f}"
        )

    def test_perturbed_cameras_increase_loss(
        self, sample_joints_3d, triangulator, device
    ):
        """Perturbing camera translations should increase the loss."""
        joints_3d, n_joints = sample_joints_3d
        B = joints_3d.shape[0]
        V = 6
        IMAGE_SIZE = triangulator.renderer.image_size

        fov_per_view, R_per_view, T_per_view, aspect_ratios = \
            _make_cameras_around_origin(V, radius=3.0, fov_deg=60.0, device=device, batch_size=B)

        # Project with TRUE cameras
        kp_2d = _project_joints_to_views(
            joints_3d, fov_per_view, R_per_view, T_per_view,
            aspect_ratios, IMAGE_SIZE, device,
        )
        visibility = torch.ones(B, V, n_joints, device=device)

        # Triangulate with TRUE cameras → baseline loss
        tri_true, valid_true = triangulator._triangulate_joints_dlt(
            kp_2d, visibility,
            fov_per_view, R_per_view, T_per_view,
            aspect_ratio_per_view=aspect_ratios,
        )
        diff_sq_true = (tri_true - joints_3d.detach()) ** 2
        loss_true = (diff_sq_true * valid_true.float().unsqueeze(-1)).sum() / \
                    (valid_true.sum().float() * 3 + 1e-8)

        # Triangulate with PERTURBED cameras
        T_perturbed = [t + 0.3 * torch.randn_like(t) for t in T_per_view]
        tri_pert, valid_pert = triangulator._triangulate_joints_dlt(
            kp_2d, visibility,
            fov_per_view, R_per_view, T_perturbed,
            aspect_ratio_per_view=aspect_ratios,
        )
        diff_sq_pert = (tri_pert - joints_3d.detach()) ** 2
        # Only use joints valid in both
        both_valid = valid_true & valid_pert
        loss_pert = (diff_sq_pert * both_valid.float().unsqueeze(-1)).sum() / \
                    (both_valid.sum().float() * 3 + 1e-8)

        assert loss_pert.item() > loss_true.item(), (
            f"Perturbed cameras should give higher loss: "
            f"true={loss_true.item():.6f}, perturbed={loss_pert.item():.6f}"
        )

    def test_gradient_descent_reduces_loss(
        self, sample_joints_3d, triangulator, device
    ):
        """A few steps of gradient descent on camera translations should
        reduce the triangulation loss — verifying the loss provides a
        useful optimisation signal."""
        joints_3d, n_joints = sample_joints_3d
        B = joints_3d.shape[0]
        V = 4
        IMAGE_SIZE = triangulator.renderer.image_size

        fov_per_view, R_per_view, T_per_view, aspect_ratios = \
            _make_cameras_around_origin(V, radius=3.0, fov_deg=60.0, device=device, batch_size=B)

        # Project with TRUE cameras to get GT 2D observations
        kp_2d = _project_joints_to_views(
            joints_3d, fov_per_view, R_per_view, T_per_view,
            aspect_ratios, IMAGE_SIZE, device,
        ).detach()
        visibility = torch.ones(B, V, n_joints, device=device)
        joints_target = joints_3d.detach()

        # Start with perturbed camera translations
        T_optim = [
            (t + 0.2 * torch.randn_like(t)).requires_grad_(True)
            for t in T_per_view
        ]
        optimizer = torch.optim.Adam(T_optim, lr=0.05)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            tri, valid = triangulator._triangulate_joints_dlt(
                kp_2d, visibility,
                fov_per_view, R_per_view, T_optim,
                aspect_ratio_per_view=aspect_ratios,
            )
            diff_sq = (tri - joints_target) ** 2
            loss = (diff_sq * valid.float().unsqueeze(-1)).sum() / \
                   (valid.sum().float() * 3 + 1e-8)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.5, (
            f"20 steps of Adam should reduce loss by at least 50%: "
            f"initial={losses[0]:.6f}, final={losses[-1]:.6f}"
        )


class TestTriangulationEdgeCases:
    """Numerical stability and edge cases."""

    def test_no_nan_with_zero_visibility(self, triangulator, device):
        """Fully invisible joints should produce finite output (even if
        invalid), never NaN."""
        B, V, J = 2, 4, 10

        fov_per_view, R_per_view, T_per_view, aspect_ratios = \
            _make_cameras_around_origin(V, radius=3.0, fov_deg=60.0, device=device, batch_size=B)

        kp_2d = torch.rand(B, V, J, 2, device=device)
        visibility = torch.zeros(B, V, J, device=device)  # all invisible

        triangulated, valid_mask = triangulator._triangulate_joints_dlt(
            kp_2d, visibility,
            fov_per_view, R_per_view, T_per_view,
            aspect_ratio_per_view=aspect_ratios,
        )

        assert not valid_mask.any(), "All-zero visibility → nothing valid"
        assert torch.isfinite(triangulated).all(), (
            "Triangulated values should be finite even for invisible joints "
            "(damped solve should prevent NaN)"
        )

    def test_output_shapes(self, triangulator, device):
        """Verify output tensor shapes match the documented API."""
        B, V, J = 3, 5, 8

        fov_per_view, R_per_view, T_per_view, aspect_ratios = \
            _make_cameras_around_origin(V, radius=3.0, fov_deg=60.0, device=device, batch_size=B)

        kp_2d = torch.rand(B, V, J, 2, device=device)
        visibility = torch.ones(B, V, J, device=device)

        triangulated, valid_mask = triangulator._triangulate_joints_dlt(
            kp_2d, visibility,
            fov_per_view, R_per_view, T_per_view,
            aspect_ratio_per_view=aspect_ratios,
        )

        assert triangulated.shape == (B, J, 3)
        assert valid_mask.shape == (B, J)
        assert valid_mask.dtype == torch.bool
