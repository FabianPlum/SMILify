"""Tests for Renderer.orbit_view, the second-view placement in SMALFitter collages (#106).

The old second view moved the mesh to the world origin and rotated it 180 deg about y.
That is the camera itself for camera-centric checkpoints, so the panel rendered from
inside the mesh. orbit_view works in camera space: for any camera, the orbited subject
must end up in front of the near plane and inside the (narrower) field of view.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch3d.renderer import look_at_view_transform  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from smal_fitter.p3d_renderer import Renderer  # noqa: E402


def stick(centre, length):
    """Elongated point cloud (body along x), roughly a stick insect of the given length."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.5, 0.5, size=(400, 3)) * np.array([length, 0.15 * length, 0.1 * length])
    return torch.tensor(pts + np.asarray(centre), dtype=torch.float32).unsqueeze(0)


def camera_centric_renderer(aspect=None):
    r = Renderer(64, "cpu")
    r.set_camera_parameters(R=torch.eye(3)[None], T=torch.zeros(1, 3), fov=torch.tensor([35.33]), aspect_ratio=aspect)
    return r


def model_centric_renderer():
    r = Renderer(64, "cpu")
    R, T = look_at_view_transform(2.7, 30.0, 45.0)
    r.set_camera_parameters(R=R, T=T, fov=torch.tensor([60.0]))
    return r


def assert_in_front_and_in_frame(renderer, verts, fill=0.8):
    view = renderer.cameras.get_world_to_view_transform().transform_points(verts)
    assert (view[..., 2] > renderer.DEFAULT_ZNEAR).all()
    ndc = renderer.cameras.transform_points(verts)[..., :2]
    assert ndc.abs().max() <= 1.0

    # Framing is defined by the bounding sphere (an end-on stick can still look small):
    # its angular radius is `fill` of the narrower half field of view.
    centre = view.mean(dim=1)
    radius = (view - centre[:, None]).norm(dim=-1).max()
    half_fov = torch.deg2rad(renderer.cameras.fov.reshape(-1)[0]) / 2
    half_fov = torch.atan(torch.tan(half_fov) * renderer.cameras.aspect_ratio.reshape(-1)[0].clamp(max=1.0))
    assert torch.isclose(torch.asin(radius / centre.norm()), fill * half_fov, rtol=1e-3)
    assert torch.allclose(centre[0, :2], torch.zeros(2), atol=1e-5)  # centred on the optical axis


@pytest.mark.parametrize(
    "make_renderer, verts",
    [
        (camera_centric_renderer, stick([0.003, 0.014, 0.18], 0.04)),  # SMILySTICKS frame: 18 cm in front
        (model_centric_renderer, stick([0.0, 0.0, 0.0], 1.0)),  # subject at the origin, camera 2.7 out
        (lambda: camera_centric_renderer(aspect=0.6), stick([0.003, 0.014, 0.18], 0.04)),  # non-square
    ],
    ids=["camera_centric", "model_centric", "non_square_aspect"],
)
@pytest.mark.parametrize("axis, angle", [("camera_x", 90.0), ("camera_y", 90.0), ("world_y", 180.0)])
def test_orbit_view_keeps_subject_in_front_and_in_frame(make_renderer, verts, axis, angle):
    renderer = make_renderer()
    orbit_verts, _ = renderer.orbit_view(verts, verts[:, :5], angle_deg=angle, axis=axis)
    assert_in_front_and_in_frame(renderer, orbit_verts)


def test_old_reverse_view_was_behind_the_camera_for_camera_centric():
    """Documents the bug: re-centring on the world origin puts the mesh on the camera."""
    renderer = camera_centric_renderer()
    verts = stick([0.003, 0.014, 0.18], 0.04)
    recentred = verts - verts.mean(dim=1, keepdim=True)
    depth = renderer.cameras.get_world_to_view_transform().transform_points(recentred)[..., 2]
    assert (depth < renderer.DEFAULT_ZNEAR).any()


def test_world_y_180_reproduces_the_old_turn_shape():
    """world_y / 180 deg rotates the subject like the old view; only its placement differs."""
    renderer = model_centric_renderer()
    verts = stick([0.0, 0.0, 0.0], 1.0)
    orbit_verts, _ = renderer.orbit_view(verts, verts[:, :5], angle_deg=180.0, axis="world_y")
    rot = torch.tensor(Rotation.from_euler("y", 180, degrees=True).as_matrix(), dtype=torch.float32)
    expected = (verts - verts[:, :1]) @ rot.T
    assert torch.allclose(orbit_verts - orbit_verts[:, :1], expected, atol=1e-4)


def test_camera_x_90_is_a_quarter_turn_in_camera_space():
    renderer = camera_centric_renderer()
    verts = stick([0.003, 0.014, 0.18], 0.04)
    orbit_verts, _ = renderer.orbit_view(verts, verts[:, :5])
    # camera-centric: world == camera space
    rot = torch.tensor(Rotation.from_euler("x", 90, degrees=True).as_matrix(), dtype=torch.float32)
    expected = (verts - verts[:, :1]) @ rot.T
    assert torch.allclose(orbit_verts - orbit_verts[:, :1], expected, atol=1e-5)


def test_unknown_axis_raises():
    with pytest.raises(ValueError, match="Unknown orbit axis"):
        camera_centric_renderer().orbit_view(stick([0, 0, 0.2], 0.04), torch.zeros(1, 1, 3), axis="z")
