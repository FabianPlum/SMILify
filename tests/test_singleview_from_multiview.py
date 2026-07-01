"""Tests for single-view-from-multiview camera-centric sampling.

The single-view-from-multiview sampler re-expresses ONE calibrated camera as the
world origin (camera-at-origin): the sampled view's camera becomes the fixed
PyTorch3D identity camera, and the shared 3D keypoints (and, for synthetic data,
the model root pose) are transformed into that camera's frame. This is the
`recanonicalize_single_view` step in `smal_fitter/multiview_common/canonical_frame.py`.

The load-bearing property — the whole reason the convention is safe — is that the
transform is *rigid*, so it must be **reprojection-invariant**: projecting the
re-canonicalised 3D keypoints through the re-canonicalised (identity) camera must
land on exactly the same pixels as the original `(K, R, t)` reprojection of the
original 3D keypoints. If that holds, any 2D-reprojection / 3D error computed in
the camera-centric frame is identical to the one computed in the original frame,
so nothing is lost by re-anchoring.

These tests build a fully synthetic multi-camera sample (known cameras, known 3D
points, known 2D projections) so the invariant can be checked to numerical
precision without any dataset on disk.
"""

import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports — canonical_frame has no heavy (config / SMAL model) dependency.
# ---------------------------------------------------------------------------
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from smal_fitter.multiview_common.canonical_frame import (  # noqa: E402
    RZ_180,
    project_world_to_pixel,
    recanonicalize_single_view,
)


# ---------------------------------------------------------------------------
# Synthetic multi-camera sample
# ---------------------------------------------------------------------------
def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly-random proper rotation (det = +1) via QR."""
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def _make_synthetic_sample(
    rng: np.random.Generator,
    n_views: int = 5,
    n_joints: int = 24,
    n_sentinel: int = 3,
    img: int = 512,
    focal: float = 800.0,
):
    """Build a synthetic (kp3d, K, sz, sentinel_idx, cams) multi-cam sample.

    3D keypoints are clustered near the origin; every camera is placed at a
    positive, dominant +z depth so all real joints land in front of it. A few
    joints are set to the `(0, 0, 0)` "no GT" sentinel used by the SLEAP /
    replicAnt convention.
    """
    kp3d = rng.uniform(-1.0, 1.0, size=(n_joints, 3)).astype(np.float64)
    sentinel_idx = rng.choice(n_joints, size=n_sentinel, replace=False)
    kp3d[sentinel_idx] = 0.0

    K = np.array(
        [[focal, 0.0, img / 2.0], [0.0, focal, img / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    sz = np.array([img, img], dtype=np.float64)

    cams = []
    for _ in range(n_views):
        R = _random_rotation(rng)
        # Dominant +z translation → world points near origin project in front.
        t = np.array(
            [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(8.0, 12.0)],
            dtype=np.float64,
        )
        cams.append((R, t))
    return kp3d, K, sz, sentinel_idx, cams


def _non_sentinel_mask(n_joints: int, sentinel_idx: np.ndarray) -> np.ndarray:
    mask = np.ones(n_joints, dtype=bool)
    mask[sentinel_idx] = False
    return mask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_reprojection_invariance_under_recanonicalization():
    """Re-canonicalising to any single view must NOT change the reprojection.

    This is the exact property requested for the single-view sampler: the
    reprojection of the re-canonicalised 3D through the re-canonicalised camera
    stays identical (to numerical precision) to the original 2D/3D projection.
    """
    rng = np.random.default_rng(20260701)
    kp3d, K, _sz, sentinel_idx, cams = _make_synthetic_sample(rng)
    real = _non_sentinel_mask(kp3d.shape[0], sentinel_idx)

    worst_px = 0.0
    for v, (R, t) in enumerate(cams):
        # Re-anchor the world onto this view (camera-at-origin).
        kp3d_view, R_cv_out, t_cv_out, _R0, _t0 = recanonicalize_single_view(R, t, kp3d)

        # Sentinel (0,0,0) rows land at z=0 in the canonical frame and are
        # returned as NaN by project_world_to_pixel; the div-by-zero is expected
        # and masked, so silence its benign warning here.
        with np.errstate(divide="ignore", invalid="ignore"):
            # Original "2D data": projection through the view's real camera.
            p_orig = project_world_to_pixel(kp3d, R, t, K)
            # Reproject the re-canonicalised 3D through the re-canonicalised camera.
            p_canon = project_world_to_pixel(kp3d_view, R_cv_out, t_cv_out, K)

        # Real joints must be finite and pixel-identical in both frames.
        assert not np.isnan(p_orig[real]).any(), f"view {v}: original projection has NaNs"
        assert not np.isnan(p_canon[real]).any(), f"view {v}: canonical projection has NaNs"
        assert np.allclose(p_orig[real], p_canon[real], atol=1e-6, rtol=0.0), (
            f"view {v}: reprojection changed after re-canonicalization"
        )

        worst_px = max(worst_px, float(np.max(np.abs(p_orig[real] - p_canon[real]))))

    # Sanity: the residual is numerical noise, not a real error.
    assert worst_px < 1e-6, f"max reprojection drift {worst_px} px exceeds tolerance"


def test_recanonicalized_camera_is_fixed_identity():
    """The re-canonicalised camera must be the fixed PyTorch3D identity camera.

    `recanonicalize_single_view` returns the aligned OpenCV camera `(Rz_180, 0)`;
    the reader's OpenCV→PyTorch3D conversion (`R_p3d = R_cv.T @ Rz_180`,
    `T_p3d = Rz_180 @ t_cv`, mirrored here) must therefore yield `(I, 0)`.
    """
    rng = np.random.default_rng(7)
    kp3d, _K, _sz, _sentinel_idx, cams = _make_synthetic_sample(rng)

    for v, (R, t) in enumerate(cams):
        _kp3d_view, R_cv_out, t_cv_out, _R0, _t0 = recanonicalize_single_view(R, t, kp3d)

        # Aligned OpenCV camera is exactly (Rz_180, 0), independent of the input view.
        assert np.allclose(R_cv_out, RZ_180, atol=1e-9), f"view {v}: aligned R != Rz_180"
        assert np.allclose(t_cv_out, 0.0, atol=1e-9), f"view {v}: aligned t != 0"

        # OpenCV → PyTorch3D (mirrors SLEAPMultiViewDataset._sleap_to_pytorch3d_camera).
        R_p3d = R_cv_out.T @ RZ_180
        T_p3d = RZ_180 @ t_cv_out
        assert np.allclose(R_p3d, np.eye(3), atol=1e-9), f"view {v}: PyTorch3D R != I"
        assert np.allclose(T_p3d, 0.0, atol=1e-9), f"view {v}: PyTorch3D T != 0"


def test_sentinel_joints_preserved():
    """`(0, 0, 0)` no-GT sentinels stay zero; real joints move into the frame."""
    rng = np.random.default_rng(11)
    kp3d, _K, _sz, sentinel_idx, cams = _make_synthetic_sample(rng)
    R, t = cams[2]

    kp3d_view, _R_cv_out, _t_cv_out, _R0, _t0 = recanonicalize_single_view(R, t, kp3d)

    assert np.all(kp3d_view[sentinel_idx] == 0.0), "sentinel joints were transformed"
    real = _non_sentinel_mask(kp3d.shape[0], sentinel_idx)
    assert np.any(kp3d_view[real] != 0.0), "real joints were not transformed"


def test_root_translation_matches_keypoint_transform():
    """The dataset's `M @ root_loc + b` closed form must equal the keypoint transform.

    In camera_centric mode the model's GT root location is moved into the view
    frame via `M = RZ_180 @ R_0`, `b = RZ_180 @ t_0`. That must agree exactly
    with treating the root as one more 3D keypoint through
    `recanonicalize_single_view`, otherwise the 3D-keypoint supervision and the
    root-translation supervision would live in different frames.
    """
    rng = np.random.default_rng(3)
    kp3d, _K, _sz, _sentinel_idx, cams = _make_synthetic_sample(rng)
    R, t = cams[1]

    root_loc = rng.uniform(-1.0, 1.0, size=3)  # non-zero → not a sentinel

    # Path A: transform the root as a keypoint.
    kp_view, _R_cv_out, _t_cv_out, R0, t0 = recanonicalize_single_view(R, t, root_loc.reshape(1, 3))

    # Path B: the closed form the dataset applies to root_loc.
    M = RZ_180 @ R0
    b = RZ_180 @ t0
    root_loc_view = M @ root_loc + b

    assert np.allclose(kp_view[0], root_loc_view, atol=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
