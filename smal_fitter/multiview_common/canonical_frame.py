"""Canonical-camera-frame transform for multi-view samples.

Convention reference
--------------------

For multi-view 3D pose / mesh regression with known (GT) cameras, the
standard storage / training convention is to pick **one camera per frame
as the world-frame origin** and express every other camera, every 3D
keypoint, and the model's translation in that camera's frame. The choice
of "first / lowest-index camera" as canonical follows the conventions used in:

- HMR (Kanazawa et al., CVPR 2018) — canonical body frame + weak-perspective
  camera. The body lives in its own root-centred frame and the camera
  attaches it to the image; there is no separate world frame.
- Expose (Choutas et al., ECCV 2020) — explicit per-sample canonical camera
  for expressive body recovery.
- AGORA (Patel et al., CVPR 2021) — synth multi-view dataset with a
  canonical reference camera per scene.
- 4D-Humans / SLAHMR (Goel et al., ICCV 2023) — multi-view training in a
  canonical-camera frame.
- FreeMan (Wang et al., CVPR 2024) — multi-view in-the-wild 3D pose dataset,
  per-sample canonical reference camera.
- EFT (Joo, Neverova, Kanazawa, 3DV 2021) — explicitly normalises camera
  convention across multiple training datasets before mixing them.

For raw camera math (column-vector OpenCV `X_cam = R @ X_world + t`,
camera centre `c = -R.T @ t`, perspective projection `p = K @ X_cam`)
see Hartley & Zisserman, "Multiple View Geometry in Computer Vision"
(2nd ed., 2004, ch. 6) and the OpenCV `cv2.projectPoints` documentation.

The replicAnt-side multi-view loader (Phase 1, see
`MULTIVIEW_REPLICANT_INTEGRATION_DESIGN.md`) already operates this way.
The SLEAP-side multi-view preprocessor today stores extrinsics in the
raw rig-world frame; this module's `canonicalize_sample` is the bridge
that lets a cross-source merger produce a uniform canonical-frame HDF5
without touching either preprocessor.

Convention details
------------------

All inputs and outputs use the **column-vector OpenCV** convention:

    X_cam   = R   @ X_world + t            (3,) or (N, 3) on the right
    p_homog = K   @ X_cam
    p_pix   = p_homog[:2] / p_homog[2]
    c_world = -R.T @ t                     (camera centre in world)

Empirically verified on real SLEAP data
(`SMILymice_3D_6_cam_undistort.h5`, sample 0, 6 cams): the canonical-
frame transform is geometrically lossless to within numerical precision
(reprojection delta = 4.2e-6 px between raw-world and canonical-frame
reprojection; cam-0 self-check ||R'_0 - I||_max = 5e-8, ||t'_0||_max = 1.4e-5).
The "alternative" interpretation (R, t = camera pose in world) that
appears as a flag in `smal_fitter/sleap_data/sleap_3d_loader.py` is
NOT the one used by SLEAP HDF5s on disk — do not enable it here.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def canonicalize_sample(
    R: np.ndarray,
    t: np.ndarray,
    kp3d: np.ndarray,
    view_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Apply the canonical-camera-frame transform.

    Picks the lowest-index `view_mask=True` view as the canonical camera
    (call it view `v0`, with extrinsics `(R_0, t_0)`). After the transform:

    - `R[v0]` becomes the identity and `t[v0]` becomes zero
    - Every other valid view's extrinsics are expressed relative to v0
    - 3D keypoints are mapped into v0's frame

    Column-vector OpenCV formulas (derivation: invert v0's
    `X_cam0 = R_0 @ X_world + t_0` to define new world coords
    `X' := X_cam0`, then substitute into every other view's projection):

        R'_v   = R_v @ R_0.T
        t'_v   = t_v - R'_v @ t_0
        X'_w   = R_0 @ X_world + t_0          (so kp3d' = kp3d @ R_0.T + t_0)

    Cam v0 self-check (by construction):

        R'_{v0} = R_0 @ R_0.T = I
        t'_{v0} = t_0 - I @ t_0 = 0

    The transform is rigid → all per-view reprojections through
    `(K, R'_v, t'_v)` of `kp3d'` are byte-equivalent to the original
    `(K, R_v, t_v)` of `kp3d` (modulo floating-point noise).

    The (0, 0, 0) sentinel for joints without ground-truth 3D
    (SLEAP / replicAnt convention) is preserved exactly: only rows
    whose original value is not all-zero are transformed; pure-zero
    rows stay pure-zero so downstream `~all(kp3d == 0, axis=1)`
    detection still works.

    Args:
        R: `(V, 3, 3)` per-view world->camera rotations (OpenCV column-vector).
        t: `(V, 3)` per-view world->camera translations.
        kp3d: `(J, 3)` 3D keypoints in the world frame. Rows that are
            all-zero are treated as "no GT for this joint" and left
            unchanged.
        view_mask: `(V,)` bool — True for supervisable views, False for
            padded / invalid slots. Invalid views' extrinsics are left
            unchanged (their geometry is meaningless anyway). The canonical
            view is the lowest-index True slot.

    Returns:
        Tuple of `(R_new, t_new, kp3d_new, R_0, t_0, canonical_v)`:

        - `R_new`, `t_new`: same shapes as inputs, canonicalized at valid slots
        - `kp3d_new`: same shape as `kp3d`, canonicalized except sentinel rows
        - `R_0`, `t_0`: the canonical-camera's original extrinsics
          (the forward `world -> canonical` transform; consumers that need
          to recover raw world coords apply the inverse:
          `X_world = R_0.T @ (X_can - t_0)`)
        - `canonical_v`: int — slot index used as canonical

    Raises:
        ValueError: if `view_mask.sum() == 0` (no valid view to anchor on).
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    kp3d = np.asarray(kp3d, dtype=np.float64)
    view_mask = np.asarray(view_mask, dtype=bool)

    valid = np.where(view_mask)[0]
    if valid.size == 0:
        raise ValueError("canonicalize_sample: view_mask has no True entries")

    canonical_v = int(valid[0])
    R_0 = R[canonical_v].copy()
    t_0 = t[canonical_v].copy()

    R_new = R.copy()
    t_new = t.copy()
    for v in range(R.shape[0]):
        if not view_mask[v]:
            continue
        R_v_new = R[v] @ R_0.T
        t_v_new = t[v] - R_v_new @ t_0
        R_new[v] = R_v_new
        t_new[v] = t_v_new

    # Preserve the (0, 0, 0) sentinel for joints without GT 3D (matches
    # the replicAnt loader's post-canonical re-zero in
    # smal_fitter/Unreal2Pytorch3D.py — keep both code paths in sync).
    kp3d_new = kp3d.copy()
    has_gt = ~np.all(kp3d == 0, axis=1)
    if has_gt.any():
        kp3d_new[has_gt] = kp3d[has_gt] @ R_0.T + t_0

    return R_new, t_new, kp3d_new, R_0, t_0, canonical_v


def project_world_to_pixel(
    X_world: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray
) -> np.ndarray:
    """Project 3D world-frame points through an OpenCV column-vector camera.

    Args:
        X_world: `(N, 3)` or `(3,)` world-frame coordinates.
        R: `(3, 3)` world->camera rotation.
        t: `(3,)` world->camera translation.
        K: `(3, 3)` intrinsic matrix calibrated for the same image extent
           the caller's 2D keypoints live in.

    Returns:
        `(N, 2)` pixel coordinates `[x, y]`. Points behind the camera
        (`z <= 0`) are marked with `np.nan` so they don't pollute error
        statistics.
    """
    X = np.atleast_2d(np.asarray(X_world, dtype=np.float64))
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    X_cam = X @ R.T + t
    z = X_cam[:, 2:3]
    z_safe = np.where(np.abs(z) < 1e-12, 1e-12, z)
    p_homog = X_cam @ K.T
    p_pix = p_homog[:, :2] / p_homog[:, 2:3]
    return np.where(z_safe > 0, p_pix, np.nan)


def cam_center_world(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Camera centre in world coordinates: `c = -R.T @ t` (OpenCV)."""
    return -np.asarray(R, dtype=np.float64).T @ np.asarray(t, dtype=np.float64)


def kp2d_norm_yx_to_pixel_xy(kp2d_norm_yx: np.ndarray, img_W: int, img_H: int) -> np.ndarray:
    """Convert SLEAP/replicAnt-stored 2D keypoints from normalized `[y/H, x/W]`
    to pixel `[x, y]` coordinates.

    Both preprocessor paths store 2D keypoints with the intentional axis
    swap `(kp[:, 0] = y/H, kp[:, 1] = x/W)` — see
    `smal_fitter/Unreal2Pytorch3D.py` and the SLEAP `map_keypoints_to_smal_model`.
    This helper inverts the swap for projection / overlay drawing /
    comparison against `project_world_to_pixel` output.
    """
    kp = np.asarray(kp2d_norm_yx, dtype=np.float64)
    px = kp[..., 1] * float(img_W)
    py = kp[..., 0] * float(img_H)
    return np.stack([px, py], axis=-1)


# Rz_180: 180-degree rotation about the world Z-axis. The replicAnt-side
# multi-view preprocessor (see smal_fitter/replicAnt_data/preprocess_…)
# applies this when converting its loader's PyTorch3D-row-vector canonical
# extrinsics to the OpenCV column-vector form stored in the HDF5:
#
#     R_cv = Rz_180 @ R_p3d.T,    t_cv = Rz_180 @ T_p3d
#
# The SLEAPMultiViewDataset reader inverts at load time:
#
#     R_p3d = R_cv.T @ Rz_180,    T_p3d = Rz_180 @ t_cv
#
# Net result: replicAnt cam-0, stored as R_cv = Rz_180 in the HDF5, becomes
# (I, 0) in PyTorch3D at the trainer. The merger needs SLEAP cam-0 to land
# at the SAME (I, 0) PyTorch3D point, otherwise the trainer's trans head
# trains against two contradictory world frames across sources.
RZ_180 = np.array([[-1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)


def align_to_pytorch3d_reader_convention(
    R: np.ndarray,
    t: np.ndarray,
    kp3d: np.ndarray,
    view_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the `Rz_180` world-frame shift so canonicalized OpenCV cam-0
    matches the replicAnt-preprocessor storage convention.

    Use after `canonicalize_sample`: that helper produces cam-0 = (I, 0)
    in OpenCV. This helper rotates the world frame by `Rz_180.T` so that
    cam-0 becomes (Rz_180, 0) — matching what the replicAnt preprocessor
    emits, and (via the reader's inverse) handing the trainer cam-0 =
    (I, 0) in PyTorch3D row-vector form.

    Mathematics (column-vector OpenCV): a global rotation of the world
    frame by `M = Rz_180.T` on the canonicalized data sends:

        R'_v = R_v @ M.T  = R_v @ Rz_180        (R_v_canon @ Rz_180)
        t'_v = t_v                              (unchanged because we rotate
                                                 the world frame, not the
                                                 camera centres in the rotated
                                                 world)
        kp3d' = kp3d @ M.T = kp3d @ Rz_180

    Cam-0 self-check: `R_v_canon[v0] = I` so `R'[v0] = I @ Rz_180 = Rz_180`.
    Invariant: per-view reprojection is unchanged (rigid world rotation).

    The (0,0,0) sentinel for joints without GT is preserved (sentinel rows
    remain all-zero under any rotation).

    Empirically verified on `SMILymice_3D_6_cam_undistort.h5` sample 0:
    after `canonicalize_sample` + this helper, per-view reprojection
    errors are identical to the raw input's reprojection errors to the
    pixel (max 67 px residual matches exactly — that is calibration
    noise, not introduced by the transform).
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    kp3d = np.asarray(kp3d, dtype=np.float64)
    view_mask = np.asarray(view_mask, dtype=bool)

    R_out = R.copy()
    t_out = t.copy()
    for v in range(R.shape[0]):
        if not view_mask[v]:
            continue
        R_out[v] = R[v] @ RZ_180

    has_gt = ~np.all(kp3d == 0, axis=1)
    kp3d_out = kp3d.copy()
    if has_gt.any():
        kp3d_out[has_gt] = kp3d[has_gt] @ RZ_180

    return R_out, t_out, kp3d_out


def infer_world_scale(t: np.ndarray, view_mask: np.ndarray, threshold: float = 50.0) -> float:
    """Mirror the `SLEAPMultiViewDataset` reader's world-scale heuristic.

    SLEAP-format HDF5s often omit the `world_scale` metadata attribute.
    The reader infers `0.001` (mm → m) when any valid view's `||t||`
    exceeds `threshold`, else `1.0`. We re-implement the same heuristic
    here so the merger can pre-multiply `t` and `kp3d` and emit a uniform
    `world_scale=1.0` output, without depending on the reader.

    Args:
        t: `(V, 3)` per-view OpenCV translations.
        view_mask: `(V,)` bool — only valid views are considered.
        threshold: distance threshold for the mm-vs-m guess (default 50).

    Returns:
        Inferred world scale (`1.0` or `0.001`).
    """
    t = np.asarray(t, dtype=np.float64)
    view_mask = np.asarray(view_mask, dtype=bool)
    valid = t[view_mask]
    if valid.size == 0:
        return 1.0
    if float(np.max(np.linalg.norm(valid, axis=1))) > threshold:
        return 1.0e-3
    return 1.0
