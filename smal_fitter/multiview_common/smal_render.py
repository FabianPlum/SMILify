"""SMAL forward + PyTorch3D mesh/silhouette rendering for the dataset viewer.

Thin wrapper around `smal_model.smal_torch.SMAL` (the SMAL/SMIL forward
model) and `smal_fitter.p3d_renderer.Renderer` (the existing PyTorch3D
silhouette/colour renderer used by `SMALFitter`). The viewer uses this to:

- Forward (betas, global_rot, joint_rot, trans) from an HDF5 sample into
  posed vertices + joints.
- Add a Mesh3d trace + posed-joint scatter to the Plotly 3D viewer.
- Project posed joints through each stored camera for a per-view overlay
  (cheap — pure numpy).
- Rasterise the posed mesh through each stored camera into a silhouette
  mask and alpha-blend onto the input image (slow — gated behind a
  button in the viewer).

Conventions
-----------

Stored cameras are in OpenCV column-vector form (see
multiview_common/canonical_frame.py). The SLEAPMultiViewDataset reader's
`_sleap_to_pytorch3d_camera` converts to PyTorch3D row-vector form
(`X_cam = X_world @ R_p3d + T_p3d`, `fov_y` in degrees, `aspect_ratio`
for non-square pixels). We use the SAME conversion here so silhouette
overlays land on the same anatomy as the stored 2D keypoints.

The SMAL model holds load-bearing global state in `config.SMAL_FILE`
(via `apply_smal_file_override`). Callers must apply the override
BEFORE instantiating SMAL; this wrapper does that in the constructor.
Loading a SECOND pkl with a different path mutates the same global —
fine for the viewer (one dataset at a time) but be aware.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class PosedSample:
    """The output of a single SMAL forward + translate call."""
    vertices_world: np.ndarray   # (V, 3)
    joints_world: np.ndarray     # (J, 3)
    faces: np.ndarray            # (F, 3) int


class SMALRendererWrapper:
    """SMAL forward + (optional) PyTorch3D silhouette rendering.

    Construct once per SMAL pkl path. `forward(...)` is cheap (CPU
    matmul). `render_silhouette(...)` is the expensive PyTorch3D call;
    callers should gate it behind a button.
    """

    def __init__(self, smal_file: str, render_size: int, device: str = "cpu",
                 shape_family: Optional[int] = None) -> None:
        # The SMAL constructor reads `config.SMAL_FILE` (global). Apply the
        # override BEFORE the import-chain touches config.dd.

        # Defer these until paths are set.
        from smal_fitter.neuralSMIL.configs.config_utils import apply_smal_file_override  # noqa: E402
        apply_smal_file_override(smal_file, shape_family=shape_family)

        import config  # noqa: E402  — re-imported after override so attrs reflect the pkl
        from smal_model.smal_torch import SMAL  # noqa: E402
        from smal_fitter.p3d_renderer import Renderer  # noqa: E402

        self.device = torch.device(device)
        self.smal = SMAL(self.device, shape_family_id=shape_family if shape_family is not None else -1)
        self.smal.eval()
        # The Renderer constructor uses a single integer image_size (square
        # rendering target). Calibration frames may be non-square but the
        # FoVPerspectiveCameras `aspect_ratio` parameter handles that; the
        # rendered silhouette is square and we stretch it to the calibration
        # extent at overlay time.
        self.render_size = int(render_size)
        self.renderer = Renderer(self.render_size, self.device)
        self.config = config
        self.smal_file = str(smal_file)

    # ------------------------------------------------------------------
    # SMAL forward.
    # ------------------------------------------------------------------

    def forward(
        self,
        betas: np.ndarray,           # (n_betas,)
        global_rot: np.ndarray,      # (3,) axis-angle
        joint_rot: np.ndarray,       # (n_pose+1, 3) or (n_pose, 3) axis-angle (we strip the leading row if it matches global_rot)
        trans: np.ndarray,           # (3,)
        propagate_scaling: bool = False,
    ) -> PosedSample:
        """Run SMAL once on CPU and return world-frame vertices + joints."""
        betas_t = torch.as_tensor(betas, dtype=torch.float32, device=self.device).reshape(1, -1)
        global_t = torch.as_tensor(global_rot, dtype=torch.float32, device=self.device).reshape(1, 1, 3)
        joints_t = torch.as_tensor(joint_rot, dtype=torch.float32, device=self.device).reshape(-1, 3)

        # The HDF5 stores joint_rot with the global rotation already pulled
        # out into its own field — but some preprocessors stash the global
        # rotation as the leading row of joint_rot. Detect by row count:
        # n_joints expected by SMAL is `config.N_POSE + 1` (root + the
        # posable joints). If we already have that many rows, the leading
        # one IS the global rotation; strip it and we'll re-prepend
        # `global_rot`. Otherwise concat as-is.
        n_pose = int(self.config.N_POSE) if hasattr(self.config, "N_POSE") else (joints_t.shape[0] - 1 if joints_t.shape[0] >= 1 else 0)
        if joints_t.shape[0] == n_pose + 1:
            # Drop the existing leading row, replace with the explicit global.
            joints_t = joints_t[1:]
        theta = torch.cat([global_t, joints_t.unsqueeze(0)], dim=1)  # (1, n_pose+1, 3)

        with torch.no_grad():
            verts, joints, _, _ = self.smal(
                betas_t, theta, propagate_scaling=propagate_scaling,
            )

        # Recenter at the posed root joint before adding trans. This matches
        # the use_ue_scaling=True and `allow_mesh_scaling` branches in
        # `multiview_smil_regressor` (lines ~2147-2156), which always do
        #     (verts - root_joint) * scale + trans
        # so the body root lands at `trans`. The HDF5 stores `parameters/trans`
        # as the GT root joint position in canonical-frame world coords, so
        # not recentering puts the mesh at `rest_root + trans` and produces a
        # consistent offset (e.g. the SMAL Falkner-conv mouse pkl has rest
        # root at (0, 0.72, 0.24) — that exact offset shows up on every view
        # without this recenter).
        # The `verts + trans` branch in the trainer is only reachable when the
        # model is trained to predict `trans` that already compensates for
        # `rest_root` — irrelevant for the viewer, which always consumes the
        # raw GT params from disk.
        trans_t = torch.as_tensor(trans, dtype=torch.float32, device=self.device).reshape(1, 1, 3)
        root_joint = joints[:, 0:1, :]  # (1, 1, 3)
        verts = (verts - root_joint) + trans_t
        joints = (joints - root_joint) + trans_t

        faces = self.smal.faces.detach().cpu().numpy().astype(np.int64)
        return PosedSample(
            vertices_world=verts[0].detach().cpu().numpy().astype(np.float32),
            joints_world=joints[0].detach().cpu().numpy().astype(np.float32),
            faces=faces,
        )

    # ------------------------------------------------------------------
    # Silhouette rendering (expensive — gate behind a UI button).
    # ------------------------------------------------------------------

    def render_silhouette(
        self,
        verts_world: np.ndarray,     # (V, 3)
        faces: np.ndarray,           # (F, 3) int
        R_cv: np.ndarray, t_cv: np.ndarray, K: np.ndarray,
        image_size_wh: np.ndarray,   # (W, H) of the calibration frame
    ) -> np.ndarray:
        """Rasterise the posed mesh through one OpenCV camera and return
        a (render_size, render_size) silhouette mask in [0, 1] (float32).

        The renderer is square; for non-square calibration frames the
        caller stretches the result to the calibration extent before
        overlay. The aspect_ratio plumbed into FoVPerspectiveCameras
        handles the non-square camera intrinsics correctly.

        The OpenCV column-vec -> PyTorch3D row-vec conversion is the
        same one `SLEAPMultiViewDataset._sleap_to_pytorch3d_camera`
        does at trainer load-time; we import that to keep the viewer's
        projection convention byte-equivalent to the trainer's.
        """
        # Imported lazily so module-import doesn't pull in h5py + the
        # whole SLEAPMultiViewDataset class graph when callers only
        # need `forward(...)`.
        from smal_fitter.sleap_data.sleap_multiview_dataset import SLEAPMultiViewDataset
        R_p3d, T_p3d, fov_y, aspect_ratio = SLEAPMultiViewDataset._sleap_to_pytorch3d_camera(
            R_cv, t_cv, K, image_size_wh
        )
        R_t = torch.from_numpy(R_p3d).unsqueeze(0).to(self.device)
        T_t = torch.from_numpy(T_p3d).unsqueeze(0).to(self.device)
        fov_t = torch.tensor([fov_y], dtype=torch.float32, device=self.device)
        self.renderer.set_camera_parameters(R_t, T_t, fov_t, aspect_ratio=aspect_ratio)

        verts_t = torch.as_tensor(verts_world, dtype=torch.float32, device=self.device).unsqueeze(0)
        faces_t = torch.as_tensor(faces, dtype=torch.long, device=self.device).unsqueeze(0)
        # Renderer.forward needs a `points` arg even when we don't care;
        # pass the verts again as a placeholder.
        with torch.no_grad():
            sil, _ = self.renderer(verts_t, verts_t, faces_t)
        # sil: (1, 1, H, W) in [0, 1] (the alpha from SoftSilhouetteShader)
        sil_np = sil[0, 0].detach().cpu().numpy().astype(np.float32)
        return np.clip(sil_np, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Module-level helper for the viewer.
# ---------------------------------------------------------------------------


def overlay_silhouette(
    canvas_rgb: np.ndarray,         # (H_calib, W_calib, 3) uint8
    silhouette: np.ndarray,         # (S, S) float32 in [0, 1]
    image_size_wh: np.ndarray,      # (W_calib, H_calib)
    colour: Tuple[int, int, int] = (255, 60, 60),
    alpha: float = 0.45,
) -> np.ndarray:
    """Alpha-blend a square silhouette mask onto a (typically non-square)
    canvas. Stretches the silhouette to the canvas's calibration extent."""
    import cv2
    W_calib, H_calib = int(image_size_wh[0]), int(image_size_wh[1])
    if silhouette.shape != (H_calib, W_calib):
        sil_full = cv2.resize(silhouette, (W_calib, H_calib), interpolation=cv2.INTER_LINEAR)
    else:
        sil_full = silhouette
    sil_full = np.clip(sil_full, 0.0, 1.0)[..., None]   # (H, W, 1)
    colour_img = np.zeros_like(canvas_rgb, dtype=np.float32)
    colour_img[..., 0] = colour[0]
    colour_img[..., 1] = colour[1]
    colour_img[..., 2] = colour[2]
    blended = (canvas_rgb.astype(np.float32) * (1.0 - alpha * sil_full)
               + colour_img * (alpha * sil_full))
    return np.clip(blended, 0, 255).astype(np.uint8)
