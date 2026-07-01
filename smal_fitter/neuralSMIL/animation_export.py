"""Export recorded SMIL inference parameters to an AMASS-compatible .npz + sidecar .json.

Used by run_singleview_inference.py and run_multiview_inference.py to persist the
full parametric output of the neural regressors (pose, trans, betas, per-joint
scale/trans, cameras) alongside the existing MP4 previews. The on-disk schema is
documented in .claude/plans/iterative-kindling-swan.md (Phase 1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


SCHEMA_VERSION = "1.1"


def rotation_6d_to_axis_angle(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to axis-angle.

    Mirrors `smil_image_regressor.rotation_6d_to_axis_angle` but avoids importing
    that module (and its heavy dependency chain) for consumers that only need the
    conversion. Uses pytorch3d's transforms directly.
    """
    from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

    return matrix_to_axis_angle(rotation_6d_to_matrix(d6))


def _to_numpy(t: Any) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().float().numpy()
    return np.asarray(t, dtype=np.float32)


class AnimationRecorder:
    """Accumulates per-frame predicted_params dicts and writes .npz + .json.

    Each `record(predicted_params)` call appends one frame. Batch dim of 1 is
    assumed — callers running batched inference must loop and record per batch
    element. Rotations are normalised to axis-angle on write regardless of the
    inbound `rotation_representation`.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        rotation_representation: str,
        n_joints: int,
        n_betas: int,
        joint_names: List[str],
        parents: List[int],
        fps: float,
        static_joint_locs: bool,
        ignore_hardcoded_body: bool,
        source_checkpoint: Optional[str] = None,
        source_input: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        if rotation_representation not in ("axis_angle", "6d"):
            raise ValueError(f"rotation_representation must be 'axis_angle' or '6d', got {rotation_representation!r}")
        self.output_path = Path(output_path)
        self.rotation_representation = rotation_representation
        self.n_joints = int(n_joints)
        self.n_betas = int(n_betas)
        self.joint_names = [str(n) for n in joint_names]
        self.parents = [int(p) for p in parents]
        self.fps = float(fps)
        self.static_joint_locs = bool(static_joint_locs)
        self.ignore_hardcoded_body = bool(ignore_hardcoded_body)
        self.source_checkpoint = source_checkpoint
        self.source_input = source_input
        self.model_id = model_id

        self._poses: List[np.ndarray] = []
        self._trans: List[np.ndarray] = []
        self._betas: List[np.ndarray] = []
        self._log_beta_scales: List[np.ndarray] = []
        self._betas_trans: List[np.ndarray] = []
        self._mesh_scale: List[np.ndarray] = []

        # Per-frame camera buffers (singleview). Multi-view cameras are stored
        # via set_cameras() because they're static per view for a given clip.
        self._cam_rot: List[np.ndarray] = []
        self._cam_trans: List[np.ndarray] = []
        self._fov: List[np.ndarray] = []

        self._cameras_sidecar: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ record

    def _rot_to_aa(self, rot: torch.Tensor) -> torch.Tensor:
        if self.rotation_representation == "6d":
            return rotation_6d_to_axis_angle(rot)
        return rot

    def record(self, predicted_params: Dict[str, Any]) -> None:
        global_rot = predicted_params["global_rot"]
        joint_rot = predicted_params["joint_rot"]

        if isinstance(global_rot, torch.Tensor):
            global_rot = global_rot.detach().cpu()
        if isinstance(joint_rot, torch.Tensor):
            joint_rot = joint_rot.detach().cpu()

        global_rot_aa = self._rot_to_aa(global_rot)
        joint_rot_aa = self._rot_to_aa(joint_rot)

        if global_rot_aa.dim() == 2:  # (B, 3) -> (B, 1, 3)
            global_rot_aa = global_rot_aa.unsqueeze(1)
        poses = torch.cat([global_rot_aa, joint_rot_aa], dim=1)  # (B, N_JOINTS, 3)

        self._poses.append(_to_numpy(poses[0]))
        self._trans.append(_to_numpy(predicted_params["trans"][0]))
        self._betas.append(_to_numpy(predicted_params["betas"][0]))

        if predicted_params.get("log_beta_scales", None) is not None:
            self._log_beta_scales.append(_to_numpy(predicted_params["log_beta_scales"][0]))
        if predicted_params.get("betas_trans", None) is not None:
            self._betas_trans.append(_to_numpy(predicted_params["betas_trans"][0]))
        if predicted_params.get("mesh_scale", None) is not None:
            self._mesh_scale.append(_to_numpy(predicted_params["mesh_scale"][0]).reshape(-1))

        # Singleview cameras (one per frame); averaged on write.
        if predicted_params.get("cam_rot", None) is not None:
            self._cam_rot.append(_to_numpy(predicted_params["cam_rot"][0]))
        if predicted_params.get("cam_trans", None) is not None:
            self._cam_trans.append(_to_numpy(predicted_params["cam_trans"][0]))
        if predicted_params.get("fov", None) is not None:
            self._fov.append(_to_numpy(predicted_params["fov"][0]))

    def set_cameras(self, cameras: List[Dict[str, Any]]) -> None:
        """Explicitly set the cameras block of the sidecar (multi-view path).

        Each entry: {"view_name": str, "R": 3x3 list, "t": 3 list, "fov": float}.
        """
        self._cameras_sidecar = list(cameras)

    def num_frames(self) -> int:
        return len(self._poses)

    # ------------------------------------------------------------------- write

    def _build_averaged_singleview_camera(self) -> List[Dict[str, Any]]:
        if not self._cam_rot:
            return []
        R = np.stack(self._cam_rot).mean(axis=0)
        t = np.stack(self._cam_trans).mean(axis=0) if self._cam_trans else np.zeros(3, np.float32)
        fov = float(np.mean(self._fov)) if self._fov else 0.0
        return [
            {
                "view_name": "view_0",
                "R": R.tolist(),
                "t": t.flatten().tolist(),
                "fov": fov,
            }
        ]

    def write(self) -> Dict[str, Path]:
        if not self._poses:
            raise RuntimeError("AnimationRecorder has no frames to write.")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        npz_path = self.output_path.with_suffix(".npz")
        json_path = self.output_path.with_suffix(".json")

        poses = np.stack(self._poses).astype(np.float32)
        trans = np.stack(self._trans).astype(np.float32)
        betas_per_frame = np.stack(self._betas).astype(np.float32)
        betas_avg = betas_per_frame.mean(axis=0).astype(np.float32)

        payload: Dict[str, Any] = {
            "poses": poses,
            "trans": trans,
            "betas": betas_avg,
            "betas_per_frame": betas_per_frame,
            "fps": np.float32(self.fps),
        }
        if self._log_beta_scales:
            payload["log_beta_scales"] = np.stack(self._log_beta_scales).astype(np.float32)
        if self._betas_trans:
            payload["betas_trans"] = np.stack(self._betas_trans).astype(np.float32)
        if self._mesh_scale:
            # Store as (F,) — global isotropic scale applied around the root joint:
            # rendered_v = (v - J0) * mesh_scale + trans. Importers must subtract the
            # rest-pose root joint position before scaling and re-apply the trans.
            payload["mesh_scale"] = np.stack(self._mesh_scale).astype(np.float32).reshape(-1)

        np.savez(npz_path, **payload)

        cameras = self._cameras_sidecar or self._build_averaged_singleview_camera()

        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "model_id": self.model_id,
            "source_checkpoint": self.source_checkpoint,
            "source_input": self.source_input,
            "n_frames": int(poses.shape[0]),
            "n_joints": self.n_joints,
            "n_betas": self.n_betas,
            "joint_names": self.joint_names,
            "parents": self.parents,
            "rotation_representation": "axis_angle",
            "root_joint_index": 0,
            "static_joint_locs": self.static_joint_locs,
            "ignore_hardcoded_body": self.ignore_hardcoded_body,
            "fps": self.fps,
            "cameras": cameras,
        }
        with open(json_path, "w") as f:
            json.dump(sidecar, f, indent=2)

        return {"npz": npz_path, "json": json_path}


def build_multiview_cameras(
    all_predictions: List,
    camera_order: List[str],
) -> List[Dict[str, Any]]:
    """Average `cam_{rot,trans,fov}_per_view` across frames into a static sidecar block.

    Args:
        all_predictions: list of (global_idx, predicted_params_cpu) — as produced by
            run_multiview_inference.run_inference_phase.
        camera_order: dataset.get_canonical_camera_order(), names per view index.
    """
    if not all_predictions:
        return []

    first = all_predictions[0][1]
    if "cam_rot_per_view" not in first:
        return []

    n_views = len(first["cam_rot_per_view"])
    R_sum = [np.zeros((3, 3), np.float64) for _ in range(n_views)]
    t_sum = [np.zeros(3, np.float64) for _ in range(n_views)]
    fov_sum = [0.0] * n_views
    counts = [0] * n_views

    for _, params in all_predictions:
        if "cam_rot_per_view" not in params:
            continue
        for v in range(n_views):
            R = _to_numpy(params["cam_rot_per_view"][v][0]).reshape(3, 3)
            t = _to_numpy(params["cam_trans_per_view"][v][0]).reshape(-1)
            fov = float(_to_numpy(params["fov_per_view"][v][0]).reshape(-1)[0])
            R_sum[v] += R
            t_sum[v] += t
            fov_sum[v] += fov
            counts[v] += 1

    cameras: List[Dict[str, Any]] = []
    for v in range(n_views):
        if counts[v] == 0:
            continue
        name = camera_order[v] if v < len(camera_order) else f"view_{v}"
        cameras.append(
            {
                "view_name": str(name),
                "R": (R_sum[v] / counts[v]).tolist(),
                "t": (t_sum[v] / counts[v]).tolist(),
                "fov": fov_sum[v] / counts[v],
            }
        )
    return cameras


def build_recorder_from_config(
    output_path: Union[str, Path],
    rotation_representation: str,
    fps: float,
    source_checkpoint: Optional[str] = None,
    source_input: Optional[str] = None,
    model_id: Optional[str] = None,
) -> AnimationRecorder:
    """Convenience constructor that pulls joint metadata from the global `config`."""
    import config as smil_config  # local import to keep this module importable in tests

    dd = smil_config.dd
    joint_names = list(dd.get("J_names", [f"J_{i}" for i in range(smil_config.N_POSE + 1)]))
    parents = list(np.asarray(dd["kintree_table"][0]).astype(int).tolist())
    n_joints = smil_config.N_POSE + 1  # root + per-joint pose entries

    return AnimationRecorder(
        output_path=output_path,
        rotation_representation=rotation_representation,
        n_joints=n_joints,
        n_betas=smil_config.N_BETAS,
        joint_names=joint_names,
        parents=parents,
        fps=fps,
        static_joint_locs=bool(getattr(smil_config, "STATIC_JOINT_LOCATIONS", False)),
        ignore_hardcoded_body=bool(getattr(smil_config, "ignore_hardcoded_body", False)),
        source_checkpoint=source_checkpoint,
        source_input=source_input,
        model_id=model_id,
    )
