"""
Helpers shared by run_singleview_inference.py and run_multiview_inference.py.

Kept free of model / pytorch3d imports so the helpers can be imported and
tested on their own.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Default upper bound for render_resolution in video / image-folder inference.
# An explicit --render_resolution is used as given.
RENDER_RESOLUTION_CAP = 1024


def crop_to_network_view(frame: np.ndarray, transform_info: Optional[Dict[str, Any]]) -> np.ndarray:
    """Return the part of *frame* the network saw, before it was resized.

    ``centred`` and ``bbox_crop`` cut out ``crop_size`` (h, w) at ``crop_offset``
    (y, x), zero-padded where a box runs past the frame edge. ``default`` mode
    resizes the whole frame, so it (and a missing *transform_info*) returns the
    frame unchanged.
    """
    if not transform_info or transform_info.get("mode") not in ("centred", "bbox_crop"):
        return frame
    crop_h, crop_w = (int(round(v)) for v in transform_info["crop_size"])
    y0, x0 = (int(round(v)) for v in transform_info["crop_offset"])

    view = np.zeros((crop_h, crop_w) + frame.shape[2:], dtype=frame.dtype)
    fy0, fx0 = max(y0, 0), max(x0, 0)
    fy1, fx1 = min(y0 + crop_h, frame.shape[0]), min(x0 + crop_w, frame.shape[1])
    if fy1 > fy0 and fx1 > fx0:
        view[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0] = frame[fy0:fy1, fx0:fx1]
    return view


def resolve_render_resolution(requested: Optional[int], view_hw: Tuple[int, int]) -> int:
    """Square render size: *requested* if given, else the view size capped at RENDER_RESOLUTION_CAP."""
    if requested:
        return int(requested)
    return min(min(view_hw), RENDER_RESOLUTION_CAP)


class PredictionSmoother:
    """Temporal smoother that applies moving average over predicted parameters.

    Maintains a ring buffer of the last ``window_size`` predictions and returns
    their element-wise mean.  Metadata keys (non-tensor values and lists whose
    length varies across frames) are passed through from the latest frame.
    Per-view keys only occur in multi-view predictions.
    """

    _PER_VIEW_KEYS = {"fov_per_view", "cam_rot_per_view", "cam_trans_per_view"}
    _METADATA_KEYS = {"num_views", "view_mask", "camera_indices"}

    def __init__(self, window_size: int):
        self.window_size = window_size
        self._buffer: List[Dict[str, Any]] = []

    def __call__(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add *params* to the buffer and return the smoothed result."""
        if self.window_size <= 0:
            return params

        self._buffer.append(params)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

        if len(self._buffer) == 1:
            return params

        smoothed: Dict[str, Any] = {}
        for key in params:
            if key in self._METADATA_KEYS:
                smoothed[key] = params[key]
            elif key in self._PER_VIEW_KEYS:
                num_views = len(params[key])
                smoothed_list = []
                for v in range(num_views):
                    tensors = [buf[key][v] for buf in self._buffer if key in buf and v < len(buf[key])]
                    if tensors:
                        smoothed_list.append(torch.stack(tensors).mean(dim=0))
                    else:
                        smoothed_list.append(params[key][v])
                smoothed[key] = smoothed_list
            elif isinstance(params[key], torch.Tensor):
                tensors = [buf[key] for buf in self._buffer if key in buf]
                smoothed[key] = torch.stack(tensors).mean(dim=0)
            else:
                smoothed[key] = params[key]

        return smoothed


def params_to_cpu(params: Dict[str, Any]) -> Dict[str, Any]:
    """Detach and move all tensors in a predicted_params dict to CPU."""
    out: Dict[str, Any] = {}
    for key, val in params.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.detach().cpu()
        elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
            out[key] = [t.detach().cpu() for t in val]
        else:
            out[key] = val
    return out


def params_to_device(params: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Move all tensors in a predicted_params dict to *device*."""
    out: Dict[str, Any] = {}
    for key, val in params.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.to(device)
        elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
            out[key] = [t.to(device) for t in val]
        else:
            out[key] = val
    return out


def pad_or_resize(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Force *frame* to ``(width, height)`` so every video frame matches."""
    target_w, target_h = target_size
    if frame.shape[1] == target_w and frame.shape[0] == target_h:
        return frame
    if frame.shape[1] > target_w or frame.shape[0] > target_h:
        return cv2.resize(frame, (target_w, target_h))
    padded = np.ones((target_h, target_w, 3), dtype=np.uint8) * 40
    h = min(target_h, frame.shape[0])
    w = min(target_w, frame.shape[1])
    padded[:h, :w] = frame[:h, :w]
    return padded


class InMemoryImageExporter:
    """Capture SMALFitter's collage instead of writing it to disk."""

    def __init__(self):
        self.image = None

    def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces, img_idx=0, epoch=None):
        self.image = collage_np
