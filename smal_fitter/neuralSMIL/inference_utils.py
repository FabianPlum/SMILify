"""
Helpers shared by run_singleview_inference.py and run_multiview_inference.py.

Kept free of model / pytorch3d imports so the helpers can be imported and
tested on their own.
"""

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch


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
