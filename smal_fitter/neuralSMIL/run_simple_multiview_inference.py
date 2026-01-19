#!/usr/bin/env python3
"""
Simple multi-view inference + visualization using a preprocessed SLEAP dataset.

This script mirrors the visualization logic used during training in
`train_multiview_regressor.py`, but runs over all samples in a preprocessed
multi-view HDF5 dataset and writes two videos in the current working directory:

  - "<DATASET>_multiview_inference.mp4" (multi-view grid visualization)
  - "<DATASET>_smultiview_first_camera_render.mp4" (single-view render for view 0)
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import cv2
import torch

from multiview_smil_regressor import create_multiview_regressor, MultiViewSMILImageRegressor
from smil_image_regressor import rotation_6d_to_axis_angle
from smal_fitter import SMALFitter
from sleap_data.sleap_multiview_dataset import SLEAPMultiViewDataset
import config


DEFAULT_CHECKPOINTS = [
    "multiview_checkpoints/best_model.pth",
    "multiview_checkpoints/final_model.pth",
]


def _find_default_checkpoint() -> Path:
    for rel_path in DEFAULT_CHECKPOINTS:
        path = Path(rel_path)
        if path.exists():
            return path
    return Path(DEFAULT_CHECKPOINTS[0])


def load_multiview_model_from_checkpoint(checkpoint_path: Path,
                                          device: str,
                                          max_views: int,
                                          canonical_camera_order: List[str]) -> MultiViewSMILImageRegressor:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    ckpt_config = checkpoint.get("config", {})

    backbone_name = ckpt_config.get("backbone_name", "vit_large_patch16_224")
    head_type = ckpt_config.get("head_type", "transformer_decoder")
    hidden_dim = ckpt_config.get("hidden_dim", 512)
    rotation_representation = ckpt_config.get("rotation_representation", "6d")
    scale_trans_mode = ckpt_config.get("scale_trans_mode", "separate")
    freeze_backbone = ckpt_config.get("freeze_backbone", True)
    use_unity_prior = ckpt_config.get("use_unity_prior", False)
    use_ue_scaling = ckpt_config.get("use_ue_scaling", False)
    allow_mesh_scaling = ckpt_config.get("allow_mesh_scaling", False)
    mesh_scale_init = ckpt_config.get("mesh_scale_init", 1.0)

    cross_attention_layers = ckpt_config.get("cross_attention_layers", 2)
    cross_attention_heads = ckpt_config.get("cross_attention_heads", 8)
    cross_attention_dropout = ckpt_config.get("cross_attention_dropout", 0.1)
    transformer_config = ckpt_config.get("transformer_config", {})

    input_resolution = 224 if backbone_name.startswith("vit") else 512

    model = create_multiview_regressor(
        device=device,
        batch_size=1,
        shape_family=config.SHAPE_FAMILY,
        use_unity_prior=use_unity_prior,
        max_views=max_views,
        canonical_camera_order=canonical_camera_order,
        cross_attention_layers=cross_attention_layers,
        cross_attention_heads=cross_attention_heads,
        cross_attention_dropout=cross_attention_dropout,
        backbone_name=backbone_name,
        freeze_backbone=freeze_backbone,
        head_type=head_type,
        hidden_dim=hidden_dim,
        rotation_representation=rotation_representation,
        scale_trans_mode=scale_trans_mode,
        use_ue_scaling=use_ue_scaling,
        input_resolution=input_resolution,
        transformer_config=transformer_config,
        allow_mesh_scaling=allow_mesh_scaling,
        mesh_scale_init=mesh_scale_init,
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    smal_optimization_params = [
        "global_rotation", "joint_rotations", "trans", "log_beta_scales",
        "betas_trans", "betas", "fov", "target_joints", "target_visibility"
    ]
    nn_state_dict = {
        k: v for k, v in state_dict.items()
        if not any(k == param or k.startswith(param + ".") for param in smal_optimization_params)
    }
    model.load_state_dict(nn_state_dict, strict=False)
    model.eval()
    return model


def create_multiview_visualization(model: MultiViewSMILImageRegressor,
                                   x_data: dict,
                                   y_data: dict,
                                   device: str) -> Optional[np.ndarray]:
    images = x_data.get("images", [])
    num_views = len(images)
    if num_views == 0:
        return None

    cam_indices = x_data.get("camera_indices", list(range(num_views)))
    if isinstance(cam_indices, np.ndarray):
        cam_indices = cam_indices.tolist()

    images_per_view = []
    for img in images:
        img_tensor = model.preprocess_image(img).to(device)  # (1, 3, H, W)
        images_per_view.append(img_tensor.squeeze(0))  # (3, H, W)

    images_tensors = [img.unsqueeze(0) for img in images_per_view]
    camera_indices_tensor = torch.tensor([cam_indices], device=device)
    view_mask = torch.ones(1, num_views, dtype=torch.bool, device=device)

    with torch.no_grad():
        predicted_params = model.forward_multiview(images_tensors, camera_indices_tensor, view_mask)

    img_size = 224
    margin = 5
    grid_width = num_views * img_size + (num_views + 1) * margin
    grid_height = 2 * img_size + 3 * margin

    canvas = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 40
    target_keypoints = y_data.get("keypoints_2d", None)
    target_visibility = y_data.get("keypoint_visibility", None)

    for v in range(num_views):
        x_offset = margin + v * (img_size + margin)
        input_img = images[v]
        if isinstance(input_img, np.ndarray):
            if input_img.shape[0] != img_size or input_img.shape[1] != img_size:
                from PIL import Image
                pil_img = Image.fromarray((input_img * 255).astype(np.uint8) if input_img.max() <= 1 else input_img.astype(np.uint8))
                pil_img = pil_img.resize((img_size, img_size), Image.BILINEAR)
                input_img = np.array(pil_img)
            if input_img.max() <= 1.0:
                input_img = (input_img * 255).astype(np.uint8)
            else:
                input_img = input_img.astype(np.uint8)
            if len(input_img.shape) == 2:
                input_img = np.stack([input_img] * 3, axis=-1)
            elif input_img.shape[-1] == 4:
                input_img = input_img[:, :, :3]
            canvas[margin:margin + img_size, x_offset:x_offset + img_size] = input_img

        try:
            rendered_img = create_rendered_view_with_keypoints(
                model, predicted_params, v,
                target_keypoints, target_visibility,
                device, img_size
            )
            canvas[2 * margin + img_size:2 * margin + 2 * img_size,
                   x_offset:x_offset + img_size] = rendered_img
        except Exception as e:
            print(f"Warning: Could not render view {v}: {e}")
            placeholder = np.ones((img_size, img_size, 3), dtype=np.uint8) * 128
            canvas[2 * margin + img_size:2 * margin + 2 * img_size,
                   x_offset:x_offset + img_size] = placeholder

    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
        draw.text((5, margin + img_size // 2 - 6), "Input", fill=(255, 255, 255), font=font)
        draw.text((5, 2 * margin + img_size + img_size // 2 - 6), "Pred", fill=(255, 255, 255), font=font)
        for v in range(num_views):
            x_pos = margin + v * (img_size + margin) + img_size // 2 - 10
            cam_name = x_data.get("camera_names", [f"V{v}"])[v] if v < len(x_data.get("camera_names", [])) else f"V{v}"
            draw.text((x_pos, 2), str(cam_name)[:8], fill=(255, 255, 255), font=font)
        canvas = np.array(pil_canvas)
    except Exception:
        pass

    return canvas


def create_rendered_view_with_keypoints(model: MultiViewSMILImageRegressor,
                                        predicted_params: dict,
                                        view_idx: int,
                                        target_keypoints: np.ndarray,
                                        target_visibility: np.ndarray,
                                        device: str,
                                        img_size: int) -> np.ndarray:
    fov = predicted_params["fov_per_view"][view_idx]
    cam_rot = predicted_params["cam_rot_per_view"][view_idx]
    cam_trans = predicted_params["cam_trans_per_view"][view_idx]

    pred_kps = None
    try:
        with torch.no_grad():
            rendered_joints = model._render_keypoints_with_camera(
                predicted_params, fov, cam_rot, cam_trans
            )
        pred_kps = rendered_joints[0].detach().cpu().numpy()
        pred_kps = pred_kps * img_size
    except Exception as e:
        print(f"Keypoint rendering failed: {e}")

    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 60
    for i in range(img_size):
        img[i, :, 0] = min(255, 60 + view_idx * 30)

    gt_kps = None
    gt_vis = None
    if target_keypoints is not None:
        if len(target_keypoints.shape) == 3:
            if view_idx < target_keypoints.shape[0]:
                gt_kps = target_keypoints[view_idx] * img_size
                if target_visibility is not None and view_idx < target_visibility.shape[0]:
                    gt_vis = target_visibility[view_idx]
        elif len(target_keypoints.shape) == 2 and view_idx == 0:
            gt_kps = target_keypoints * img_size
            gt_vis = target_visibility

    from PIL import Image, ImageDraw
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    if gt_kps is not None:
        for j, (y, x) in enumerate(gt_kps):
            if gt_vis is None or gt_vis[j] > 0.5:
                x, y = float(x), float(y)
                if 0 <= x < img_size and 0 <= y < img_size:
                    draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline="green", width=2)

    if pred_kps is not None:
        for j, (y, x) in enumerate(pred_kps):
            x, y = float(x), float(y)
            if 0 <= x < img_size and 0 <= y < img_size:
                draw.line([x - 4, y, x + 4, y], fill="red", width=2)
                draw.line([x, y - 4, x, y + 4], fill="red", width=2)

    return np.array(pil_img)


class _InMemoryImageExporter:
    def __init__(self):
        self.image = None

    def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces, img_idx=0):
        self.image = collage_np


def render_singleview_collage(model: MultiViewSMILImageRegressor,
                              x_data: dict,
                              y_data: dict,
                              device: str,
                              view_idx: int = 0) -> Optional[np.ndarray]:
    images = x_data.get("images", [])
    num_views = len(images)
    if num_views == 0 or view_idx >= num_views:
        return None

    cam_indices = x_data.get("camera_indices", list(range(num_views)))
    if isinstance(cam_indices, np.ndarray):
        cam_indices = cam_indices.tolist()

    images_per_view = []
    for img in images:
        img_tensor = model.preprocess_image(img).to(device)
        images_per_view.append(img_tensor.squeeze(0))

    images_tensors = [img.unsqueeze(0) for img in images_per_view]
    camera_indices_tensor = torch.tensor([cam_indices], device=device)
    view_mask = torch.ones(1, num_views, dtype=torch.bool, device=device)

    with torch.no_grad():
        predicted_params = model.forward_multiview(images_tensors, camera_indices_tensor, view_mask)

    fov_per_view = predicted_params.get("fov_per_view", None)
    cam_rot_per_view = predicted_params.get("cam_rot_per_view", None)
    cam_trans_per_view = predicted_params.get("cam_trans_per_view", None)

    target_size = int(getattr(model.renderer, "image_size", 224))

    original_image = images[view_idx]
    from PIL import Image
    pil_img = Image.fromarray((original_image * 255).astype(np.uint8))
    pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
    resized_image = np.array(pil_img).astype(np.float32) / 255.0

    resized_image = np.clip(resized_image, 0.0, 1.0)
    resized_image_bgr = resized_image[:, :, [2, 1, 0]]
    rgb = torch.from_numpy(resized_image_bgr).permute(2, 0, 1).unsqueeze(0).float()

    keypoints_2d = y_data.get("keypoints_2d", None)
    visibility = y_data.get("keypoint_visibility", None)

    view_keypoints = None
    view_visibility = None
    if keypoints_2d is not None:
        if len(keypoints_2d.shape) == 3:
            view_keypoints = keypoints_2d[view_idx] if view_idx < keypoints_2d.shape[0] else None
            if visibility is not None and view_idx < visibility.shape[0]:
                view_visibility = visibility[view_idx]
        else:
            view_keypoints = keypoints_2d if view_idx == 0 else None
            view_visibility = visibility if view_idx == 0 else None

    sil = torch.zeros(1, 1, target_size, target_size)
    if view_keypoints is not None and view_visibility is not None:
        pixel_coords = view_keypoints.copy()
        pixel_coords[:, 0] = pixel_coords[:, 0] * target_size
        pixel_coords[:, 1] = pixel_coords[:, 1] * target_size
        num_joints = len(view_keypoints)
        joints = torch.tensor(pixel_coords.reshape(1, num_joints, 2), dtype=torch.float32)
        vis = torch.tensor(view_visibility.reshape(1, num_joints), dtype=torch.float32)
        temp_batch = (rgb, sil, joints, vis)
        rgb_only = False
    else:
        temp_batch = rgb
        rgb_only = True

    temp_fitter = SMALFitter(
        device=device,
        data_batch=temp_batch,
        batch_size=1,
        shape_family=config.SHAPE_FAMILY,
        use_unity_prior=False,
        rgb_only=rgb_only,
    )

    if view_keypoints is not None and view_visibility is not None:
        pixel_coords = view_keypoints.copy()
        pixel_coords[:, 0] = pixel_coords[:, 0] * target_size
        pixel_coords[:, 1] = pixel_coords[:, 1] * target_size
        temp_fitter.target_joints = torch.tensor(pixel_coords, dtype=torch.float32, device=device).unsqueeze(0)
        temp_fitter.target_visibility = torch.tensor(view_visibility, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        n_joints = temp_fitter.joint_rotations.shape[1] + 1
        temp_fitter.target_joints = torch.zeros((1, n_joints, 2), device=device)
        temp_fitter.target_visibility = torch.zeros((1, n_joints), device=device)

    if model.rotation_representation == "6d":
        global_rot_aa = rotation_6d_to_axis_angle(predicted_params["global_rot"][0:1].detach())
        joint_rot_aa = rotation_6d_to_axis_angle(predicted_params["joint_rot"][0:1].detach())
    else:
        global_rot_aa = predicted_params["global_rot"][0:1].detach()
        joint_rot_aa = predicted_params["joint_rot"][0:1].detach()

    temp_fitter.global_rotation.data = global_rot_aa.to(device)
    temp_fitter.joint_rotations.data = joint_rot_aa.to(device)
    temp_fitter.betas.data = predicted_params["betas"][0].detach().to(device)
    temp_fitter.trans.data = predicted_params["trans"][0:1].detach().to(device)

    if fov_per_view is not None and view_idx < len(fov_per_view):
        fov_val = fov_per_view[view_idx][0, 0].detach().to(device)
        temp_fitter.fov.data = fov_val.unsqueeze(0)
    elif "fov" in predicted_params:
        temp_fitter.fov.data = predicted_params["fov"][0:1].detach().to(device)

    if "log_beta_scales" in predicted_params and "betas_trans" in predicted_params:
        if model.scale_trans_mode not in ["separate", "ignore"]:
            temp_fitter.log_beta_scales.data = predicted_params["log_beta_scales"][0:1].detach().to(device)
            temp_fitter.betas_trans.data = predicted_params["betas_trans"][0:1].detach().to(device)

    if cam_rot_per_view is not None and cam_trans_per_view is not None and view_idx < len(cam_rot_per_view):
        cam_rot = cam_rot_per_view[view_idx][0:1].detach().to(device)
        cam_trans = cam_trans_per_view[view_idx][0:1].detach().to(device)
        if fov_per_view is not None and view_idx < len(fov_per_view):
            view_fov_val = fov_per_view[view_idx][0, 0].detach().to(device)
            view_fov = view_fov_val.unsqueeze(0)
            temp_fitter.fov.data = view_fov
        else:
            view_fov = temp_fitter.fov.data

        aspect = None
        try:
            if y_data.get("cam_aspect_per_view") is not None:
                aspect = float(np.array(y_data["cam_aspect_per_view"][view_idx]).reshape(-1)[0])
        except Exception:
            aspect = None

        temp_fitter.renderer.set_camera_parameters(
            R=cam_rot, T=cam_trans, fov=view_fov, aspect_ratio=aspect
        )

    exporter = _InMemoryImageExporter()
    vis_mesh_scale = None
    if model.allow_mesh_scaling and "mesh_scale" in predicted_params:
        vis_mesh_scale = predicted_params["mesh_scale"][0:1].detach()
    temp_fitter.generate_visualization(
        exporter,
        apply_UE_transform=False,
        img_idx=view_idx,
        mesh_scale=vis_mesh_scale,
    )
    return exporter.image


def _pad_or_resize(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
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


def main():
    parser = argparse.ArgumentParser(
        description="Run simple multi-view inference on a preprocessed dataset"
    )
    parser.add_argument("--dataset", required=True, type=str, help="Path to preprocessed SLEAP HDF5 dataset")
    parser.add_argument("--fps", type=int, default=60, help="Output video FPS (default: 60)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = _find_default_checkpoint()

    dataset = SLEAPMultiViewDataset(
        hdf5_path=str(dataset_path),
        rotation_representation="6d",
        num_views_to_use=None,
        random_view_sampling=True,
    )

    max_views = dataset.get_max_views_in_dataset()
    canonical_camera_order = dataset.get_canonical_camera_order()

    model = load_multiview_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        max_views=max_views,
        canonical_camera_order=canonical_camera_order,
    )

    dataset_name = dataset_path.stem
    multiview_out = Path(f"{dataset_name}_multiview_inference.mp4")
    singleview_out = Path(f"{dataset_name}_smultiview_first_camera_render.mp4")

    multiview_writer = None
    singleview_writer = None

    img_size = 224
    margin = 5
    grid_width = max_views * img_size + (max_views + 1) * margin
    grid_height = 2 * img_size + 3 * margin

    for idx in range(len(dataset)):
        x_data, y_data = dataset[idx]

        mv_frame = create_multiview_visualization(model, x_data, y_data, device)
        if mv_frame is not None:
            mv_frame = _pad_or_resize(mv_frame, (grid_width, grid_height))
            mv_bgr = cv2.cvtColor(mv_frame, cv2.COLOR_RGB2BGR)
            if multiview_writer is None:
                multiview_writer = cv2.VideoWriter(
                    str(multiview_out),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    (mv_bgr.shape[1], mv_bgr.shape[0]),
                )
            multiview_writer.write(mv_bgr)

        sv_frame = render_singleview_collage(model, x_data, y_data, device, view_idx=0)
        if sv_frame is not None:
            sv_frame = _pad_or_resize(sv_frame, (sv_frame.shape[1], sv_frame.shape[0]))
            sv_bgr = cv2.cvtColor(sv_frame, cv2.COLOR_RGB2BGR)
            if singleview_writer is None:
                singleview_writer = cv2.VideoWriter(
                    str(singleview_out),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    (sv_bgr.shape[1], sv_bgr.shape[0]),
                )
            singleview_writer.write(sv_bgr)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(dataset)} samples")

    if multiview_writer is not None:
        multiview_writer.release()
        print(f"Wrote {multiview_out}")
    if singleview_writer is not None:
        singleview_writer.release()
        print(f"Wrote {singleview_out}")


if __name__ == "__main__":
    main()
