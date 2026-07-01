"""Shared multi-view visualization helpers.

Both ``train_multiview_regressor.py`` and ``run_multiview_inference.py`` need to
build the same input/render grid for a multi-view sample. The pipelines used to
keep near-duplicate copies of these helpers, which drifted (e.g. the inference
side painted a red background that clashed with the red predicted-keypoint
markers). This module hosts the single canonical implementation; both callers
import from here.
"""

from typing import Optional

import numpy as np
import torch



def run_forward_multiview_single_sample(model, x_data: dict, y_data: dict, device: str) -> Optional[dict]:
    """Run a single-sample multi-view forward pass and return ``predicted_params``."""
    images = x_data.get("images", [])
    num_views = len(images)
    if num_views == 0:
        return None

    cam_indices = x_data.get("camera_indices", list(range(num_views)))
    if isinstance(cam_indices, np.ndarray):
        cam_indices = cam_indices.tolist()
    if len(cam_indices) != num_views:
        if len(cam_indices) > num_views:
            cam_indices = cam_indices[:num_views]
        else:
            cam_indices = list(cam_indices) + list(range(len(cam_indices), num_views))

    images_per_view = []
    for img in images:
        img_tensor = model.preprocess_image(img).to(device)
        images_per_view.append(img_tensor.squeeze(0))

    images_tensors = [img.unsqueeze(0) for img in images_per_view]
    camera_indices_tensor = torch.tensor([cam_indices], device=device)
    view_mask = torch.ones(1, num_views, dtype=torch.bool, device=device)

    with torch.no_grad():
        return model.forward_multiview(images_tensors, camera_indices_tensor, view_mask, target_data=[y_data])


def create_rendered_view_with_keypoints(
    model,
    predicted_params: dict,
    view_idx: int,
    target_keypoints: Optional[np.ndarray],
    target_visibility: Optional[np.ndarray],
    device: str,
    img_size: int,
    aspect_ratio: Optional[float] = None,
    disable_scaling: bool = False,
    disable_translation: bool = False,
) -> np.ndarray:
    """Render a single view's predicted 2D keypoints over a blue tint background.

    GT keypoints are drawn as green circles, predicted keypoints as red crosses.
    The background uses a per-view blue gradient — red is intentionally kept low
    so the predicted-keypoint markers remain legible.
    """
    vis_params = predicted_params
    if disable_scaling or disable_translation:
        vis_params = predicted_params.copy()
        if disable_scaling and "log_beta_scales" in vis_params:
            vis_params["log_beta_scales"] = torch.zeros_like(vis_params["log_beta_scales"])
        if disable_translation and "betas_trans" in vis_params:
            vis_params["betas_trans"] = torch.zeros_like(vis_params["betas_trans"])

    fov = vis_params["fov_per_view"][view_idx]
    cam_rot = vis_params["cam_rot_per_view"][view_idx]
    cam_trans = vis_params["cam_trans_per_view"][view_idx]

    pred_kps = None
    try:
        with torch.no_grad():
            aspect_tensor = None
            if aspect_ratio is not None:
                aspect_tensor = torch.tensor([aspect_ratio], dtype=torch.float32, device=device)

            rendered_joints = model._render_keypoints_with_camera(
                vis_params, fov, cam_rot, cam_trans, aspect_ratio=aspect_tensor
            )
        pred_kps = rendered_joints[0].detach().cpu().numpy() * img_size
    except Exception as e:
        print(f"Keypoint rendering failed: {e}")

    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 50
    img[:, :, 2] = min(255, 70 + view_idx * 25)
    img[:, :, 1] = min(255, 55 + view_idx * 8)
    img[:, :, 0] = 45

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

    from PIL import Image, ImageDraw, ImageFont

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

    try:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except Exception:
            font = ImageFont.load_default()
        draw.text((5, img_size - 25), "O GT", fill="green", font=font)
        draw.text((5, img_size - 12), "+ Pred", fill="red", font=font)
    except Exception:
        pass

    return np.array(pil_img)


_MULTIVIEW_WRAP_THRESHOLD = 12
_MULTIVIEW_WRAP_COLS = 6


def compute_multiview_grid_layout(total_view_slots: int, img_size: int, margin: int = 5) -> dict:
    """Pre-compute the multi-view grid layout once for a given slot count.

    Views > ``_MULTIVIEW_WRAP_THRESHOLD`` wrap to ``_MULTIVIEW_WRAP_COLS`` columns
    and stack into multiple input/render row-pairs so wide grids stay readable
    in standard media players. Inference call sites should invoke this once at
    startup with ``total_view_slots = model.max_views`` and reuse the returned
    dimensions for every frame — that keeps the video stream uniform even when
    individual samples have fewer views than the model supports.
    """
    if total_view_slots <= 0:
        raise ValueError(f"total_view_slots must be > 0, got {total_view_slots}")

    cols = _MULTIVIEW_WRAP_COLS if total_view_slots > _MULTIVIEW_WRAP_THRESHOLD else total_view_slots
    num_blocks = (total_view_slots + cols - 1) // cols  # ceil
    grid_width = cols * img_size + (cols + 1) * margin
    grid_height = num_blocks * (2 * img_size + 2 * margin) + margin
    return {
        "cols": cols,
        "num_blocks": num_blocks,
        "grid_width": grid_width,
        "grid_height": grid_height,
        "img_size": img_size,
        "margin": margin,
        "total_view_slots": total_view_slots,
    }


def _multiview_cell_offsets(view_idx: int, layout: dict) -> tuple:
    """Pixel offsets (x, input_y, render_y) for ``view_idx`` in the given layout."""
    cols = layout["cols"]
    img_size = layout["img_size"]
    margin = layout["margin"]
    block = view_idx // cols
    col = view_idx % cols
    x = margin + col * (img_size + margin)
    input_y = margin + block * (2 * img_size + 2 * margin)
    render_y = input_y + img_size + margin
    return x, input_y, render_y


def create_multiview_visualization(
    model,
    x_data: dict,
    y_data: dict,
    device: str,
    predicted_params: Optional[dict] = None,
    disable_scaling: bool = False,
    disable_translation: bool = False,
    total_view_slots: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Build the input/predicted grid for a multi-view sample.

    Single-row layout (≤ 12 slots)::

        +-----------+-----------+-----------+-----+
        |  Input V0 |  Input V1 |  Input V2 | ... |
        +-----------+-----------+-----------+-----+
        | Render V0 | Render V1 | Render V2 | ... |
        +-----------+-----------+-----------+-----+

    Wrapped layout (> 12 slots, 6 cols per row-pair). Slots beyond ``num_views``
    are rendered as empty (dark) cells so video output stays size-consistent.

    ``total_view_slots`` controls the grid; when None it defaults to the
    sample's actual ``num_views``. Inference passes ``model.max_views`` here so
    every frame has identical dimensions.

    If ``predicted_params`` is not provided, a forward pass is run internally.
    """
    images = x_data.get("images", [])
    num_views = len(images)
    if num_views == 0:
        return None

    if predicted_params is None:
        predicted_params = run_forward_multiview_single_sample(model, x_data, y_data, device)
        if predicted_params is None:
            return None

    img_size = int(model.renderer.image_size)
    margin = 5
    if total_view_slots is None:
        total_view_slots = num_views
    layout = compute_multiview_grid_layout(total_view_slots, img_size, margin)
    grid_width = layout["grid_width"]
    grid_height = layout["grid_height"]

    canvas = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 40
    target_keypoints = y_data.get("keypoints_2d", None)
    target_visibility = y_data.get("keypoint_visibility", None)

    for v in range(num_views):
        x_offset, input_y, render_y = _multiview_cell_offsets(v, layout)

        input_img = images[v]
        if isinstance(input_img, np.ndarray):
            if input_img.shape[0] != img_size or input_img.shape[1] != img_size:
                from PIL import Image

                pil_img = Image.fromarray(
                    (input_img * 255).astype(np.uint8) if input_img.max() <= 1 else input_img.astype(np.uint8)
                )
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
            canvas[input_y : input_y + img_size, x_offset : x_offset + img_size] = input_img

        try:
            aspect_ratio = None
            try:
                if y_data.get("cam_aspect_per_view") is not None:
                    aspect_ratio = float(np.array(y_data["cam_aspect_per_view"][v]).reshape(-1)[0])
            except Exception:
                aspect_ratio = None

            rendered_img = create_rendered_view_with_keypoints(
                model,
                predicted_params,
                v,
                target_keypoints,
                target_visibility,
                device,
                img_size,
                aspect_ratio=aspect_ratio,
                disable_scaling=disable_scaling,
                disable_translation=disable_translation,
            )
            canvas[render_y : render_y + img_size, x_offset : x_offset + img_size] = rendered_img
        except Exception as e:
            print(f"Warning: Could not render view {v}: {e}")
            placeholder = np.ones((img_size, img_size, 3), dtype=np.uint8) * 128
            canvas[render_y : render_y + img_size, x_offset : x_offset + img_size] = placeholder

    try:
        from PIL import Image, ImageDraw, ImageFont

        pil_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
        cam_names = x_data.get("camera_names", [])
        # "Input"/"Pred" row labels on every block. Camera-name labels go on
        # block 0 only — the wrapped layout reuses the same column ordering for
        # every block, and inter-block gaps are too tight to fit a label row.
        for block in range(layout["num_blocks"]):
            _, input_y, render_y = _multiview_cell_offsets(block * layout["cols"], layout)
            draw.text((5, input_y + img_size // 2 - 6), "Input", fill=(255, 255, 255), font=font)
            draw.text((5, render_y + img_size // 2 - 6), "Pred", fill=(255, 255, 255), font=font)
        for col in range(layout["cols"]):
            v = col
            if v >= total_view_slots:
                break
            x_pos = margin + col * (img_size + margin) + img_size // 2 - 10
            cam_name = cam_names[v] if v < len(cam_names) else f"V{v}"
            draw.text((x_pos, 2), str(cam_name)[:8], fill=(255, 255, 255), font=font)
        canvas = np.array(pil_canvas)
    except Exception:
        pass

    return canvas


def print_joint_scale_diagnostics(model, predicted_params: dict, label: str = "") -> None:
    """Print per-joint log-beta scales for a predicted sample (training-only debug)."""
    if "log_beta_scales" not in predicted_params:
        return
    try:
        import config
        from smal_fitter.neuralSMIL.training_config import TrainingConfig

        scale_trans_config = TrainingConfig.get_scale_trans_config()
        use_pca_transformation = scale_trans_config.get("separate", {}).get("use_pca_transformation", True)

        if model.scale_trans_mode == "separate" and use_pca_transformation:
            scale_weights = predicted_params["log_beta_scales"][0]
            trans_weights = predicted_params.get("betas_trans", None)
            if trans_weights is not None:
                trans_weights = trans_weights[0:1]
            log_beta_scales_joint, _ = model._transform_separate_pca_weights_to_joint_values(
                scale_weights.unsqueeze(0), trans_weights
            )
            log_beta_scales_joint = log_beta_scales_joint[0]
        else:
            log_beta_scales_joint = predicted_params["log_beta_scales"][0]

        scales_joint = torch.exp(log_beta_scales_joint)
        joint_names = config.dd["J_names"]

        header = f"Joint Scales{(' for ' + label) if label else ''}"
        print(f"\n=== {header} ===")
        print(f"Mode: {model.scale_trans_mode}")
        print(f"{'Joint Name':<20} {'Scale X':>10} {'Scale Y':>10} {'Scale Z':>10} {'Mean Scale':>12}")
        print("-" * 70)
        for joint_idx, joint_name in enumerate(joint_names):
            if joint_idx < scales_joint.shape[0]:
                scale_xyz = scales_joint[joint_idx].cpu().numpy()
                print(
                    f"{joint_name:<20} {scale_xyz[0]:>10.4f} {scale_xyz[1]:>10.4f} "
                    f"{scale_xyz[2]:>10.4f} {scale_xyz.mean():>12.4f}"
                )

        all_scales = scales_joint.cpu().numpy()
        print("\nSummary Statistics:")
        print(f"  Mean scale (all joints, all axes): {all_scales.mean():.4f}")
        print(f"  Std scale (all joints, all axes): {all_scales.std():.4f}")
        print(f"  Min scale: {all_scales.min():.4f}")
        print(f"  Max scale: {all_scales.max():.4f}")
        print(f"  Joints with scale > 1.1: {((all_scales > 1.1).any(axis=1).sum().item())}")
        print(f"  Joints with scale < 0.9: {((all_scales < 0.9).any(axis=1).sum().item())}")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"Warning: Failed to print joint scales: {e}")
        import traceback

        traceback.print_exc()
