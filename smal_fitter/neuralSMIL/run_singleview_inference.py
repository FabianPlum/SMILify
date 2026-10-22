#!/usr/bin/env python3
"""
SMIL Image Regressor Inference Script

Loads a trained SMILImageRegressor checkpoint and runs inference on one of three
inputs: a folder of images, a video, or a preprocessed HDF5 dataset. Results are
written to an output folder as visualizations, per-frame parameters and videos.

Usage:
    # Folder of images
    python -m smal_fitter.neuralSMIL.run_singleview_inference -c ckpt.pth -i images/ -o out/

    # Video
    python -m smal_fitter.neuralSMIL.run_singleview_inference -c ckpt.pth -v clip.mp4 -o out/

    # Preprocessed dataset (mirrors run_multiview_inference.py)
    python -m smal_fitter.neuralSMIL.run_singleview_inference -c ckpt.pth -d dataset.h5 -o out/ \
        --view_indices 0,2 --smoothing_window 5

Input modes
-----------
`--input_folder` / `--input_video` take uncalibrated footage: the image must be
cropped here (``--crop_mode``), there is no ground-truth camera, and a
camera-centric checkpoint therefore has to be told a FOV (``--fov``, default 60).

`--dataset` takes a preprocessed HDF5 and is the mode that reproduces training
conditions, because everything the convention depends on comes from the data and
the checkpoint rather than from CLI guesses:

  * images are used exactly as the preprocessor stored them, under the dataset's
    own ``crop_mode`` — no second crop;
  * the dataset is opened under the checkpoint's ``frame_convention``, so a
    camera-centric checkpoint gets views whose camera is re-anchored to the world
    origin, and the render camera is the identity with the FOV (and aspect)
    derived from that view's calibrated intrinsics;
  * a model-centric checkpoint instead renders through its own predicted camera;
  * predicted joint scales/translations are expanded from PCA weights or used
    per-joint according to the checkpoint's ``scale_trans_mode``;
  * mesh placement follows the checkpoint: legacy 10x UE scaling, or the
    predicted per-sample ``mesh_scale``.

Each requested camera view slot yields its own temporally-ordered output video.
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import h5py
import json
import pickle as pkl

import torch
import numpy as np
import cv2
import imageio
from tqdm import tqdm

# Set matplotlib backend BEFORE any other imports to prevent tkinter issues
import matplotlib

matplotlib.use("Agg")


from smal_fitter.neuralSMIL.smil_image_regressor import SMILImageRegressor, rotation_6d_to_axis_angle
from smal_fitter.neuralSMIL.training_config import TrainingConfig
from smal_fitter.fitter import SMALFitter
import config
from smal_fitter.neuralSMIL.animation_export import AnimationRecorder, build_recorder_from_config
from smal_fitter.neuralSMIL.configs import resolve_smal_file_for_checkpoint
from smal_fitter.neuralSMIL.data_splits import add_inference_split_arg, split_dataset_from_settings
from smal_fitter.neuralSMIL.inference_utils import (
    RENDER_RESOLUTION_CAP,
    InMemoryImageExporter,
    PredictionSmoother,
    crop_to_network_view,
    pad_or_resize,
    params_to_cpu,
    params_to_device,
    resolve_render_resolution,
)
from sleap_data_loader import SLEAPDataLoader
import importlib.util
import importlib


def _import_sleap_preprocessor():
    try:
        module = importlib.import_module("smal_fitter.sleap_data.preprocess_sleap_dataset")
        return module.SLEAPDatasetPreprocessor
    except ModuleNotFoundError:
        module_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "sleap_data", "preprocess_sleap_dataset.py")
        )
        spec = importlib.util.spec_from_file_location("preprocess_sleap_dataset", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not locate preprocess_sleap_dataset at {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SLEAPDatasetPreprocessor


SLEAPDatasetPreprocessor = _import_sleap_preprocessor()


class SLEAPCroppingHelper:
    """Leverage preprocessing logic from preprocess_sleap_dataset for bbox cropping."""

    def __init__(
        self,
        project_path: str,
        crop_mode: str,
        target_resolution: int,
        backbone_name: str,
        use_reprojections: bool = True,
    ):
        self.project_path = Path(project_path)
        self.crop_mode = crop_mode
        self.preprocessor = SLEAPDatasetPreprocessor(
            joint_lookup_table_path=None,
            shape_betas_table_path=None,
            target_resolution=target_resolution,
            backbone_name=backbone_name,
            crop_mode=crop_mode,
            use_reprojections=use_reprojections,
        )

        session_paths = self.preprocessor.discover_sleap_sessions(project_path)
        if not session_paths:
            raise ValueError(f"No SLEAP sessions found in {project_path}")

        self.session_map: Dict[str, Path] = {Path(s).name: Path(s) for s in session_paths}
        self.single_session_name = next(iter(self.session_map)) if len(self.session_map) == 1 else None

        self.session_loaders: Dict[str, SLEAPDataLoader] = {}
        self.camera_data_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.reprojection_cache: Dict[str, Tuple[Optional[h5py.File], Dict[str, h5py.File]]] = {}
        self._warned_missing_camera: Set[str] = set()
        self._warned_missing_frame: Set[Tuple[str, int]] = set()

    def close(self):
        """Close any open reprojection file handles."""
        for root_handle, sub_handles in self.reprojection_cache.values():
            try:
                if root_handle is not None:
                    root_handle.close()
            except Exception:
                pass
            for handle in sub_handles.values():
                try:
                    handle.close()
                except Exception:
                    pass

    def list_cameras(self) -> List[str]:
        cameras = set()
        for session_name in self.session_map:
            loader = self._get_loader(session_name)
            cameras.update(loader.camera_views)
        return sorted(cameras)

    def _resolve_session(self, media_path: str) -> Tuple[str, Path]:
        for session_name, session_path in self.session_map.items():
            if session_name in media_path:
                return session_name, session_path
        if self.single_session_name is not None:
            session_path = self.session_map[self.single_session_name]
            return self.single_session_name, session_path
        raise ValueError(f"Could not infer SLEAP session for path: {media_path}")

    def _get_loader(self, session_name: str) -> SLEAPDataLoader:
        if session_name not in self.session_loaders:
            session_path = self.session_map[session_name]
            self.session_loaders[session_name] = SLEAPDataLoader(
                project_path=str(session_path),
                lookup_table_path=self.preprocessor.joint_lookup_table_path,
                shape_betas_path=self.preprocessor.shape_betas_table_path,
            )
        return self.session_loaders[session_name]

    def _get_reprojection_handles(self, session_name: str) -> Tuple[Optional[h5py.File], Dict[str, h5py.File]]:
        if session_name in self.reprojection_cache:
            return self.reprojection_cache[session_name]

        session_path = self.session_map[session_name]
        root_handle = None
        root_reproj = session_path / "reprojections.h5"
        if root_reproj.exists():
            try:
                root_handle = h5py.File(str(root_reproj), "r")
            except Exception as exc:
                print(f"Warning: Failed to open reprojections file {root_reproj}: {exc}")
                root_handle = None

        sub_handles = (
            self.preprocessor._find_all_reprojection_files(str(session_path))
            if self.preprocessor.use_reprojections
            else {}
        )
        self.reprojection_cache[session_name] = (root_handle, sub_handles)
        return self.reprojection_cache[session_name]

    def _get_camera_data(self, session_name: str, camera_name: str, loader: SLEAPDataLoader) -> Dict[str, Any]:
        key = (session_name, camera_name)
        if key not in self.camera_data_cache:
            self.camera_data_cache[key] = loader.load_camera_data(camera_name)
        return self.camera_data_cache[key]

    def _infer_camera(self, media_path: str, loader: SLEAPDataLoader, explicit_camera: Optional[str]) -> Optional[str]:
        if explicit_camera:
            return explicit_camera

        path_lower = str(media_path).lower()
        stem_lower = Path(media_path).stem.lower()
        parent_names = {p.name.lower() for p in Path(media_path).parents}
        for camera in loader.camera_views:
            cam_lower = camera.lower()
            if f"_cam{cam_lower}" in path_lower or f"-cam{cam_lower}" in path_lower:
                return camera
            if cam_lower in stem_lower.split("_"):
                return camera
            # Also check directory names in the path (e.g., .../Camera2/...)
            if cam_lower in parent_names:
                return camera
        if len(loader.camera_views) == 1:
            return loader.camera_views[0]
        return None

    def _get_reprojection_handle_for_camera(self, session_name: str, camera_name: str) -> Optional[h5py.File]:
        root_handle, sub_handles = self._get_reprojection_handles(session_name)
        if root_handle is not None:
            return root_handle
        session_path = self.session_map[session_name]
        camera_subdir = self.preprocessor._find_camera_subdir(str(session_path), camera_name)
        if camera_subdir and camera_subdir in sub_handles:
            return sub_handles[camera_subdir]
        return None

    def _extract_keypoints(
        self, loader: SLEAPDataLoader, camera_data: Dict[str, Any], camera_name: str, frame_idx: int, session_name: str
    ) -> Optional[np.ndarray]:
        reproj_handle = None
        if self.preprocessor.use_reprojections:
            reproj_handle = self._get_reprojection_handle_for_camera(session_name, camera_name)
        try:
            keypoints_2d, visibility = self.preprocessor._extract_2d_keypoints_for_frame(
                loader=loader,
                camera_data=camera_data,
                camera_name=camera_name,
                frame_idx=frame_idx,
                reproj_handle=reproj_handle,
            )
        except Exception as exc:
            key = (camera_name, frame_idx)
            if key not in self._warned_missing_frame:
                print(f"Warning: Failed to extract keypoints for camera '{camera_name}', frame {frame_idx}: {exc}")
                self._warned_missing_frame.add(key)
            return None

        keypoints_2d = self.preprocessor._sanitize_array(keypoints_2d, default_value=np.nan)
        if keypoints_2d is None:
            return None

        keypoints = np.asarray(keypoints_2d, dtype=np.float32)
        vis_mask = None
        if visibility is not None:
            visibility = np.asarray(visibility)
            if visibility.dtype == bool:
                vis_mask = visibility
            else:
                vis_mask = visibility > 0.5
        if vis_mask is not None and vis_mask.shape[0] == keypoints.shape[0]:
            keypoints = keypoints[vis_mask]

        if keypoints.size == 0:
            return None

        valid_mask = np.isfinite(keypoints).all(axis=1)
        keypoints = keypoints[valid_mask]
        keypoints = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
        return keypoints if keypoints.size > 0 else None

    def preprocess_image(
        self, image: np.ndarray, media_path: str, frame_idx: int, explicit_camera: Optional[str] = None
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        session_name, _ = self._resolve_session(media_path)
        loader = self._get_loader(session_name)
        camera_name = self._infer_camera(media_path, loader, explicit_camera)
        if camera_name is None:
            if media_path not in self._warned_missing_camera:
                print(f"Warning: Could not infer SLEAP camera for '{media_path}'")
                self._warned_missing_camera.add(media_path)
            return None

        camera_data = self._get_camera_data(session_name, camera_name, loader)
        keypoints = self._extract_keypoints(loader, camera_data, camera_name, frame_idx, session_name)
        if keypoints is None or len(keypoints) == 0:
            return None

        if image.max() <= 1.0:
            work_image = (image * 255.0).astype(np.uint8)
        else:
            work_image = image.astype(np.uint8)

        processed_image, transform_info = self.preprocessor._preprocess_image(work_image, keypoints)
        return processed_image, transform_info


class InferenceImageExporter:
    """Enhanced image exporter for inference results."""

    def __init__(self, output_dir: str):
        """
        Initialize the image exporter.

        Args:
            output_dir: Directory to save visualization images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export(
        self,
        collage_np: np.ndarray,
        batch_id: int,
        global_id: int,
        img_parameters: Dict[str, Any],
        vertices: torch.Tensor,
        faces: np.ndarray,
        img_idx: int = 0,
        image_name: str = "image",
        **kwargs,  # tolerate extra kwargs (e.g. epoch) passed by SMALFitter.generate_visualization
    ):
        """
        Export visualization image and parameters.

        Args:
            collage_np: Visualization collage as numpy array
            batch_id: Batch ID
            global_id: Global ID
            img_parameters: Dictionary of SMIL parameters
            vertices: Model vertices
            faces: Model faces
            img_idx: Image index
            image_name: Base name for the image
        """
        # Save visualization image
        vis_filename = f"{image_name}_visualization.png"
        vis_path = os.path.join(self.output_dir, vis_filename)
        imageio.imsave(vis_path, collage_np)

        # Save parameters as JSON (for human readability)
        params_filename = f"{image_name}_parameters.json"
        params_path = os.path.join(self.output_dir, params_filename)

        # Convert numpy arrays / tensors to lists for JSON serialization, recursing
        # into nested dicts/lists (a nested tensor previously escaped the top-level
        # check and raised "Object of type Tensor is not JSON serializable").
        def _to_jsonable(v):
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().numpy().tolist()
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            if isinstance(v, dict):
                return {k: _to_jsonable(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_to_jsonable(x) for x in v]
            return v

        json_parameters = _to_jsonable(img_parameters)

        with open(params_path, "w") as f:
            json.dump(json_parameters, f, indent=2, default=str)

        # Save parameters as pickle (for exact reproduction)
        pkl_filename = f"{image_name}_parameters.pkl"
        pkl_path = os.path.join(self.output_dir, pkl_filename)
        with open(pkl_path, "wb") as f:
            pkl.dump(img_parameters, f)

        print(f"Saved results for {image_name}:")
        print(f"  Visualization: {vis_path}")
        print(f"  Parameters (JSON): {params_path}")
        print(f"  Parameters (PKL): {pkl_path}")


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str,
    cli_smal_file: Optional[str] = None,
    cli_shape_family: Optional[int] = None,
) -> Tuple[SMILImageRegressor, Dict[str, Any]]:
    """
    Load a trained SMILImageRegressor model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: PyTorch device ('cuda' or 'cpu')
        cli_smal_file: --smal_file; overrides the checkpoint's recorded smal_file
        cli_shape_family: --shape_family; overrides the checkpoint's shape_family

    Returns:
        Tuple of (loaded_model, model_config)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("Checkpoint loaded successfully")

        # Prefer config from checkpoint so inference matches training; fall back to training_config if missing
        ckpt_config = checkpoint.get("config", {})
        training_config_fallback = TrainingConfig.get_all_config()
        fallback_model = training_config_fallback["model_config"].copy()
        fallback_params = training_config_fallback["training_params"]

        if ckpt_config:
            model_config = {**fallback_model, **ckpt_config.get("model_config", {})}
            rotation_representation = (ckpt_config.get("training_params") or {}).get(
                "rotation_representation"
            ) or fallback_params.get("rotation_representation", "6d")
            scale_trans_mode = ckpt_config.get("scale_trans_mode") or TrainingConfig.get_scale_trans_mode()
            config_source = "checkpoint (fallback: training_config for missing keys)"
        else:
            model_config = fallback_model
            rotation_representation = fallback_params["rotation_representation"]
            scale_trans_mode = TrainingConfig.get_scale_trans_mode()
            config_source = "training_config.py (no config in checkpoint)"

        # Frame convention + camera flags (persisted at the TOP level of
        # checkpoint["config"] by the trainer, not inside model_config).
        # camera_centric checkpoints use a fixed identity camera and were
        # trained without the 10x UE scaling.
        frame_convention = ckpt_config.get("frame_convention", "model_centric") if ckpt_config else "model_centric"
        fixed_camera = bool(ckpt_config.get("fixed_camera", frame_convention == "camera_centric"))
        use_ue_scaling = bool(ckpt_config.get("use_ue_scaling", not fixed_camera))
        # Mesh-scale: prefer the persisted flag; fall back to detecting the
        # mesh_scale head in the state dict so older checkpoints (saved before the
        # flag was persisted) still rebuild with the head instead of dropping it.
        _has_mesh_scale_head = any("mesh_scale_head" in k for k in checkpoint.get("model_state_dict", {}))
        allow_mesh_scaling = bool(ckpt_config.get("allow_mesh_scaling", _has_mesh_scale_head))
        mesh_scale_init = float(ckpt_config.get("init_mesh_scale", 1.0))
        model_config["frame_convention"] = frame_convention
        model_config["fixed_camera"] = fixed_camera
        model_config["use_ue_scaling"] = use_ue_scaling
        model_config["allow_mesh_scaling"] = allow_mesh_scaling
        model_config["rotation_representation"] = rotation_representation
        model_config["scale_trans_mode"] = scale_trans_mode
        # Split settings the trainer stored (seed, ratios, split_method, holdout_fold), for --split
        model_config["split_settings"] = (ckpt_config.get("training_params") or {}) if ckpt_config else {}

        # SMAL/SMIL model: --smal_file > checkpoint's smal_file > config.py default.
        # Must run before the model is built: head widths follow config.N_BETAS / N_POSE.
        shape_family = resolve_smal_file_for_checkpoint(ckpt_config or {}, cli_smal_file, cli_shape_family)

        print(f"Configuration from {config_source}:")
        print(f"  frame_convention: {frame_convention} (fixed_camera={fixed_camera}, use_ue_scaling={use_ue_scaling})")
        print(f"  backbone_name: {model_config['backbone_name']}")
        print(f"  head_type: {model_config.get('head_type', 'mlp')}")
        print(f"  rotation_representation: {rotation_representation}")
        print(f"  scale_trans_mode: {scale_trans_mode}")
        print(f"  shape_family: {shape_family}")

        # Verify this matches the checkpoint by checking state dict keys
        state_dict = checkpoint["model_state_dict"]

        # Check for transformer head
        has_transformer_head = any("transformer_head" in key for key in state_dict.keys())

        # Infer backbone type from feature dimensions in the checkpoint
        # Note: Backbone weights are NOT saved (frozen pretrained weights are re-downloaded on load)
        # So we detect backbone type from the input dimension of the regression head
        inferred_backbone = None
        if has_transformer_head:
            # Check transformer head input dimension from token_embedding weight
            token_emb_key = "transformer_head.token_embedding.weight"
            if token_emb_key in state_dict:
                feature_dim = state_dict[token_emb_key].shape[0]
                if feature_dim == 1024:
                    inferred_backbone = "vit_large"
                elif feature_dim == 768:
                    inferred_backbone = "vit_base"
                elif feature_dim == 2048:
                    inferred_backbone = "resnet"
                print(f"Inferred backbone from checkpoint feature dim ({feature_dim}): {inferred_backbone}")

        # Validate config matches inferred backbone
        config_backbone = model_config["backbone_name"]
        if inferred_backbone:
            config_is_vit_large = "vit_large" in config_backbone
            config_is_vit_base = "vit_base" in config_backbone
            config_is_resnet = config_backbone.startswith("resnet")

            if inferred_backbone == "vit_large" and not config_is_vit_large:
                print(f"WARNING: Checkpoint was trained with ViT-Large but config specifies {config_backbone}")
            elif inferred_backbone == "vit_base" and not config_is_vit_base:
                print(f"WARNING: Checkpoint was trained with ViT-Base but config specifies {config_backbone}")
            elif inferred_backbone == "resnet" and not config_is_resnet:
                print(f"WARNING: Checkpoint was trained with ResNet but config specifies {config_backbone}")
            else:
                print(f"Backbone configuration matches checkpoint: {config_backbone}")

        if model_config["head_type"] == "transformer_decoder" and not has_transformer_head:
            print("WARNING: Config specifies transformer_decoder but checkpoint doesn't contain transformer_head")

        print("Checkpoint verification:")
        print(f"  Inferred backbone: {inferred_backbone or 'unknown'}")
        print(f"  Contains transformer_head: {has_transformer_head}")

        print("Model configuration:")
        for key, value in model_config.items():
            if key != "transformer_config":
                print(f"  {key}: {value}")

        print(f"Using rotation representation: {rotation_representation}")

        # For inference, always use batch_size=1
        # The checkpoint may have been saved with a different batch size during training,
        # but for inference we process one image at a time
        batch_size = 1
        print(f"Using batch size: {batch_size} (inference mode)")

        # Create placeholder data for model initialization
        placeholder_data = torch.zeros((batch_size, 3, 512, 512))

        # Determine input resolution from the centralized backbone factory
        from smal_fitter.neuralSMIL.backbone_factory import BackboneFactory

        input_resolution = BackboneFactory.get_default_input_resolution(model_config["backbone_name"])

        print(f"Creating model with input resolution: {input_resolution}")

        # Initialize model with detected configuration
        model = SMILImageRegressor(
            device=device,
            data_batch=placeholder_data,
            batch_size=batch_size,
            shape_family=shape_family,
            use_unity_prior=model_config.get("use_unity_prior", False),
            rgb_only=model_config.get("rgb_only", True),
            freeze_backbone=model_config.get("freeze_backbone", True),
            hidden_dim=model_config.get("hidden_dim", 1024),
            # Legacy replicAnt single-view uses 10x UE scaling; camera-centric
            # (multi-view-derived) checkpoints do not (scale baked via world_scale).
            use_ue_scaling=use_ue_scaling,
            rotation_representation=rotation_representation,
            input_resolution=input_resolution,
            backbone_name=model_config["backbone_name"],
            head_type=model_config.get("head_type", "mlp"),
            transformer_config=model_config.get("transformer_config", {}),
            scale_trans_mode=scale_trans_mode,  # Critical for correct output dimensions
            fixed_camera=fixed_camera,  # camera-centric: pin camera to identity
            allow_mesh_scaling=allow_mesh_scaling,  # rebuild the mesh_scale head
            mesh_scale_init=mesh_scale_init,
        ).to(device)

        # Load model state, handling batch size differences
        # For inference, we need the neural network weights, but skip SMAL optimization parameters
        state_dict = checkpoint["model_state_dict"]

        # Filter out SMAL optimization parameters that have batch size dependencies
        # These are specific to the SMALFitter optimization process, not the neural network
        smal_optimization_params = [
            "global_rotation",
            "joint_rotations",
            "trans",
            "log_beta_scales",
            "betas_trans",
            "betas",
            "fov",
            "target_joints",
            "target_visibility",
        ]

        # Keep all neural network parameters (backbone, transformer_head, fc layers, etc.)
        nn_state_dict = {}
        skipped_params = []

        for k, v in state_dict.items():
            # Skip SMAL optimization parameters that are specific to the optimization process
            if any(k == param or k.startswith(param + ".") for param in smal_optimization_params):
                skipped_params.append(k)
            else:
                nn_state_dict[k] = v

        print(f"Loading {len(nn_state_dict)} neural network parameters")
        print(
            f"Skipping {len(skipped_params)} SMAL optimization parameters: {skipped_params[:5]}{'...' if len(skipped_params) > 5 else ''}"
        )

        # Load the neural network weights
        missing_keys, unexpected_keys = model.load_state_dict(nn_state_dict, strict=False)

        if missing_keys:
            print(
                f"Missing keys (will use random initialization): {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}"
            )
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''}")
        model.eval()

        print("Model loaded and set to evaluation mode")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Model statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Head type: {model.head_type}")
        print(f"  Backbone: {model.backbone_name}")
        print(f"  Input resolution: {input_resolution}")

        return model, model_config

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")


def find_image_files(input_folder: str, supported_extensions: List[str] = None) -> List[str]:
    """
    Find all image files in the input folder.

    Args:
        input_folder: Path to folder containing images
        supported_extensions: List of supported file extensions

    Returns:
        List of image file paths
    """
    if supported_extensions is None:
        supported_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".tif",
            ".JPG",
            ".JPEG",
            ".PNG",
            ".BMP",
            ".TIFF",
            ".TIF",
        ]

    image_files = []
    input_path = Path(input_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_folder}")

    # Find all image files
    for ext in supported_extensions:
        pattern = f"*{ext}"
        image_files.extend(input_path.glob(pattern))

    # Convert to strings and sort
    image_files = [str(f) for f in image_files]
    image_files.sort()

    print(f"Found {len(image_files)} image files in {input_folder}")

    return image_files


def preprocess_frame(
    image: np.ndarray, target_resolution: int, crop_mode: str = "centred"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Preprocess a frame using centred or default resize.

    Args:
        image: Input image (H, W, C) in range [0, 255] or [0, 1]
        target_resolution: Target resolution for model input
        crop_mode: 'centred' or 'default'

    Returns:
        Tuple of (preprocessed_image, transform_info)
    """
    if crop_mode == "bbox_crop":
        raise ValueError(
            "bbox_crop should be handled via SLEAPCroppingHelper; preprocess_frame only supports 'centred' or 'default'."
        )

    # Ensure image is in [0, 255] range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    original_h, original_w = image.shape[:2]
    transform_info = {
        "original_size": (original_h, original_w),
        "crop_offset": (0, 0),
        "crop_size": (original_h, original_w),
        "scale_factor": 1.0,
        "mode": crop_mode,
    }

    if crop_mode == "centred":
        crop_size = min(original_h, original_w)
        y_offset = (original_h - crop_size) // 2
        x_offset = (original_w - crop_size) // 2
        image = image[y_offset : y_offset + crop_size, x_offset : x_offset + crop_size]
        transform_info["crop_offset"] = (y_offset, x_offset)
        transform_info["crop_size"] = (crop_size, crop_size)
        scale_factor = target_resolution / crop_size
        transform_info["scale_factor"] = scale_factor
        image = cv2.resize(image, (target_resolution, target_resolution))

    else:
        scale_y = target_resolution / original_h
        scale_x = target_resolution / original_w
        transform_info["scale_factor"] = (scale_y, scale_x)
        image = cv2.resize(image, (target_resolution, target_resolution))

    image = image.astype(np.float32) / 255.0
    return image, transform_info


def load_and_preprocess_image(
    image_path: str, model: SMILImageRegressor, crop_mode: str = "centred", keypoints_2d: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, torch.Tensor, Dict[str, Any]]:
    """
    Load and preprocess an image for inference.

    Args:
        image_path: Path to the image file
        model: SMILImageRegressor model for preprocessing
        crop_mode: Cropping mode ('centred' or 'default')

    Returns:
        Tuple of (original_image_array, preprocessed_tensor, transform_info)
    """
    try:
        # Load image
        image_data = imageio.v2.imread(image_path)

        # Keep original for visualization
        original_image = image_data.copy()

        # Preprocess with proper cropping
        target_resolution = model.input_resolution
        preprocessed_image, transform_info = preprocess_frame(image_data, target_resolution, crop_mode)

        # Convert to tensor (C, H, W) format
        preprocessed_tensor = torch.from_numpy(preprocessed_image).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

        return original_image, preprocessed_tensor, transform_info

    except Exception as e:
        raise RuntimeError(f"Failed to load/preprocess image {image_path}: {e}")


def run_inference_on_image(
    model: SMILImageRegressor,
    image_tensor: torch.Tensor,
    device: str,
    gt_fov: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run inference on a preprocessed image tensor.

    Args:
        model: SMILImageRegressor model
        image_tensor: Preprocessed image tensor (1, C, H, W)
        device: PyTorch device
        gt_fov: Calibrated vertical FOV (degrees) for this view, when known.
            Dataset inference passes the view's FOV derived from the stored
            camera intrinsics; raw images/videos have no calibration and pass
            None, falling back to ``model._inference_fov`` (--fov or 60.0).
            Only used by camera-centric (fixed_camera) checkpoints.

    Returns:
        Dictionary of predicted SMIL parameters
    """
    try:
        with torch.no_grad():
            # Move to device
            image_tensor = image_tensor.to(device)

            # Get batch size from tensor
            image_tensor.shape[0]

            # Run inference through the model's forward pass
            # The model's forward() method handles batches correctly
            predicted_params = model.forward(image_tensor)

            # Camera-centric checkpoints: pin the camera to the PyTorch3D identity
            # and inject the known FOV. Inference calls forward() directly (not
            # predict_from_batch), so we re-implement that method's fixed_camera
            # override here. The camera heads are UNSUPERVISED in this mode, so
            # their outputs must never reach the renderer.
            #
            # FOV source, in order of preference:
            #   1. gt_fov  — the view's calibrated FOV (dataset inference; this
            #      is exactly what predict_from_batch uses via target["fov"]).
            #   2. model._inference_fov — --fov, else 60.0 (raw image / video,
            #      where no calibration exists).
            if getattr(model, "fixed_camera", False):
                bs = predicted_params["global_rot"].shape[0]
                predicted_params["cam_rot"] = torch.eye(3, device=device).unsqueeze(0).expand(bs, 3, 3).contiguous()
                predicted_params["cam_trans"] = torch.zeros(bs, 3, device=device)
                fov_deg = float(gt_fov) if gt_fov is not None else float(getattr(model, "_inference_fov", 60.0))
                predicted_params["fov"] = torch.full((bs, 1), fov_deg, device=device)

            # Move results back to CPU for visualization
            cpu_params = {}
            for key, value in predicted_params.items():
                if isinstance(value, torch.Tensor):
                    cpu_params[key] = value.cpu()
                else:
                    cpu_params[key] = value

            return cpu_params

    except Exception as e:
        import traceback

        print("\n" + "=" * 60)
        print("INFERENCE ERROR DEBUG INFO")
        print("=" * 60)
        print(f"Input tensor shape: {image_tensor.shape}")
        print(f"Device: {device}")
        print(f"Model batch_size attribute: {model.batch_size}")
        print(f"Model head_type: {model.head_type}")
        print(f"Model rotation_representation: {model.rotation_representation}")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        raise RuntimeError(f"Inference failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared parameter-application helpers
#
# These centralise the three conventions that MUST agree between training,
# multi-view inference and single-view inference, and that used to be applied
# inconsistently across this file's three render paths:
#
#   1. scale_trans_mode  — 'separate' predicts PCA *weights* that have to be
#      expanded to per-joint (n_joints, 3) values before they can be written to
#      a SMALFitter; 'entangled_with_betas' already predicts per-joint values;
#      'ignore' means scales/translations are not used at all. Writing raw PCA
#      weights straight into `log_beta_scales` silently produced a differently
#      shaped (and geometrically meaningless) mesh.
#   2. mesh scaling      — `use_ue_scaling` (legacy replicAnt 10x) XOR the
#      predicted per-sample `mesh_scale` (camera-centric). Neither is optional:
#      without one the mesh renders at native SMAL size.
#   3. camera            — camera-centric checkpoints render through a FIXED
#      PyTorch3D identity camera with a *known* FOV; model-centric checkpoints
#      render through the camera the network predicted.
#
# Mirrors `render_singleview_collage()` in run_multiview_inference.py and
# `visualize_training_progress()` in train_smil_regressor.py.
# ─────────────────────────────────────────────────────────────────────────────


def apply_scale_trans_to_fitter(
    model: SMILImageRegressor,
    temp_fitter: SMALFitter,
    predicted_params: Dict[str, torch.Tensor],
    device: str,
    disable_scaling: bool = False,
    disable_translation: bool = False,
) -> None:
    """Write predicted joint scales / translations onto *temp_fitter*.

    Handles all three ``scale_trans_mode`` values coherently with training:

    - ``ignore``               → nothing is applied.
    - ``separate``             → predictions are PCA weights ``(B, N_BETAS)``
      when ``use_pca_transformation`` is on and must be expanded to per-joint
      ``(B, n_joints, 3)`` values via the model's own PCA basis; otherwise they
      are already per-joint and are used directly.
    - ``entangled_with_betas`` → predictions are already per-joint.

    The two heads are handled independently: a checkpoint that predicts only
    scales, or only translations, still gets the one it has applied.
    """
    has_scales = "log_beta_scales" in predicted_params
    has_trans = "betas_trans" in predicted_params
    if not (has_scales or has_trans):
        return

    mode = getattr(model, "scale_trans_mode", "separate")
    if mode == "ignore":
        # Scales/translations are not part of this model's shape space.
        return

    scales = predicted_params["log_beta_scales"][0:1].detach() if has_scales else None
    trans = predicted_params["betas_trans"][0:1].detach() if has_trans else None

    if mode == "separate":
        scale_trans_config = TrainingConfig.get_scale_trans_config()
        use_pca_transformation = scale_trans_config.get("separate", {}).get("use_pca_transformation", True)
        if use_pca_transformation:
            # PCA weights -> per-joint values. Same call the trainer makes. The
            # expansion needs scale weights; zero weights give unit scale
            # (log-scale 0), so a translation-only checkpoint keeps identity scales.
            scale_weights = scales if scales is not None else torch.zeros_like(trans)
            try:
                scales_joint, trans_joint = model._transform_separate_pca_weights_to_joint_values(scale_weights, trans)
            except Exception as e:
                print(f"Warning: Failed to expand PCA limb scales for visualization: {e}")
                return
            scales = scales_joint if has_scales else None
            trans = trans_joint if has_trans else None

    if not disable_scaling and scales is not None:
        temp_fitter.log_beta_scales.data = scales.to(device)
    if not disable_translation and trans is not None:
        temp_fitter.betas_trans.data = trans.to(device)


def resolve_mesh_scale(
    model: SMILImageRegressor, predicted_params: Dict[str, torch.Tensor], device: str
) -> Optional[torch.Tensor]:
    """Return the per-sample mesh scale to render with, or None.

    Only meaningful when the checkpoint was trained with ``allow_mesh_scaling``
    (camera-centric). ``use_ue_scaling`` (legacy replicAnt 10x) is mutually
    exclusive with it and is applied by SMALFitter via ``apply_UE_transform``.
    """
    if not getattr(model, "allow_mesh_scaling", False):
        return None
    if "mesh_scale" not in predicted_params:
        return None
    return predicted_params["mesh_scale"][0:1].detach().to(device)


def apply_camera_to_fitter(
    model: SMILImageRegressor,
    temp_fitter: SMALFitter,
    predicted_params: Dict[str, torch.Tensor],
    device: str,
    aspect_ratio: Optional[float] = None,
) -> None:
    """Set the render camera from *predicted_params*.

    ``run_inference_on_image()`` has already normalised the camera-centric case
    (identity R, zero T, calibrated/CLI FOV injected into ``predicted_params``),
    so both conventions are applied identically here. *aspect_ratio* carries the
    non-square intrinsics of a calibrated dataset view (``cam_aspect``); it is
    None for raw images / videos, where the renderer assumes square pixels.
    """
    fov = predicted_params["fov"][0:1].detach().to(device)
    temp_fitter.fov.data = fov
    if "cam_rot" in predicted_params and "cam_trans" in predicted_params:
        temp_fitter.renderer.set_camera_parameters(
            R=predicted_params["cam_rot"][0:1].detach().to(device),
            T=predicted_params["cam_trans"][0:1].detach().to(device),
            fov=fov,
            aspect_ratio=aspect_ratio,
        )


# Render resolutions that already ran out of GPU memory (warned once each).
_WARNED_RENDER_OOM: set = set()


def warn_render_oom(render_size: int) -> None:
    """Free cached GPU memory and explain, once per size, how to avoid the OOM."""
    torch.cuda.empty_cache()
    if render_size not in _WARNED_RENDER_OOM:
        _WARNED_RENDER_OOM.add(render_size)
        print(
            f"WARNING: GPU out of memory while rendering at {render_size}x{render_size} px. "
            f"Pass a smaller --render_resolution (the default caps at {RENDER_RESOLUTION_CAP}). "
            f"Frames that fail to render are written without the overlay."
        )


def render_model_only(
    model: SMILImageRegressor, predicted_params: Dict[str, torch.Tensor], device: str, render_size: int
) -> np.ndarray:
    """
    Render only the predicted 3D model without any background.

    Args:
        model: SMILImageRegressor model
        predicted_params: Dictionary of predicted SMIL parameters
        device: PyTorch device
        render_size: Target render resolution

    Returns:
        Rendered model image (render_size, render_size, 3) in RGB, range [0, 255]
    """
    try:
        # Convert rotations to axis-angle if they're in 6D representation
        if model.rotation_representation == "6d":
            global_rot_aa = rotation_6d_to_axis_angle(predicted_params["global_rot"])
            joint_rot_aa = rotation_6d_to_axis_angle(predicted_params["joint_rot"])
        else:
            global_rot_aa = predicted_params["global_rot"]
            joint_rot_aa = predicted_params["joint_rot"]

        # Create a blank RGB tensor for rendering
        rgb_tensor = torch.zeros((1, 3, render_size, render_size), device=device)

        # Create temporary SMALFitter for rendering
        temp_fitter = SMALFitter(
            device=device,
            data_batch=rgb_tensor,
            batch_size=1,
            shape_family=config.SHAPE_FAMILY,
            use_unity_prior=False,
            rgb_only=True,
        )

        # CRITICAL: Match propagate_scaling to the training model's setting.
        # The model learns scales with propagate_scaling=True (set in SMILImageRegressor.__init__),
        # so visualization must also use propagate_scaling=True for consistent geometry.
        temp_fitter.propagate_scaling = model.propagate_scaling

        # Set the predicted parameters
        temp_fitter.global_rotation.data = global_rot_aa.to(device)
        temp_fitter.joint_rotations.data = joint_rot_aa.to(device)
        temp_fitter.betas.data = predicted_params["betas"].to(device)
        temp_fitter.trans.data = predicted_params["trans"].to(device)

        # Joint scales / translations, expanded from PCA weights when the
        # checkpoint's scale_trans_mode calls for it (see helper docstring).
        apply_scale_trans_to_fitter(model, temp_fitter, predicted_params, device)

        # Camera (identity + calibrated FOV for camera-centric checkpoints,
        # predicted camera for model-centric ones).
        apply_camera_to_fitter(model, temp_fitter, predicted_params, device)

        # Render the model
        with torch.no_grad():
            # Get vertices and joints from SMAL model
            verts, joints, Rs, v_shaped = temp_fitter.smal_model(
                temp_fitter.betas,
                torch.cat([temp_fitter.global_rotation.unsqueeze(1), temp_fitter.joint_rotations], dim=1),
                betas_logscale=temp_fitter.log_beta_scales,
                betas_trans=temp_fitter.betas_trans,
                propagate_scaling=temp_fitter.propagate_scaling,
            )

            root_joint = joints[:, 0:1, :]
            mesh_scale = resolve_mesh_scale(model, predicted_params, device)
            if model.use_ue_scaling:
                # Apply UE scaling transformation (10x scale) — legacy replicAnt
                verts = (verts - root_joint) * 10 + temp_fitter.trans.unsqueeze(1)
                joints = (joints - root_joint) * 10 + temp_fitter.trans.unsqueeze(1)
            elif mesh_scale is not None:
                # Camera-centric: the predicted per-sample mesh scale IS the
                # size of the animal. Skipping it renders the mesh at native
                # SMAL size (orders of magnitude off the metric 3D), which is
                # what SMALFitter.generate_visualization does via mesh_scale.
                scale_val = mesh_scale.reshape(-1, 1, 1)
                verts = (verts - root_joint) * scale_val + temp_fitter.trans.unsqueeze(1)
                joints = (joints - root_joint) * scale_val + temp_fitter.trans.unsqueeze(1)
            else:
                # No UE scaling and no mesh-scale head: plain translation.
                verts = verts + temp_fitter.trans.unsqueeze(1)
                joints = joints + temp_fitter.trans.unsqueeze(1)

            # Get canonical model joints
            canonical_joints = joints[:, config.CANONICAL_MODEL_JOINTS]

            # Prepare faces
            faces_batch = temp_fitter.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1)

            # Render with texture
            rendered_silhouettes, rendered_joints, rendered_image = temp_fitter.renderer(
                verts, canonical_joints, faces_batch, render_texture=True
            )

        # Convert rendered image to numpy (already in (B, C, H, W) format)
        rendered_np = rendered_image[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        rendered_np = np.clip(rendered_np, 0, 1)

        # Convert to [0, 255] range
        return (rendered_np * 255).astype(np.uint8)

    except torch.cuda.OutOfMemoryError:
        warn_render_oom(render_size)
        return np.zeros((render_size, render_size, 3), dtype=np.uint8)
    except Exception as e:
        print(f"Warning: Failed to render model: {e}")
        # Return black image on error
        return np.zeros((render_size, render_size, 3), dtype=np.uint8)


def render_prediction_on_frame(
    model: SMILImageRegressor,
    predicted_params: Dict[str, torch.Tensor],
    original_frame: np.ndarray,
    device: str,
    transform_info: Optional[Dict[str, Any]] = None,
    render_resolution: Optional[int] = None,
) -> np.ndarray:
    """
    Render the predicted 3D model onto the original frame.

    Args:
        model: SMILImageRegressor model
        predicted_params: Dictionary of predicted SMIL parameters
        original_frame: Original frame (H, W, 3) in RGB, range [0, 255]
        device: PyTorch device
        transform_info: Crop applied before inference; the overlay is placed in that region
        render_resolution: Square render size; default is the crop size capped at
            RENDER_RESOLUTION_CAP. The camera is FOV-based, so this only changes sharpness.

    Returns:
        Rendered frame with 3D model overlay (H, W, 3) in RGB, range [0, 255]
    """
    network_view = crop_to_network_view(original_frame, transform_info)
    render_size = resolve_render_resolution(render_resolution, network_view.shape[:2])
    try:
        # Convert rotations to axis-angle if they're in 6D representation
        if model.rotation_representation == "6d":
            global_rot_aa = rotation_6d_to_axis_angle(predicted_params["global_rot"])
            joint_rot_aa = rotation_6d_to_axis_angle(predicted_params["joint_rot"])
        else:
            global_rot_aa = predicted_params["global_rot"]
            joint_rot_aa = predicted_params["joint_rot"]

        # Get frame dimensions
        frame_h, frame_w = original_frame.shape[:2]

        # Prepare image for rendering: the view the network saw, at render size
        # (SMALFitter renders at the size of this image)
        if network_view.max() > 1.0:
            rgb_image = network_view.astype(np.float32) / 255.0
        else:
            rgb_image = network_view.astype(np.float32)

        rgb_resized = cv2.resize(rgb_image, (render_size, render_size))

        # Convert to tensor format expected by SMALFitter
        rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        # Create temporary SMALFitter for rendering
        temp_fitter = SMALFitter(
            device=device,
            data_batch=rgb_tensor,
            batch_size=1,
            shape_family=config.SHAPE_FAMILY,
            use_unity_prior=False,
            rgb_only=True,
        )

        # CRITICAL: Match propagate_scaling to the training model's setting.
        # The model learns scales with propagate_scaling=True (set in SMILImageRegressor.__init__),
        # so visualization must also use propagate_scaling=True for consistent geometry.
        temp_fitter.propagate_scaling = model.propagate_scaling

        # Set the predicted parameters
        temp_fitter.global_rotation.data = global_rot_aa.to(device)
        temp_fitter.joint_rotations.data = joint_rot_aa.to(device)
        temp_fitter.betas.data = predicted_params["betas"].to(device)
        temp_fitter.trans.data = predicted_params["trans"].to(device)

        # Joint scales / translations, expanded from PCA weights when the
        # checkpoint's scale_trans_mode calls for it (see helper docstring).
        apply_scale_trans_to_fitter(model, temp_fitter, predicted_params, device)

        # Camera (identity + calibrated FOV for camera-centric checkpoints,
        # predicted camera for model-centric ones).
        apply_camera_to_fitter(model, temp_fitter, predicted_params, device)

        # Render the model
        with torch.no_grad():
            # Get vertices and joints from SMAL model
            verts, joints, Rs, v_shaped = temp_fitter.smal_model(
                temp_fitter.betas,
                torch.cat([temp_fitter.global_rotation.unsqueeze(1), temp_fitter.joint_rotations], dim=1),
                betas_logscale=temp_fitter.log_beta_scales,
                betas_trans=temp_fitter.betas_trans,
                propagate_scaling=temp_fitter.propagate_scaling,
            )

            root_joint = joints[:, 0:1, :]
            mesh_scale = resolve_mesh_scale(model, predicted_params, device)
            if model.use_ue_scaling:
                # Apply UE scaling transformation (10x scale) — legacy replicAnt
                verts = (verts - root_joint) * 10 + temp_fitter.trans.unsqueeze(1)
                joints = (joints - root_joint) * 10 + temp_fitter.trans.unsqueeze(1)
            elif mesh_scale is not None:
                # Camera-centric: the predicted per-sample mesh scale IS the
                # size of the animal. Skipping it renders the mesh at native
                # SMAL size (orders of magnitude off the metric 3D), which is
                # what SMALFitter.generate_visualization does via mesh_scale.
                scale_val = mesh_scale.reshape(-1, 1, 1)
                verts = (verts - root_joint) * scale_val + temp_fitter.trans.unsqueeze(1)
                joints = (joints - root_joint) * scale_val + temp_fitter.trans.unsqueeze(1)
            else:
                # No UE scaling and no mesh-scale head: plain translation.
                verts = verts + temp_fitter.trans.unsqueeze(1)
                joints = joints + temp_fitter.trans.unsqueeze(1)

            # Get canonical model joints
            canonical_joints = joints[:, config.CANONICAL_MODEL_JOINTS]

            # Prepare faces
            faces_batch = temp_fitter.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1)

            # Render with texture
            rendered_silhouettes, rendered_joints, rendered_image = temp_fitter.renderer(
                verts, canonical_joints, faces_batch, render_texture=True
            )

        # Convert rendered image to numpy (already in (B, C, H, W) format)
        rendered_np = rendered_image[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        rendered_np = np.clip(rendered_np, 0, 1)

        alpha = 0.6  # Transparency of the overlay
        overlay_base = original_frame.astype(np.float32) / 255.0

        placed_overlay = overlay_base.copy()

        if transform_info is not None and transform_info.get("mode") in ("centred", "bbox_crop"):
            crop_height, crop_width = transform_info.get("crop_size", (frame_h, frame_w))
            y_offset, x_offset = transform_info.get("crop_offset", (0, 0))

            crop_height = int(round(crop_height))
            crop_width = int(round(crop_width))
            y_offset = int(round(y_offset))
            x_offset = int(round(x_offset))

            if crop_height <= 0 or crop_width <= 0:
                rendered_resized = cv2.resize(rendered_np, (frame_w, frame_h))
                blended = alpha * rendered_resized + (1 - alpha) * overlay_base
                return (blended * 255).astype(np.uint8)

            overlay_y_end = min(y_offset + crop_height, frame_h)
            overlay_x_end = min(x_offset + crop_width, frame_w)

            if overlay_y_end <= y_offset or overlay_x_end <= x_offset:
                rendered_resized = cv2.resize(rendered_np, (frame_w, frame_h))
                blended = alpha * rendered_resized + (1 - alpha) * overlay_base
                return (blended * 255).astype(np.uint8)

            target_height = overlay_y_end - y_offset
            target_width = overlay_x_end - x_offset

            rendered_resized = cv2.resize(rendered_np, (target_width, target_height))

            base_region = overlay_base[y_offset:overlay_y_end, x_offset:overlay_x_end]
            blended_region = alpha * rendered_resized + (1 - alpha) * base_region
            placed_overlay[y_offset:overlay_y_end, x_offset:overlay_x_end] = blended_region
            return (placed_overlay * 255).astype(np.uint8)
        else:
            rendered_resized = cv2.resize(rendered_np, (frame_w, frame_h))
            blended = alpha * rendered_resized + (1 - alpha) * overlay_base
            return (blended * 255).astype(np.uint8)

    except torch.cuda.OutOfMemoryError:
        warn_render_oom(render_size)
        return original_frame
    except Exception as e:
        print(f"Warning: Failed to render prediction: {e}")
        # Return original frame on error
        return original_frame


def generate_visualization(
    model: SMILImageRegressor,
    predicted_params: Dict[str, torch.Tensor],
    original_image: np.ndarray,
    image_exporter: InferenceImageExporter,
    image_name: str,
    device: str,
    transform_info: Optional[Dict[str, Any]] = None,
    render_resolution: Optional[int] = None,
) -> None:
    """
    Generate visualization using the SMIL model and predicted parameters.

    Args:
        model: SMILImageRegressor model
        predicted_params: Dictionary of predicted SMIL parameters
        original_image: Original input image
        image_exporter: Image exporter for saving results
        image_name: Base name for the image
        device: PyTorch device
        transform_info: Crop applied before inference. The collage shows that
            crop, not the whole (possibly non-square) frame squashed to a square.
        render_resolution: Square collage panel size; default is the crop size
            capped at RENDER_RESOLUTION_CAP.
    """
    network_view = crop_to_network_view(original_image, transform_info)
    render_size = resolve_render_resolution(render_resolution, network_view.shape[:2])
    try:
        # Convert rotations to axis-angle if they're in 6D representation
        if model.rotation_representation == "6d":
            global_rot_aa = rotation_6d_to_axis_angle(predicted_params["global_rot"])
            joint_rot_aa = rotation_6d_to_axis_angle(predicted_params["joint_rot"])
        else:
            global_rot_aa = predicted_params["global_rot"]
            joint_rot_aa = predicted_params["joint_rot"]

        # Create a simplified SMALFitter for visualization, using the view the
        # network saw as RGB input
        if network_view.max() > 1.0:
            rgb_image = network_view.astype(np.float32) / 255.0
        else:
            rgb_image = network_view.astype(np.float32)

        if rgb_image.shape[:2] != (render_size, render_size):
            rgb_image = cv2.resize(rgb_image, (render_size, render_size))

        # Convert to tensor format expected by SMALFitter
        rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        # Create temporary SMALFitter for visualization
        temp_fitter = SMALFitter(
            device=device,
            data_batch=rgb_tensor,
            batch_size=1,
            shape_family=config.SHAPE_FAMILY,
            use_unity_prior=False,
            rgb_only=True,
        )

        # CRITICAL: Match propagate_scaling to the training model's setting.
        # The model learns scales with propagate_scaling=True (set in SMILImageRegressor.__init__),
        # so visualization must also use propagate_scaling=True for consistent geometry.
        temp_fitter.propagate_scaling = model.propagate_scaling

        # Set the predicted parameters (ensure they're on the right device)
        temp_fitter.global_rotation.data = global_rot_aa.to(device)
        temp_fitter.joint_rotations.data = joint_rot_aa.to(device)
        temp_fitter.betas.data = predicted_params["betas"].to(device)
        temp_fitter.trans.data = predicted_params["trans"].to(device)

        # Joint scales / translations, expanded from PCA weights when the
        # checkpoint's scale_trans_mode calls for it (see helper docstring).
        apply_scale_trans_to_fitter(model, temp_fitter, predicted_params, device)

        # Camera (identity + calibrated FOV for camera-centric checkpoints,
        # predicted camera for model-centric ones).
        apply_camera_to_fitter(model, temp_fitter, predicted_params, device)

        # Set dummy target joints and visibility for visualization
        temp_fitter.target_joints = torch.zeros((1, config.N_POSE, 2), device=device)
        temp_fitter.target_visibility = torch.ones((1, config.N_POSE), device=device)

        # Generate visualization with custom image exporter wrapper
        class NamedImageExporter:
            def __init__(self, base_exporter, image_name):
                self.base_exporter = base_exporter
                self.image_name = image_name

            def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces, img_idx=0, **kwargs):
                # Call the base exporter with the specific image name; forward any
                # extra kwargs (e.g. epoch) from SMALFitter.generate_visualization.
                self.base_exporter.export(
                    collage_np,
                    batch_id,
                    global_id,
                    img_parameters,
                    vertices,
                    faces,
                    img_idx=img_idx,
                    image_name=self.image_name,
                    **kwargs,
                )

        named_exporter = NamedImageExporter(image_exporter, image_name)
        # Apply the predicted per-sample mesh scale (camera_centric); without it the
        # mesh renders at native size (~35x too large vs the metric 3D).
        mesh_scale_viz = resolve_mesh_scale(model, predicted_params, device)
        temp_fitter.generate_visualization(
            named_exporter, apply_UE_transform=model.use_ue_scaling, img_idx=0, mesh_scale=mesh_scale_viz
        )

        print(f"Generated visualization for {image_name}")

    except Exception as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            warn_render_oom(render_size)
        else:
            print(f"Warning: Failed to generate visualization for {image_name}: {e}")
        # Save just the parameters without visualization
        img_parameters = {
            k: v.cpu().data.numpy() if isinstance(v, torch.Tensor) else v for k, v in predicted_params.items()
        }

        # Create a simple visualization showing the original image
        simple_vis = original_image.copy()
        if simple_vis.max() > 1.0:
            simple_vis = (simple_vis).astype(np.uint8)
        else:
            simple_vis = (simple_vis * 255).astype(np.uint8)

        image_exporter.export(
            simple_vis,
            0,
            0,
            img_parameters,
            torch.zeros(1, 1000, 3),
            np.zeros((1000, 3), dtype=int),
            img_idx=0,
            image_name=image_name,
        )


def process_images_batch(
    model: SMILImageRegressor,
    image_files: List[str],
    output_folder: str,
    device: str,
    crop_mode: str = "centred",
    batch_size: int = 1,
    sleap_helper: Optional[SLEAPCroppingHelper] = None,
    sleap_camera: Optional[str] = None,
    render_resolution: Optional[int] = None,
) -> None:
    """
    Process a batch of images for inference.

    Args:
        model: SMILImageRegressor model
        image_files: List of image file paths
        output_folder: Output folder for results
        device: PyTorch device
        crop_mode: Cropping mode ('centred', 'default', or 'bbox_crop')
        batch_size: Batch size for processing (currently only supports 1)
        sleap_helper: Optional helper for bbox_crop using SLEAP keypoints
        sleap_camera: Optional camera override for bbox_crop
        render_resolution: Collage panel size (default: crop size, capped)
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Create image exporter
    image_exporter = InferenceImageExporter(output_folder)

    print(f"Processing {len(image_files)} images...")
    print(f"Crop mode: {crop_mode}")

    # Process images with progress bar
    for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Get image name for output files
            image_name = Path(image_path).stem

            print(f"\nProcessing image {i + 1}/{len(image_files)}: {image_name}")

            if crop_mode == "bbox_crop" and sleap_helper is not None:
                original_image = imageio.v2.imread(image_path)
                if original_image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")
                preprocess_result = sleap_helper.preprocess_image(
                    original_image, image_path, frame_idx=i, explicit_camera=sleap_camera
                )
                if preprocess_result is None:
                    print("Warning: bbox_crop requested but no keypoints available; falling back to centred crop")
                    preprocessed_image, transform_info = preprocess_frame(
                        original_image, model.input_resolution, crop_mode="centred"
                    )
                else:
                    preprocessed_image, transform_info = preprocess_result
                preprocessed_tensor = torch.from_numpy(preprocessed_image).permute(2, 0, 1).unsqueeze(0)
            else:
                original_image, preprocessed_tensor, transform_info = load_and_preprocess_image(
                    image_path, model, crop_mode
                )

            # Run inference
            predicted_params = run_inference_on_image(model, preprocessed_tensor, device)

            # Generate visualization with unique image name
            generate_visualization(
                model,
                predicted_params,
                original_image,
                image_exporter,
                image_name,
                device,
                transform_info=transform_info,
                render_resolution=render_resolution,
            )

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    print(f"\nProcessing complete! Results saved to: {output_folder}")


def smooth_camera_parameters(
    predicted_params: Dict[str, torch.Tensor], camera_buffer: Dict[str, List], window_size: int
) -> Dict[str, torch.Tensor]:
    """
    Apply moving average smoothing to camera parameters.

    Args:
        predicted_params: Dictionary of predicted parameters
        camera_buffer: Buffer storing recent camera parameters
        window_size: Size of the moving average window

    Returns:
        Smoothed parameters dictionary
    """
    smoothed_params = predicted_params.copy()

    # Add current predictions to buffers
    camera_buffer["cam_rot"].append(predicted_params["cam_rot"].clone())
    camera_buffer["cam_trans"].append(predicted_params["cam_trans"].clone())
    camera_buffer["fov"].append(predicted_params["fov"].clone())

    # Keep only the last window_size frames
    if len(camera_buffer["cam_rot"]) > window_size:
        camera_buffer["cam_rot"].pop(0)
        camera_buffer["cam_trans"].pop(0)
        camera_buffer["fov"].pop(0)

    # Compute moving average
    if len(camera_buffer["cam_rot"]) > 0:
        smoothed_params["cam_rot"] = torch.stack(camera_buffer["cam_rot"]).mean(dim=0)
        smoothed_params["cam_trans"] = torch.stack(camera_buffer["cam_trans"]).mean(dim=0)
        smoothed_params["fov"] = torch.stack(camera_buffer["fov"]).mean(dim=0)

    return smoothed_params


def process_video(
    model: SMILImageRegressor,
    video_path: str,
    output_folder: str,
    device: str,
    crop_mode: str = "centred",
    fps: Optional[int] = None,
    save_frames: bool = False,
    max_frames: int = -1,
    camera_smoothing_window: int = 10,
    sleap_helper: Optional[SLEAPCroppingHelper] = None,
    sleap_camera: Optional[str] = None,
    video_export_mode: str = "overlay",
    animation_recorder: Optional[AnimationRecorder] = None,
    render_resolution: Optional[int] = None,
    smoothing_window: int = 0,
) -> None:
    """
    Process a video file for inference.

    Args:
        model: SMILImageRegressor model
        video_path: Path to input video file
        output_folder: Output folder for results
        device: PyTorch device
        crop_mode: Cropping mode ('centred', 'default', or 'bbox_crop')
        fps: Output video FPS (None = same as input)
        save_frames: Whether to save individual frame results
        max_frames: Maximum number of frames to process (-1 for all frames)
        camera_smoothing_window: Number of frames for moving average of camera parameters (default: 10)
        sleap_helper: Optional helper for bbox_crop using SLEAP keypoints
        sleap_camera: Optional camera override when using bbox_crop
        video_export_mode: Export mode ('overlay' or 'side_by_side')
        render_resolution: Square render size (default: crop size, capped at RENDER_RESOLUTION_CAP)
        smoothing_window: Moving average over all predicted parameters before rendering
            (0 = off). The animation export records the raw predictions.
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_fps = fps if fps is not None else input_fps

    # Determine how many frames to process
    if max_frames > 0:
        frames_to_process = min(max_frames, total_frames)
    else:
        frames_to_process = total_frames

    print("Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  Frames to process: {frames_to_process}")
    print(f"  Input FPS: {input_fps}")
    print(f"  Output FPS: {output_fps}")
    print(f"  Resolution: {frame_width}x{frame_height}")
    print(f"  Crop mode: {crop_mode}")
    print(f"  Camera smoothing window: {camera_smoothing_window} frames")
    print(f"  Parameter smoothing window: {smoothing_window} frames")
    print(f"  Video export mode: {video_export_mode}")

    # Render size: the crop is at most min(frame_h, frame_w) for centred crops,
    # so this matches the per-frame resolution used by the overlay renders.
    render_size = resolve_render_resolution(render_resolution, (frame_height, frame_width))
    print(f"  Render resolution: {render_size}x{render_size}")

    # Determine output video dimensions based on export mode
    if video_export_mode == "side_by_side":
        # For side-by-side: the network's view and the render, each render_size square
        output_height = render_size
        output_width = render_size * 2
    else:
        # For overlay mode: keep original video dimensions
        output_height = frame_height
        output_width = frame_width

    # Create video writer for output
    output_video_path = os.path.join(output_folder, Path(video_path).stem + "_inference.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (output_width, output_height))

    # Initialize moving average buffers for camera parameters
    camera_buffer = {"cam_rot": [], "cam_trans": [], "fov": []}
    smoother = PredictionSmoother(smoothing_window)

    # Optionally create frame exporter
    if save_frames:
        frames_folder = os.path.join(output_folder, "frames")
        os.makedirs(frames_folder, exist_ok=True)
        frame_exporter = InferenceImageExporter(frames_folder)
    else:
        frame_exporter = None

    # Process frames
    frame_idx = 0

    try:
        pbar = tqdm(total=frames_to_process, desc="Processing video")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Stop if we've reached max_frames limit
            if max_frames > 0 and frame_idx >= max_frames:
                break

            try:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Preprocess frame
                target_resolution = model.input_resolution
                if crop_mode == "bbox_crop" and sleap_helper is not None:
                    preprocess_result = sleap_helper.preprocess_image(
                        frame_rgb, video_path, frame_idx=frame_idx, explicit_camera=sleap_camera
                    )
                    if preprocess_result is None:
                        print(
                            f"Warning: Missing SLEAP keypoints for frame {frame_idx}; "
                            "falling back to centred crop for this frame"
                        )
                        preprocessed_image, transform_info = preprocess_frame(
                            frame_rgb, target_resolution, crop_mode="centred"
                        )
                    else:
                        preprocessed_image, transform_info = preprocess_result
                else:
                    preprocessed_image, transform_info = preprocess_frame(frame_rgb, target_resolution, crop_mode)

                # Convert to tensor
                preprocessed_tensor = torch.from_numpy(preprocessed_image).permute(2, 0, 1).unsqueeze(0)

                # Run inference
                predicted_params = run_inference_on_image(model, preprocessed_tensor, device)

                # Record raw (pre-smoothing) parameters for Phase 1 animation export.
                if animation_recorder is not None:
                    animation_recorder.record(predicted_params)

                # Smooth all parameters (no-op at window 0), then the camera on top
                smoothed_params = smoother(predicted_params)
                if camera_smoothing_window > 0:
                    smoothed_params = smooth_camera_parameters(smoothed_params, camera_buffer, camera_smoothing_window)

                # Process frame based on export mode
                if video_export_mode == "side_by_side":
                    # Render model only
                    rendered_model = render_model_only(model, smoothed_params, device, render_size)

                    # The view the network saw (cropped, not squashed), at render_size
                    input_resized = cv2.resize(
                        crop_to_network_view(frame_rgb, transform_info), (render_size, render_size)
                    )

                    # Create side-by-side visualization
                    side_by_side = np.hstack([input_resized, rendered_model])

                    # Convert RGB to BGR for OpenCV
                    output_frame_bgr = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)
                else:
                    # Render prediction onto frame (using smoothed camera parameters)
                    rendered_frame = render_prediction_on_frame(
                        model,
                        smoothed_params,
                        frame_rgb,
                        device,
                        transform_info=transform_info,
                        render_resolution=render_resolution,
                    )

                    # Convert RGB back to BGR for OpenCV
                    output_frame_bgr = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)

                # Write output frame to video
                out.write(output_frame_bgr)

                # Optionally save frame results
                if save_frames and frame_idx % 10 == 0:  # Save every 10th frame
                    frame_name = f"frame_{frame_idx:06d}"
                    try:
                        generate_visualization(
                            model,
                            predicted_params,
                            frame_rgb,
                            frame_exporter,
                            frame_name,
                            device,
                            transform_info=transform_info,
                            render_resolution=render_resolution,
                        )
                    except Exception as e:
                        print(f"Warning: Failed to save frame {frame_idx}: {e}")

                frame_idx += 1
                pbar.update(1)

            except Exception as e:
                print(f"Warning: Failed to process frame {frame_idx}: {e}")
                # Write original frame on error
                out.write(frame)
                frame_idx += 1
                pbar.update(1)
                continue

        pbar.close()

    finally:
        cap.release()
        out.release()

    print("\nVideo processing complete!")
    print(f"  Output video: {output_video_path}")
    print(f"  Processed {frame_idx} frames")
    if save_frames:
        print(f"  Frame results: {frames_folder}")


# ═════════════════════════════════════════════════════════════════════════════
# Dataset inference
#
# Mirrors run_multiview_inference.py: consume a preprocessed HDF5 dataset
# directly instead of raw images/video, so single-view inference is evaluated
# under exactly the conventions the checkpoint was trained with.
#
# The three things that differ from the image/video paths — and that this
# section exists to get right:
#
#   * Camera. A dataset carries real calibration. For a *camera-centric*
#     checkpoint the dataset re-anchors the sampled view's camera to the world
#     origin, so the render camera is the PyTorch3D identity and the FOV comes
#     from that view's intrinsics K (NOT --fov / 60 deg, which is only a
#     fallback for uncalibrated footage). For a *model-centric* checkpoint the
#     network predicts its own camera and the dataset's camera is not used.
#     Non-square intrinsics are carried through as `cam_aspect`.
#   * Shape space. `log_beta_scales` / `betas_trans` are PCA weights in
#     'separate' mode and per-joint values in 'entangled_with_betas'; see
#     apply_scale_trans_to_fitter().
#   * Cropping + mesh scale. Dataset images are ALREADY cropped and resized by
#     the preprocessor according to the dataset's own `crop_mode`, so --crop_mode
#     does not apply and no second crop is performed. Mesh placement follows the
#     checkpoint: legacy 10x UE scaling, or the predicted per-sample mesh_scale.
# ═════════════════════════════════════════════════════════════════════════════


def build_singleview_dataset(
    dataset_path: str,
    model: SMILImageRegressor,
    model_config: Dict[str, Any],
    preferred_view: int = 0,
):
    """Open *dataset_path* as a stream of single-view items.

    The dataset MUST be constructed with the same convention the checkpoint was
    trained under, because the convention changes the *data*, not just the
    render: with ``camera_centric=True`` SLEAPMultiViewDataset re-expresses each
    sampled view's camera as the world origin and transforms the 3D keypoints
    and root pose into that camera's frame. Building a camera-centric checkpoint's
    dataset in model-centric mode (or vice versa) silently yields a valid-looking
    but wrong camera.

    Multi-view HDF5s are opened in single-view mode with ``expand_all_views`` so
    every camera of every frame is its own item; single-view HDF5s are opened
    directly by the unified factory.
    """
    from smal_fitter.neuralSMIL.smil_datasets import UnifiedSMILDataset

    frame_convention = model_config.get("frame_convention", "model_centric")
    camera_centric = frame_convention == "camera_centric"
    rotation_representation = getattr(model, "rotation_representation", "6d")

    is_multiview_h5 = False
    if str(dataset_path).endswith((".h5", ".hdf5")):
        with h5py.File(dataset_path, "r") as f:
            if "metadata" in f:
                is_multiview_h5 = bool(f["metadata"].attrs.get("is_multiview", False))

    kwargs: Dict[str, Any] = dict(
        rotation_representation=rotation_representation,
        backbone_name=model.backbone_name,
    )
    if is_multiview_h5:
        kwargs.update(
            return_single_view=True,
            camera_centric=camera_centric,
            expand_all_views=True,
            preferred_view=preferred_view,
            augment=False,
        )
    elif camera_centric:
        raise ValueError(
            f"Checkpoint was trained with frame_convention='camera_centric', which requires a "
            f"multi-view HDF5 (the camera is re-anchored per view). {dataset_path} is not one."
        )

    dataset = UnifiedSMILDataset.from_path(str(dataset_path), **kwargs)
    return dataset, is_multiview_h5


def compute_view_item_indices(dataset, view_indices: List[int]) -> Dict[int, List[int]]:
    """Map each requested camera slot to its dataset item indices, in frame order.

    In ``expand_all_views`` mode items are ``(sample, view)`` pairs enumerated
    sample-major, so consecutive items are *different cameras of the same frame*.
    Filtering to one view slot recovers a temporally-ordered clip per camera —
    which is what makes an output video (and temporal smoothing) meaningful.

    For a single-view HDF5 there are no view slots; everything maps to slot 0.
    """
    sv_items = getattr(dataset, "_sv_items", None)
    if sv_items is None:
        return {view_indices[0] if view_indices else 0: list(range(len(dataset)))}

    per_view: Dict[int, List[int]] = {v: [] for v in view_indices}
    for item_idx, (_sample_idx, view_slot) in enumerate(sv_items):
        if view_slot in per_view:
            per_view[view_slot].append(item_idx)
    return per_view


def compute_subclip_ranges(
    num_items: int,
    max_frames: Optional[int],
    num_subclips: int,
) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` index ranges (end exclusive) for each subclip.

    Same semantics as run_multiview_inference.py: ``num_subclips`` evenly-spaced
    starts, each running for ``max_frames`` items. Falls back to one full clip.
    """
    if num_subclips <= 1:
        end = min(max_frames, num_items) if max_frames else num_items
        return [(0, end)]

    if not max_frames:
        print(f"WARNING: --generate_num_subclips={num_subclips} requires --max_frames; using a single full clip.")
        return [(0, num_items)]

    slot_size = num_items // num_subclips
    if slot_size < max_frames:
        print(
            f"WARNING: {num_items} frames cannot hold {num_subclips} subclips of {max_frames} "
            f"frames ({slot_size} per slot); using a single full clip."
        )
        return [(0, num_items)]

    return [
        (i * num_items // num_subclips, min(i * num_items // num_subclips + max_frames, num_items))
        for i in range(num_subclips)
    ]


# Non-square calibrated aspects already warned about (see render_dataset_sample_collage).
_WARNED_NONSQUARE_ASPECTS: set = set()


def render_dataset_sample_collage(
    model: SMILImageRegressor,
    x_data: Dict[str, Any],
    y_data: Dict[str, Any],
    predicted_params: Dict[str, torch.Tensor],
    device: str,
    disable_scaling: bool = False,
    disable_translation: bool = False,
    render_resolution: Optional[int] = None,
    img_idx: int = 0,
    use_calibrated_aspect: bool = False,
) -> Optional[np.ndarray]:
    """Render one dataset item's mesh overlay collage.

    The single-view counterpart of ``render_singleview_collage()`` in
    run_multiview_inference.py, and deliberately identical to
    ``visualize_training_progress()`` in train_smil_regressor.py so an inference
    frame and a training-visualisation frame of the same sample agree.

    NOTE ON CROPPING: ``x_data['input_image_data']`` is the image the
    preprocessor already cropped and resized per the dataset's own ``crop_mode``
    — the exact pixels the network is fed. It is used as-is; the only resampling
    here is the optional cosmetic upscale to ``render_resolution``.
    """
    image = x_data.get("input_image_data")
    if image is None:
        return None

    native_size = int(image.shape[0])
    target_size = int(render_resolution) if render_resolution else native_size

    if target_size != native_size:
        from PIL import Image

        pil_img = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))
        pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
        image = np.array(pil_img).astype(np.float32) / 255.0

    image = np.clip(image.astype(np.float32), 0.0, 1.0)
    rgb = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

    # Ground-truth 2D keypoints, scaled to the resolution the mesh is actually
    # rendered at (SMALFitter renders at rgb's size, not model.input_resolution).
    keypoints_2d = y_data.get("keypoints_2d")
    visibility = y_data.get("keypoint_visibility")
    has_keypoints = keypoints_2d is not None and visibility is not None

    if has_keypoints:
        pixel_coords = np.asarray(keypoints_2d, dtype=np.float32).copy()
        pixel_coords[:, 0] = pixel_coords[:, 0] * target_size
        pixel_coords[:, 1] = pixel_coords[:, 1] * target_size
        num_joints = pixel_coords.shape[0]
        joints_t = torch.tensor(pixel_coords.reshape(1, num_joints, 2), dtype=torch.float32)
        vis_t = torch.tensor(np.asarray(visibility).reshape(1, num_joints), dtype=torch.float32)
        sil = torch.zeros(1, 1, target_size, target_size)
        temp_batch = (rgb, sil, joints_t, vis_t)
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

    # CRITICAL: the model learned its scales with propagate_scaling=True; the
    # visualisation must use the same setting or the kinematic chain differs.
    temp_fitter.propagate_scaling = model.propagate_scaling

    if has_keypoints:
        temp_fitter.target_joints = joints_t.to(device)
        temp_fitter.target_visibility = vis_t.to(device)
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
    temp_fitter.betas.data = predicted_params["betas"][0:1].detach().to(device)
    temp_fitter.trans.data = predicted_params["trans"][0:1].detach().to(device)

    apply_scale_trans_to_fitter(
        model,
        temp_fitter,
        predicted_params,
        device,
        disable_scaling=disable_scaling,
        disable_translation=disable_translation,
    )

    # Camera aspect ratio.
    #
    # The dataset carries the view's true `cam_aspect` = (W*fy)/(H*fx), which the
    # MULTI-view pipeline feeds to FoVPerspectiveCameras. The SINGLE-view
    # regressor does not: neither smil_image_regressor.py nor
    # train_smil_regressor.py ever touches aspect_ratio, so every single-view
    # checkpoint was trained with the renderer default of 1.0. Rendering
    # inference with the calibrated aspect would therefore NOT reproduce the
    # checkpoint's own convention — it would silently reproject the mesh
    # differently from the loss that trained it.
    #
    # So: match training (aspect = 1.0) by default, and warn once when the
    # dataset's calibration is meaningfully non-square, because that is a real
    # data/model mismatch the user should know about rather than a render bug.
    # `use_calibrated_aspect` opts into the dataset's value for A/B comparison.
    aspect = None
    raw_aspect = y_data.get("cam_aspect")
    if raw_aspect is not None:
        try:
            dataset_aspect = float(np.asarray(raw_aspect).reshape(-1)[0])
        except Exception:
            dataset_aspect = None
        if dataset_aspect is not None:
            if use_calibrated_aspect:
                aspect = dataset_aspect
            elif abs(dataset_aspect - 1.0) > 0.01 and round(dataset_aspect, 3) not in _WARNED_NONSQUARE_ASPECTS:
                _WARNED_NONSQUARE_ASPECTS.add(round(dataset_aspect, 3))
                print(
                    f"WARNING: this view's calibrated aspect ratio is {dataset_aspect:.4f}, not 1.0, but the "
                    f"single-view regressor is trained with aspect_ratio=1.0 (it never reads cam_aspect). "
                    f"Rendering at 1.0 to match training. Pass --use_calibrated_aspect to render with "
                    f"{dataset_aspect:.4f} instead — the overlay may fit better, but it will no longer "
                    f"reflect the projection the model was optimised under."
                )

    apply_camera_to_fitter(model, temp_fitter, predicted_params, device, aspect_ratio=aspect)

    exporter = InMemoryImageExporter()
    temp_fitter.generate_visualization(
        exporter,
        # MUST match the model setting — this is the mesh placement convention.
        apply_UE_transform=model.use_ue_scaling,
        img_idx=img_idx,
        mesh_scale=resolve_mesh_scale(model, predicted_params, device),
    )
    return exporter.image


def run_dataset_inference_phase(
    dataset,
    model: SMILImageRegressor,
    device: str,
    indices: List[int],
    desc: str = "Running inference",
) -> List[Tuple[int, dict]]:
    """Forward-pass the given item indices; return ``(idx, params_on_cpu)``."""
    model.eval()
    raw_predictions: List[Tuple[int, dict]] = []

    for item_idx in tqdm(indices, desc=desc):
        try:
            x_data, y_data = dataset[item_idx]
            image = x_data.get("input_image_data")
            if image is None:
                continue

            image_tensor = model.preprocess_image(image).to(device)

            # Calibrated FOV for this view. Only consumed by camera-centric
            # checkpoints, where the camera heads are unsupervised and the FOV
            # must come from the intrinsics (mirrors predict_from_batch).
            gt_fov = y_data.get("cam_fov")
            if isinstance(gt_fov, (list, tuple, np.ndarray)):
                gt_fov = float(np.asarray(gt_fov).reshape(-1)[0])

            predicted_params = run_inference_on_image(model, image_tensor, device, gt_fov=gt_fov)
            raw_predictions.append((item_idx, params_to_cpu(predicted_params)))
        except Exception as e:
            print(f"Warning: inference failed for item {item_idx}: {e}")
            continue

    return raw_predictions


def process_dataset(
    model: SMILImageRegressor,
    model_config: Dict[str, Any],
    dataset_path: str,
    output_folder: str,
    device: str,
    view_indices: List[int],
    fps: int = 30,
    max_frames: Optional[int] = None,
    num_subclips: int = 1,
    smoothing_window: int = 0,
    disable_scaling: bool = False,
    disable_translation: bool = False,
    render_resolution: Optional[int] = None,
    export_animation: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    save_frames: bool = False,
    use_calibrated_aspect: bool = False,
    split: str = "all",
) -> None:
    """Run inference over a preprocessed dataset and write one video per view.

    ``split`` other than "all" keeps only the frames of that part of the
    checkpoint's train/val/test split (see data_splits.add_inference_split_arg).
    """
    os.makedirs(output_folder, exist_ok=True)

    dataset, is_multiview_h5 = build_singleview_dataset(dataset_path, model, model_config)

    dataset_crop_mode = getattr(dataset, "crop_mode", "unknown")
    frame_convention = model_config.get("frame_convention", "model_centric")

    print("\nDataset inference")
    print(f"  Dataset: {dataset_path}")
    print(f"  Items: {len(dataset)}" + (" (one per camera view per frame)" if is_multiview_h5 else ""))
    print(f"  Dataset crop_mode: {dataset_crop_mode} (images used as stored — --crop_mode does not apply)")
    print(f"  Dataset target resolution: {dataset.get_target_resolution()}")
    print(f"  Frame convention: {frame_convention}")
    if frame_convention == "camera_centric":
        print("    → fixed identity camera; FOV taken from each view's calibrated intrinsics")
    else:
        print("    → camera predicted by the network")
    print(f"  scale_trans_mode: {getattr(model, 'scale_trans_mode', 'separate')}")
    print(
        f"  Camera aspect ratio: "
        f"{'calibrated (--use_calibrated_aspect)' if use_calibrated_aspect else '1.0 (matches single-view training)'}"
    )
    print(
        f"  Mesh placement: {'UE 10x scaling' if model.use_ue_scaling else 'predicted mesh_scale' if getattr(model, 'allow_mesh_scaling', False) else 'translation only'}"
    )
    if hasattr(dataset, "world_scale"):
        print(f"  World scale (baked into dataset 3D/camera units): {dataset.world_scale}")

    per_view_items = compute_view_item_indices(dataset, view_indices)
    if split != "all":
        _, sample_split = split_dataset_from_settings(dataset, model_config["split_settings"])
        keep = set(getattr(sample_split, split))
        item_samples = getattr(dataset, "item_sample_indices", None)
        sample_of = (lambda i: int(item_samples[i])) if item_samples is not None else (lambda i: i)
        per_view_items = {v: [i for i in idxs if sample_of(i) in keep] for v, idxs in per_view_items.items()}
        print(f"  Frames: {split} split only ({sample_split.description})")
    available = {v: idxs for v, idxs in per_view_items.items() if idxs}
    if not available:
        raise ValueError(
            f"No dataset items for view indices {view_indices}. "
            f"Dataset has view slots 0..{getattr(dataset, 'max_views', 1) - 1}."
        )
    for v in view_indices:
        if v not in available:
            print(f"WARNING: view index {v} has no items in this dataset; skipping.")

    dataset_name = Path(dataset_path).stem if split == "all" else f"{Path(dataset_path).stem}_{split}"
    multi_view_out = len(available) > 1

    for view_idx, item_indices in available.items():
        subclip_ranges = compute_subclip_ranges(len(item_indices), max_frames, num_subclips)
        multi_subclip = len(subclip_ranges) > 1

        for start_idx, end_idx in subclip_ranges:
            clip_items = item_indices[start_idx:end_idx]
            if not clip_items:
                continue

            range_suffix = f"_frames{start_idx:06d}-{end_idx:06d}" if multi_subclip else ""
            view_suffix = f"_view{view_idx}" if multi_view_out else ""
            label = f"view {view_idx}" + (f" frames [{start_idx}, {end_idx})" if multi_subclip else "")
            print(f"\n── {label}: {len(clip_items)} frames ──")

            # ── Phase 1: inference ──────────────────────────────────────────
            raw_predictions = run_dataset_inference_phase(
                dataset, model, device, clip_items, desc=f"Inference ({label})"
            )
            if not raw_predictions:
                print(f"  No predictions for {label}; skipping.")
                continue

            # ── Phase 1b: animation export (raw, pre-smoothing) ─────────────
            if export_animation:
                recorder = build_recorder_from_config(
                    output_path=f"{export_animation}{view_suffix}{range_suffix}",
                    rotation_representation=getattr(model, "rotation_representation", "6d"),
                    fps=float(fps),
                    source_checkpoint=str(checkpoint_path),
                    source_input=str(dataset_path),
                    model_id=getattr(model, "model_id", None),
                )
                for _, params in raw_predictions:
                    recorder.record(params)
                written = recorder.write()
                print(f"  Animation export: {written['npz']} + {written['json']} ({recorder.num_frames()} frames)")

            # ── Phase 2: temporal smoothing ─────────────────────────────────
            if smoothing_window > 0:
                smoother = PredictionSmoother(smoothing_window)
                smoothed = {
                    idx: smoother(params)
                    for idx, params in tqdm(raw_predictions, desc=f"Smoothing (window={smoothing_window})")
                }
            else:
                smoothed = dict(raw_predictions)
            del raw_predictions

            # ── Phase 3+4: render and stream straight into the video ────────
            #
            # Frames are written to the VideoWriter as they are produced rather
            # than buffered. A full pass over a large dataset is easily 10^5
            # frames per view; at 512x512x3 that is ~0.8 MB each, so buffering
            # would need hundreds of GB of RAM. The writer is opened lazily on
            # the first successful render because its frame size comes from the
            # rendered collage, and every later frame is padded/resized to match.
            frame_exporter = None
            if save_frames:
                frames_folder = os.path.join(output_folder, f"frames{view_suffix}{range_suffix}")
                os.makedirs(frames_folder, exist_ok=True)
                frame_exporter = InferenceImageExporter(frames_folder)

            out_path = os.path.join(
                output_folder, f"{dataset_name}{view_suffix}{range_suffix}_singleview_inference.mp4"
            )
            writer: Optional[cv2.VideoWriter] = None
            frame_size: Optional[Tuple[int, int]] = None
            frames_written = 0

            try:
                for i, item_idx in enumerate(tqdm(clip_items, desc=f"Rendering ({label})")):
                    if item_idx not in smoothed:
                        continue
                    try:
                        x_data, y_data = dataset[item_idx]
                        params = params_to_device(smoothed[item_idx], device)
                        collage = render_dataset_sample_collage(
                            model,
                            x_data,
                            y_data,
                            params,
                            device,
                            disable_scaling=disable_scaling,
                            disable_translation=disable_translation,
                            render_resolution=render_resolution,
                            img_idx=i,
                            use_calibrated_aspect=use_calibrated_aspect,
                        )
                        if collage is None:
                            continue

                        if writer is None:
                            frame_size = (collage.shape[1], collage.shape[0])
                            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
                            if not writer.isOpened():
                                raise RuntimeError(f"Failed to open VideoWriter for {out_path} at {frame_size}")

                        writer.write(cv2.cvtColor(pad_or_resize(collage, frame_size), cv2.COLOR_RGB2BGR))
                        frames_written += 1

                        if frame_exporter is not None and i % 10 == 0:
                            frame_exporter.export(
                                collage,
                                0,
                                item_idx,
                                {k: v.cpu().numpy() for k, v in params.items() if isinstance(v, torch.Tensor)},
                                torch.zeros(1, 1, 3),
                                np.zeros((1, 3), dtype=int),
                                img_idx=i,
                                image_name=f"item_{item_idx:06d}",
                            )
                    except Exception as e:
                        print(f"Warning: render failed for item {item_idx}: {e}")
                        continue
            finally:
                if writer is not None:
                    writer.release()

            del smoothed

            if frames_written == 0 or frame_size is None:
                print(f"  No frames rendered for {label}.")
                continue
            print(f"  Wrote {out_path} ({frames_written} frames at {frame_size[0]}x{frame_size[1]})")

    if hasattr(dataset, "close"):
        try:
            dataset.close()
        except Exception:
            pass


def main():
    """Main function for the inference script."""
    parser = argparse.ArgumentParser(
        description="Run SMIL inference on images or video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process images
  python run_singleview_inference.py --checkpoint checkpoints/best_model.pth --input_folder test_images --output_folder results
  python run_singleview_inference.py -c model.pth -i images/ -o output/ --crop_mode centred

  # Process video
  python run_singleview_inference.py --checkpoint model.pth --input_video video.mp4 --output_folder results
  python run_singleview_inference.py -c model.pth -v video.mp4 -o output/ --save_frames --fps 30

  # Process a preprocessed HDF5 dataset (mirrors run_multiview_inference.py)
  python run_singleview_inference.py -c model.pth -d dataset.h5 -o output/
  python run_singleview_inference.py -c model.pth -d dataset.h5 -o output/ \\
      --view_indices 0,2 --smoothing_window 5 --render_resolution 512

  # With different preprocessing
  python run_singleview_inference.py -c model.pth -i images/ -o output/ --crop_mode default

Supported image formats: jpg, jpeg, png, bmp, tiff, tif (case-insensitive)
Supported video formats: mp4, avi, mov, mkv (anything supported by OpenCV)
        """,
    )

    parser.add_argument(
        "-c", "--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--smal_file",
        type=str,
        default=None,
        help="Path to a SMAL/SMIL model file overriding config.SMAL_FILE. Normally unnecessary: "
        "the checkpoint's own smal_file is applied automatically. Use this only to override it, "
        "or when the recorded path does not resolve on this machine.",
    )
    parser.add_argument(
        "--shape_family",
        type=int,
        default=None,
        help="Shape family override (legacy SMAL setting). Default: the checkpoint's recorded "
        "shape_family, else config.SHAPE_FAMILY.",
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--input_folder", type=str, help="Path to folder containing input images")
    input_group.add_argument("-v", "--input_video", type=str, help="Path to input video file")
    input_group.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Path to a preprocessed HDF5 dataset (.h5). Mirrors run_multiview_inference.py: "
        "runs over every frame of the dataset and writes one video per camera view. "
        "Images are used exactly as the preprocessor stored them (its own crop_mode), "
        "and the camera/scale conventions are taken from the checkpoint — so this is the "
        "input mode that reproduces training conditions. --crop_mode and --fov are ignored.",
    )

    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to folder for saving results")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for image-folder inference (default: 1).",
    )

    parser.add_argument(
        "--fov",
        type=float,
        default=None,
        help="Vertical field-of-view in degrees for camera-centric checkpoints (fixed identity camera). "
        "Fallback chain: --fov -> 60.0 (pytorch3d / codebase default). "
        "Ignored for legacy model-centric checkpoints (which predict their own camera), and "
        "ignored in --dataset mode, where each view's FOV is derived from its stored intrinsics.",
    )

    # Preprocessing options
    parser.add_argument(
        "--crop_mode",
        type=str,
        default="centred",
        choices=["centred", "default", "bbox_crop"],
        help="Image preprocessing mode: centred=center crop (preserves aspect ratio), "
        "default=direct resize (may distort), bbox_crop=SLEAP-driven bounding box crop. "
        "Should match training preprocessing. (default: centred)",
    )
    parser.add_argument(
        "--render_resolution",
        type=int,
        default=None,
        help="Square pixel resolution the mesh is rendered at. Cosmetic only: inference always runs "
        "at the backbone's input resolution. Default for --input_folder / --input_video: the crop "
        f"size, capped at {RENDER_RESOLUTION_CAP}; an explicit value is used as given (very large "
        "values may run out of GPU memory, which is reported). Default for --dataset: the dataset's "
        "stored image resolution.",
    )

    # Processing options
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference (default: auto), using cuda if available",
    )

    # Video-specific options
    parser.add_argument("--fps", type=int, default=None, help="Output video FPS (default: same as input)")
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help="Save individual frame results (every 10th frame) when processing a video or a dataset",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Maximum number of frames to process from a video, or per view (and per subclip) "
        "in --dataset mode (default: -1 for all frames)",
    )
    parser.add_argument(
        "--camera_smoothing",
        type=int,
        default=0,
        help="Moving average window size for camera parameter smoothing (default: 0, set to 0 to disable)",
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=0,
        help="Moving-average window over ALL predicted parameters before rendering, for --input_video "
        "and --dataset (per view) (default: 0, disabled). The animation export stays unsmoothed. "
        "--camera_smoothing averages camera parameters only, which has no effect for camera-centric "
        "checkpoints (fixed camera).",
    )
    parser.add_argument(
        "--video_export_mode",
        type=str,
        default="overlay",
        choices=["overlay", "side_by_side"],
        help="Video export mode: overlay=blend model onto input (default), "
        "side_by_side=display input and rendered model side by side at same resolution",
    )

    # SLEAP-specific options
    parser.add_argument(
        "--sleap_project", type=str, default=None, help="Path to SLEAP project directory (required for bbox_crop)"
    )
    parser.add_argument(
        "--sleap_camera",
        type=str,
        default=None,
        help="Optional camera name override when using bbox_crop with SLEAP data",
    )

    # Animation export (Phase 1)
    parser.add_argument(
        "--export_animation",
        type=str,
        default=None,
        help="Optional output path stem for SMIL animation export. "
        "Writes <stem>.npz + <stem>.json alongside the MP4. "
        "Active for --input_video and --dataset (dataset mode suffixes the stem "
        "with the view index and frame range when it writes more than one clip). "
        'NOTE: any string is accepted as-is (e.g. "True" writes True.npz) — '
        "no validation is performed, so pass a real path/filename stem.",
    )

    # ── Dataset-mode options (mirroring run_multiview_inference.py) ──────────
    dataset_group = parser.add_argument_group("dataset mode (--dataset)")
    dataset_group.add_argument(
        "--view_indices",
        type=str,
        default="0",
        help="Comma-separated camera view slots to run, e.g. '0,4,11' (default: '0'). "
        "Each slot yields its own temporally-ordered output video. Multi-view HDF5s only; "
        "single-view HDF5s have one implicit slot.",
    )
    add_inference_split_arg(dataset_group)
    dataset_group.add_argument(
        "--generate_num_subclips",
        type=int,
        default=1,
        help="Generate N subclips evenly spaced across each view's frames, each --max_frames "
        "long. Output files are suffixed with the frame range. Falls back to a single full "
        "clip if the subclips do not fit. Default: 1.",
    )
    dataset_group.add_argument(
        "--disable_scaling",
        action="store_true",
        help="Do not apply predicted per-joint scaling (log_beta_scales) when rendering. Debug/comparison only.",
    )
    dataset_group.add_argument(
        "--disable_translation",
        action="store_true",
        help="Do not apply predicted per-joint translation (betas_trans) when rendering. Debug/comparison only.",
    )
    dataset_group.add_argument(
        "--use_calibrated_aspect",
        action="store_true",
        help="Render with the dataset view's calibrated aspect ratio ((W*fy)/(H*fx)) instead of 1.0. "
        "OFF by default because the single-view regressor is trained with aspect_ratio=1.0 — it never "
        "reads cam_aspect — so the default reproduces the checkpoint's own projection. Enable only to "
        "compare against the multi-view pipeline, which does use the calibrated value.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SMIL Image Regressor - Inference Script")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    if args.input_folder:
        print(f"Input folder: {args.input_folder}")
    if args.input_video:
        print(f"Input video: {args.input_video}")
    if args.dataset:
        print(f"Dataset: {args.dataset}")
    print(f"Output folder: {args.output_folder}")
    if args.dataset:
        # Dataset images are stored pre-cropped by the preprocessor; re-cropping
        # them here would desync the image from its stored keypoints/intrinsics.
        print("Crop mode: taken from the dataset (--crop_mode ignored)")
    else:
        print(f"Crop mode: {args.crop_mode}")
    if args.sleap_project:
        print(f"SLEAP project: {args.sleap_project}")
        if args.sleap_camera:
            print(f"SLEAP camera override: {args.sleap_camera}")
    if args.input_video:
        print(f"Save frames: {args.save_frames}")
        if args.fps:
            print(f"Output FPS: {args.fps}")
        if args.max_frames > 0:
            print(f"Max frames: {args.max_frames}")
        else:
            print("Max frames: All frames")

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = "cpu"

    print(f"Device: {device}")

    if args.crop_mode == "bbox_crop" and not args.sleap_project and not args.dataset:
        print("Error: bbox_crop mode requires --sleap-project to supply keypoints.")
        return 1

    sleap_helper = None
    try:
        # Load model from checkpoint (applies --smal_file, else the checkpoint's
        # own smal_file, before the model is built)
        print("\n" + "=" * 40)
        print("Loading model...")
        model, model_config = load_model_from_checkpoint(
            args.checkpoint, device, cli_smal_file=args.smal_file, cli_shape_family=args.shape_family
        )

        # Resolve the inference FOV for camera-centric checkpoints. A raw image
        # carries no GT calibration, so the fallback chain is: --fov -> 60.0
        # (the pytorch3d / codebase default). Stashed on the model for the
        # fixed-camera override in run_inference_on_image.
        chosen_fov = args.fov if args.fov is not None else 60.0
        model._inference_fov = chosen_fov
        if getattr(model, "fixed_camera", False):
            if args.dataset:
                print(
                    "Camera-centric checkpoint: fixed identity camera, FOV taken per view from the "
                    "dataset's calibrated intrinsics (--fov unused)"
                )
            else:
                src = "from --fov" if args.fov is not None else "default"
                print(f"Camera-centric checkpoint: fixed identity camera, FOV={chosen_fov} deg ({src})")

        if args.crop_mode == "bbox_crop" and not args.dataset:
            sleap_helper = SLEAPCroppingHelper(
                project_path=args.sleap_project,
                crop_mode=args.crop_mode,
                target_resolution=model.input_resolution,
                backbone_name=model.backbone_name,
            )
            print(f"SLEAP project loaded from {args.sleap_project}")
            print(f"Available SLEAP cameras: {sleap_helper.list_cameras()}")

        # Process based on input type
        if args.dataset:
            print("\n" + "=" * 40)
            print("Running inference on dataset...")
            view_indices = [int(v.strip()) for v in args.view_indices.split(",") if v.strip()]
            process_dataset(
                model=model,
                model_config=model_config,
                dataset_path=args.dataset,
                output_folder=args.output_folder,
                device=device,
                view_indices=view_indices,
                fps=args.fps if args.fps is not None else 30,
                max_frames=args.max_frames if args.max_frames > 0 else None,
                num_subclips=args.generate_num_subclips,
                smoothing_window=args.smoothing_window,
                disable_scaling=args.disable_scaling,
                disable_translation=args.disable_translation,
                render_resolution=args.render_resolution,
                export_animation=args.export_animation,
                checkpoint_path=args.checkpoint,
                save_frames=args.save_frames,
                use_calibrated_aspect=args.use_calibrated_aspect,
                split=args.split,
            )

        elif args.input_folder:
            # Find image files
            print("\n" + "=" * 40)
            print("Finding images...")
            image_files = find_image_files(args.input_folder)

            if len(image_files) == 0:
                print("No image files found in the input folder!")
                return 1

            # Process images
            print("\n" + "=" * 40)
            print("Running inference on images...")
            process_images_batch(
                model,
                image_files,
                args.output_folder,
                device,
                args.crop_mode,
                args.batch_size,
                sleap_helper=sleap_helper,
                sleap_camera=args.sleap_camera,
                render_resolution=args.render_resolution,
            )

        elif args.input_video:
            # Process video
            print("\n" + "=" * 40)
            print("Running inference on video...")

            animation_recorder: Optional[AnimationRecorder] = None
            if args.export_animation:
                output_fps = (
                    args.fps if args.fps is not None else cv2.VideoCapture(args.input_video).get(cv2.CAP_PROP_FPS)
                )
                animation_recorder = build_recorder_from_config(
                    output_path=args.export_animation,
                    rotation_representation=model.rotation_representation,
                    fps=float(output_fps),
                    source_checkpoint=args.checkpoint,
                    source_input=args.input_video,
                    model_id=getattr(model, "model_id", None),
                )
                print(f"Animation export enabled: {args.export_animation}.[npz|json]")

            process_video(
                model,
                args.input_video,
                args.output_folder,
                device,
                args.crop_mode,
                args.fps,
                args.save_frames,
                args.max_frames,
                args.camera_smoothing,
                sleap_helper=sleap_helper,
                sleap_camera=args.sleap_camera,
                video_export_mode=args.video_export_mode,
                animation_recorder=animation_recorder,
                render_resolution=args.render_resolution,
                smoothing_window=args.smoothing_window,
            )

            if animation_recorder is not None and animation_recorder.num_frames() > 0:
                written = animation_recorder.write()
                print(
                    f"Animation export written: {written['npz']} + {written['json']} "
                    f"({animation_recorder.num_frames()} frames)"
                )

        print("\n" + "=" * 60)
        print("Inference completed successfully!")
        print(f"Results saved to: {args.output_folder}")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        print("\nInference interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if sleap_helper is not None:
            sleap_helper.close()


if __name__ == "__main__":
    exit(main())
