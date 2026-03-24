"""
Multi-View SLEAP Dataset Class for SMILify Training Pipeline

This module provides a PyTorch dataset class for loading preprocessed multi-view SLEAP datasets
that is compatible with the multi-view SMILify training pipeline.
"""

import os
import torch
import h5py
import numpy as np
import cv2
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config


class SLEAPMultiViewDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset class for loading preprocessed multi-view SLEAP datasets.
    
    Each sample contains multiple synchronized camera views of the same frame,
    enabling multi-view training for improved pose and shape estimation.
    
    Key Features:
    - Variable number of views per sample (no padding, use masking)
    - Optional view sampling when more views available than needed
    - Random or fixed view selection
    - Backward compatibility mode for single-view training
    """
    
    def __init__(self,
                 hdf5_path: str,
                 rotation_representation: str = '6d',
                 num_views_to_use: Optional[int] = None,
                 return_single_view: bool = False,
                 preferred_view: int = 0,
                 random_view_sampling: bool = True,
                 backbone_name: str = None,
                 augment: bool = False,
                 augmentation_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize the multi-view SLEAP dataset.
        
        Args:
            hdf5_path: Path to the preprocessed multi-view HDF5 file
            rotation_representation: Rotation representation ('6d' or 'axis_angle')
            num_views_to_use: Maximum views to use per sample (None = use all available)
                             If sample has more views, sample this many
                             If sample has fewer views, use all available (no padding)
            return_single_view: If True, return only one view (backward compatible)
            preferred_view: Which view to return if return_single_view=True
            random_view_sampling: If True, randomly sample views when more available
            backbone_name: Backbone name (for compatibility)
            **kwargs: Additional keyword arguments (for compatibility)
        """
        self.hdf5_path = hdf5_path
        self.rotation_representation = rotation_representation
        self.num_views_to_use = num_views_to_use
        self.return_single_view = return_single_view
        self.preferred_view = preferred_view
        self.random_view_sampling = random_view_sampling
        self.augment = augment

        # Augmentation defaults (overridden by augmentation_config)
        aug = augmentation_config or {}
        self.aug_color_jitter_brightness = aug.get('color_jitter_brightness', 0.2)
        self.aug_color_jitter_contrast = aug.get('color_jitter_contrast', 0.2)
        self.aug_color_jitter_saturation = aug.get('color_jitter_saturation', 0.15)
        self.aug_gaussian_noise_std = aug.get('gaussian_noise_std', 0.015)
        self.aug_gaussian_blur_prob = aug.get('gaussian_blur_prob', 0.3)
        self.aug_gaussian_blur_kernel_range = tuple(aug.get('gaussian_blur_kernel_range', (3, 7)))
        self.aug_random_erasing_prob = aug.get('random_erasing_prob', 0.2)
        self.aug_random_erasing_scale_range = tuple(aug.get('random_erasing_scale_range', (0.02, 0.1)))
        self.aug_crop_jitter_fraction = aug.get('crop_jitter_fraction', 0.05)
        self.aug_scale_jitter_range = tuple(aug.get('scale_jitter_range', (0.9, 1.1)))
        
        # Load dataset metadata
        self._load_metadata()
        
        # File handle (lazy loading for multiprocessing compatibility)
        self._file = None
        
        # Validate dataset
        self._validate_dataset()
    
    def _load_metadata(self):
        """Load dataset metadata from HDF5 file."""
        with h5py.File(self.hdf5_path, 'r') as f:
            metadata = f['metadata']
            
            self.num_samples = metadata.attrs['num_samples']
            self.max_views = metadata.attrs['max_views']
            self.n_joints = metadata.attrs['n_joints']
            self.target_resolution = metadata.attrs['target_resolution']
            self.backbone_name = metadata.attrs['backbone_name']
            self.dataset_type = metadata.attrs['dataset_type']
            self.is_multiview = metadata.attrs.get('is_multiview', True)
            self.n_pose = metadata.attrs['n_pose']
            self.n_betas = metadata.attrs['n_betas']
            self.min_views_per_sample = metadata.attrs.get('min_views_per_sample', 2)
            self.crop_mode = metadata.attrs.get('crop_mode', 'bbox_crop')

            # Optional 3D + camera supervision flags (added by preprocess_sleap_multiview_dataset.py)
            self.has_camera_parameters = bool(metadata.attrs.get('has_camera_parameters', False))
            self.has_3d_keypoints = bool(metadata.attrs.get('has_3d_keypoints', False))
            self.load_3d_data = bool(metadata.attrs.get('load_3d_data', False))

            # World unit scale for 3D + camera translation.
            # - SMAL/SMILify generally operates in "meters-ish" units (order 1.0)
            # - SLEAP/anipose often produces mm-scale coordinates (order 100-1000)
            # We store an explicit `world_scale` if available; otherwise infer a sensible default.
            self.world_scale = float(metadata.attrs.get('world_scale', 1.0))
            if 'world_scale' not in metadata.attrs and (self.has_camera_parameters or self.has_3d_keypoints):
                try:
                    inferred = None
                    if self.has_camera_parameters and 'multiview_keypoints' in f and 'camera_extrinsics_t' in f['multiview_keypoints']:
                        t0 = f['multiview_keypoints/camera_extrinsics_t'][0, 0]  # (3,)
                        tnorm = float(np.linalg.norm(t0))
                        # Heuristic: translations >> 50 are almost certainly mm (or similar), convert to meters.
                        if tnorm > 50.0:
                            inferred = 0.001
                    if inferred is None and self.has_3d_keypoints and 'multiview_keypoints' in f and 'keypoints_3d' in f['multiview_keypoints']:
                        kp0 = f['multiview_keypoints/keypoints_3d'][0]
                        kmax = float(np.max(np.abs(kp0)))
                        if kmax > 50.0:
                            inferred = 0.001
                    if inferred is not None:
                        self.world_scale = float(inferred)
                except Exception:
                    # Fall back to 1.0 if inference fails
                    self.world_scale = float(self.world_scale)
            
            # Parse canonical camera order
            canonical_order_json = metadata.attrs.get('canonical_camera_order', '[]')
            self.canonical_camera_order = json.loads(canonical_order_json)

    @staticmethod
    def _sleap_to_pytorch3d_camera(R_cv: np.ndarray, t_cv: np.ndarray, K: np.ndarray, image_size_wh: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Convert SLEAP/OpenCV camera parameters to the PyTorch3D convention used by SMILify.

        Mirrors `convert_sleap_to_pytorch3d_camera` in `smal_fitter/sleap_data/sleap_3d_loader.py`.

        Returns:
            (R_p3d, T_p3d, fov_y_degrees, aspect_ratio)
        """
        width = float(image_size_wh[0])
        height = float(image_size_wh[1])
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        fov_y = float(2.0 * np.arctan(height / (2.0 * fy)) * 180.0 / np.pi)
        # For FoVPerspectiveCameras, `aspect_ratio` controls how horizontal FOV relates to vertical FOV.
        # Match pinhole intrinsics by using:
        #   tan(fov_x/2) = aspect_ratio * tan(fov_y/2)
        # with tan(fov_x/2) = width/(2*fx), tan(fov_y/2) = height/(2*fy)
        # => aspect_ratio = (width*fy)/(height*fx)
        aspect_ratio = float((width * fy) / (height * fx + 1e-12))

        Rz_180 = np.array([[-1, 0, 0],
                           [0, -1, 0],
                           [0, 0, 1]], dtype=np.float32)
        R_p3d = (R_cv.T @ Rz_180).astype(np.float32)
        T_p3d = (Rz_180 @ t_cv).astype(np.float32)
        return R_p3d, T_p3d, fov_y, aspect_ratio
    
    def _validate_dataset(self):
        """Validate that the dataset is properly formatted."""
        with h5py.File(self.hdf5_path, 'r') as f:
            # Check required groups
            required_groups = ['multiview_images', 'multiview_keypoints', 'parameters', 
                             'auxiliary', 'metadata']
            for group in required_groups:
                if group not in f:
                    raise ValueError(f"Missing required group: {group}")
            
            # Verify view mask exists
            if 'view_mask' not in f['multiview_images']:
                raise ValueError("Missing view_mask in multiview_images group")
            
            # Verify keypoints exist
            if 'keypoints_2d' not in f['multiview_keypoints']:
                raise ValueError("Missing keypoints_2d in multiview_keypoints group")
    
    def _ensure_file_open(self):
        """Ensure HDF5 file is open in current process."""
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r')
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (x_data, y_data) where:
            - x_data: Input data dictionary with multi-view images
            - y_data: Target data dictionary with keypoints and parameters
        """
        self._ensure_file_open()
        
        if self.return_single_view:
            return self._get_single_view_sample(idx)
        else:
            return self._get_multiview_sample(idx)
    
    def _get_multiview_sample(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get a multi-view sample."""
        f = self._file
        
        # Get view mask and number of available views
        view_mask = f['multiview_images/view_mask'][idx]  # (max_views,)
        num_available = int(view_mask.sum())
        available_view_indices = np.where(view_mask)[0]
        
        # Determine which views to use
        if self.num_views_to_use is not None and num_available > self.num_views_to_use:
            # Sample views
            if self.random_view_sampling:
                selected_indices = np.random.choice(
                    available_view_indices, 
                    size=self.num_views_to_use, 
                    replace=False
                )
                selected_indices = np.sort(selected_indices)  # Keep canonical order
            else:
                selected_indices = available_view_indices[:self.num_views_to_use]
        else:
            # Use all available views
            selected_indices = available_view_indices
        
        num_views = len(selected_indices)
        
        # Load images for selected views
        images = []
        for v in selected_indices:
            image_jpeg_bytes = f[f'multiview_images/image_jpeg_view_{v}'][idx]
            image = self._decode_jpeg_image(image_jpeg_bytes)
            images.append(image)
        
        # Load keypoints for selected views
        keypoints_2d_all = f['multiview_keypoints/keypoints_2d'][idx]  # (max_views, n_joints, 2)
        keypoint_visibility_all = f['multiview_keypoints/keypoint_visibility'][idx]  # (max_views, n_joints)
        camera_indices_all = f['multiview_keypoints/camera_indices'][idx]  # (max_views,)
        
        keypoints_2d = keypoints_2d_all[selected_indices]
        keypoint_visibility = keypoint_visibility_all[selected_indices]
        camera_indices = camera_indices_all[selected_indices]
        
        # Load shared body parameters
        betas = f['parameters/betas'][idx]
        
        # Load auxiliary data
        session_name = f['auxiliary/session_name'][idx].decode('utf-8')
        frame_idx = f['auxiliary/frame_idx'][idx]
        has_ground_truth_betas = f['auxiliary/has_ground_truth_betas'][idx]
        camera_names_str = f['auxiliary/camera_names'][idx].decode('utf-8')
        all_camera_names = camera_names_str.split(',')
        
        # Filter camera names for selected views
        camera_names = [all_camera_names[i] if i < len(all_camera_names) else f'cam_{i}' 
                       for i in range(num_available)]
        camera_names = [camera_names[np.where(available_view_indices == v)[0][0]] 
                       for v in selected_indices if v in available_view_indices]
        
        # Create view valid mask (all True for selected views)
        view_valid = np.ones(num_views, dtype=bool)
        
        # Prepare x_data (input data)
        x_data = {
            'images': images,  # List of (H, W, C) arrays in [0, 1]
            'view_mask': view_valid,  # (num_views,) all True
            'camera_names': camera_names,  # List of camera name strings
            'camera_indices': camera_indices,  # (num_views,) canonical indices
            'num_active_views': num_views,
            'session_name': session_name,
            'frame_idx': frame_idx,
            'is_multiview': True,
            'is_sleap_dataset': True,
            'available_labels': {
                'global_rot': False,
                'joint_rot': False,
                'betas': bool(has_ground_truth_betas),
                'trans': False,
                'fov': self.has_camera_parameters,
                'cam_rot': self.has_camera_parameters,
                'cam_trans': self.has_camera_parameters,
                'log_beta_scales': False,
                'betas_trans': False,
                'keypoint_2d': True,
                'keypoint_3d': self.has_3d_keypoints,
                'silhouette': False,
            },
        }
        
        # Prepare y_data (target data)
        y_data = {
            # Per-view keypoint data
            'keypoints_2d': keypoints_2d,  # (num_views, n_joints, 2)
            'keypoint_visibility': keypoint_visibility,  # (num_views, n_joints)
            'view_valid': view_valid,  # (num_views,)
            
            # Shared body parameters
            'shape_betas': betas if has_ground_truth_betas else None,
            
            # Placeholders (no ground truth for SLEAP)
            'root_rot': None,
            'joint_angles': None,
            'root_loc': None,
            'cam_fov': None,
            'cam_rot': None,
            'cam_trans': None,

            # Optional 3D + camera supervision (if present in HDF5)
            'camera_intrinsics': None,
            'camera_extrinsics_R': None,
            'camera_extrinsics_t': None,
            'image_sizes': None,
            'cam_fov_per_view': None,
            'cam_rot_per_view': None,
            'cam_trans_per_view': None,
            'keypoints_3d': None,
            'has_3d_data': False,
            
            # Metadata
            'has_ground_truth_betas': has_ground_truth_betas,
            'num_views': num_views,
        }

        # Load optional camera parameters and 3D keypoints if available in the file
        try:
            if self.has_camera_parameters and 'camera_intrinsics' in f['multiview_keypoints']:
                K_all = f['multiview_keypoints/camera_intrinsics'][idx]  # (max_views, 3, 3)
                R_all = f['multiview_keypoints/camera_extrinsics_R'][idx]  # (max_views, 3, 3)
                t_all = f['multiview_keypoints/camera_extrinsics_t'][idx]  # (max_views, 3)
                sz_all = f['multiview_keypoints/image_sizes'][idx]  # (max_views, 2) as (W,H)

                K_sel = K_all[selected_indices]
                R_sel = R_all[selected_indices]
                t_sel = t_all[selected_indices]
                sz_sel = sz_all[selected_indices]

                # Convert to PyTorch3D camera convention per view
                cam_fov = np.zeros((num_views, 1), dtype=np.float32)
                cam_rot = np.zeros((num_views, 3, 3), dtype=np.float32)
                cam_trans = np.zeros((num_views, 3), dtype=np.float32)
                cam_aspect = np.ones((num_views, 1), dtype=np.float32)
                for vi in range(num_views):
                    R_p3d, T_p3d, fov_y, aspect = self._sleap_to_pytorch3d_camera(
                        R_cv=R_sel[vi].astype(np.float32),
                        # Scale translation into SMILify's world units (typically meters)
                        t_cv=(t_sel[vi].astype(np.float32) * float(self.world_scale)),
                        K=K_sel[vi].astype(np.float32),
                        image_size_wh=sz_sel[vi].astype(np.float32),
                    )
                    cam_fov[vi, 0] = fov_y
                    cam_rot[vi] = R_p3d
                    cam_trans[vi] = T_p3d
                    cam_aspect[vi, 0] = float(aspect)

                y_data.update({
                    'camera_intrinsics': K_sel.astype(np.float32),
                    'camera_extrinsics_R': R_sel.astype(np.float32),
                    # Store scaled translation for downstream consumers (losses, debugging)
                    'camera_extrinsics_t': (t_sel.astype(np.float32) * float(self.world_scale)),
                    'image_sizes': sz_sel.astype(np.int32),
                    'cam_fov_per_view': cam_fov,
                    'cam_rot_per_view': cam_rot,
                    'cam_trans_per_view': cam_trans,
                    'cam_aspect_per_view': cam_aspect,
                })

            if self.has_3d_keypoints and 'keypoints_3d' in f['multiview_keypoints'] and 'has_3d_data' in f['auxiliary']:
                kp3d = f['multiview_keypoints/keypoints_3d'][idx]  # (n_joints, 3)
                has_3d = bool(f['auxiliary/has_3d_data'][idx])
                y_data['keypoints_3d'] = (kp3d.astype(np.float32) * float(self.world_scale))
                y_data['has_3d_data'] = has_3d
        except Exception:
            # Keep None defaults if optional datasets are missing or malformed.
            pass

        # ── Augmentation (training only) ────────────────────────────────
        if self.augment:
            images = x_data['images']
            kp2d = y_data['keypoints_2d']       # (num_views, n_joints, 2)
            vis = y_data['keypoint_visibility']  # (num_views, n_joints)
            K_aug = y_data.get('camera_intrinsics')  # (num_views, 3, 3) or None

            for vi in range(len(images)):
                # Geometric augmentation (per-view, updates K and 2D keypoints)
                if K_aug is not None:
                    images[vi], K_aug[vi], kp2d[vi], vis[vi] = (
                        self._apply_geometric_augmentation(
                            images[vi], K_aug[vi], kp2d[vi], vis[vi]
                        )
                    )
                # Photometric augmentation (per-view, independent randomness)
                images[vi] = self._apply_photometric_augmentation(images[vi])

            x_data['images'] = images
            y_data['keypoints_2d'] = kp2d
            y_data['keypoint_visibility'] = vis

            # Recompute PyTorch3D camera params from updated intrinsics
            if K_aug is not None:
                y_data['camera_intrinsics'] = K_aug
                sz = y_data['image_sizes']  # (num_views, 2) as (W, H)
                for vi in range(len(images)):
                    R_p3d, T_p3d, fov_y, aspect = self._sleap_to_pytorch3d_camera(
                        R_cv=y_data['camera_extrinsics_R'][vi],
                        t_cv=y_data['camera_extrinsics_t'][vi],
                        K=K_aug[vi],
                        image_size_wh=sz[vi].astype(np.float32),
                    )
                    y_data['cam_fov_per_view'][vi, 0] = fov_y
                    y_data['cam_rot_per_view'][vi] = R_p3d
                    y_data['cam_trans_per_view'][vi] = T_p3d
                    y_data['cam_aspect_per_view'][vi, 0] = float(aspect)

        return x_data, y_data
    
    def _get_single_view_sample(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get a single-view sample (backward compatibility mode)."""
        f = self._file
        
        # Get view mask
        view_mask = f['multiview_images/view_mask'][idx]
        available_view_indices = np.where(view_mask)[0]
        
        if len(available_view_indices) == 0:
            raise ValueError(f"No valid views for sample {idx}")
        
        # Select the preferred view, or first available
        if self.preferred_view < len(available_view_indices):
            selected_view = available_view_indices[self.preferred_view]
        else:
            selected_view = available_view_indices[0]
        
        # Load image
        image_jpeg_bytes = f[f'multiview_images/image_jpeg_view_{selected_view}'][idx]
        image = self._decode_jpeg_image(image_jpeg_bytes)
        
        # Load keypoints
        keypoints_2d = f['multiview_keypoints/keypoints_2d'][idx, selected_view]
        keypoint_visibility = f['multiview_keypoints/keypoint_visibility'][idx, selected_view]
        
        # Load shared parameters
        betas = f['parameters/betas'][idx]
        
        # Load auxiliary data
        session_name = f['auxiliary/session_name'][idx].decode('utf-8')
        frame_idx = f['auxiliary/frame_idx'][idx]
        has_ground_truth_betas = f['auxiliary/has_ground_truth_betas'][idx]
        
        # Prepare x_data (single-view format compatible with existing pipeline)
        x_data = {
            'input_image': f"{session_name}/frame_{frame_idx:04d}",
            'input_image_data': image,
            'input_image_mask': None,
            'session_name': session_name,
            'frame_idx': frame_idx,
            'is_sleap_dataset': True,
            'is_multiview': False,
            'available_labels': {
                'global_rot': False,
                'joint_rot': False,
                'betas': bool(has_ground_truth_betas),
                'trans': False,
                'fov': self.has_camera_parameters,
                'cam_rot': self.has_camera_parameters,
                'cam_trans': self.has_camera_parameters,
                'log_beta_scales': False,
                'betas_trans': False,
                'keypoint_2d': True,
                'keypoint_3d': self.has_3d_keypoints,
                'silhouette': False,
            },
        }
        
        # Prepare y_data (single-view format)
        y_data = {
            'keypoints_2d': keypoints_2d,
            'keypoint_visibility': keypoint_visibility,
            'shape_betas': betas if has_ground_truth_betas else None,
            'root_rot': None,
            'joint_angles': None,
            'root_loc': None,
            'cam_fov': None,
            'cam_rot': None,
            'cam_trans': None,
            'camera_intrinsics': None,
            'camera_extrinsics_R': None,
            'camera_extrinsics_t': None,
            'image_sizes': None,
            'cam_fov_per_view': None,
            'cam_rot_per_view': None,
            'cam_trans_per_view': None,
            'keypoints_3d': None,
            'has_3d_data': False,
            'has_ground_truth_betas': has_ground_truth_betas,
        }

        # Optional supervision for single-view mode: select the chosen view's camera params
        try:
            if self.has_camera_parameters and 'camera_intrinsics' in f['multiview_keypoints']:
                K = f['multiview_keypoints/camera_intrinsics'][idx, selected_view].astype(np.float32)
                R = f['multiview_keypoints/camera_extrinsics_R'][idx, selected_view].astype(np.float32)
                t = f['multiview_keypoints/camera_extrinsics_t'][idx, selected_view].astype(np.float32) * float(self.world_scale)
                sz = f['multiview_keypoints/image_sizes'][idx, selected_view].astype(np.int32)
                R_p3d, T_p3d, fov_y, aspect = self._sleap_to_pytorch3d_camera(R, t, K, sz.astype(np.float32))
                y_data.update({
                    'camera_intrinsics': K,
                    'camera_extrinsics_R': R,
                    'camera_extrinsics_t': t,
                    'image_sizes': sz,
                    'cam_fov': float(fov_y),
                    'cam_rot': R_p3d,
                    'cam_trans': T_p3d,
                    'cam_aspect': float(aspect),
                })

            if self.has_3d_keypoints and 'keypoints_3d' in f['multiview_keypoints'] and 'has_3d_data' in f['auxiliary']:
                kp3d = f['multiview_keypoints/keypoints_3d'][idx].astype(np.float32) * float(self.world_scale)
                y_data['keypoints_3d'] = kp3d
                y_data['has_3d_data'] = bool(f['auxiliary/has_3d_data'][idx])
        except Exception:
            pass
        
        return x_data, y_data
    
    def _decode_jpeg_image(self, jpeg_bytes) -> np.ndarray:
        """Decode JPEG bytes to image array."""
        if isinstance(jpeg_bytes, np.ndarray):
            jpeg_array = jpeg_bytes
        else:
            jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        
        image = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode JPEG image")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        return image

    # ── Augmentation ─────────────────────────────────────────────────────

    def _apply_photometric_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply photometric augmentation to a single image.

        Args:
            image: (H, W, 3) float32 RGB in [0, 1].

        Returns:
            Augmented image, same shape and dtype, clipped to [0, 1].
        """
        # Brightness: additive shift
        if self.aug_color_jitter_brightness > 0:
            delta = np.random.uniform(-self.aug_color_jitter_brightness,
                                       self.aug_color_jitter_brightness)
            image = image + delta

        # Contrast: scale around mean
        if self.aug_color_jitter_contrast > 0:
            factor = np.random.uniform(max(0, 1 - self.aug_color_jitter_contrast),
                                       1 + self.aug_color_jitter_contrast)
            mean = image.mean()
            image = (image - mean) * factor + mean

        # Saturation: blend with grayscale
        if self.aug_color_jitter_saturation > 0:
            factor = np.random.uniform(max(0, 1 - self.aug_color_jitter_saturation),
                                       1 + self.aug_color_jitter_saturation)
            gray = image.mean(axis=2, keepdims=True)
            image = gray + (image - gray) * factor

        # Gaussian noise
        if self.aug_gaussian_noise_std > 0:
            noise = np.random.normal(0, self.aug_gaussian_noise_std,
                                     image.shape).astype(np.float32)
            image = image + noise

        # Gaussian blur
        if self.aug_gaussian_blur_prob > 0 and np.random.random() < self.aug_gaussian_blur_prob:
            lo, hi = self.aug_gaussian_blur_kernel_range
            # Kernel size must be odd
            k = np.random.choice(range(lo, hi + 1, 2))
            image = cv2.GaussianBlur(image, (k, k), 0)

        # Random erasing (rectangular cutout filled with mean)
        if self.aug_random_erasing_prob > 0 and np.random.random() < self.aug_random_erasing_prob:
            h, w = image.shape[:2]
            lo_s, hi_s = self.aug_random_erasing_scale_range
            area_frac = np.random.uniform(lo_s, hi_s)
            erase_area = int(h * w * area_frac)
            aspect = np.random.uniform(0.3, 1.0 / 0.3)
            eh = int(np.sqrt(erase_area * aspect))
            ew = int(np.sqrt(erase_area / aspect))
            eh, ew = min(eh, h), min(ew, w)
            y0 = np.random.randint(0, h - eh + 1)
            x0 = np.random.randint(0, w - ew + 1)
            image[y0:y0 + eh, x0:x0 + ew] = image.mean()

        return np.clip(image, 0.0, 1.0)

    def _apply_geometric_augmentation(
        self,
        image: np.ndarray,
        K: np.ndarray,
        keypoints_2d: np.ndarray,
        visibility: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply scale jitter, updating K and 2D keypoints consistently.

        The projection relationship ``2D = K @ [R|t] @ X_3d`` is preserved by
        adjusting the intrinsics matrix K to match the new image crop/resize.
        Extrinsics R, t are never modified.

        **Important:** Only *centered* scale jitter is applied (no crop offset).
        The training pipeline uses ``FoVPerspectiveCameras`` which parametrises
        projection via fov + aspect_ratio and assumes the principal point is at
        the image centre.  A crop offset would shift cx/cy in K, but that shift
        is lost when converting to FoV parameters — creating a systematic
        mismatch between the projected keypoints and the augmented targets.
        Centered scale jitter is safe because it only changes fx/fy
        (captured by fov/aspect) while keeping cx = w/2, cy = h/2.

        To support crop jitter in the future, the camera pipeline would need to
        switch to ``PerspectiveCameras`` which accepts a full intrinsics matrix.

        Args:
            image: (H, W, 3) float32 RGB in [0, 1].
            K: (3, 3) camera intrinsics for this view.
            keypoints_2d: (n_joints, 2) **normalized [0,1] in [y, x] order**
                as stored in the HDF5 dataset.
            visibility: (n_joints,) boolean visibility mask.

        Returns:
            (image, K, keypoints_2d, visibility) — all updated in-place-safe copies.
        """
        h, w = image.shape[:2]
        K = K.copy()
        keypoints_2d = keypoints_2d.copy()
        visibility = visibility.copy()

        # 1. Random scale factor (centered zoom in/out)
        lo, hi = self.aug_scale_jitter_range
        scale = np.random.uniform(lo, hi)

        # 2. Compute source crop region — always centred (no offset).
        #    scale > 1 means zoom in (smaller source crop), < 1 means zoom out.
        src_h = h / scale
        src_w = w / scale
        x1 = (w - src_w) / 2.0
        y1 = (h - src_h) / 2.0

        # 3. Build the affine mapping: source (x1, y1, src_w, src_h) -> dest (0, 0, w, h)
        #    dest_x = (src_x - x1) * (w / src_w) = (src_x - x1) * scale
        sx = w / src_w  # == scale
        sy = h / src_h  # == scale

        M = np.array([
            [sx, 0, -x1 * sx],
            [0, sy, -y1 * sy],
        ], dtype=np.float32)
        image = cv2.warpAffine(image, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)

        # 4. Update intrinsics: K_new maps from the same world ray to the new pixel coords.
        #    For centred scale, this simplifies to scaling fx, fy, cx, cy by the scale factor,
        #    which preserves cx = w/2, cy = h/2 (required by FoVPerspectiveCameras).
        T = np.eye(3, dtype=np.float64)
        T[0, 0] = sx
        T[1, 1] = sy
        T[0, 2] = -x1 * sx
        T[1, 2] = -y1 * sy
        K = (T @ K.astype(np.float64)).astype(np.float32)

        # 5. Update 2D keypoints: convert from normalized [y,x] to pixel [x,y],
        #    apply the affine transform, then convert back to normalized [y,x].
        pixel_x = keypoints_2d[:, 1] * w   # col 1 = normalized x
        pixel_y = keypoints_2d[:, 0] * h   # col 0 = normalized y

        new_pixel_x = (pixel_x - x1) * sx
        new_pixel_y = (pixel_y - y1) * sy

        keypoints_2d[:, 1] = new_pixel_x / w  # back to normalized x
        keypoints_2d[:, 0] = new_pixel_y / h  # back to normalized y

        # 6. Mark keypoints outside the [0, 1] normalized range as invisible
        out_of_bounds = (
            (keypoints_2d[:, 0] < 0) | (keypoints_2d[:, 0] > 1.0) |
            (keypoints_2d[:, 1] < 0) | (keypoints_2d[:, 1] > 1.0)
        )
        visibility[out_of_bounds] = False

        return image, K, keypoints_2d, visibility

    def get_max_views_in_dataset(self) -> int:
        """Return the maximum number of views across all samples."""
        return self.max_views
    
    def get_canonical_camera_order(self) -> List[str]:
        """Return the canonical ordering of cameras."""
        return self.canonical_camera_order
    
    def get_target_resolution(self) -> int:
        """Get the target resolution of the dataset."""
        return self.target_resolution
    
    def get_ue_scaling_flag(self) -> bool:
        """Get the UE scaling flag (always False for SLEAP data)."""
        return False
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        return {
            'dataset_type': self.dataset_type,
            'is_multiview': self.is_multiview,
            'num_samples': self.num_samples,
            'max_views': self.max_views,
            'n_joints': self.n_joints,
            'target_resolution': self.target_resolution,
            'backbone_name': self.backbone_name,
            'crop_mode': self.crop_mode,
            'n_pose': self.n_pose,
            'n_betas': self.n_betas,
            'min_views_per_sample': self.min_views_per_sample,
            'canonical_camera_order': self.canonical_camera_order,
            'rotation_representation': self.rotation_representation,
            'num_views_to_use': self.num_views_to_use,
        }
    
    def print_dataset_summary(self):
        """Print a summary of the dataset."""
        info = self.get_dataset_info()
        
        print("\n" + "="*50)
        print("MULTI-VIEW SLEAP DATASET SUMMARY")
        print("="*50)
        print(f"Dataset type: {info['dataset_type']}")
        print(f"Number of samples: {info['num_samples']}")
        print(f"Maximum views per sample: {info['max_views']}")
        print(f"Number of joints: {info['n_joints']}")
        print(f"Target resolution: {info['target_resolution']}x{info['target_resolution']}")
        print(f"Backbone: {info['backbone_name']}")
        print(f"Crop mode: {info['crop_mode']}")
        print(f"N_POSE: {info['n_pose']}")
        print(f"N_BETAS: {info['n_betas']}")
        print(f"Min views per sample: {info['min_views_per_sample']}")
        print(f"Canonical camera order: {info['canonical_camera_order']}")
        print(f"Num views to use: {info['num_views_to_use'] or 'all'}")
        print("="*50)
    
    def __del__(self):
        """Close file handle on deletion."""
        if hasattr(self, '_file') and self._file is not None:
            try:
                self._file.close()
            except:
                pass


def multiview_collate_fn(batch: List[Tuple[Dict, Dict]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Custom collate function for multi-view batches.
    
    Handles variable number of views per sample by keeping samples as lists
    rather than stacking into tensors.
    
    Args:
        batch: List of (x_data, y_data) tuples
        
    Returns:
        Tuple of (x_data_batch, y_data_batch) where each is a list of dicts
    """
    x_data_batch = []
    y_data_batch = []
    
    for x_data, y_data in batch:
        # Stack images into array if they're in list format
        if 'images' in x_data and isinstance(x_data['images'], list):
            # Keep as list for now - model will handle conversion
            pass
        
        x_data_batch.append(x_data)
        y_data_batch.append(y_data)
    
    return x_data_batch, y_data_batch


if __name__ == "__main__":
    # Test the multi-view dataset
    import argparse
    
    parser = argparse.ArgumentParser(description="Test multi-view SLEAP dataset")
    parser.add_argument("hdf5_path", help="Path to multi-view SLEAP HDF5 dataset")
    parser.add_argument("--num_views", type=int, default=None,
                       help="Number of views to use per sample")
    parser.add_argument("--single_view", action="store_true",
                       help="Test single-view mode")
    args = parser.parse_args()
    
    # Load dataset
    dataset = SLEAPMultiViewDataset(
        args.hdf5_path,
        num_views_to_use=args.num_views,
        return_single_view=args.single_view
    )
    
    # Print summary
    dataset.print_dataset_summary()
    
    # Test loading samples
    if len(dataset) > 0:
        print(f"\nTesting sample loading...")
        x_data, y_data = dataset[0]
        
        if args.single_view:
            print(f"Single-view sample 0:")
            print(f"  Image shape: {x_data['input_image_data'].shape}")
            print(f"  Keypoints 2D shape: {y_data['keypoints_2d'].shape}")
        else:
            print(f"Multi-view sample 0:")
            print(f"  Number of views: {x_data['num_active_views']}")
            print(f"  Image shapes: {[img.shape for img in x_data['images']]}")
            print(f"  Keypoints 2D shape: {y_data['keypoints_2d'].shape}")
            print(f"  Camera names: {x_data['camera_names']}")
            print(f"  Camera indices: {x_data['camera_indices']}")
        
        print(f"  Has ground truth betas: {y_data['has_ground_truth_betas']}")
    
    print("\nMulti-view SLEAP dataset test completed!")

