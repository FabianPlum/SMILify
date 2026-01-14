"""
SMIL Image Regressor

A neural network that learns to predict SMIL parameters from input images.
Uses a frozen ResNet152 backbone as feature extractor and fully connected layers
for parameter regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from scipy.spatial.transform import Rotation

# Import from parent modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smal_fitter import SMALFitter
import config
from backbone_factory import BackboneFactory, BackboneInterface
from transformer_decoder import build_smil_transformer_decoder_head

# Import rotation utilities from PyTorch3D
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, rotation_6d_to_matrix, matrix_to_rotation_6d

# Helper function for tensor conversion
def safe_to_tensor(data, dtype=torch.float32, device='cpu'):
    """Safely convert data to PyTorch tensor."""
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype, device=device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data.astype(np.float32 if dtype == torch.float32 else data.dtype)).to(dtype=dtype, device=device)
    else:
        return torch.tensor(data, dtype=dtype, device=device)


# ----------------------- ROTATION CONVERSION UTILS ----------------------- #

def robust_axis_angle_to_matrix(axis_angle):
    """
    Converts axis-angle representation to a 3x3 rotation matrix.
    Handles potential NaNs for zero-angle rotations if using pytorch3d's matrix_to_axis_angle back and forth.
    Args:
        axis_angle: Tensor of shape (..., 3)
    Returns:
        Tensor of shape (..., 3, 3)
    """
    return axis_angle_to_matrix(axis_angle)

def robust_matrix_to_axis_angle(matrix):
    """
    Converts a 3x3 rotation matrix to axis-angle representation.
    Handles potential issues with ill-defined gradients for identity matrices by adding a small epsilon.
    Args:
        matrix: Tensor of shape (..., 3, 3)
    Returns:
        Tensor of shape (..., 3)
    """
    return matrix_to_axis_angle(matrix)

def axis_angle_to_rotation_6d(axis_angle):
    """
    Converts axis-angle representation to 6D rotation representation.
    The 6D representation consists of the first two columns of the rotation matrix.
    Args:
        axis_angle: Tensor of shape (..., 3)
    Returns:
        Tensor of shape (..., 6)
    """
    rotation_matrix = robust_axis_angle_to_matrix(axis_angle)
    return matrix_to_rotation_6d(rotation_matrix)

def rotation_6d_to_axis_angle(rotation_6d):
    """
    Converts 6D rotation representation back to axis-angle.
    Args:
        rotation_6d: Tensor of shape (..., 6)
    Returns:
        Tensor of shape (..., 3)
    """
    rotation_matrix = rotation_6d_to_matrix(rotation_6d)
    return robust_matrix_to_axis_angle(rotation_matrix)


class SMILImageRegressor(SMALFitter):
    """
    Neural network for predicting SMIL parameters from RGB images.
    
    Extends SMALFitter to inherit SMIL model functionality while adding
    a ResNet152 backbone for image feature extraction and regression head
    for parameter prediction.
    """
    
    def __init__(self, device, data_batch, batch_size, shape_family, use_unity_prior, 
                 rgb_only=True, freeze_backbone=True, hidden_dim=512, use_ue_scaling=True, 
                 rotation_representation='axis_angle', input_resolution=512, backbone_name='resnet152',
                 head_type='mlp', transformer_config=None, scale_trans_mode='separate',
                 allow_mesh_scaling=False, mesh_scale_init=1.0):
        """
        Initialize the SMIL Image Regressor.
        
        Args:
            device: PyTorch device (cuda/cpu)
            data_batch: Batch data for SMALFitter initialization (can be placeholder)
            batch_size: Batch size for processing
            shape_family: Shape family ID for SMIL model
            use_unity_prior: Whether to use unity prior
            rgb_only: Whether to use only RGB images (no silhouettes/keypoints)
            freeze_backbone: Whether to freeze backbone weights
            hidden_dim: Hidden dimension for fully connected layers
            use_ue_scaling: Whether to apply 10x UE scaling (default True for replicAnt data)
            rotation_representation: '6d' or 'axis_angle' for joint rotations (default: 'axis_angle')
            input_resolution: Input image resolution (default: 512, should match dataset resolution)
            backbone_name: Backbone network name ('resnet152', 'vit_base_patch16_224', etc.)
            head_type: Type of regression head ('mlp' or 'transformer_decoder')
            transformer_config: Configuration dict for transformer decoder (only used if head_type='transformer_decoder')
            scale_trans_mode: Mode for handling scale/translation betas ('ignore', 'separate', 'entangled_with_betas')
            allow_mesh_scaling: If True, predict a global mesh scale factor (default: False)
            mesh_scale_init: Initial value for mesh scale (default: 1.0 = no scaling)
        """
        # For rgb_only=True, SMALFitter expects data_batch to be just the RGB tensor
        if rgb_only and isinstance(data_batch, tuple):
            # Extract RGB tensor from tuple
            rgb_tensor = data_batch[0]
        else:
            rgb_tensor = data_batch
            
        # Initialize parent SMALFitter with rgb_only=True since we only use images
        super(SMILImageRegressor, self).__init__(
            device, rgb_tensor, batch_size, shape_family, use_unity_prior, rgb_only=True
        )
        
        self.freeze_backbone = freeze_backbone
        self.hidden_dim = hidden_dim
        self.use_ue_scaling = use_ue_scaling
        self.rotation_representation = rotation_representation
        self.input_resolution = input_resolution
        self.backbone_name = backbone_name
        self.head_type = head_type
        self.transformer_config = transformer_config or {}
        self.scale_trans_mode = scale_trans_mode
        self.allow_mesh_scaling = allow_mesh_scaling
        self.mesh_scale_init = mesh_scale_init
        
        # Enable scaling propagation for SMIL models (matches Unreal2Pytorch3D behavior)
        self.propagate_scaling = True
        
        # Create backbone using factory
        self.backbone = BackboneFactory.create_backbone(
            backbone_name, 
            pretrained=True, 
            freeze=freeze_backbone
        ).to(device)
        
        # Get the feature dimension from backbone
        self.feature_dim = self.backbone.get_feature_dim()
        
        # Calculate output dimensions for SMIL parameters
        self._calculate_output_dims()
        
        # Create regression head based on type
        if self.head_type == 'mlp':
            self._create_mlp_regression_head()
            self._initialize_mlp_parameters()
        elif self.head_type == 'transformer_decoder':
            self._create_transformer_decoder_head()
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}. Must be 'mlp' or 'transformer_decoder'")
    
    def _calculate_output_dims(self):
        """Calculate the output dimensions for each SMIL parameter group."""
        # Global rotation and joint rotations dimensions depend on representation
        if self.rotation_representation == '6d':
            self.global_rot_dim = 6  # 6D rotation representation
            self.joint_rot_dim = config.N_POSE * 6  # 6D for each joint
        else:  # axis_angle (default)
            self.global_rot_dim = 3  # axis-angle representation
            self.joint_rot_dim = config.N_POSE * 3  # axis-angle for each joint
        
        # Shape parameters
        self.betas_dim = config.N_BETAS
        
        # Translation
        self.trans_dim = 3
        
        # Camera FOV
        self.fov_dim = 1
        
        # Camera rotation and translation (in model space)
        self.cam_rot_dim = 9  # 9 for 3x3 rotation matrix (flattened)
        self.cam_trans_dim = 3
        
        # Handle scale and trans dimensions based on mode
        if self.scale_trans_mode == 'entangled_with_betas':
            # No separate outputs needed - betas handle everything
            self.scales_dim = 0
            self.joint_trans_dim = 0
        else:
            # Separate outputs for scales and translations
            # These are PCA weights, so same dimension as betas
            self.scales_dim = config.N_BETAS  # PCA components for scaling
            self.joint_trans_dim = config.N_BETAS  # PCA components for translation
        
        # Total output dimension
        self.total_output_dim = (self.global_rot_dim + self.joint_rot_dim + 
                                self.betas_dim + self.trans_dim + self.fov_dim +
                                self.cam_rot_dim + self.cam_trans_dim +
                                self.scales_dim + self.joint_trans_dim)
    
    def _create_mlp_regression_head(self):
        """Create the regression head with fully connected layers."""
        # First fully connected layer (using LayerNorm instead of BatchNorm for stability)
        self.fc1 = nn.Linear(self.feature_dim, self.hidden_dim)
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.ln2 = nn.LayerNorm(self.hidden_dim // 2)

        # Third fully connected layer
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4)
        self.ln3 = nn.LayerNorm(self.hidden_dim // 4)
        
        # Final regression layer
        self.regressor = nn.Linear(self.hidden_dim // 4, self.total_output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)
    
    def _initialize_mlp_parameters(self):
        """Initialize the regression head parameters."""
        # Use He initialization for ReLU layers (more appropriate for ReLU activations)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)
        
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        nn.init.constant_(self.fc3.bias, 0)

        # Use Xavier initialization for final layer (no activation function)
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.constant_(self.regressor.bias, 0)
    
    def _create_transformer_decoder_head(self):
        """Create the transformer decoder regression head."""
        # Default transformer configuration
        default_config = {
            'hidden_dim': 1024,
            'depth': 6,
            'heads': 8,
            'dim_head': 64,
            'mlp_dim': 1024,
            'dropout': 0.0,
            'ief_iters': 3,
        }
        
        # Merge with user-provided config
        config = {**default_config, **self.transformer_config}
        
        # For ViT backbones, we can use spatial features
        if self.backbone_name.startswith('vit'):
            context_dim = self.feature_dim  # Same as feature dim for ViT
        else:
            # For ResNet, we don't have spatial features, so use global features as context
            context_dim = self.feature_dim
        
        # Create transformer decoder head
        self.transformer_head = build_smil_transformer_decoder_head(
            feature_dim=self.feature_dim,
            context_dim=context_dim,
            hidden_dim=config['hidden_dim'],
            depth=config['depth'],
            heads=config['heads'],
            dim_head=config['dim_head'],
            mlp_dim=config['mlp_dim'],
            dropout=config['dropout'],
            ief_iters=config['ief_iters'],
            rotation_representation=self.rotation_representation,
            scales_scale_factor=config.get('scales_scale_factor', 1),
            trans_scale_factor=config.get('trans_scale_factor', 0.01),
            scale_trans_mode=self.scale_trans_mode,  # Pass scale_trans_mode to control output dimensions
            allow_mesh_scaling=self.allow_mesh_scaling,
            mesh_scale_init=self.mesh_scale_init
        ).to(self.device)
    
    def preprocess_image(self, image_data) -> torch.Tensor:
        """
        Preprocess input image for the network.
        
        Args:
            image_data: Input image as numpy array (H, W, C) with values in [0, 255]
                       OR list of images for batch processing
            
        Returns:
            Preprocessed image tensor (1, C, H, W) or (B, C, H, W) with values in [0, 1]
        """
        # Handle different input types
        if isinstance(image_data, list):
            # Batch processing mode
            return self._preprocess_image_batch(image_data)
        elif isinstance(image_data, torch.Tensor):
            # Convert tensor to numpy
            image_data = image_data.cpu().numpy()
        
        # Ensure it's a numpy array
        if not isinstance(image_data, np.ndarray):
            image_data = np.array(image_data)
        
        # Handle different image formats
        if len(image_data.shape) == 4:
            # Batch of images (B, H, W, C)
            if image_data.shape[3] == 3:
                pass  # Already in correct format
            elif image_data.shape[1] == 3:
                # BCHW format, convert to BHWC
                image_data = image_data.transpose(0, 2, 3, 1)
        elif len(image_data.shape) == 3:
            # Single RGB image (H, W, C)
            if image_data.shape[2] == 3:
                pass  # Already in correct format
            elif image_data.shape[0] == 3:
                # CHW format, convert to HWC
                image_data = image_data.transpose(1, 2, 0)
        elif len(image_data.shape) == 2:
            # Grayscale image, convert to RGB
            image_data = np.stack([image_data] * 3, axis=2)
        else:
            raise ValueError(f"Unsupported image shape: {image_data.shape}")
        
        # Ensure the image has the right data type and range
        if image_data.dtype == np.uint8:
            image_data = image_data.astype(np.float32) / 255.0
        elif image_data.dtype == np.float64:
            image_data = image_data.astype(np.float32)
        elif image_data.max() > 1.0:
            # Assume values are in [0, 255] range
            image_data = image_data.astype(np.float32) / 255.0
        
        # Resize to appropriate resolution based on backbone type
        if self.backbone_name.startswith('vit'):
            # Vision Transformers expect 224x224 input
            target_size = (224, 224)
        else:
            # ResNet can handle higher resolutions
            target_size = (self.input_resolution, self.input_resolution)
        if len(image_data.shape) == 4:
            # Batch of images
            batch_size = image_data.shape[0]
            resized_batch = []
            for i in range(batch_size):
                if image_data[i].shape[:2] != target_size:
                    resized_img = cv2.resize(image_data[i], target_size)
                else:
                    resized_img = image_data[i]
                resized_batch.append(resized_img)
            image_data = np.array(resized_batch)
            # Convert to tensor (B, H, W, C) -> (B, C, H, W)
            image_tensor = torch.from_numpy(image_data).permute(0, 3, 1, 2)
        else:
            # Single image
            if image_data.shape[:2] != target_size:
                image_data = cv2.resize(image_data, target_size)
            # Convert to tensor and add batch dimension (H, W, C) -> (1, C, H, W)
            image_tensor = torch.from_numpy(image_data).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor
    
    def scale_keypoint_coordinates(self, keypoints, original_size, target_size=None):
        """
        Scale keypoint coordinates when image is resized.
        
        Args:
            keypoints: Keypoint coordinates in normalized [0,1] format (n_joints, 2)
            original_size: Original image size (width, height) or (height, width)
            target_size: Target image size (width, height) or (height, width). If None, uses backbone-appropriate resolution
            
        Returns:
            Scaled keypoint coordinates in normalized [0,1] format
        """
        if target_size is None:
            if self.backbone_name.startswith('vit'):
                # Vision Transformers expect 224x224 input
                target_size = (224, 224)
            else:
                # ResNet can handle higher resolutions
                target_size = (self.input_resolution, self.input_resolution)
        
        # Handle different input formats
        if isinstance(original_size, (int, float)):
            # Single value - assume square image
            orig_w, orig_h = original_size, original_size
        elif len(original_size) == 2:
            # (width, height) or (height, width) - need to determine which
            # For now, assume it's (width, height) as that's more common
            orig_w, orig_h = original_size
        else:
            raise ValueError(f"Unsupported original_size format: {original_size}")
        
        if isinstance(target_size, (int, float)):
            # Single value - assume square image
            target_w, target_h = target_size, target_size
        elif len(target_size) == 2:
            target_w, target_h = target_size
        else:
            raise ValueError(f"Unsupported target_size format: {target_size}")
        
        # Calculate scaling factors
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        # Scale the keypoint coordinates
        # Note: keypoints are in [y_norm, x_norm] format (normalized coordinates)
        scaled_keypoints = keypoints.copy()
        scaled_keypoints[:, 0] = scaled_keypoints[:, 0] * scale_y  # y coordinates
        scaled_keypoints[:, 1] = scaled_keypoints[:, 1] * scale_x  # x coordinates
        
        return scaled_keypoints
    
    def _preprocess_image_batch(self, image_list) -> torch.Tensor:
        """
        Efficiently preprocess a batch of images in parallel.
        
        Args:
            image_list: List of image data (each as numpy array or tensor)
            
        Returns:
            Batched image tensor (B, C, H, W) with values in [0, 1]
        """
        batch_images = []
        
        for image_data in image_list:
            # Process each image individually for now (can be optimized further)
            if isinstance(image_data, torch.Tensor):
                image_data = image_data.cpu().numpy()
            
            if not isinstance(image_data, np.ndarray):
                image_data = np.array(image_data)
            
            # Handle different image formats
            if len(image_data.shape) == 3:
                # Single RGB image (H, W, C)
                if image_data.shape[2] == 3:
                    pass  # Already in correct format
                elif image_data.shape[0] == 3:
                    # CHW format, convert to HWC
                    image_data = image_data.transpose(1, 2, 0)
            elif len(image_data.shape) == 2:
                # Grayscale image, convert to RGB
                image_data = np.stack([image_data] * 3, axis=2)
            else:
                raise ValueError(f"Unsupported image shape: {image_data.shape}")
            
            # Ensure the image has the right data type and range
            if image_data.dtype == np.uint8:
                image_data = image_data.astype(np.float32) / 255.0
            elif image_data.dtype == np.float64:
                image_data = image_data.astype(np.float32)
            elif image_data.max() > 1.0:
                # Assume values are in [0, 255] range
                image_data = image_data.astype(np.float32) / 255.0
            
            # Resize to appropriate resolution based on backbone type
            if self.backbone_name.startswith('vit'):
                # Vision Transformers expect 224x224 input
                target_size = (224, 224)
            else:
                # ResNet can handle higher resolutions
                target_size = (self.input_resolution, self.input_resolution)
            if image_data.shape[:2] != target_size:
                image_data = cv2.resize(image_data, target_size)
            
            batch_images.append(image_data)
        
        # Convert batch to tensor (B, H, W, C) -> (B, C, H, W)
        batch_array = np.array(batch_images)
        image_tensor = torch.from_numpy(batch_array).permute(0, 3, 1, 2)
        
        return image_tensor
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the network.
        
        Args:
            images: Input images tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Dictionary containing predicted SMIL parameters:
                - 'global_rot': Global rotation (batch_size, 3)
                - 'joint_rot': Joint rotations (batch_size, N_POSE, 3)
                - 'betas': Shape parameters (batch_size, N_BETAS)
                - 'trans': Translation (batch_size, 3)
                - 'fov': Camera FOV (batch_size, 1)
                - 'log_beta_scales': Joint scales (batch_size, N_JOINTS, 3) [if available]
                - 'betas_trans': Joint translations (batch_size, N_JOINTS, 3) [if available]
        """
        batch_size = images.size(0)
        
        if self.head_type == 'mlp':
            return self._forward_mlp(images, batch_size)
        elif self.head_type == 'transformer_decoder':
            return self._forward_transformer_decoder(images, batch_size)
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")
    
    def _forward_mlp(self, images: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        """Forward pass for MLP regression head."""
        # Extract features using backbone (ResNet or ViT)
        features = self.backbone(images)  # (batch_size, feature_dim)
        
        # Pass through regression head
        x = F.relu(self.ln1(self.fc1(features)))
        x = self.dropout(x)
        
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        
        # Final regression
        output = self.regressor(x)  # (batch_size, total_output_dim)
        
        # Parse output into parameter groups
        params = {}
        idx = 0
        
        # Global rotation
        params['global_rot'] = output[:, idx:idx + self.global_rot_dim]
        idx += self.global_rot_dim
        
        # Joint rotations
        joint_rot_flat = output[:, idx:idx + self.joint_rot_dim]
        if self.rotation_representation == '6d':
            params['joint_rot'] = joint_rot_flat.view(batch_size, config.N_POSE, 6)
        else:  # axis_angle
            params['joint_rot'] = joint_rot_flat.view(batch_size, config.N_POSE, 3)
        idx += self.joint_rot_dim
        
        # Shape parameters
        params['betas'] = output[:, idx:idx + self.betas_dim]
        idx += self.betas_dim
        
        # Translation
        params['trans'] = output[:, idx:idx + self.trans_dim]
        idx += self.trans_dim
        
        # Camera FOV
        params['fov'] = output[:, idx:idx + self.fov_dim]
        idx += self.fov_dim
        
        # Debug: Check FOV shape occasionally
        if hasattr(self, '_debug_shapes') and torch.rand(1).item() < 0.01:
            print(f"DEBUG - Network FOV output shape: {params['fov'].shape}")
        
        # Camera rotation (in model space) - reshape to 3x3 matrix
        cam_rot_flat = output[:, idx:idx + self.cam_rot_dim]
        params['cam_rot'] = cam_rot_flat.view(batch_size, 3, 3)
        idx += self.cam_rot_dim
        
        # Camera translation (in model space)
        params['cam_trans'] = output[:, idx:idx + self.cam_trans_dim]
        idx += self.cam_trans_dim
        
        # Joint scales (if available)
        if self.scales_dim > 0:
            scales_flat = output[:, idx:idx + self.scales_dim]
            n_joints = self.scales_dim // 3
            params['log_beta_scales'] = scales_flat.view(batch_size, n_joints, 3)
            idx += self.scales_dim
        
        # Joint translations (if available)
        if self.joint_trans_dim > 0:
            trans_flat = output[:, idx:idx + self.joint_trans_dim]
            n_joints = self.joint_trans_dim // 3
            params['betas_trans'] = trans_flat.view(batch_size, n_joints, 3)
            idx += self.joint_trans_dim
        
        return params
    
    def _forward_transformer_decoder(self, images: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        """Forward pass for transformer decoder regression head."""
        # Extract features using backbone
        if self.backbone_name.startswith('vit'):
            # For ViT, get both global and spatial features
            global_features, spatial_features = self.backbone.forward_with_spatial(images)
        else:
            # For ResNet, only global features available
            global_features = self.backbone(images)
            spatial_features = None
        
        # Pass through transformer decoder head
        params = self.transformer_head(global_features, spatial_features)
        
        # Handle different modes by transforming PCA weights to per-joint values
        if self.scale_trans_mode == 'entangled_with_betas':
            if 'betas' in params:
                log_beta_scales, betas_trans = self._transform_betas_to_joint_values(params['betas'])
                params['log_beta_scales'] = log_beta_scales
                params['betas_trans'] = betas_trans
        elif self.scale_trans_mode == 'separate':
            # In separate mode, keep the PCA weights as-is for loss computation
            # The transformer decoder outputs PCA weights directly
            # We'll compute loss on these PCA weights, not on derived per-joint values
            pass
        
        return params
    
    def predict_from_batch(self, x_data_batch, y_data_batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
        """
        Process an entire batch efficiently for training/validation.
        
        This method implements dual-layer protection against penalizing samples without ground truth:
        
        1. **Implicit Protection (None Detection)**:
           - If y_data['joint_angles'] is None, target is set to None
           - _combine_target_parameters_batch creates availability masks from None detection
           - Ensures single-dataset training never penalizes missing ground truth
        
        2. **Explicit Protection (available_labels)**:
           - Multi-dataset training can provide 'available_labels' in x_data
           - These labels explicitly declare which parameters have ground truth
           - Example: {'joint_rot': False, 'betas': True, 'keypoint_2d': True}
        
        3. **Safety Merge (AND Logic)**:
           - Both mechanisms are combined using AND logic
           - A sample is only penalized if BOTH mechanisms agree it has ground truth
           - This protects against dataset configuration errors
           - Example: If available_labels says joint_rot is available but y_data['joint_angles'] is None,
             the sample is PROTECTED from joint rotation loss
        
        This ensures SLEAP data (and any other partial-label data) is never penalized for missing parameters
        in both single-dataset and mixed-batch multi-dataset training scenarios.
        
        Args:
            x_data_batch: List of x_data dictionaries (one per sample)
                - Can optionally include 'available_labels' dict for explicit protection
            y_data_batch: List of y_data dictionaries (one per sample)
                - Parameters set to None trigger implicit protection
            
        Returns:
            Tuple of (predicted_params, target_params_batch, auxiliary_data)
        """
        # Extract image data from all samples
        batch_images = []
        batch_target_params = []
        batch_auxiliary_data = {
            'keypoint_data': [], 
            'silhouette_data': [], 
            'is_sleap_dataset': [],
            'dataset_sources': []  # For multi-dataset tracking
        }
        
        for x_data, y_data in zip(x_data_batch, y_data_batch):
            # Skip samples without image data
            if x_data['input_image_data'] is None:
                continue
                
            batch_images.append(x_data['input_image_data'])
            
            # Extract target parameters for this sample
            target_params = self._extract_target_parameters_single(y_data)
            batch_target_params.append(target_params)
            
            # Extract auxiliary data for loss computation
            keypoint_data = None
            if 'keypoints_2d' in y_data and 'keypoint_visibility' in y_data:
                keypoint_data = {
                    'keypoints_2d': y_data['keypoints_2d'],
                    'keypoint_visibility': y_data['keypoint_visibility']
                }
                # Add 3D keypoints if available
                if 'keypoints_3d' in y_data:
                    keypoint_data['keypoints_3d'] = y_data['keypoints_3d']
            batch_auxiliary_data['keypoint_data'].append(keypoint_data)
            
            silhouette_data = x_data.get("input_image_mask")
            batch_auxiliary_data['silhouette_data'].append(silhouette_data)
            
            # Extract dataset source (for multi-dataset training)
            dataset_source = x_data.get('dataset_source', 'unknown')
            batch_auxiliary_data['dataset_sources'].append(dataset_source)
            
            # Extract/derive SLEAP dataset flag
            # Support both legacy 'is_sleap_dataset' and new 'dataset_source' fields
            is_sleap = x_data.get('is_sleap_dataset', False)
            if not is_sleap and 'dataset_source' in x_data:
                # If dataset_source contains 'sleap', mark as SLEAP data
                is_sleap = 'sleap' in dataset_source.lower()
            batch_auxiliary_data['is_sleap_dataset'].append(is_sleap)

            # Carry available_labels so the loss can mask unavailable targets
            if 'available_labels' in x_data:
                if 'available_labels' not in batch_auxiliary_data:
                    batch_auxiliary_data['available_labels'] = []
                batch_auxiliary_data['available_labels'].append(x_data['available_labels'])
        
        if not batch_images:
            # No valid samples in batch
            return None, None, None
        
        # Preprocess all images at once
        image_tensor = self.preprocess_image(batch_images).to(self.device)
        
        # Forward pass on entire batch
        predicted_params = self.forward(image_tensor)
        
        # Combine target parameters into batched format
        target_params_batch = self._combine_target_parameters_batch(batch_target_params)

        # If per-sample available_labels provided, convert them to availability masks
        # so unavailable parameters are fully masked from loss
        if 'available_labels' in batch_auxiliary_data:
            # Get implicit availability masks from None detection (already in target_params_batch)
            implicit_masks = target_params_batch.get('_availability_masks', {})
            
            # Build explicit availability masks from available_labels
            explicit_masks = {}
            labels_list = batch_auxiliary_data['available_labels']
            num_samples = len(labels_list)
            # For each parameter in targets, build a boolean mask over samples
            for param_name in ['global_rot','joint_rot','betas','trans','fov','cam_rot','cam_trans','log_beta_scales','betas_trans','keypoint_2d','keypoint_3d','silhouette']:
                mask_vals = []
                for i in range(num_samples):
                    mask_vals.append(bool(labels_list[i].get(param_name, False)))
                explicit_masks[param_name] = torch.tensor(mask_vals, dtype=torch.bool, device=self.device)
            
            # Merge explicit and implicit masks using AND logic for safety
            # A sample is only considered available if BOTH mechanisms agree
            # This prevents penalizing samples with None ground truth even if available_labels incorrectly marks them as available
            availability_masks = {}
            for param_name in explicit_masks.keys():
                if param_name in implicit_masks:
                    # Use AND logic: only available if both implicit (None detection) AND explicit (available_labels) say so
                    merged_mask = explicit_masks[param_name] & implicit_masks[param_name]
                    availability_masks[param_name] = merged_mask
                    
                    # Detect potential dataset configuration errors where available_labels claims data is available
                    # but the actual ground truth is None (detected by implicit mask)
                    mismatches = explicit_masks[param_name] & ~implicit_masks[param_name]
                    if mismatches.any():
                        num_mismatches = mismatches.sum().item()
                        # Only warn occasionally to avoid spam (1% of batches)
                        if torch.rand(1).item() < 0.01:
                            print(f"⚠️  WARNING: Dataset configuration mismatch detected for '{param_name}':")
                            print(f"   {num_mismatches} sample(s) marked as available in 'available_labels' but have None ground truth")
                            print(f"   These samples will be PROTECTED from loss computation (using AND logic)")
                            print(f"   Please fix the dataset's 'available_labels' to match actual ground truth availability")
                else:
                    # No implicit mask for this param (e.g., keypoint_2d, silhouette), use explicit only
                    availability_masks[param_name] = explicit_masks[param_name]
            
            # Also preserve any implicit masks not in explicit (shouldn't happen, but for safety)
            for param_name in implicit_masks.keys():
                if param_name not in availability_masks:
                    availability_masks[param_name] = implicit_masks[param_name]
            
            # Attach merged masks to target_params_batch
            target_params_batch['_availability_masks'] = availability_masks
        
        return predicted_params, target_params_batch, batch_auxiliary_data
    
    def _extract_target_parameters_single(self, y_data):
        """Extract target parameters from a single sample (helper method)."""
        # This is the original extract_target_parameters logic for a single sample
        targets = {}
        
        # Global rotation (root rotation)
        # Return None if placeholder data (will be excluded from loss)
        if y_data.get('root_rot') is None:
            targets['global_rot'] = None
        else:
            targets['global_rot'] = safe_to_tensor(y_data['root_rot'], device=self.device)
        
        # Joint rotations (excluding root joint)
        # Return None if placeholder data (will be excluded from loss)
        if y_data.get('joint_angles') is None:
            targets['joint_rot'] = None
        else:
            joint_angles = safe_to_tensor(y_data['joint_angles'], device=self.device)
            targets['joint_rot'] = joint_angles[1:]  # Exclude root joint
        
        # Shape parameters
        # Return None if placeholder data (will be excluded from loss)
        if y_data.get('shape_betas') is None:
            targets['betas'] = None
        else:
            targets['betas'] = safe_to_tensor(y_data['shape_betas'], device=self.device)
        
        # Translation (root location)
        # Return None if placeholder data (will be excluded from loss)
        if y_data.get('root_loc') is None:
            targets['trans'] = None
        else:
            targets['trans'] = safe_to_tensor(y_data['root_loc'], device=self.device)
        
        # Camera FOV
        # Return None if placeholder data (will be excluded from loss)
        if y_data.get('cam_fov') is None:
            targets['fov'] = None
        else:
            fov_value = y_data['cam_fov']
            if isinstance(fov_value, list):
                fov_value = fov_value[0]  # Take first element if it's a list
            targets['fov'] = torch.tensor([fov_value], dtype=torch.float32).to(self.device)
        
        # Camera rotation and translation
        # Return None if placeholder data (will be excluded from loss)
        if y_data.get('cam_rot') is None:
            targets['cam_rot'] = None
        else:
            cam_rot_matrix = y_data['cam_rot']
            if hasattr(cam_rot_matrix, 'shape') and len(cam_rot_matrix.shape) == 2 and cam_rot_matrix.shape == (3, 3):
                targets['cam_rot'] = safe_to_tensor(cam_rot_matrix, device=self.device)
            else:
                if hasattr(cam_rot_matrix, 'shape') and cam_rot_matrix.shape == (3,):
                    r = Rotation.from_rotvec(cam_rot_matrix)
                    cam_rot_matrix = r.as_matrix()
                    targets['cam_rot'] = safe_to_tensor(cam_rot_matrix, device=self.device)
                else:
                    targets['cam_rot'] = safe_to_tensor(cam_rot_matrix, device=self.device)
        
        if y_data.get('cam_trans') is None:
            targets['cam_trans'] = None
        else:
            targets['cam_trans'] = safe_to_tensor(y_data['cam_trans'], device=self.device)
        
        # Handle scale and translation parameters based on mode
        if self.scale_trans_mode == 'ignore':
            # Set to zeros - no scaling or translation
            # Use PCA weights (5 parameters) set to zero to match predicted dimensions
            targets['log_beta_scales'] = torch.zeros(config.N_BETAS).to(self.device)
            targets['betas_trans'] = torch.zeros(config.N_BETAS).to(self.device)
            
        elif self.scale_trans_mode == 'separate':
            # In separate mode, use PCA weights directly as targets
            if y_data['scale_weights'] is not None and y_data['trans_weights'] is not None:
                # Use the PCA weights directly (5 parameters each)
                targets['log_beta_scales'] = torch.from_numpy(y_data['scale_weights']).float().to(self.device)
                targets['betas_trans'] = torch.from_numpy(y_data['trans_weights']).float().to(self.device)
            else:
                # No PCA weights available, use zeros
                targets['log_beta_scales'] = torch.zeros(config.N_BETAS).to(self.device)
                targets['betas_trans'] = torch.zeros(config.N_BETAS).to(self.device)
                
        elif self.scale_trans_mode == 'entangled_with_betas':
            # For entangled mode, we still need to compute the per-joint values
            # but we'll use the same betas for all three PCA spaces
            if y_data['scale_weights'] is not None and y_data['trans_weights'] is not None:
                from Unreal2Pytorch3D import sample_pca_transforms_from_dirs
                translation_out, scale_out = sample_pca_transforms_from_dirs(
                    config.dd, y_data['scale_weights'], y_data['trans_weights']
                )
                # Check for NaN or infinite values in PCA results
                if not np.isfinite(scale_out).all():
                    print(f"Warning: Non-finite values in scale_out, replacing with ones")
                    scale_out = np.nan_to_num(scale_out, nan=1.0, posinf=1.0, neginf=1.0)
                if not np.isfinite(translation_out).all():
                    print(f"Warning: Non-finite values in translation_out, replacing with zeros")
                    translation_out = np.nan_to_num(translation_out, nan=0.0, posinf=0.0, neginf=0.0)
                targets['log_beta_scales'] = torch.from_numpy(np.log(np.maximum(scale_out, 1e-8))).float().to(self.device)
                targets['betas_trans'] = torch.from_numpy(translation_out * y_data['translation_factor']).float().to(self.device)
            else:
                n_joints = len(config.dd["J_names"])
                targets['log_beta_scales'] = torch.zeros(n_joints, 3).to(self.device)
                targets['betas_trans'] = torch.zeros(n_joints, 3).to(self.device)
        
        return targets
    
    def _combine_target_parameters_batch(self, target_params_list):
        """
        Combine list of target parameters into batched tensors with availability masks.
        
        Handles None values (placeholder data) by:
        1. Creating zero/identity placeholders for None values
        2. Creating availability masks to indicate which samples have real data
        3. Using masks in loss computation (complete masking - Option A)
        """
        if not target_params_list:
            return {}
        
        batch_size = len(target_params_list)
        batch_targets = {}
        batch_availability = {}  # Track which samples have real data for each parameter
        
        # Get all parameter names from first sample
        param_names = set()
        for targets in target_params_list:
            param_names.update(targets.keys())
        
        for param_name in param_names:
            # Collect tensors and track availability
            param_tensors = []
            availability_mask = []
            
            for targets in target_params_list:
                if targets.get(param_name) is not None:
                    param_tensors.append(targets[param_name])
                    availability_mask.append(True)
                else:
                    # Create placeholder tensor with appropriate shape
                    placeholder = self._create_placeholder_tensor(param_name, targets)
                    param_tensors.append(placeholder)
                    availability_mask.append(False)
            
            # Stack all tensors (including placeholders)
            batch_targets[param_name] = torch.stack(param_tensors, dim=0)
            
            # Store availability mask
            batch_availability[param_name] = torch.tensor(
                availability_mask, dtype=torch.bool, device=self.device
            )
        
        # Add availability masks to batch_targets with special key
        batch_targets['_availability_masks'] = batch_availability
        
        return batch_targets
    
    def _create_placeholder_tensor(self, param_name: str, targets: Dict) -> torch.Tensor:
        """
        Create a placeholder tensor for a parameter with None value.
        
        Args:
            param_name: Name of the parameter
            targets: Target dictionary (used to infer shapes from other parameters)
            
        Returns:
            Placeholder tensor with appropriate shape and values
        """
        # Use zeros for most parameters, identity for rotations
        if param_name == 'global_rot':
            if self.rotation_representation == '6d':
                # Identity rotation in 6D: [1, 0, 0, 0, 1, 0]
                return torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
                                  dtype=torch.float32, device=self.device)
            else:
                # Identity rotation in axis-angle: [0, 0, 0]
                return torch.zeros(3, dtype=torch.float32, device=self.device)
        
        elif param_name == 'joint_rot':
            # Need to know number of joints
            n_joints = config.N_POSE
            if self.rotation_representation == '6d':
                return torch.zeros(n_joints, 6, dtype=torch.float32, device=self.device)
            else:
                return torch.zeros(n_joints, 3, dtype=torch.float32, device=self.device)
        
        elif param_name == 'betas':
            return torch.zeros(config.N_BETAS, dtype=torch.float32, device=self.device)
        
        elif param_name == 'trans':
            return torch.zeros(3, dtype=torch.float32, device=self.device)
        
        elif param_name == 'fov':
            return torch.tensor([60.0], dtype=torch.float32, device=self.device)  # Default FOV
        
        elif param_name == 'cam_rot':
            # Identity rotation matrix
            return torch.eye(3, dtype=torch.float32, device=self.device)
        
        elif param_name == 'cam_trans':
            return torch.zeros(3, dtype=torch.float32, device=self.device)
        
        elif param_name == 'log_beta_scales':
            # For entangled mode or when not available
            n_joints = len(config.dd["J_names"]) if hasattr(config.dd, '__getitem__') else 32
            return torch.zeros(n_joints, 3, dtype=torch.float32, device=self.device)
        
        elif param_name == 'betas_trans':
            n_joints = len(config.dd["J_names"]) if hasattr(config.dd, '__getitem__') else 32
            return torch.zeros(n_joints, 3, dtype=torch.float32, device=self.device)
        
        else:
            # Unknown parameter - return scalar zero
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
    
    def _transform_betas_to_joint_values(self, betas, translation_factor=0.01):
        """
        Transform betas through PCA spaces to get per-joint scale and translation values.
        
        Args:
            betas: Tensor of shape (batch_size, num_betas) - PCA weights
            translation_factor: Factor to apply to translation values
            
        Returns:
            tuple: (log_beta_scales, betas_trans) both of shape (batch_size, num_joints, 3)
        """
        if not hasattr(config.dd, 'scaledirs') or not hasattr(config.dd, 'transdirs'):
            # No PCA data available, return zeros
            n_joints = len(config.dd["J_names"])
            batch_size = betas.shape[0]
            log_beta_scales = torch.zeros(batch_size, n_joints, 3, device=betas.device)
            betas_trans = torch.zeros(batch_size, n_joints, 3, device=betas.device)
            return log_beta_scales, betas_trans
        
        # Convert betas to numpy for PCA transformation
        betas_np = betas.detach().cpu().numpy()
        batch_size = betas_np.shape[0]
        
        # Check for NaN or infinite values in betas
        if not np.isfinite(betas_np).all():
            print(f"Warning: Non-finite values in betas, replacing with zeros")
            betas_np = np.nan_to_num(betas_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        log_beta_scales_list = []
        betas_trans_list = []
        
        for i in range(batch_size):
            # Use the same betas for both scale and translation PCA spaces
            scale_weights = betas_np[i]
            trans_weights = betas_np[i]
            
            from Unreal2Pytorch3D import sample_pca_transforms_from_dirs
            translation_out, scale_out = sample_pca_transforms_from_dirs(
                config.dd, scale_weights, trans_weights
            )
            
            # Check for NaN or infinite values in PCA results
            if not np.isfinite(scale_out).all():
                print(f"Warning: Non-finite values in scale_out, replacing with ones")
                scale_out = np.nan_to_num(scale_out, nan=1.0, posinf=1.0, neginf=1.0)
            if not np.isfinite(translation_out).all():
                print(f"Warning: Non-finite values in translation_out, replacing with zeros")
                translation_out = np.nan_to_num(translation_out, nan=0.0, posinf=0.0, neginf=0.0)
            
            log_beta_scales_list.append(np.log(np.maximum(scale_out, 1e-8)))
            betas_trans_list.append(translation_out * translation_factor)
        
        # Convert back to tensors
        log_beta_scales = torch.from_numpy(np.stack(log_beta_scales_list)).float().to(betas.device)
        betas_trans = torch.from_numpy(np.stack(betas_trans_list)).float().to(betas.device)
        
        return log_beta_scales, betas_trans
    
    def _transform_separate_pca_weights_to_joint_values(self, scale_weights, trans_weights, translation_factor=0.01):
        """
        Transform separate PCA weights for scales and translations to per-joint values.
        
        Args:
            scale_weights: Tensor of shape (batch_size, num_betas) - PCA weights for scaling
            trans_weights: Tensor of shape (batch_size, num_betas) - PCA weights for translation
            translation_factor: Factor to apply to translation values
            
        Returns:
            tuple: (log_beta_scales, betas_trans) both of shape (batch_size, num_joints, 3)
        """
        if not hasattr(config.dd, 'scaledirs') or not hasattr(config.dd, 'transdirs'):
            # No PCA data available, return zeros
            n_joints = len(config.dd["J_names"])
            batch_size = scale_weights.shape[0]
            log_beta_scales = torch.zeros(batch_size, n_joints, 3, device=scale_weights.device)
            betas_trans = torch.zeros(batch_size, n_joints, 3, device=scale_weights.device)
            return log_beta_scales, betas_trans
        
        # Convert to numpy for PCA transformation
        scale_weights_np = scale_weights.detach().cpu().numpy()
        trans_weights_np = trans_weights.detach().cpu().numpy()
        batch_size = scale_weights_np.shape[0]
        
        log_beta_scales_list = []
        betas_trans_list = []
        
        for i in range(batch_size):
            # Use separate weights for scale and translation PCA spaces
            scale_weights_i = scale_weights_np[i]
            trans_weights_i = trans_weights_np[i]
            
            from Unreal2Pytorch3D import sample_pca_transforms_from_dirs
            translation_out, scale_out = sample_pca_transforms_from_dirs(
                config.dd, scale_weights_i, trans_weights_i
            )
            
            # Check for NaN or infinite values in PCA results
            if not np.isfinite(scale_out).all():
                print(f"Warning: Non-finite values in scale_out, replacing with ones")
                scale_out = np.nan_to_num(scale_out, nan=1.0, posinf=1.0, neginf=1.0)
            if not np.isfinite(translation_out).all():
                print(f"Warning: Non-finite values in translation_out, replacing with zeros")
                translation_out = np.nan_to_num(translation_out, nan=0.0, posinf=0.0, neginf=0.0)
            
            log_beta_scales_list.append(np.log(np.maximum(scale_out, 1e-8)))
            betas_trans_list.append(translation_out * translation_factor)
        
        # Convert back to tensors
        log_beta_scales = torch.from_numpy(np.stack(log_beta_scales_list)).float().to(scale_weights.device)
        betas_trans = torch.from_numpy(np.stack(betas_trans_list)).float().to(scale_weights.device)
        
        return log_beta_scales, betas_trans
    
    def predict_from_image(self, image_data: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Predict SMIL parameters from a single image.
        
        Args:
            image_data: Input image as numpy array (H, W, C)
            
        Returns:
            Dictionary containing predicted SMIL parameters
        """
        self.eval()
        with torch.no_grad():
            # Preprocess image
            image_tensor = self.preprocess_image(image_data).to(self.device)
            
            # Predict parameters
            params = self.forward(image_tensor)
            
            return params
    
    def set_smil_parameters(self, params: Dict[str, torch.Tensor], batch_idx: int = 0):
        """
        Set the predicted parameters to the SMALFitter model.
        
        Args:
            params: Dictionary containing predicted SMIL parameters
            batch_idx: Index of the batch to set parameters for
        """
        # Set global rotation (keep gradients)
        self.global_rotation[batch_idx] = params['global_rot'][batch_idx]
        
        # Set joint rotations (keep gradients)
        self.joint_rotations[batch_idx] = params['joint_rot'][batch_idx]
        
        # Set shape parameters (betas) (keep gradients)
        self.betas = params['betas'][batch_idx]
        
        # Set translation (keep gradients)
        self.trans[batch_idx] = params['trans'][batch_idx]
        
        # Set camera FOV (keep gradients)
        self.fov[batch_idx] = params['fov'][batch_idx]
        
        # Set joint scales (if available) (keep gradients)
        if 'log_beta_scales' in params and hasattr(self, 'log_beta_scales'):
            self.log_beta_scales[batch_idx] = params['log_beta_scales'][batch_idx]
        
        # Set joint translations (if available) (keep gradients)
        if 'betas_trans' in params and hasattr(self, 'betas_trans'):
            self.betas_trans[batch_idx] = params['betas_trans'][batch_idx]
    
    def compute_batch_loss(self, predicted_params: Dict[str, torch.Tensor], 
                          target_params_batch: Dict[str, torch.Tensor], 
                          auxiliary_data: Dict = None, return_components=False,
                          loss_weights: Dict[str, float] = None) -> torch.Tensor:
        """
        Compute loss for an entire batch efficiently.
        
        Args:
            predicted_params: Dictionary containing predicted parameters (batch tensors)
            target_params_batch: Dictionary containing target parameters (batch tensors)
            auxiliary_data: Dictionary containing keypoint and silhouette data for the batch
            return_components: If True, return both total loss and individual components
            loss_weights: Dictionary of loss weights for different components
            
        Returns:
            Total loss as a scalar tensor, or (total_loss, loss_components) if return_components=True
        """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_components = {}
        
        if loss_weights is None:
            # Define loss weights for different parameter types
            loss_weights = {
                'global_rot': 0.02,
                'joint_rot': 0.02,
                'betas': 0.01,
                'trans': 0.001,
                'fov': 0.001,
                'cam_rot': 0.01,
                'cam_trans': 0.001,
                'log_beta_scales': 0.1,
                'betas_trans': 0.1,
                'keypoint_2d': 0.0,
                'keypoint_3d': 0.0,
                'silhouette': 0.0
            }
        
        batch_size = predicted_params['global_rot'].shape[0]
        
        # Validate sample visibility before computing any losses
        # This prevents the model from being penalized for things it cannot possibly know
        sample_validity_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        filtered_samples_count = 0
        
        if auxiliary_data is not None:
            keypoint_data_list = auxiliary_data.get('keypoint_data', [])
            silhouette_data_list = auxiliary_data.get('silhouette_data', [])
            is_sleap_list = auxiliary_data.get('is_sleap_dataset', [])
            
            for i in range(batch_size):
                # Get keypoint and silhouette data for this sample
                keypoint_data = keypoint_data_list[i] if i < len(keypoint_data_list) else None
                silhouette_data = silhouette_data_list[i] if i < len(silhouette_data_list) else None
                is_sleap = is_sleap_list[i] if i < len(is_sleap_list) else False
                
                # Validate sample visibility
                is_valid = self._validate_sample_visibility(
                    keypoint_data=keypoint_data,
                    silhouette_data=silhouette_data,
                    min_visible_keypoints=5,
                    min_pixel_coverage=0.05,
                    is_sleap_dataset=is_sleap
                )
                
                if not is_valid:
                    sample_validity_mask[i] = False
                    filtered_samples_count += 1
        
        # Log filtering statistics (occasionally to avoid spam)
        if hasattr(self, '_debug_shapes') and torch.rand(1).item() < 0.01:  # 1% of the time
            print(f"DEBUG - Sample filtering: {filtered_samples_count}/{batch_size} samples filtered out")
            if filtered_samples_count > 0:
                print(f"DEBUG - Filtering rate: {filtered_samples_count/batch_size*100:.1f}%")
        
        # Track filtering statistics for monitoring
        if not hasattr(self, '_filtering_stats'):
            self._filtering_stats = {
                'total_batches': 0,
                'total_samples': 0,
                'filtered_samples': 0,
                'batches_with_filtering': 0
            }
        
        self._filtering_stats['total_batches'] += 1
        self._filtering_stats['total_samples'] += batch_size
        self._filtering_stats['filtered_samples'] += filtered_samples_count
        if filtered_samples_count > 0:
            self._filtering_stats['batches_with_filtering'] += 1
        
        # If all samples are invalid, return zero loss
        if not sample_validity_mask.any():
            print(f"WARNING: All {batch_size} samples in batch filtered out (invalid)")
            eps = 1e-8
            zero_loss = torch.tensor(eps, device=self.device, requires_grad=True)
            if return_components:
                # Return zero for all loss components
                loss_components = {
                    'global_rot': zero_loss,
                    'joint_rot': zero_loss,
                    'betas': zero_loss,
                    'trans': zero_loss,
                    'fov': zero_loss,
                    'cam_rot': zero_loss,
                    'cam_trans': zero_loss,
                    'log_beta_scales': zero_loss,
                    'betas_trans': zero_loss,
                    'keypoint_2d': zero_loss,
                    'silhouette': zero_loss
                }
                return zero_loss, loss_components
            return zero_loss
        
        # Process keypoint and silhouette losses efficiently for the whole batch
        need_keypoint_loss = (auxiliary_data is not None and 'keypoint_data' in auxiliary_data and 
                             loss_weights['keypoint_2d'] > 0)
        need_silhouette_loss = (auxiliary_data is not None and 'silhouette_data' in auxiliary_data and 
                               loss_weights['silhouette'] > 0)
        need_keypoint_3d_loss = (auxiliary_data is not None and 'keypoint_data' in auxiliary_data and 
                                loss_weights['keypoint_3d'] > 0)
        
        # Single rendering pass for 2D keypoints, silhouette, and 3D keypoints if needed
        rendered_joints = None
        rendered_silhouette = None
        joints_3d = None
        if need_keypoint_loss or need_silhouette_loss or need_keypoint_3d_loss:
            try:
                rendered_joints, rendered_silhouette, joints_3d = self._compute_rendered_outputs(
                    predicted_params, compute_joints=need_keypoint_loss, 
                    compute_silhouette=need_silhouette_loss, compute_joints_3d=need_keypoint_3d_loss
                )
            except Exception as e:
                print(f"Warning: Failed to compute rendered outputs for batch: {e}")
                # Continue without rendered outputs
                rendered_joints = None
                rendered_silhouette = None
                joints_3d = None
        
        # Apply validity mask to parameters for loss computation
        # Only compute losses for valid samples to prevent penalizing the model for impossible predictions
        valid_predicted_params = {}
        valid_target_params = {}
        
        for key in predicted_params:
            if key in target_params_batch:
                # Apply validity mask to both predicted and target parameters
                valid_predicted_params[key] = predicted_params[key][sample_validity_mask]
                valid_target_params[key] = target_params_batch[key][sample_validity_mask]
        
        # Get number of valid samples for proper averaging
        num_valid_samples = sample_validity_mask.sum().item()
        eps = 1e-8
        
        # Extract availability masks if present (for multi-dataset training)
        # This replaces the legacy SLEAP batch detection - availability masks handle
        # per-sample filtering automatically, allowing mixed batches
        availability_masks = target_params_batch.get('_availability_masks', {})
        
        # Basic parameter losses (only for valid samples)
        
        # Global rotation loss
        if 'global_rot' in valid_target_params and num_valid_samples > 0:
            # Apply availability masking (complete masking - Option A)
            pred_masked, target_masked, num_available = self._apply_availability_mask(
                valid_predicted_params['global_rot'],
                valid_target_params['global_rot'],
                availability_masks.get('global_rot', None),
                sample_validity_mask
            )
            
            if num_available > 0:
                if self.rotation_representation == '6d':
                    pred_global_matrix = rotation_6d_to_matrix(pred_masked)
                    target_global_matrix = rotation_6d_to_matrix(target_masked)
                    matrix_diff_loss = torch.norm(pred_global_matrix - target_global_matrix, p='fro', dim=(-2, -1))
                    loss = matrix_diff_loss.mean()
                else:
                    loss = F.mse_loss(pred_masked, target_masked)
                
                loss_components['global_rot'] = loss
                total_loss = total_loss + loss_weights['global_rot'] * loss
            else:
                # No available data for this parameter
                loss_components['global_rot'] = torch.tensor(eps, device=self.device, requires_grad=True)
        else:
            # No valid samples or no global rotation data
            loss_components['global_rot'] = torch.tensor(eps, device=self.device, requires_grad=True)
        
        # Joint rotation loss (with visibility awareness)
        # IMPORTANT: This loss is PROTECTED by availability masks to ensure samples without joint angle ground truth
        # (e.g., SLEAP data) are NEVER penalized. The _apply_availability_mask filters out samples where:
        #   - y_data['joint_angles'] was None (implicit protection via None detection)
        #   - OR x_data['available_labels']['joint_rot'] was False (explicit protection)
        # This works for both single-dataset and mixed-batch multi-dataset training.
        if 'joint_rot' in valid_target_params and num_valid_samples > 0:
            # Respect availability mask (skip entirely if no samples have ground truth)
            pred_masked_jr, target_masked_jr, num_available_joint = self._apply_availability_mask(
                valid_predicted_params['joint_rot'],
                valid_target_params['joint_rot'],
                availability_masks.get('joint_rot', None),
                sample_validity_mask
            )
            if num_available_joint == 0:
                # No samples in batch have joint rotation ground truth - return epsilon loss (no penalty)
                loss_components['joint_rot'] = torch.tensor(eps, device=self.device, requires_grad=True)
            else:
                # Check if we have visibility information for joint-specific loss
                if auxiliary_data is not None and 'keypoint_data' in auxiliary_data and auxiliary_data['keypoint_data']:
                    # Build mask over original batch for samples that are both valid and available
                    if availability_masks.get('joint_rot', None) is not None:
                        joint_avail_mask_valid = availability_masks['joint_rot'][sample_validity_mask]
                    else:
                        # If no availability provided, default to all True (shouldn't happen here)
                        joint_avail_mask_valid = torch.ones(sample_validity_mask.sum().item(), dtype=torch.bool, device=self.device)

                    # First filter keypoint_data by sample_validity_mask, then by availability
                    temp_keypoint_data = [auxiliary_data['keypoint_data'][i] for i in range(len(auxiliary_data['keypoint_data'])) if sample_validity_mask[i]]
                    valid_keypoint_data = [kd for kd, keep in zip(temp_keypoint_data, joint_avail_mask_valid.tolist()) if keep]
                    
                    # Use visibility-aware joint rotation loss on masked tensors and filtered keypoint data
                    loss = self._compute_visibility_aware_joint_rotation_loss_batch(
                        pred_masked_jr, target_masked_jr, valid_keypoint_data
                    )
                else:
                    # Fallback to standard joint rotation loss if no visibility data
                    if self.rotation_representation == '6d':
                        pred_matrices = rotation_6d_to_matrix(pred_masked_jr)
                        target_matrices = rotation_6d_to_matrix(target_masked_jr)
                        matrix_diff_loss = torch.norm(pred_matrices - target_matrices, p='fro', dim=(-2, -1))
                        loss = matrix_diff_loss.mean()
                    else:
                        loss = F.mse_loss(pred_masked_jr, target_masked_jr)
                
                loss_components['joint_rot'] = loss
                total_loss = total_loss + loss_weights['joint_rot'] * loss
        else:
            # No valid samples or no joint rotation data
            loss_components['joint_rot'] = torch.tensor(eps, device=self.device, requires_grad=True)
        
        # Shape parameter loss
        if 'betas' in valid_target_params and num_valid_samples > 0:
            # Apply availability masking
            pred_masked, target_masked, num_available = self._apply_availability_mask(
                valid_predicted_params['betas'],
                valid_target_params['betas'],
                availability_masks.get('betas', None),
                sample_validity_mask
            )
            
            if num_available > 0:
                loss = F.mse_loss(pred_masked, target_masked)
                loss_components['betas'] = loss
                total_loss = total_loss + loss_weights['betas'] * loss
            else:
                loss_components['betas'] = torch.tensor(eps, device=self.device, requires_grad=True)
        else:
            loss_components['betas'] = torch.tensor(eps, device=self.device, requires_grad=True)
        
        # Translation loss
        if 'trans' in valid_target_params and num_valid_samples > 0:
            # Apply availability masking
            pred_masked, target_masked, num_available = self._apply_availability_mask(
                valid_predicted_params['trans'],
                valid_target_params['trans'],
                availability_masks.get('trans', None),
                sample_validity_mask
            )
            
            if num_available > 0:
                loss = F.mse_loss(pred_masked, target_masked)
                loss_components['trans'] = loss
                total_loss = total_loss + loss_weights['trans'] * loss
            else:
                loss_components['trans'] = torch.tensor(eps, device=self.device, requires_grad=True)
        else:
            loss_components['trans'] = torch.tensor(eps, device=self.device, requires_grad=True)
        
        # FOV loss
        if 'fov' in valid_target_params and num_valid_samples > 0:
            # Apply availability masking
            pred_masked, target_masked, num_available = self._apply_availability_mask(
                valid_predicted_params['fov'],
                valid_target_params['fov'],
                availability_masks.get('fov', None),
                sample_validity_mask
            )
            
            if num_available > 0:
                # Handle shape mismatch
                pred_fov = pred_masked
                target_fov = target_masked
                
                if pred_fov.shape != target_fov.shape:
                    if len(pred_fov.shape) != len(target_fov.shape):
                        if len(pred_fov.shape) > len(target_fov.shape):
                            target_fov = target_fov.unsqueeze(-1)
                        else:
                            pred_fov = pred_fov.unsqueeze(-1)
                
                loss = F.mse_loss(pred_fov, target_fov)
                loss_components['fov'] = loss
                total_loss = total_loss + loss_weights['fov'] * loss
            else:
                loss_components['fov'] = torch.tensor(eps, device=self.device, requires_grad=True)
        else:
            loss_components['fov'] = torch.tensor(eps, device=self.device, requires_grad=True)
        
        # Camera rotation loss
        if 'cam_rot' in valid_target_params and num_valid_samples > 0:
            # Apply availability masking
            pred_masked, target_masked, num_available = self._apply_availability_mask(
                valid_predicted_params['cam_rot'],
                valid_target_params['cam_rot'],
                availability_masks.get('cam_rot', None),
                sample_validity_mask
            )
            
            if num_available > 0:
                loss = F.mse_loss(pred_masked, target_masked)
                loss_components['cam_rot'] = loss
                total_loss = total_loss + loss_weights['cam_rot'] * loss
            else:
                loss_components['cam_rot'] = torch.tensor(eps, device=self.device, requires_grad=True)
        else:
            loss_components['cam_rot'] = torch.tensor(eps, device=self.device, requires_grad=True)
        
        # Camera translation loss
        if 'cam_trans' in valid_target_params and num_valid_samples > 0:
            # Apply availability masking
            pred_masked, target_masked, num_available = self._apply_availability_mask(
                valid_predicted_params['cam_trans'],
                valid_target_params['cam_trans'],
                availability_masks.get('cam_trans', None),
                sample_validity_mask
            )
            
            if num_available > 0:
                loss = F.mse_loss(pred_masked, target_masked)
                loss_components['cam_trans'] = loss
                total_loss = total_loss + loss_weights['cam_trans'] * loss
            else:
                loss_components['cam_trans'] = torch.tensor(eps, device=self.device, requires_grad=True)
        else:
            loss_components['cam_trans'] = torch.tensor(eps, device=self.device, requires_grad=True)
        
        # Handle scale and translation losses based on mode
        if self.scale_trans_mode == 'entangled_with_betas':
            # In entangled mode, we only supervise the betas directly
            # The scale and translation values are derived from betas, so no separate losses
            loss_components['log_beta_scales'] = torch.tensor(eps, device=self.device, requires_grad=True)
            loss_components['betas_trans'] = torch.tensor(eps, device=self.device, requires_grad=True)
        else:
            # Original logic for separate/ignore modes
            # Joint scales loss (if available)
            if 'log_beta_scales' in valid_target_params and 'log_beta_scales' in valid_predicted_params and num_valid_samples > 0:
                # Apply availability masking
                pred_masked, target_masked, num_available = self._apply_availability_mask(
                    valid_predicted_params['log_beta_scales'],
                    valid_target_params['log_beta_scales'],
                    availability_masks.get('log_beta_scales', None),
                    sample_validity_mask
                )
                
                if num_available > 0:
                    loss = F.mse_loss(pred_masked, target_masked)
                    loss_components['log_beta_scales'] = loss
                    total_loss = total_loss + loss_weights['log_beta_scales'] * loss
                else:
                    loss_components['log_beta_scales'] = torch.tensor(eps, device=self.device, requires_grad=True)
            else:
                loss_components['log_beta_scales'] = torch.tensor(eps, device=self.device, requires_grad=True)
            
            # Joint translations loss (if available)
            if 'betas_trans' in valid_target_params and 'betas_trans' in valid_predicted_params and num_valid_samples > 0:
                # Apply availability masking
                pred_masked, target_masked, num_available = self._apply_availability_mask(
                    valid_predicted_params['betas_trans'],
                    valid_target_params['betas_trans'],
                    availability_masks.get('betas_trans', None),
                    sample_validity_mask
                )
                
                if num_available > 0:
                    loss = F.mse_loss(pred_masked, target_masked)
                    loss_components['betas_trans'] = loss
                    total_loss = total_loss + loss_weights['betas_trans'] * loss
                else:
                    loss_components['betas_trans'] = torch.tensor(eps, device=self.device, requires_grad=True)
            else:
                loss_components['betas_trans'] = torch.tensor(eps, device=self.device, requires_grad=True)
        
        # Batched 2D keypoint loss (only for valid samples with available 2D keypoints)
        if need_keypoint_loss and rendered_joints is not None and auxiliary_data['keypoint_data'] and num_valid_samples > 0:
            # Apply availability mask - only compute loss for samples with real 2D keypoint ground truth
            keypoint_2d_availability = availability_masks.get('keypoint_2d', None)
            
            if keypoint_2d_availability is not None:
                # Filter to samples with available 2D keypoints AND valid visibility
                combined_mask = sample_validity_mask & keypoint_2d_availability
                
                if combined_mask.any():
                    try:
                        # Filter rendered joints and keypoint data to only include samples with available data
                        valid_rendered_joints = rendered_joints[combined_mask]
                        valid_keypoint_data = [auxiliary_data['keypoint_data'][i] for i in range(len(auxiliary_data['keypoint_data'])) 
                                             if i < len(combined_mask) and combined_mask[i]]
                        
                        loss = self._compute_batch_keypoint_loss(valid_rendered_joints, valid_keypoint_data)
                        if torch.isfinite(loss):
                            loss_components['keypoint_2d'] = loss
                            total_loss = total_loss + loss_weights['keypoint_2d'] * loss
                        else:
                            loss_components['keypoint_2d'] = torch.tensor(eps, device=self.device, requires_grad=True)
                    except Exception as e:
                        print(f"Warning: Failed to compute batch keypoint loss: {e}")
                        import traceback
                        traceback.print_exc()
                        loss_components['keypoint_2d'] = torch.tensor(eps, device=self.device, requires_grad=True)
                else:
                    # No samples with available 2D keypoints
                    loss_components['keypoint_2d'] = torch.tensor(eps, device=self.device, requires_grad=True)
            else:
                # No availability masks - legacy behavior (try all valid samples)
                try:
                    valid_rendered_joints = rendered_joints[sample_validity_mask]
                    valid_keypoint_data = [auxiliary_data['keypoint_data'][i] for i in range(len(auxiliary_data['keypoint_data'])) 
                                         if sample_validity_mask[i]]
                    
                    loss = self._compute_batch_keypoint_loss(valid_rendered_joints, valid_keypoint_data)
                    if torch.isfinite(loss):
                        loss_components['keypoint_2d'] = loss
                        total_loss = total_loss + loss_weights['keypoint_2d'] * loss
                    else:
                        loss_components['keypoint_2d'] = torch.tensor(eps, device=self.device, requires_grad=True)
                except Exception as e:
                    print(f"Warning: Failed to compute batch keypoint loss: {e}")
                    import traceback
                    traceback.print_exc()
                    loss_components['keypoint_2d'] = torch.tensor(eps, device=self.device, requires_grad=True)
        else:
            loss_components['keypoint_2d'] = torch.tensor(eps, device=self.device, requires_grad=True)
        
        # Batched silhouette loss (only for valid samples with available silhouette)
        if need_silhouette_loss and rendered_silhouette is not None and auxiliary_data['silhouette_data'] and num_valid_samples > 0:
            # Apply availability mask - only compute loss for samples with real silhouette ground truth
            silhouette_availability = availability_masks.get('silhouette', None)
            
            if silhouette_availability is not None:
                # Filter to samples with available silhouette AND valid visibility
                combined_mask = sample_validity_mask & silhouette_availability
                
                if combined_mask.any():
                    try:
                        # Filter rendered silhouette and silhouette data to only include samples with available data
                        valid_rendered_silhouette = rendered_silhouette[combined_mask]
                        valid_silhouette_data = [auxiliary_data['silhouette_data'][i] for i in range(len(auxiliary_data['silhouette_data'])) 
                                               if i < len(combined_mask) and combined_mask[i]]
                        
                        loss = self._compute_batch_silhouette_loss(valid_rendered_silhouette, valid_silhouette_data)
                        if torch.isfinite(loss):
                            loss_components['silhouette'] = loss
                            total_loss = total_loss + loss_weights['silhouette'] * loss
                        else:
                            loss_components['silhouette'] = torch.tensor(eps, device=self.device, requires_grad=True)
                    except Exception as e:
                        print(f"Warning: Failed to compute batch silhouette loss: {e}")
                        loss_components['silhouette'] = torch.tensor(eps, device=self.device, requires_grad=True)
                else:
                    # No samples with available silhouette
                    loss_components['silhouette'] = torch.tensor(eps, device=self.device, requires_grad=True)
            else:
                # No availability masks - legacy behavior (try all valid samples)
                try:
                    valid_rendered_silhouette = rendered_silhouette[sample_validity_mask]
                    valid_silhouette_data = [auxiliary_data['silhouette_data'][i] for i in range(len(auxiliary_data['silhouette_data'])) 
                                           if sample_validity_mask[i]]
                    
                    loss = self._compute_batch_silhouette_loss(valid_rendered_silhouette, valid_silhouette_data)
                    if torch.isfinite(loss):
                        loss_components['silhouette'] = loss
                        total_loss = total_loss + loss_weights['silhouette'] * loss
                    else:
                        loss_components['silhouette'] = torch.tensor(eps, device=self.device, requires_grad=True)
                except Exception as e:
                    print(f"Warning: Failed to compute batch silhouette loss: {e}")
                    loss_components['silhouette'] = torch.tensor(eps, device=self.device, requires_grad=True)
        else:
            loss_components['silhouette'] = torch.tensor(eps, device=self.device, requires_grad=True)
        
        # Batched 3D keypoint loss (only for valid samples with available 3D keypoints)
        if need_keypoint_3d_loss and joints_3d is not None and auxiliary_data['keypoint_data'] and num_valid_samples > 0:
            # Apply availability mask - only compute loss for samples with real 3D keypoint ground truth
            keypoint_3d_availability = availability_masks.get('keypoint_3d', None)
            
            if keypoint_3d_availability is not None:
                # Filter to samples with available 3D keypoints AND valid visibility
                combined_mask = sample_validity_mask & keypoint_3d_availability
                
                if combined_mask.any():
                    try:
                        # Filter 3D joints and keypoint data to only include samples with available data
                        valid_joints_3d = joints_3d[combined_mask]
                        valid_keypoint_data = [auxiliary_data['keypoint_data'][i] for i in range(len(auxiliary_data['keypoint_data'])) 
                                             if i < len(combined_mask) and combined_mask[i]]
                        
                        loss = self._compute_batch_keypoint_3d_loss(valid_joints_3d, valid_keypoint_data)
                        if torch.isfinite(loss):
                            loss_components['keypoint_3d'] = loss
                            total_loss = total_loss + loss_weights['keypoint_3d'] * loss
                        else:
                            loss_components['keypoint_3d'] = torch.tensor(eps, device=self.device, requires_grad=True)
                    except Exception as e:
                        print(f"Warning: Failed to compute batch 3D keypoint loss: {e}")
                        loss_components['keypoint_3d'] = torch.tensor(eps, device=self.device, requires_grad=True)
                else:
                    # No samples with available 3D keypoints
                    loss_components['keypoint_3d'] = torch.tensor(eps, device=self.device, requires_grad=True)
            else:
                # No availability masks - legacy behavior (try all valid samples)
                try:
                    valid_joints_3d = joints_3d[sample_validity_mask]
                    valid_keypoint_data = [auxiliary_data['keypoint_data'][i] for i in range(len(auxiliary_data['keypoint_data'])) 
                                         if sample_validity_mask[i]]
                    
                    loss = self._compute_batch_keypoint_3d_loss(valid_joints_3d, valid_keypoint_data)
                    if torch.isfinite(loss):
                        loss_components['keypoint_3d'] = loss
                        total_loss = total_loss + loss_weights['keypoint_3d'] * loss
                    else:
                        loss_components['keypoint_3d'] = torch.tensor(eps, device=self.device, requires_grad=True)
                except Exception as e:
                    print(f"Warning: Failed to compute batch 3D keypoint loss: {e}")
                    loss_components['keypoint_3d'] = torch.tensor(eps, device=self.device, requires_grad=True)
        else:
            loss_components['keypoint_3d'] = torch.tensor(eps, device=self.device, requires_grad=True)
        
        # Final safety check for total loss
        if not torch.isfinite(total_loss):
            print(f"Warning: Non-finite total loss detected: {total_loss.item()}, replacing with small epsilon")
            total_loss = torch.tensor(1e-6, device=self.device, requires_grad=True)
        
        # Diagnostic: Log zero or near-zero loss batches
        if total_loss.item() < 1e-6:
            print(f"\nWARNING: Near-zero total loss detected: {total_loss.item():.2e}")
            print(f"  Valid samples: {num_valid_samples}/{batch_size}")
            
            # Show dataset composition if available
            if auxiliary_data and 'dataset_sources' in auxiliary_data:
                from collections import Counter
                sources = Counter(auxiliary_data['dataset_sources'])
                print(f"  Dataset composition: {dict(sources)}")
            
            print(f"  Availability masks present: {'_availability_masks' in target_params_batch}")
            if '_availability_masks' in target_params_batch:
                avail_masks = target_params_batch['_availability_masks']
                print(f"  Availability summary:")
                for key, mask in avail_masks.items():
                    if isinstance(mask, torch.Tensor):
                        num_avail = mask.sum().item()
                        print(f"    {key}: {num_avail}/{len(mask)} samples have real data")
            
            print(f"  Loss weights: keypoint_2d={loss_weights.get('keypoint_2d', 0)}, betas={loss_weights.get('betas', 0)}")
            print(f"  need_keypoint_loss={need_keypoint_loss}, need_silhouette_loss={need_silhouette_loss}")
            
            print(f"  Loss components (non-zero only):")
            for key, val in loss_components.items():
                if isinstance(val, torch.Tensor) and val.item() > 1e-8:
                    print(f"    {key}: {val.item():.6f} (weight={loss_weights.get(key, 0)})")
        
        if return_components:
            return total_loss, loss_components
        return total_loss

    def compute_prediction_loss(self, predicted_params: Dict[str, torch.Tensor], 
                               target_params: Dict[str, torch.Tensor], pose_data=None, silhouette_data=None, return_components=False,
                               loss_weights: Dict[str, float] = None) -> torch.Tensor:
        """
        Compute loss between predicted and target SMIL parameters.
        
        Args:
            predicted_params: Dictionary containing predicted parameters
            target_params: Dictionary containing target parameters
            pose_data: Optional dictionary containing 2D keypoint data and visibility for keypoint loss computation
            silhouette_data: Optional tensor containing target silhouette mask for silhouette loss computation
            return_components: If True, return both total loss and individual components
            
        Returns:
            Total loss as a scalar tensor, or (total_loss, loss_components) if return_components=True
        """
        total_loss = 0.0
        loss_components = {}
        
        if loss_weights is None:
            # Define loss weights for different parameter types
            loss_weights = {
                'global_rot': 0.02,
                'joint_rot': 0.02,  # Joint rotations are typically smaller values
                'betas': 0.01,     # Shape parameters need higher weight
                'trans': 0.001,
                'fov': 0.001,     # FOV is typically a large (constant) value (degrees)
                'cam_rot': 0.01,    # Camera rotation
                'cam_trans': 0.001, # Camera translation
                'log_beta_scales': 0.1,
                'betas_trans': 0.1,
                'keypoint_2d': 0.0,  # 2D keypoint loss weight (higher since normalized coordinates are small)
                'keypoint_3d': 0.0,  # 3D keypoint loss weight
                'silhouette': 0.0     # Silhouette loss weight - disabled due to gradient instability
            }
        
        # Check if we need to compute rendered outputs (keypoints and/or silhouette)
        need_keypoint_loss = (pose_data is not None and 'keypoints_2d' in pose_data and 
                             'keypoint_visibility' in pose_data and loss_weights['keypoint_2d'] > 0)
        need_silhouette_loss = (silhouette_data is not None and loss_weights['silhouette'] > 0)
        need_keypoint_3d_loss = (pose_data is not None and 'keypoints_3d' in pose_data and 
                                'keypoint_visibility' in pose_data and loss_weights['keypoint_3d'] > 0)
        
        # Single rendering pass for 2D keypoints, silhouette, and 3D keypoints if needed
        rendered_joints = None
        rendered_silhouette = None
        joints_3d = None
        if need_keypoint_loss or need_silhouette_loss or need_keypoint_3d_loss:
            try:
                rendered_joints, rendered_silhouette, joints_3d = self._compute_rendered_outputs(
                    predicted_params, compute_joints=need_keypoint_loss, 
                    compute_silhouette=need_silhouette_loss, compute_joints_3d=need_keypoint_3d_loss
                )
            except Exception as e:
                print(f"Warning: Failed to compute rendered outputs: {e}")
                if hasattr(self, '_debug_shapes'):
                    import traceback
                    traceback.print_exc()
        
        # Global rotation loss
        if 'global_rot' in target_params:
            if self.rotation_representation == '6d':
                # Convert 6D representations to rotation matrices for comparison
                pred_global_matrix = rotation_6d_to_matrix(predicted_params['global_rot'])
                target_global_matrix = rotation_6d_to_matrix(target_params['global_rot'])
                
                # Compute Frobenius norm of the difference
                matrix_diff_loss = torch.norm(pred_global_matrix - target_global_matrix, p='fro', dim=(-2, -1))
                loss = matrix_diff_loss.mean()  # Mean over batch
            else:
                # Use MSE for axis-angle representation
                loss = F.mse_loss(predicted_params['global_rot'], target_params['global_rot'])
            
            loss_components['global_rot'] = loss
            total_loss += loss_weights['global_rot'] * loss
        
        # Joint rotation loss (with visibility awareness)
        if 'joint_rot' in target_params:
            # Check if we have visibility information for joint-specific loss
            if pose_data is not None and 'keypoint_visibility' in pose_data:
                # Use visibility-aware joint rotation loss
                loss = self._compute_visibility_aware_joint_rotation_loss_single(
                    predicted_params['joint_rot'], target_params['joint_rot'], pose_data['keypoint_visibility']
                )
            else:
                # Fallback to standard joint rotation loss if no visibility data
                if self.rotation_representation == '6d':
                    # Convert 6D representations to rotation matrices for comparison
                    pred_matrices = rotation_6d_to_matrix(predicted_params['joint_rot'])
                    target_matrices = rotation_6d_to_matrix(target_params['joint_rot'])
                    
                    # Compute Frobenius norm of the difference
                    matrix_diff_loss = torch.norm(pred_matrices - target_matrices, p='fro', dim=(-2, -1))
                    loss = matrix_diff_loss.mean()  # Mean over batch and joints
                else:
                    # Use MSE for axis-angle representation
                    loss = F.mse_loss(predicted_params['joint_rot'], target_params['joint_rot'])
            
            loss_components['joint_rot'] = loss
            total_loss += loss_weights['joint_rot'] * loss
        
        # Shape parameter loss
        if 'betas' in target_params:
            loss = F.mse_loss(predicted_params['betas'], target_params['betas'])
            loss_components['betas'] = loss
            total_loss += loss_weights['betas'] * loss
        
        # Translation loss
        if 'trans' in target_params:
            loss = F.mse_loss(predicted_params['trans'], target_params['trans'])
            loss_components['trans'] = loss
            total_loss += loss_weights['trans'] * loss
        
        # FOV loss
        if 'fov' in target_params:
            # Ensure both tensors have the same shape
            pred_fov = predicted_params['fov']
            target_fov = target_params['fov']
            
            # Handle shape mismatch
            if pred_fov.shape != target_fov.shape:
                if pred_fov.shape[0] == 1 and target_fov.shape[0] == 1:
                    # Both are batch size 1, but different dimensions
                    if len(pred_fov.shape) > len(target_fov.shape):
                        target_fov = target_fov.unsqueeze(-1)
                    elif len(target_fov.shape) > len(pred_fov.shape):
                        pred_fov = pred_fov.unsqueeze(-1)
            
            loss = F.mse_loss(pred_fov, target_fov)
            loss_components['fov'] = loss
            total_loss += loss_weights['fov'] * loss
        
        # Camera rotation loss
        if 'cam_rot' in target_params:
            loss = F.mse_loss(predicted_params['cam_rot'], target_params['cam_rot'])
            loss_components['cam_rot'] = loss
            total_loss += loss_weights['cam_rot'] * loss
        
        # Camera translation loss
        if 'cam_trans' in target_params:
            loss = F.mse_loss(predicted_params['cam_trans'], target_params['cam_trans'])
            loss_components['cam_trans'] = loss
            total_loss += loss_weights['cam_trans'] * loss
        
        # Joint scales loss (if available)
        if 'log_beta_scales' in target_params and 'log_beta_scales' in predicted_params:
            loss = F.mse_loss(predicted_params['log_beta_scales'], target_params['log_beta_scales'])
            loss_components['log_beta_scales'] = loss
            total_loss += loss_weights['log_beta_scales'] * loss
        
        # Joint translations loss (if available)
        if 'betas_trans' in target_params and 'betas_trans' in predicted_params:
            loss = F.mse_loss(predicted_params['betas_trans'], target_params['betas_trans'])
            loss_components['betas_trans'] = loss
            total_loss += loss_weights['betas_trans'] * loss
        
        # 2D keypoint loss (if available and computed)
        if need_keypoint_loss and rendered_joints is not None:
            try:
                # Get target keypoints and visibility
                target_keypoints = safe_to_tensor(pose_data['keypoints_2d'], device=self.device)
                visibility = safe_to_tensor(pose_data['keypoint_visibility'], device=self.device)
                
                # Ensure batch dimensions match
                batch_size = predicted_params['global_rot'].shape[0]
                
                # Debug: Check rendered_joints shape to understand the issue
                if hasattr(self, '_debug_shapes') and torch.rand(1).item() < 0.01:
                    print(f"DEBUG - Initial shapes: rendered_joints={rendered_joints.shape}, target_keypoints={target_keypoints.shape}, visibility={visibility.shape}")
                    print(f"DEBUG - Expected batch_size={batch_size}")
                
                # Handle target_keypoints dimensions safely
                if target_keypoints.dim() == 2:  # Shape (n_joints, 2)
                    # Only expand if batch size is different than 1
                    if batch_size > 1:
                        target_keypoints = target_keypoints.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, n_joints, 2)
                    else:
                        target_keypoints = target_keypoints.unsqueeze(0)  # (1, n_joints, 2)
                elif target_keypoints.dim() == 3 and target_keypoints.shape[0] != batch_size:
                    # If already 3D but wrong batch size, take first sample and expand
                    target_keypoints = target_keypoints[0:1].expand(batch_size, -1, -1)
                
                # Handle visibility dimensions safely  
                if visibility.dim() == 1:  # Shape (n_joints,)
                    if batch_size > 1:
                        visibility = visibility.unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_joints)
                    else:
                        visibility = visibility.unsqueeze(0)  # (1, n_joints)
                elif visibility.dim() == 2 and visibility.shape[0] != batch_size:
                    # If already 2D but wrong batch size, take first sample and expand
                    visibility = visibility[0:1].expand(batch_size, -1)
                
                # Final safety check: ensure rendered_joints and target_keypoints have compatible shapes
                if rendered_joints.shape[0] != target_keypoints.shape[0]:
                    print(f"Warning: Batch size mismatch in keypoint loss - rendered: {rendered_joints.shape}, target: {target_keypoints.shape}")
                    # Use minimum batch size to avoid errors
                    min_batch = min(rendered_joints.shape[0], target_keypoints.shape[0])
                    rendered_joints = rendered_joints[:min_batch]
                    target_keypoints = target_keypoints[:min_batch]
                    visibility = visibility[:min_batch]
                
                # Debug: Print shapes to understand the issue (only occasionally)
                if hasattr(self, '_debug_shapes') and torch.rand(1).item() < 0.01:  # Only 1% of the time
                    print(f"DEBUG - rendered_joints shape: {rendered_joints.shape}")
                    print(f"DEBUG - target_keypoints shape: {target_keypoints.shape}")
                    print(f"DEBUG - visibility shape: {visibility.shape}")
                    print(f"DEBUG - rendered_joints range: [{rendered_joints.min():.3f}, {rendered_joints.max():.3f}]")
                    print(f"DEBUG - target_keypoints range: [{target_keypoints.min():.3f}, {target_keypoints.max():.3f}]")
                
                # Apply visibility mask - only compute loss for visible joints
                visible_mask = visibility.bool()
                
                # Check if ground truth keypoints are within reasonable image bounds [0, 1]
                # Only compute loss for keypoints where ground truth is within bounds
                gt_in_bounds_mask = (target_keypoints >= 0.0) & (target_keypoints <= 1.0)
                gt_in_bounds_mask = gt_in_bounds_mask.all(dim=-1)  # Both x and y must be in bounds

                # Track non-finite predictions before sanitising so we can report them
                non_finite_prediction_mask = ~torch.isfinite(rendered_joints).all(dim=-1)
                if non_finite_prediction_mask.any():
                    print(
                        "[KeypointLoss] Detected non-finite projected joints; "
                        f"sanitising {non_finite_prediction_mask.sum().item()} entries"
                    )
                
                # Replace NaNs/Infs with zeros to keep gradients flowing
                rendered_joints = torch.nan_to_num(rendered_joints, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Only keep joints that are visible AND have ground truth within bounds
                valid_mask = visible_mask & gt_in_bounds_mask
                
                # Debug: print validation info occasionally
                if hasattr(self, '_debug_shapes') and torch.rand(1).item() < 0.01:  # 1% of the time
                    gt_in_bounds_count = gt_in_bounds_mask.sum().item()
                    print(f"DEBUG - Keypoint loss: visible={visible_mask.sum().item()}, gt_in_bounds={gt_in_bounds_count}, valid={valid_mask.sum().item()}")
                    print(f"  Rendered joints range: [{rendered_joints.min():.3f}, {rendered_joints.max():.3f}]")
                    print(f"  Target keypoints range: [{target_keypoints.min():.3f}, {target_keypoints.max():.3f}]")
                
                if valid_mask.any():
                    # Use masking that preserves gradients - multiply by mask weights instead of indexing
                    # Convert boolean mask to float weights (1.0 for valid, 0.0 for invalid)
                    joint_weights = valid_mask.float().unsqueeze(-1)  # Shape: (batch_size, n_joints, 1)
                    
                    # Compute weighted MSE loss that preserves gradients
                    # Only valid joints contribute to loss (invalid ones get weight 0)
                    diff_squared = (rendered_joints - target_keypoints) ** 2
                    weighted_diff = diff_squared * joint_weights
                    
                    # Average over all valid joints with numerical stability
                    num_valid = valid_mask.sum().float()
                    eps = 1e-8  # Small epsilon for numerical stability
                    
                    if num_valid > 0:
                        # Compute loss only for valid joints, preserving gradients
                        # Sum over all joints (weighted by visibility) and divide by number of valid joints
                        loss = weighted_diff.sum() / (num_valid * 2 + eps)  # Divide by 2 for x,y coordinates
                        # Add small epsilon to prevent very small losses that can cause numerical issues
                        loss = loss + eps
                    else:
                        # No valid joints - return small epsilon with gradients
                        loss = torch.tensor(eps, device=self.device, requires_grad=True)
                    
                    # Check for NaN/inf in loss before proceeding
                    if torch.isfinite(loss):
                        loss_components['keypoint_2d'] = loss
                        total_loss += loss_weights['keypoint_2d'] * loss
                    else:
                        print(f"Warning: Non-finite keypoint loss detected: {loss.item()}, skipping")
                        loss_components['keypoint_2d'] = torch.tensor(1e-8, device=self.device, requires_grad=True)
                    
                    if hasattr(self, '_debug_shapes') and torch.rand(1).item() < 0.01:  # Only 1% of the time
                        print(f"DEBUG - Valid joints: {valid_mask.sum().item()}/{valid_mask.numel()}")
                        print(f"DEBUG - Loss value: {loss.item():.6f}")
                else:
                    # No valid joints, set loss to small epsilon but still add it to components
                    eps = 1e-8
                    loss = torch.tensor(eps, device=self.device, requires_grad=True)
                    loss_components['keypoint_2d'] = loss
                    print(
                        "[KeypointLoss] No valid 2D joints remaining for supervision "
                        f"(visible={visible_mask.sum().item()}, "
                        f"in_bounds={(gt_in_bounds_mask & visible_mask).sum().item()}, "
                        f"non_finite_preds={(non_finite_prediction_mask & visible_mask).sum().item()})"
                    )
                    
            except Exception as e:
                # If keypoint loss computation fails, continue without it
                print(f"Warning: Failed to compute keypoint loss: {e}")
                import traceback
                traceback.print_exc()
                loss_components['keypoint_2d'] = torch.tensor(1e-8, device=self.device, requires_grad=True)
        
        # Silhouette loss (if available and computed)
        if need_silhouette_loss and rendered_silhouette is not None:
            try:
                # Ensure target silhouette has correct format and device
                target_silhouette = safe_to_tensor(silhouette_data, device=self.device)
                
                # Ensure batch dimensions match
                batch_size = predicted_params['global_rot'].shape[0]
                
                # Debug: Check rendered_silhouette shape to understand the issue
                if hasattr(self, '_debug_shapes') and torch.rand(1).item() < 0.01:
                    print(f"DEBUG - Silhouette shapes: rendered={rendered_silhouette.shape}, target={target_silhouette.shape}")
                    print(f"DEBUG - Expected batch_size={batch_size}")
                
                if target_silhouette.dim() == 3:  # Shape (height, width, channels) or (channels, height, width)
                    if target_silhouette.shape[0] != batch_size:
                        # Assume it's (height, width, channels) - add batch dimension
                        if batch_size > 1:
                            target_silhouette = target_silhouette.unsqueeze(0).expand(batch_size, -1, -1, -1)
                        else:
                            target_silhouette = target_silhouette.unsqueeze(0)
                elif target_silhouette.dim() == 2:  # Shape (height, width)
                    # Add batch and channel dimensions
                    if batch_size > 1:
                        target_silhouette = target_silhouette.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
                    else:
                        target_silhouette = target_silhouette.unsqueeze(0).unsqueeze(0)
                
                # Final safety check: ensure rendered_silhouette and target_silhouette have compatible shapes
                if rendered_silhouette.shape[0] != target_silhouette.shape[0]:
                    print(f"Warning: Batch size mismatch in silhouette loss - rendered: {rendered_silhouette.shape}, target: {target_silhouette.shape}")
                    # Use minimum batch size to avoid errors
                    min_batch = min(rendered_silhouette.shape[0], target_silhouette.shape[0])
                    rendered_silhouette = rendered_silhouette[:min_batch]
                    target_silhouette = target_silhouette[:min_batch]
                
                # Ensure target silhouette has the same shape as rendered silhouette
                if target_silhouette.shape != rendered_silhouette.shape:
                    # If target has more than 1 channel (e.g., RGB), convert to single channel
                    if target_silhouette.shape[1] > 1:
                        # Take the mean across channels or the first channel
                        target_silhouette = target_silhouette.mean(dim=1, keepdim=True)
                    
                    # Resize if necessary
                    if target_silhouette.shape[-2:] != rendered_silhouette.shape[-2:]:
                        target_silhouette = F.interpolate(
                            target_silhouette, 
                            size=rendered_silhouette.shape[-2:], 
                            mode='bilinear', 
                            align_corners=False
                        )
                
                # Convert to binary mask if necessary (threshold at 0.5)
                if target_silhouette.max() > 1.0:
                    target_silhouette = target_silhouette / 255.0  # Normalize from [0, 255] to [0, 1]
                
                # Add small epsilon for numerical stability
                eps = 1e-8
                
                # Clamp values to prevent extreme gradients
                rendered_silhouette_clamped = torch.clamp(rendered_silhouette, 0.0, 1.0)
                target_silhouette_clamped = torch.clamp(target_silhouette, 0.0, 1.0)
                
                # Compute L1 loss (same as in SMALFitter)
                loss = F.l1_loss(rendered_silhouette_clamped, target_silhouette_clamped)
                
                # Add small epsilon to prevent numerical issues with very small losses
                loss = loss + eps
                
                # Check for NaN/inf in loss before proceeding
                if torch.isfinite(loss):
                    loss_components['silhouette'] = loss
                    total_loss += loss_weights['silhouette'] * loss
                else:
                    print(f"Warning: Non-finite silhouette loss detected: {loss.item()}, skipping")
                    loss_components['silhouette'] = torch.tensor(1e-8, device=self.device, requires_grad=True)
                
                # Debug output (occasionally)
                if hasattr(self, '_debug_shapes') and torch.rand(1).item() < 0.01:  # Only 1% of the time
                    print(f"DEBUG - Rendered silhouette shape: {rendered_silhouette.shape}")
                    print(f"DEBUG - Target silhouette shape: {target_silhouette.shape}")
                    print(f"DEBUG - Silhouette loss: {loss.item():.6f}")
                    print(f"DEBUG - Rendered range: [{rendered_silhouette.min():.3f}, {rendered_silhouette.max():.3f}]")
                    print(f"DEBUG - Target range: [{target_silhouette.min():.3f}, {target_silhouette.max():.3f}]")
                    
            except Exception as e:
                # If silhouette loss computation fails, continue without it
                print(f"Warning: Failed to compute silhouette loss: {e}")
                import traceback
                traceback.print_exc()
                loss_components['silhouette'] = torch.tensor(1e-8, device=self.device, requires_grad=True)
        
        # 3D keypoint loss (if available and computed)
        if need_keypoint_3d_loss and joints_3d is not None:
            try:
                # Get target 3D keypoints and visibility
                target_keypoints_3d = safe_to_tensor(pose_data['keypoints_3d'], device=self.device)
                visibility = safe_to_tensor(pose_data['keypoint_visibility'], device=self.device)
                
                # Ensure batch dimensions match
                batch_size = predicted_params['global_rot'].shape[0]
                
                # Handle target_keypoints_3d dimensions safely
                if target_keypoints_3d.dim() == 2:  # Shape (n_joints, 3)
                    # Only expand if batch size is different than 1
                    if batch_size > 1:
                        target_keypoints_3d = target_keypoints_3d.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, n_joints, 3)
                    else:
                        target_keypoints_3d = target_keypoints_3d.unsqueeze(0)  # (1, n_joints, 3)
                elif target_keypoints_3d.dim() == 3 and target_keypoints_3d.shape[0] != batch_size:
                    # If already 3D but wrong batch size, take first sample and expand
                    target_keypoints_3d = target_keypoints_3d[0:1].expand(batch_size, -1, -1)
                
                # Handle visibility dimensions safely  
                if visibility.dim() == 1:  # Shape (n_joints,)
                    if batch_size > 1:
                        visibility = visibility.unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_joints)
                    else:
                        visibility = visibility.unsqueeze(0)  # (1, n_joints)
                elif visibility.dim() == 2 and visibility.shape[0] != batch_size:
                    # If already 2D but wrong batch size, take first sample and expand
                    visibility = visibility[0:1].expand(batch_size, -1)
                
                # Final safety check: ensure joints_3d and target_keypoints_3d have compatible shapes
                if joints_3d.shape[0] != target_keypoints_3d.shape[0]:
                    print(f"Warning: Batch size mismatch in 3D keypoint loss - predicted: {joints_3d.shape}, target: {target_keypoints_3d.shape}")
                    # Use minimum batch size to avoid errors
                    min_batch = min(joints_3d.shape[0], target_keypoints_3d.shape[0])
                    joints_3d = joints_3d[:min_batch]
                    target_keypoints_3d = target_keypoints_3d[:min_batch]
                    visibility = visibility[:min_batch]
                
                # Apply visibility mask - only compute loss for visible joints
                visible_mask = visibility.bool()
                
                # Additional safety check for finite 3D joints (prevent NaN/inf in loss)
                finite_mask = torch.isfinite(joints_3d).all(dim=-1) & torch.isfinite(target_keypoints_3d).all(dim=-1)
                
                # Only keep joints that are visible AND finite
                valid_mask = visible_mask & finite_mask
                
                if valid_mask.any():
                    # Use masking that preserves gradients - multiply by mask weights instead of indexing
                    # Convert boolean mask to float weights (1.0 for valid, 0.0 for invalid)
                    joint_weights = valid_mask.float().unsqueeze(-1)  # Shape: (batch_size, n_joints, 1)
                    
                    # Compute weighted MSE loss that preserves gradients
                    # Only valid joints contribute to loss (invalid ones get weight 0)
                    diff_squared = (joints_3d - target_keypoints_3d) ** 2
                    weighted_diff = diff_squared * joint_weights
                    
                    # Average over all valid joints with numerical stability
                    num_valid = valid_mask.sum().float()
                    eps = 1e-8  # Small epsilon for numerical stability
                    
                    if num_valid > 0:
                        # Compute loss only for valid joints, preserving gradients
                        # Sum over all joints (weighted by visibility) and divide by number of valid joints
                        loss = weighted_diff.sum() / (num_valid * 3 + eps)  # Divide by 3 for x,y,z coordinates
                        # Add small epsilon to prevent very small losses that can cause numerical issues
                        loss = loss + eps
                    else:
                        # No valid joints - return small epsilon with gradients
                        loss = torch.tensor(eps, device=self.device, requires_grad=True)
                    
                    # Check for NaN/inf in loss before proceeding
                    if torch.isfinite(loss):
                        loss_components['keypoint_3d'] = loss
                        total_loss += loss_weights['keypoint_3d'] * loss
                    else:
                        print(f"Warning: Non-finite 3D keypoint loss detected: {loss.item()}, skipping")
                        loss_components['keypoint_3d'] = torch.tensor(1e-8, device=self.device, requires_grad=True)
                else:
                    # No valid joints, set loss to small epsilon but still add it to components
                    eps = 1e-8
                    loss = torch.tensor(eps, device=self.device, requires_grad=True)
                    loss_components['keypoint_3d'] = loss
                    
            except Exception as e:
                # If 3D keypoint loss computation fails, continue without it
                print(f"Warning: Failed to compute 3D keypoint loss: {e}")
                import traceback
                traceback.print_exc()
                loss_components['keypoint_3d'] = torch.tensor(1e-8, device=self.device, requires_grad=True)
        
        # Final safety check for total loss
        if not torch.isfinite(total_loss):
            print(f"Warning: Non-finite total loss detected: {total_loss.item()}, replacing with small epsilon")
            total_loss = torch.tensor(1e-6, device=self.device, requires_grad=True)
        
        if return_components:
            return total_loss, loss_components
        return total_loss
    
    def _compute_rendered_outputs(self, predicted_params: Dict[str, torch.Tensor], 
                                 compute_joints: bool = True, compute_silhouette: bool = True, 
                                 compute_joints_3d: bool = False) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute rendered outputs (joints and/or silhouette) from predicted SMIL parameters in a single efficient pass.
        
        Args:
            predicted_params: Dictionary containing predicted SMIL parameters
            compute_joints: Whether to compute and return rendered 2D joint positions
            compute_silhouette: Whether to compute and return rendered silhouette
            compute_joints_3d: Whether to compute and return 3D joint positions
            
        Returns:
            Tuple of (rendered_joints, rendered_silhouette, joints_3d)
            - rendered_joints: 2D joint positions as tensor of shape (batch_size, n_joints, 2) or None
            - rendered_silhouette: Silhouette tensor of shape (batch_size, 1, height, width) or None
            - joints_3d: 3D joint positions as tensor of shape (batch_size, n_joints, 3) or None
        """
        if not compute_joints and not compute_silhouette and not compute_joints_3d:
            return None, None, None
            
        # Create batch parameters for SMAL model
        batch_size = predicted_params['global_rot'].shape[0]
        
        # Convert rotations to axis-angle format for SMAL model (which expects axis-angle)
        if self.rotation_representation == '6d':
            # Check for NaN or infinite values in 6D rotations
            if not torch.isfinite(predicted_params['global_rot']).all():
                print(f"Warning: Non-finite values in global_rot: {predicted_params['global_rot']}")
                # Replace with identity rotation
                predicted_params['global_rot'] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], 
                                                             device=predicted_params['global_rot'].device, 
                                                             dtype=predicted_params['global_rot'].dtype).expand_as(predicted_params['global_rot'])
            
            if not torch.isfinite(predicted_params['joint_rot']).all():
                print(f"Warning: Non-finite values in joint_rot: {predicted_params['joint_rot']}")
                # Replace with identity rotations
                predicted_params['joint_rot'] = torch.zeros_like(predicted_params['joint_rot'])
            
            try:
                global_rot_aa = rotation_6d_to_axis_angle(predicted_params['global_rot'])
                joint_rot_aa = rotation_6d_to_axis_angle(predicted_params['joint_rot'])
                
                # Check for NaN in converted rotations
                if not torch.isfinite(global_rot_aa).all():
                    print(f"Warning: Non-finite values in converted global_rot_aa: {global_rot_aa}")
                    global_rot_aa = torch.zeros_like(global_rot_aa)
                
                if not torch.isfinite(joint_rot_aa).all():
                    print(f"Warning: Non-finite values in converted joint_rot_aa: {joint_rot_aa}")
                    joint_rot_aa = torch.zeros_like(joint_rot_aa)
                    
            except Exception as e:
                print(f"Error converting 6D rotations to axis-angle: {e}")
                # Fallback to zero rotations
                global_rot_aa = torch.zeros_like(predicted_params['global_rot'][:, :3])
                joint_rot_aa = torch.zeros_like(predicted_params['joint_rot'][:, :, :3])
        else:
            global_rot_aa = predicted_params['global_rot']
            joint_rot_aa = predicted_params['joint_rot']
        
        batch_params = {
            'global_rotation': global_rot_aa,
            'joint_rotations': joint_rot_aa,
            'betas': predicted_params['betas'],
            'trans': predicted_params['trans'],
            'fov': predicted_params['fov']
        }
        
        # Add joint scales and translations if available
        if 'log_beta_scales' in predicted_params:
            batch_params['log_betascale'] = predicted_params['log_beta_scales']
        if 'betas_trans' in predicted_params:
            batch_params['betas_trans'] = predicted_params['betas_trans']
        
        # Run SMAL model to get vertices and joints (single pass)
        # Use torch.no_grad() for intermediate computations when only 3D joints are needed
        if compute_joints_3d and not compute_joints and not compute_silhouette:
            # Optimization: if we only need 3D joints, we can skip some gradient computations
            with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for SMAL to prevent precision issues
                verts, joints, Rs, v_shaped = self.smal_model(
                    batch_params['betas'],
                    torch.cat([
                        batch_params['global_rotation'].unsqueeze(1),
                        batch_params['joint_rotations']], dim=1),
                    betas_logscale=batch_params.get('log_betascale', None),
                    betas_trans=batch_params.get('betas_trans', None),
                    propagate_scaling=self.propagate_scaling)
        else:
            # Standard computation for rendering
            verts, joints, Rs, v_shaped = self.smal_model(
                batch_params['betas'],
                torch.cat([
                    batch_params['global_rotation'].unsqueeze(1),
                    batch_params['joint_rotations']], dim=1),
                betas_logscale=batch_params.get('log_betascale', None),
                betas_trans=batch_params.get('betas_trans', None),
                propagate_scaling=self.propagate_scaling)
        
        # Apply transformation based on scaling configuration
        if self.use_ue_scaling:
            # Apply UE transform (10x scale) - needed to match replicAnt model size
            # This aligns the model at the root joint and scales it to the replicAnt model size
            verts = (verts - joints[:, 0, :].unsqueeze(1)) * 10 + batch_params['trans'].unsqueeze(1)
            joints = (joints - joints[:, 0, :].unsqueeze(1)) * 10 + batch_params['trans'].unsqueeze(1)
        elif self.allow_mesh_scaling and 'mesh_scale' in predicted_params:
            # Apply predicted mesh scale - centers at root, scales, then translates
            # mesh_scale is (batch_size, 1), expand for broadcasting
            mesh_scale = predicted_params['mesh_scale']  # (batch_size, 1)
            root_joint = joints[:, 0, :].unsqueeze(1)  # (batch_size, 1, 3)
            verts = (verts - root_joint) * mesh_scale.unsqueeze(-1) + batch_params['trans'].unsqueeze(1)
            joints = (joints - root_joint) * mesh_scale.unsqueeze(-1) + batch_params['trans'].unsqueeze(1)
        else:
            # Standard transformation without scaling
            verts = verts + batch_params['trans'].unsqueeze(1)
            joints = joints + batch_params['trans'].unsqueeze(1)
        
        # Get canonical model joints
        canonical_model_joints = joints[:, config.CANONICAL_MODEL_JOINTS]
        
        # Set camera parameters from predicted values
        self.renderer.set_camera_parameters(
            R=predicted_params['cam_rot'],
            T=predicted_params['cam_trans'],
            fov=predicted_params['fov']
        )
        
        # Single renderer call to get both silhouette and joints
        # Ensure faces tensor is correctly shaped for batch
        faces_tensor = self.smal_model.faces
        if faces_tensor.dim() == 2:
            # Add batch dimension and expand to match vertex batch size
            faces_batch = faces_tensor.unsqueeze(0).expand(verts.shape[0], -1, -1)
        else:
            faces_batch = faces_tensor
        
        # Debug: print tensor shapes occasionally
        if hasattr(self, '_debug_shapes') and torch.rand(1).item() < 0.01:
            print(f"DEBUG - Renderer input shapes: verts={verts.shape}, canonical_joints={canonical_model_joints.shape}, faces={faces_batch.shape}")
        
        rendered_silhouettes, rendered_joints_raw = self.renderer(
            verts, canonical_model_joints, faces_batch)
        
        
        # Process rendered joints if requested
        rendered_joints = None
        if compute_joints:
            # Normalize rendered joints to match ground truth coordinate system
            # PyTorch3D transform_points_screen outputs [x_pixel, y_pixel], but the renderer swaps to [y_pixel, x_pixel]
            # Ground truth format (from Unreal2Pytorch3D.py): [y_norm, x_norm] = [y/height, x/width] with range [0, 1]
            image_size = self.renderer.image_size
            
            # rendered_joints_raw is already in [y_pixel, x_pixel] format due to renderer swap
            # Normalize to [0, 1] range using the rendered image size (512x512)
            # This ensures consistency with the target keypoints which are normalized by the original image size (512x512)
            # The renderer always outputs 512x512 images, so we normalize by that size
            eps = 1e-8
            rendered_image_size = 512  # The renderer always outputs 512x512 images
            
            rendered_joints_final = rendered_joints_raw / (rendered_image_size + eps)
            
            # Clamp to reasonable range to prevent extreme values
            rendered_joints_final = torch.clamp(rendered_joints_final, -10.0, 10.0)
            
            # Debug: Check if normalization produces reasonable values (only occasionally)
            if hasattr(self, '_debug_shapes') and torch.rand(1).item() < 0.01:  # Only 1% of the time
                print(f"DEBUG - Raw rendered_joints range: [{rendered_joints_raw.min():.3f}, {rendered_joints_raw.max():.3f}]")
                print(f"DEBUG - Renderer image size: {image_size}")
                print(f"DEBUG - Normalization size: {rendered_image_size}")
                print(f"DEBUG - Final normalized range: [{rendered_joints_final.min():.3f}, {rendered_joints_final.max():.3f}]")
                print(f"DEBUG - Coordinate mapping: [y_pixel, x_pixel] -> [y_norm, x_norm] (normalized by {rendered_image_size}x{rendered_image_size})")
                
                # Check for out-of-bounds joints in pixel space
                in_bounds = (rendered_joints_raw >= 0) & (rendered_joints_raw <= image_size)
                in_bounds_count = in_bounds.all(dim=-1).sum().item()
                total_joints = rendered_joints_raw.shape[0] * rendered_joints_raw.shape[1]
                print(f"DEBUG - Joints in bounds: {in_bounds_count}/{total_joints}")
            
            rendered_joints = rendered_joints_final
        
        # Process rendered silhouette if requested
        rendered_silhouette = None
        if compute_silhouette:
            rendered_silhouette = rendered_silhouettes
        
        # Process 3D joints if requested
        joints_3d = None
        if compute_joints_3d:
            # Extract 3D joint positions (already transformed appropriately above)
            joints_3d = joints  # Shape: (batch_size, n_joints, 3)
        
        return rendered_joints, rendered_silhouette, joints_3d
    
    def _validate_sample_visibility(self, keypoint_data: dict, silhouette_data=None, 
                                   min_visible_keypoints: int = 5, min_pixel_coverage: float = 0.05,
                                   is_sleap_dataset: bool = False) -> bool:
        """
        Validate if a sample has sufficient visibility for training.
        
        This method checks if the sample has enough visible keypoints and/or sufficient
        pixel coverage in the silhouette to be useful for training. Samples that fail
        these checks should have all losses set to zero to prevent the model from
        being penalized for things it cannot possibly know.
        
        Args:
            keypoint_data: Dictionary containing 'keypoints_2d' and 'keypoint_visibility'
            silhouette_data: Optional silhouette mask tensor for pixel coverage calculation
            min_visible_keypoints: Minimum number of visible keypoints required (default: 5)
            min_pixel_coverage: Minimum pixel coverage percentage required (default: 0.05 = 5%)
            is_sleap_dataset: If True, use more lenient validation (SLEAP data is high quality)
            
        Returns:
            bool: True if sample is valid for training, False otherwise
        """
        if keypoint_data is None:
            return False
            
        # Check keypoint visibility
        visibility = safe_to_tensor(keypoint_data['keypoint_visibility'], device=self.device)
        target_keypoints = safe_to_tensor(keypoint_data['keypoints_2d'], device=self.device)
        
        # Apply the same visibility logic as in the existing keypoint loss computation
        visible_mask = visibility.bool()
        
        # Check if ground truth keypoints are within reasonable image bounds [0, 1]
        gt_in_bounds_mask = (target_keypoints >= 0.0) & (target_keypoints <= 1.0)
        gt_in_bounds_mask = gt_in_bounds_mask.all(dim=-1)  # Both x and y must be in bounds
        
        # Only keep joints that are visible AND have ground truth within bounds
        valid_keypoint_mask = visible_mask & gt_in_bounds_mask
        num_valid_keypoints = valid_keypoint_mask.sum().item()
        
        # For SLEAP datasets, use more lenient threshold (2 keypoints minimum)
        # SLEAP data is high quality but may have legitimate occlusions
        if is_sleap_dataset:
            min_sleap_keypoints = 2  # Very lenient - just need some keypoints
            return num_valid_keypoints >= min_sleap_keypoints
        
        # Check keypoint threshold for non-SLEAP data
        keypoint_valid = num_valid_keypoints >= min_visible_keypoints
        
        # Check pixel coverage if silhouette data is available (only for non-SLEAP datasets)
        pixel_coverage_valid = True  # Default to True if no silhouette data
        if silhouette_data is not None:
            try:
                target_silhouette = safe_to_tensor(silhouette_data, device=self.device)
                
                # Ensure proper format for pixel counting
                if target_silhouette.dim() == 2:  # Shape (height, width)
                    silhouette_2d = target_silhouette
                elif target_silhouette.dim() == 3:  # Shape (height, width, channels) or (channels, height, width)
                    if target_silhouette.shape[0] > target_silhouette.shape[-1]:
                        # Likely (channels, height, width) - take first channel
                        silhouette_2d = target_silhouette[0]
                    else:
                        # Likely (height, width, channels) - take first channel
                        silhouette_2d = target_silhouette[:, :, 0]
                else:
                    # Unexpected format, assume invalid
                    pixel_coverage_valid = False
                
                if pixel_coverage_valid:
                    # Calculate pixel coverage percentage
                    total_pixels = silhouette_2d.numel()
                    animal_pixels = (silhouette_2d > 0.5).sum().item()  # Pixels with value > 0.5
                    coverage_percentage = animal_pixels / total_pixels if total_pixels > 0 else 0.0
                    pixel_coverage_valid = coverage_percentage >= min_pixel_coverage
                    
            except Exception as e:
                # If silhouette processing fails, assume invalid
                pixel_coverage_valid = False
        
        # Sample is valid if it passes both keypoint and pixel coverage checks
        return keypoint_valid and pixel_coverage_valid

    def get_filtering_statistics(self) -> dict:
        """
        Get statistics about sample filtering during training.
        
        Returns:
            Dictionary containing filtering statistics
        """
        if not hasattr(self, '_filtering_stats'):
            return {
                'total_batches': 0,
                'total_samples': 0,
                'filtered_samples': 0,
                'batches_with_filtering': 0,
                'filtering_rate': 0.0,
                'batch_filtering_rate': 0.0
            }
        
        stats = self._filtering_stats.copy()
        if stats['total_samples'] > 0:
            stats['filtering_rate'] = stats['filtered_samples'] / stats['total_samples']
        if stats['total_batches'] > 0:
            stats['batch_filtering_rate'] = stats['batches_with_filtering'] / stats['total_batches']
        
        return stats

    def reset_filtering_statistics(self):
        """Reset the filtering statistics."""
        if hasattr(self, '_filtering_stats'):
            self._filtering_stats = {
                'total_batches': 0,
                'total_samples': 0,
                'filtered_samples': 0,
                'batches_with_filtering': 0
            }

    def _apply_availability_mask(self, predicted: torch.Tensor, target: torch.Tensor, 
                                 availability_mask: Optional[torch.Tensor],
                                 sample_validity_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Apply availability mask to filter out samples without real ground truth data.
        
        This implements complete masking (Option A) - samples without real data
        contribute nothing to the loss.
        
        Args:
            predicted: Predicted parameter tensor
            target: Target parameter tensor  
            availability_mask: Boolean mask indicating which samples have real data (or None if all available)
            sample_validity_mask: Boolean mask indicating which samples are valid
            
        Returns:
            Tuple of (masked_predicted, masked_target, num_available_samples)
        """
        if availability_mask is None:
            # No availability information - assume all samples are available
            return predicted, target, predicted.shape[0]
        
        # Apply sample validity mask to availability mask
        # Only consider samples that are both valid AND have real data
        combined_mask = availability_mask[sample_validity_mask]
        
        if not combined_mask.any():
            # No samples have real data - return empty tensors
            return predicted[:0], target[:0], 0
        
        # Filter to only include samples with real data
        filtered_pred = predicted[combined_mask]
        filtered_target = target[combined_mask]
        num_available = combined_mask.sum().item()
        
        return filtered_pred, filtered_target, num_available
    
    def _compute_batch_keypoint_loss(self, rendered_joints: torch.Tensor, keypoint_data_list: list) -> torch.Tensor:
        """
        Compute batched 2D keypoint loss efficiently.
        
        Args:
            rendered_joints: Tensor of shape (batch_size, n_joints, 2)
            keypoint_data_list: List of keypoint data dictionaries for each sample
            
        Returns:
            Average keypoint loss across valid samples in the batch
        """
        batch_size = rendered_joints.shape[0]
        total_loss = 0.0
        valid_samples = 0
        eps = 1e-8
        
        for i, keypoint_data in enumerate(keypoint_data_list):
            if keypoint_data is None or i >= batch_size:
                continue
                
            # Get target keypoints and visibility for this sample
            target_keypoints = safe_to_tensor(keypoint_data['keypoints_2d'], device=self.device)
            visibility = safe_to_tensor(keypoint_data['keypoint_visibility'], device=self.device)
            
            # Ensure proper shapes
            if target_keypoints.dim() == 2:  # Shape (n_joints, 2)
                target_keypoints = target_keypoints  # Keep as is for this sample
            if visibility.dim() == 1:  # Shape (n_joints,)
                visibility = visibility  # Keep as is for this sample
            
            # Apply visibility mask and bounds checking
            visible_mask = visibility.bool()
            gt_in_bounds_mask = (target_keypoints >= 0.0) & (target_keypoints <= 1.0)
            gt_in_bounds_mask = gt_in_bounds_mask.all(dim=-1)

            non_finite_prediction_mask = ~torch.isfinite(rendered_joints[i]).all(dim=-1)
            if non_finite_prediction_mask.any():
                print(
                    "[KeypointLoss] (batch) Detected non-finite projected joints; "
                    f"sanitising {non_finite_prediction_mask.sum().item()} entries"
                )
            rendered_joints_sample = torch.nan_to_num(
                rendered_joints[i], nan=0.0, posinf=0.0, neginf=0.0
            )

            valid_mask = visible_mask & gt_in_bounds_mask
            
            if valid_mask.any():
                # Compute loss for this sample using masking
                joint_weights = valid_mask.float().unsqueeze(-1)  # Shape: (n_joints, 1)
                diff_squared = (rendered_joints_sample - target_keypoints) ** 2
                weighted_diff = diff_squared * joint_weights
                
                num_valid = valid_mask.sum().float()
                if num_valid > 0:
                    sample_loss = weighted_diff.sum() / (num_valid * 2 + eps)
                    total_loss += sample_loss
                    valid_samples += 1
            else:
                print(
                    "[KeypointLoss] (batch) No valid 2D joints remaining for supervision "
                    f"(visible={visible_mask.sum().item()}, "
                    f"in_bounds={(gt_in_bounds_mask & visible_mask).sum().item()}, "
                    f"non_finite_preds={(non_finite_prediction_mask & visible_mask).sum().item()})"
                )
        
        if valid_samples > 0:
            return total_loss / valid_samples + eps
        else:
            return torch.tensor(eps, device=self.device, requires_grad=True)
    
    def _compute_visibility_aware_joint_rotation_loss_single(self, predicted_joint_rot, target_joint_rot, visibility):
        """
        Compute joint rotation loss with visibility awareness for single sample.
        
        Args:
            predicted_joint_rot: Predicted joint rotations (batch_size, n_joints, rot_dim)
            target_joint_rot: Target joint rotations (batch_size, n_joints, rot_dim)
            visibility: Joint visibility (batch_size, n_joints) or (n_joints,) - can be numpy array or tensor
            
        Returns:
            Visibility-aware joint rotation loss
        """
        # Convert visibility to tensor if it's a numpy array
        visibility = safe_to_tensor(visibility, device=self.device)
        
        # Ensure visibility has correct dimensions
        if visibility.dim() == 1:
            # Single sample case - expand to batch dimension
            batch_size = predicted_joint_rot.shape[0]
            visibility = visibility.unsqueeze(0).expand(batch_size, -1)
        
        # Exclude root joint from visibility (first element) since joint rotations exclude root
        # Root joint rotation is handled separately as global rotation
        if visibility.shape[1] > 1:
            visibility = visibility[:, 1:]  # Remove root joint (first element) for all samples in batch
        else:
            print(f"Warning: Visibility array too small - expected at least 2 joints, got {visibility.shape[1]}")
            return torch.tensor(1e-8, device=self.device, requires_grad=True)
        
        # Check dimension compatibility between joint rotations and visibility
        n_joints_pred = predicted_joint_rot.shape[1]  # Number of joints in predictions
        n_joints_vis = visibility.shape[1]  # Number of joints in visibility (after removing root)
        
        if n_joints_pred != n_joints_vis:
            print(f"Warning: Joint count mismatch in single sample - predicted: {n_joints_pred}, visibility: {n_joints_vis}")
            # Return small epsilon with gradients if dimensions don't match
            return torch.tensor(1e-8, device=self.device, requires_grad=True)
        
        # Convert visibility to boolean mask
        visible_mask = visibility.bool()
        
        # Handle different rotation representations
        if self.rotation_representation == '6d':
            # Convert 6D representations to rotation matrices for comparison
            pred_matrices = rotation_6d_to_matrix(predicted_joint_rot)
            target_matrices = rotation_6d_to_matrix(target_joint_rot)
            
            # Compute Frobenius norm of the difference for each joint
            matrix_diff_loss = torch.norm(pred_matrices - target_matrices, p='fro', dim=(-2, -1))
        else:
            # Use MSE for axis-angle representation
            matrix_diff_loss = torch.sum((predicted_joint_rot - target_joint_rot) ** 2, dim=-1)
        
        # Apply visibility mask - only compute loss for visible joints
        joint_weights = visible_mask.float()  # Shape: (batch_size, n_joints)
        weighted_loss = matrix_diff_loss * joint_weights
        
        # Average over visible joints only
        num_visible = visible_mask.sum().float()
        eps = 1e-8
        
        if num_visible > 0:
            loss = weighted_loss.sum() / (num_visible + eps) + eps
        else:
            # No visible joints - return small epsilon with gradients
            loss = torch.tensor(eps, device=self.device, requires_grad=True)
        
        return loss
    
    def _compute_visibility_aware_joint_rotation_loss_batch(self, predicted_joint_rot, target_joint_rot, keypoint_data_list):
        """
        Compute batched visibility-aware joint rotation loss.
        
        Args:
            predicted_joint_rot: Predicted joint rotations (batch_size, n_joints, rot_dim)
            target_joint_rot: Target joint rotations (batch_size, n_joints, rot_dim)
            keypoint_data_list: List of keypoint data dictionaries for each sample
            
        Returns:
            Average visibility-aware joint rotation loss across valid samples
        """
        batch_size = predicted_joint_rot.shape[0]
        total_loss = 0.0
        valid_samples = 0
        eps = 1e-8
        
        for i, keypoint_data in enumerate(keypoint_data_list):
            if keypoint_data is None or i >= batch_size:
                continue
            
            # Get visibility for this sample
            visibility = safe_to_tensor(keypoint_data['keypoint_visibility'], device=self.device)
            
            # Ensure proper shape
            if visibility.dim() == 1:
                visibility = visibility  # Keep as is for this sample
            
            # Exclude root joint from visibility (first element) since joint rotations exclude root
            # Root joint rotation is handled separately as global rotation
            if visibility.shape[0] > 1:
                visibility = visibility[1:]  # Remove root joint (first element)
            else:
                print(f"Warning: Visibility array too small - expected at least 2 joints, got {visibility.shape[0]}")
                continue
            
            # Check dimension compatibility between joint rotations and visibility
            n_joints_pred = predicted_joint_rot.shape[1]  # Number of joints in predictions
            n_joints_vis = visibility.shape[0]  # Number of joints in visibility (after removing root)
            
            if n_joints_pred != n_joints_vis:
                print(f"Warning: Joint count mismatch - predicted: {n_joints_pred}, visibility: {n_joints_vis}")
                # Skip this sample if dimensions don't match
                continue
            
            # Convert visibility to boolean mask
            visible_mask = visibility.bool()
            
            # Handle different rotation representations
            if self.rotation_representation == '6d':
                # Convert 6D representations to rotation matrices for comparison
                pred_matrices = rotation_6d_to_matrix(predicted_joint_rot[i:i+1])
                target_matrices = rotation_6d_to_matrix(target_joint_rot[i:i+1])
                
                # Compute Frobenius norm of the difference for each joint
                matrix_diff_loss = torch.norm(pred_matrices - target_matrices, p='fro', dim=(-2, -1))
            else:
                # Use MSE for axis-angle representation
                matrix_diff_loss = torch.sum((predicted_joint_rot[i:i+1] - target_joint_rot[i:i+1]) ** 2, dim=-1)
            
            # Apply visibility mask - only compute loss for visible joints
            joint_weights = visible_mask.float()  # Shape: (n_joints,)
            weighted_loss = matrix_diff_loss.squeeze(0) * joint_weights  # Remove batch dim for multiplication
            
            # Average over visible joints only
            num_visible = visible_mask.sum().float()
            
            if num_visible > 0:
                sample_loss = weighted_loss.sum() / (num_visible + eps)
                total_loss += sample_loss
                valid_samples += 1
        
        if valid_samples > 0:
            return total_loss / valid_samples + eps
        else:
            return torch.tensor(eps, device=self.device, requires_grad=True)
    
    def _compute_batch_silhouette_loss(self, rendered_silhouette: torch.Tensor, silhouette_data_list: list) -> torch.Tensor:
        """
        Compute batched silhouette loss efficiently.
        
        Args:
            rendered_silhouette: Tensor of shape (batch_size, 1, height, width)
            silhouette_data_list: List of silhouette data for each sample
            
        Returns:
            Average silhouette loss across valid samples in the batch
        """
        batch_size = rendered_silhouette.shape[0]
        total_loss = 0.0
        valid_samples = 0
        eps = 1e-8
        
        for i, silhouette_data in enumerate(silhouette_data_list):
            if silhouette_data is None or i >= batch_size:
                continue
                
            # Get target silhouette for this sample
            target_silhouette = safe_to_tensor(silhouette_data, device=self.device)
            
            # Ensure proper format - add batch and channel dimensions if needed
            if target_silhouette.dim() == 2:  # Shape (height, width)
                target_silhouette = target_silhouette.unsqueeze(0).unsqueeze(0)  # (1, 1, height, width)
            elif target_silhouette.dim() == 3:  # Shape (height, width, channels) or (channels, height, width)
                if target_silhouette.shape[0] != 1:
                    # Assume it's (height, width, channels)
                    target_silhouette = target_silhouette.unsqueeze(0).permute(0, 3, 1, 2)
                else:
                    # Already has batch dimension
                    if target_silhouette.shape[1] > target_silhouette.shape[0]:
                        # Likely (1, height, width) - add channel dimension
                        target_silhouette = target_silhouette.unsqueeze(1)
            
            # Resize if necessary
            if target_silhouette.shape[-2:] != rendered_silhouette.shape[-2:]:
                target_silhouette = F.interpolate(
                    target_silhouette, 
                    size=rendered_silhouette.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Handle multi-channel targets
            if target_silhouette.shape[1] > 1:
                target_silhouette = target_silhouette.mean(dim=1, keepdim=True)
            
            # Normalize if necessary
            if target_silhouette.max() > 1.0:
                target_silhouette = target_silhouette / 255.0
            
            # Clamp values and compute L1 loss for this sample
            rendered_sample = torch.clamp(rendered_silhouette[i:i+1], 0.0, 1.0)
            target_sample = torch.clamp(target_silhouette, 0.0, 1.0)
            
            sample_loss = F.l1_loss(rendered_sample, target_sample)
            total_loss += sample_loss + eps
            valid_samples += 1
        
        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(eps, device=self.device, requires_grad=True)

    def _compute_batch_keypoint_3d_loss(self, joints_3d: torch.Tensor, keypoint_data_list: list) -> torch.Tensor:
        """
        Compute batched 3D keypoint loss efficiently with visibility awareness.
        
        Args:
            joints_3d: Predicted 3D joint positions tensor of shape (batch_size, n_joints, 3)
            keypoint_data_list: List of keypoint data dictionaries for each sample
            
        Returns:
            Average 3D keypoint loss across valid samples in the batch
        """
        batch_size = joints_3d.shape[0]
        total_loss = 0.0
        valid_samples = 0
        eps = 1e-8
        
        for i, keypoint_data in enumerate(keypoint_data_list):
            if keypoint_data is None or i >= batch_size:
                continue
            
            # Check if 3D keypoints are available in the keypoint data
            # Must check both that the key exists AND that the value is not None
            if 'keypoints_3d' not in keypoint_data or keypoint_data['keypoints_3d'] is None:
                continue
                
            # Get target 3D keypoints and visibility for this sample
            target_keypoints_3d = safe_to_tensor(keypoint_data['keypoints_3d'], device=self.device)
            visibility = safe_to_tensor(keypoint_data['keypoint_visibility'], device=self.device)
            
            # Ensure proper shapes
            if target_keypoints_3d.dim() == 2:  # Shape (n_joints, 3)
                target_keypoints_3d = target_keypoints_3d  # Keep as is for this sample
            if visibility.dim() == 1:  # Shape (n_joints,)
                visibility = visibility  # Keep as is for this sample
            
            # Apply visibility mask - only compute loss for visible joints
            visible_mask = visibility.bool()
            finite_mask = torch.isfinite(joints_3d[i]).all(dim=-1) & torch.isfinite(target_keypoints_3d).all(dim=-1)
            valid_mask = visible_mask & finite_mask
            
            if valid_mask.any():
                # Compute loss for this sample using masking (same approach as 2D keypoints)
                joint_weights = valid_mask.float().unsqueeze(-1)  # Shape: (n_joints, 1)
                
                # Compute 3D distance loss (MSE in 3D space)
                diff_squared = (joints_3d[i] - target_keypoints_3d) ** 2
                weighted_diff = diff_squared * joint_weights
                
                num_valid = valid_mask.sum().float()
                if num_valid > 0:
                    # Average over valid joints and 3D coordinates (x, y, z)
                    sample_loss = weighted_diff.sum() / (num_valid * 3 + eps)
                    total_loss += sample_loss
                    valid_samples += 1
        
        if valid_samples > 0:
            return total_loss / valid_samples + eps
        else:
            return torch.tensor(eps, device=self.device, requires_grad=True)
    
    def enable_debug(self, enable=True):
        """Enable debug output for keypoint loss computation."""
        if enable:
            self._debug_shapes = True
        else:
            if hasattr(self, '_debug_shapes'):
                delattr(self, '_debug_shapes')
    
    def get_trainable_parameters(self):
        """
        Get trainable parameters (excludes frozen backbone if freeze_backbone=True).
        
        Returns:
            List of trainable parameter groups
        """
        trainable_params = []
        
        # Add backbone parameters if not frozen
        if not self.freeze_backbone:
            backbone_params = self.backbone.get_trainable_parameters()
            if backbone_params:
                trainable_params.append({'params': backbone_params, 'lr': 1e-5})  # Lower LR for backbone
        
        # Add regression head parameters based on type
        if self.head_type == 'mlp':
            trainable_params.extend([
                {'params': self.fc1.parameters()},
                {'params': self.fc2.parameters()},
                {'params': self.fc3.parameters()},
                {'params': self.regressor.parameters()},
                {'params': self.ln1.parameters()},
                {'params': self.ln2.parameters()},
                {'params': self.ln3.parameters()},
            ])
        elif self.head_type == 'transformer_decoder':
            trainable_params.append({'params': self.transformer_head.parameters()})
        
        return trainable_params

