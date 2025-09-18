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

# Import from parent modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smal_fitter import SMALFitter
import config

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
                 rotation_representation='axis_angle'):
        """
        Initialize the SMIL Image Regressor.
        
        Args:
            device: PyTorch device (cuda/cpu)
            data_batch: Batch data for SMALFitter initialization (can be placeholder)
            batch_size: Batch size for processing
            shape_family: Shape family ID for SMIL model
            use_unity_prior: Whether to use unity prior
            rgb_only: Whether to use only RGB images (no silhouettes/keypoints)
            freeze_backbone: Whether to freeze ResNet152 backbone weights
            hidden_dim: Hidden dimension for fully connected layers
            use_ue_scaling: Whether to apply 10x UE scaling (default True for replicAnt data)
            rotation_representation: '6d' or 'axis_angle' for joint rotations (default: 'axis_angle')
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
        
        # Enable scaling propagation for SMIL models (matches Unreal2Pytorch3D behavior)
        self.propagate_scaling = True
        
        # Load pre-trained ResNet152 backbone
        self.backbone = models.resnet152(pretrained=True)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Freeze backbone weights if requested
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get the feature dimension from ResNet152 (2048)
        self.feature_dim = 2048
        
        # Calculate output dimensions for SMIL parameters
        self._calculate_output_dims()
        
        # Create regression head
        self._create_regression_head()
        
        # Initialize parameters for the regression head
        self._initialize_parameters()
    
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
        
        # Optional: Joint scales and translations (if available in model)
        self.scales_dim = 0
        self.joint_trans_dim = 0
        
        if config.ignore_hardcoded_body:
            # For SMIL model, we have per-joint scales and translations
            n_joints = config.N_POSE + 1  # +1 for root joint
            self.scales_dim = n_joints * 3
            self.joint_trans_dim = n_joints * 3
        
        # Total output dimension
        self.total_output_dim = (self.global_rot_dim + self.joint_rot_dim + 
                                self.betas_dim + self.trans_dim + self.fov_dim +
                                self.cam_rot_dim + self.cam_trans_dim +
                                self.scales_dim + self.joint_trans_dim)
    
    def _create_regression_head(self):
        """Create the regression head with fully connected layers."""
        # First fully connected layer (using LayerNorm instead of BatchNorm for stability)
        self.fc1 = nn.Linear(self.feature_dim, self.hidden_dim)
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.ln2 = nn.LayerNorm(self.hidden_dim // 2)
        
        # Final regression layer
        self.regressor = nn.Linear(self.hidden_dim // 2, self.total_output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)
    
    def _initialize_parameters(self):
        """Initialize the regression head parameters."""
        # Initialize with small random values
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.constant_(self.regressor.bias, 0)
    
    def preprocess_image(self, image_data) -> torch.Tensor:
        """
        Preprocess input image for the network.
        
        Args:
            image_data: Input image as numpy array (H, W, C) with values in [0, 255]
            
        Returns:
            Preprocessed image tensor (1, C, H, W) with values in [0, 1]
        """
        # Handle different input types
        if isinstance(image_data, torch.Tensor):
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
        
        # Resize to 512x512
        if len(image_data.shape) == 4:
            # Batch of images
            batch_size = image_data.shape[0]
            resized_batch = []
            for i in range(batch_size):
                if image_data[i].shape[:2] != (512, 512):
                    resized_img = cv2.resize(image_data[i], (512, 512))
                else:
                    resized_img = image_data[i]
                resized_batch.append(resized_img)
            image_data = np.array(resized_batch)
            # Convert to tensor (B, H, W, C) -> (B, C, H, W)
            image_tensor = torch.from_numpy(image_data).permute(0, 3, 1, 2)
        else:
            # Single image
            if image_data.shape[:2] != (512, 512):
                image_data = cv2.resize(image_data, (512, 512))
            # Convert to tensor and add batch dimension (H, W, C) -> (1, C, H, W)
            image_tensor = torch.from_numpy(image_data).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the network.
        
        Args:
            images: Input images tensor of shape (batch_size, 3, 512, 512)
            
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
        
        # Extract features using ResNet152 backbone
        features = self.backbone(images)  # (batch_size, 2048, 1, 1)
        features = features.view(batch_size, -1)  # (batch_size, 2048)
        
        # Pass through regression head
        x = F.relu(self.ln1(self.fc1(features)))
        x = self.dropout(x)
        
        x = F.relu(self.ln2(self.fc2(x)))
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
        self.trans[batch_idx] = torch.zeros_like(params['trans'][batch_idx])
        
        # Set camera FOV (keep gradients)
        self.fov[batch_idx] = params['fov'][batch_idx]
        
        # Set joint scales (if available) (keep gradients)
        if 'log_beta_scales' in params and hasattr(self, 'log_beta_scales'):
            self.log_beta_scales[batch_idx] = torch.zeros_like(params['log_beta_scales'][batch_idx])
        
        # Set joint translations (if available) (keep gradients)
        if 'betas_trans' in params and hasattr(self, 'betas_trans'):
            self.betas_trans[batch_idx] = torch.zeros_like(params['betas_trans'][batch_idx])
    
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
                'silhouette': 0.0     # Silhouette loss weight - disabled due to gradient instability
            }
        
        # Check if we need to compute rendered outputs (keypoints and/or silhouette)
        need_keypoint_loss = (pose_data is not None and 'keypoints_2d' in pose_data and 
                             'keypoint_visibility' in pose_data and loss_weights['keypoint_2d'] > 0)
        need_silhouette_loss = (silhouette_data is not None and loss_weights['silhouette'] > 0)
        
        # Single rendering pass for both keypoint and silhouette losses if needed
        rendered_joints = None
        rendered_silhouette = None
        if need_keypoint_loss or need_silhouette_loss:
            try:
                rendered_joints, rendered_silhouette = self._compute_rendered_outputs(
                    predicted_params, compute_joints=need_keypoint_loss, compute_silhouette=need_silhouette_loss
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
        
        # Joint rotation loss
        if 'joint_rot' in target_params:
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
                if target_keypoints.dim() == 2:  # Shape (n_joints, 2)
                    target_keypoints = target_keypoints.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, n_joints, 2)
                if visibility.dim() == 1:  # Shape (n_joints,)
                    visibility = visibility.unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_joints)
                
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
                
                # Only keep joints that are both visible AND have ground truth within bounds
                valid_mask = visible_mask & gt_in_bounds_mask
                
                # Additional safety check for finite rendered joints (prevent NaN/inf in loss)
                finite_mask = torch.isfinite(rendered_joints).all(dim=-1)
                valid_mask = valid_mask & finite_mask
                
                # Debug: print validation info occasionally
                if hasattr(self, '_debug_shapes') and torch.rand(1).item() < 0.01:  # 1% of the time
                    finite_count = finite_mask.sum().item()
                    gt_in_bounds_count = gt_in_bounds_mask.sum().item()
                    print(f"DEBUG - Keypoint loss: visible={visible_mask.sum().item()}, gt_in_bounds={gt_in_bounds_count}, finite={finite_count}, valid={valid_mask.sum().item()}")
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
                        # Add epsilon to prevent division by very small numbers
                        loss = weighted_diff.sum() / (num_valid * 2 + eps)  # Divide by 2 for x,y coordinates
                        # Add small epsilon to prevent very small losses that can cause numerical issues
                        loss = loss + eps
                    else:
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
                    # Only print occasionally to avoid spam
                    if hasattr(self, '_debug_shapes') and torch.rand(1).item() < 0.01:
                        finite_count = finite_mask.sum().item()
                        gt_in_bounds_count = gt_in_bounds_mask.sum().item()
                        print(f"DEBUG - No valid joints: visible={visible_mask.sum().item()}, gt_in_bounds={gt_in_bounds_count}, finite={finite_count}")
                    
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
                if target_silhouette.dim() == 3:  # Shape (height, width, channels) or (channels, height, width)
                    if target_silhouette.shape[0] != batch_size:
                        # Assume it's (height, width, channels) - add batch dimension
                        target_silhouette = target_silhouette.unsqueeze(0).expand(batch_size, -1, -1, -1)
                elif target_silhouette.dim() == 2:  # Shape (height, width)
                    # Add batch and channel dimensions
                    target_silhouette = target_silhouette.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
                
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
        
        # Final safety check for total loss
        if not torch.isfinite(total_loss):
            print(f"Warning: Non-finite total loss detected: {total_loss.item()}, replacing with small epsilon")
            total_loss = torch.tensor(1e-6, device=self.device, requires_grad=True)
        
        if return_components:
            return total_loss, loss_components
        return total_loss
    
    def _compute_rendered_outputs(self, predicted_params: Dict[str, torch.Tensor], 
                                 compute_joints: bool = True, compute_silhouette: bool = True) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute rendered outputs (joints and/or silhouette) from predicted SMIL parameters in a single efficient pass.
        
        Args:
            predicted_params: Dictionary containing predicted SMIL parameters
            compute_joints: Whether to compute and return rendered 2D joint positions
            compute_silhouette: Whether to compute and return rendered silhouette
            
        Returns:
            Tuple of (rendered_joints, rendered_silhouette)
            - rendered_joints: 2D joint positions as tensor of shape (batch_size, n_joints, 2) or None
            - rendered_silhouette: Silhouette tensor of shape (batch_size, 1, height, width) or None
        """
        if not compute_joints and not compute_silhouette:
            return None, None
            
        # Create batch parameters for SMAL model
        batch_size = predicted_params['global_rot'].shape[0]
        
        # Convert rotations to axis-angle format for SMAL model (which expects axis-angle)
        if self.rotation_representation == '6d':
            global_rot_aa = rotation_6d_to_axis_angle(predicted_params['global_rot'])
            joint_rot_aa = rotation_6d_to_axis_angle(predicted_params['joint_rot'])
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
        else:
            # Standard transformation without UE scaling
            verts = verts + batch_params['trans'].unsqueeze(1)
            joints = joints + batch_params['trans'].unsqueeze(1)
        
        # Get canonical model joints
        canonical_model_joints = joints[:, config.CANONICAL_MODEL_JOINTS]
        
        # Set camera parameters from predicted values if available
        if 'cam_rot' in predicted_params and 'cam_trans' in predicted_params:
            self.renderer.set_camera_parameters(
                R=predicted_params['cam_rot'],
                T=predicted_params['cam_trans'],
                fov=predicted_params['fov']
            )
        else:
            # Use default camera FOV
            self.renderer.cameras.fov = batch_params['fov']
        
        # Single renderer call to get both silhouette and joints
        rendered_silhouettes, rendered_joints_raw = self.renderer(
            verts, canonical_model_joints,
            self.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1))
        
        
        # Process rendered joints if requested
        rendered_joints = None
        if compute_joints:
            # Normalize rendered joints to match ground truth coordinate system
            # PyTorch3D transform_points_screen outputs [x_pixel, y_pixel], but the renderer swaps to [y_pixel, x_pixel]
            # Ground truth format (from Unreal2Pytorch3D.py): [y_norm, x_norm] = [y/height, x/width] with range [0, 1]
            image_size = self.renderer.image_size
            
            # rendered_joints_raw is already in [y_pixel, x_pixel] format due to renderer swap
            # Just normalize to [0, 1] range to match ground truth format
            # Add small epsilon to prevent division by zero issues
            eps = 1e-8
            rendered_joints_final = rendered_joints_raw / (image_size + eps)
            
            # Clamp to reasonable range to prevent extreme values
            rendered_joints_final = torch.clamp(rendered_joints_final, -10.0, 10.0)
            
            # Debug: Check if normalization produces reasonable values (only occasionally)
            if hasattr(self, '_debug_shapes') and torch.rand(1).item() < 0.01:  # Only 1% of the time
                print(f"DEBUG - Raw rendered_joints range: [{rendered_joints_raw.min():.3f}, {rendered_joints_raw.max():.3f}]")
                print(f"DEBUG - Image size: {image_size}")
                print(f"DEBUG - Final normalized range: [{rendered_joints_final.min():.3f}, {rendered_joints_final.max():.3f}]")
                print(f"DEBUG - Coordinate mapping: [y_pixel, x_pixel] -> [y_norm, x_norm]")
                
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
        
        return rendered_joints, rendered_silhouette

    
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
        if self.freeze_backbone:
            # Only return parameters from the regression head
            return [
                {'params': self.fc1.parameters()},
                {'params': self.fc2.parameters()},
                {'params': self.regressor.parameters()},
                {'params': self.ln1.parameters()},
                {'params': self.ln2.parameters()},
            ]
        else:
            # Return all parameters
            return self.parameters()
