"""
Multi-View SMIL Image Regressor

A neural network that learns to predict SMIL parameters from multiple synchronized camera views.
Uses cross-attention between views for feature fusion and separate camera heads per canonical view.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any

# Import from parent modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smil_image_regressor import SMILImageRegressor, safe_to_tensor, rotation_6d_to_axis_angle
import config


class CrossViewAttention(nn.Module):
    """
    Cross-attention module for multi-view feature fusion.
    
    Each view attends to all other views to aggregate information,
    enabling the model to understand spatial relationships between perspectives.
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize cross-view attention.
        
        Args:
            feature_dim: Dimension of input features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Layer norm and dropout
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        
        # FFN for post-attention processing
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor, view_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply cross-view attention.
        
        Args:
            features: Feature tensor of shape (batch_size, num_views, feature_dim)
            view_mask: Optional boolean mask of shape (batch_size, num_views)
                      True = valid view, False = masked view
                      
        Returns:
            Attended features of shape (batch_size, num_views, feature_dim)
        """
        batch_size, num_views, _ = features.shape
        
        # Pre-norm
        normed = self.norm1(features)
        
        # Compute Q, K, V
        Q = self.q_proj(normed)  # (B, V, D)
        K = self.k_proj(normed)  # (B, V, D)
        V = self.v_proj(normed)  # (B, V, D)
        
        # Reshape for multi-head attention: (B, V, H, D_h) -> (B, H, V, D_h)
        Q = Q.view(batch_size, num_views, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_views, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_views, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, V, V)
        
        # Apply mask if provided
        if view_mask is not None:
            # Create attention mask: (B, 1, 1, V) for broadcasting
            attn_mask = view_mask.unsqueeze(1).unsqueeze(2).float()  # (B, 1, 1, V)
            attn_mask = attn_mask.masked_fill(~view_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_scores = attn_scores + attn_mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, V)  # (B, H, V, D_h)
        
        # Reshape back: (B, H, V, D_h) -> (B, V, H, D_h) -> (B, V, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_views, -1)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Residual connection
        features = features + attn_output
        
        # FFN with residual
        features = features + self.ffn(self.norm2(features))
        
        return features


class MultiViewFeatureFusion(nn.Module):
    """
    Feature fusion module that combines multiple view features using cross-attention.
    """
    
    def __init__(self, feature_dim: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize multi-view feature fusion.
        
        Args:
            feature_dim: Dimension of input features
            num_layers: Number of cross-attention layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            CrossViewAttention(feature_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, features: torch.Tensor, view_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply cross-attention fusion across views.
        
        Args:
            features: Feature tensor of shape (batch_size, num_views, feature_dim)
            view_mask: Optional boolean mask of shape (batch_size, num_views)
            
        Returns:
            Fused features of shape (batch_size, num_views, feature_dim)
        """
        for layer in self.layers:
            features = layer(features, view_mask)
        
        return self.final_norm(features)


class CameraHead(nn.Module):
    """
    Camera parameter prediction head for a single canonical view.
    
    Outputs camera parameters with sensible constraints:
    - FOV: Clamped to [10, 60] degrees (typical range for perspective cameras)
    - Rotation: 6D representation for continuous optimization, converted to 3x3 matrix
    - Translation: Scaled to reasonable range
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 default_fov: float = 30.0, fov_range: Tuple[float, float] = (10.0, 60.0),
                 trans_scale: float = 5.0):
        """
        Initialize camera head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            default_fov: Default FOV in degrees
            fov_range: (min, max) FOV in degrees
            trans_scale: Scale factor for translation output
        """
        super().__init__()
        
        self.default_fov = default_fov
        self.fov_min, self.fov_max = fov_range
        self.trans_scale = trans_scale
        
        # Camera parameters: FOV delta (1) + rotation 6D (6) + translation (3) = 10
        self.fov_dim = 1
        self.cam_rot_dim = 6  # 6D rotation representation for continuous optimization
        self.cam_trans_dim = 3
        self.total_cam_dim = self.fov_dim + self.cam_rot_dim + self.cam_trans_dim
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, self.total_cam_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with sensible defaults."""
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)
        
        # Initialize last layer with small weights for stable training
        last_linear = self.layers[-1]
        nn.init.xavier_uniform_(last_linear.weight, gain=0.01)
        nn.init.zeros_(last_linear.bias)
    
    def _rotation_6d_to_matrix(self, rotation_6d: torch.Tensor) -> torch.Tensor:
        """
        Convert 6D rotation representation to 3x3 rotation matrix.
        
        The 6D representation consists of the first two columns of the rotation matrix.
        The third column is computed via cross product.
        
        Args:
            rotation_6d: (..., 6) tensor
            
        Returns:
            (..., 3, 3) rotation matrix
        """
        a1 = rotation_6d[..., :3]  # First column
        a2 = rotation_6d[..., 3:6]  # Second column
        
        # Gram-Schmidt orthogonalization
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        
        return torch.stack([b1, b2, b3], dim=-1)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict camera parameters from features.
        
        Args:
            features: Feature tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary with 'fov', 'cam_rot', 'cam_trans'
        """
        output = self.layers(features)
        
        idx = 0
        
        # FOV: Apply sigmoid to get [0,1], then scale to [fov_min, fov_max]
        fov_raw = output[:, idx:idx + self.fov_dim]
        fov = self.fov_min + torch.sigmoid(fov_raw) * (self.fov_max - self.fov_min)
        idx += self.fov_dim
        
        # Rotation: 6D representation -> orthonormal 3x3 matrix
        rot_6d = output[:, idx:idx + self.cam_rot_dim]
        # Initialize close to identity: add identity's first two columns
        # Identity 6D: [1,0,0, 0,1,0]
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
                                   device=output.device, dtype=output.dtype)
        rot_6d = rot_6d + identity_6d
        cam_rot = self._rotation_6d_to_matrix(rot_6d)
        idx += self.cam_rot_dim
        
        # Translation: Scale with tanh to bound range, then scale
        cam_trans_raw = output[:, idx:idx + self.cam_trans_dim]
        cam_trans = torch.tanh(cam_trans_raw) * self.trans_scale
        
        return {
            'fov': fov,
            'cam_rot': cam_rot,
            'cam_trans': cam_trans
        }


class MultiViewSMILImageRegressor(SMILImageRegressor):
    """
    Multi-view extension of SMILImageRegressor.
    
    Processes multiple synchronized camera views to predict:
    - Shared body parameters (shape, pose, translation) from cross-attention fused features
    - Per-view camera parameters from separate heads for each canonical view
    
    Key design decisions:
    - Cross-attention between views for feature fusion (not simple concatenation)
    - N separate camera heads, one per canonical view position
    - Single body parameter prediction from aggregated features
    - Per-view 2D keypoint loss with visibility weighting
    """
    
    def __init__(self, device, data_batch, batch_size, shape_family, use_unity_prior,
                 max_views: int = 4,
                 canonical_camera_order: Optional[List[str]] = None,
                 cross_attention_layers: int = 2,
                 cross_attention_heads: int = 8,
                 cross_attention_dropout: float = 0.1,
                 **kwargs):
        """
        Initialize the multi-view SMIL regressor.
        
        Args:
            device: PyTorch device
            data_batch: Batch data for initialization
            batch_size: Batch size
            shape_family: SMIL shape family
            use_unity_prior: Whether to use unity prior
            max_views: Maximum number of views to support
            canonical_camera_order: List of canonical camera names for consistent ordering
            cross_attention_layers: Number of cross-attention layers for feature fusion
            cross_attention_heads: Number of attention heads
            cross_attention_dropout: Dropout for attention layers
            **kwargs: Additional arguments passed to parent SMILImageRegressor
        """
        # Initialize parent single-view regressor
        super().__init__(device, data_batch, batch_size, shape_family, use_unity_prior, **kwargs)
        
        self.max_views = max_views
        self.canonical_camera_order = canonical_camera_order or []
        self.num_canonical_cameras = len(self.canonical_camera_order) if self.canonical_camera_order else max_views
        
        # Cross-attention feature fusion
        self.view_fusion = MultiViewFeatureFusion(
            feature_dim=self.feature_dim,
            num_layers=cross_attention_layers,
            num_heads=cross_attention_heads,
            dropout=cross_attention_dropout
        ).to(device)
        
        # Body parameter aggregation: pool fused features across views
        self.body_aggregator = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU()
        ).to(device)
        
        # Separate camera heads for each canonical view position
        self.camera_heads = nn.ModuleList([
            CameraHead(self.feature_dim, hidden_dim=256)
            for _ in range(self.num_canonical_cameras)
        ]).to(device)
        
        # View embedding to help identify which camera a view belongs to
        # This helps during inference when camera assignment might be ambiguous
        self.view_embeddings = nn.Embedding(self.num_canonical_cameras, self.feature_dim).to(device)
        nn.init.normal_(self.view_embeddings.weight, mean=0.0, std=0.02)
    
    def forward_multiview(self, images_per_view: List[torch.Tensor], 
                          camera_indices: torch.Tensor,
                          view_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Forward pass for multi-view input.
        
        Args:
            images_per_view: List of image tensors, one per view
                            Each tensor has shape (batch_size, 3, H, W)
            camera_indices: Tensor of canonical camera indices for each view
                           Shape: (batch_size, num_views) with values in [0, num_canonical_cameras)
            view_mask: Optional boolean mask indicating valid views
                      Shape: (batch_size, num_views), True = valid view
                      
        Returns:
            Dictionary containing:
                - Shared body parameters (global_rot, joint_rot, betas, trans, log_beta_scales, betas_trans)
                - Per-view camera parameters (fov, cam_rot, cam_trans) as lists
        """
        batch_size = images_per_view[0].size(0)
        num_views = len(images_per_view)
        
        # Extract features from each view using shared backbone
        view_features = []
        for view_idx, view_images in enumerate(images_per_view):
            features = self.backbone(view_images)  # (batch_size, feature_dim)
            view_features.append(features)
        
        # Stack into (batch_size, num_views, feature_dim)
        stacked_features = torch.stack(view_features, dim=1)
        
        # Add view embeddings based on camera indices
        # camera_indices: (batch_size, num_views)
        view_embeds = self.view_embeddings(camera_indices)  # (batch_size, num_views, feature_dim)
        stacked_features = stacked_features + view_embeds
        
        # Apply cross-view attention for feature fusion
        fused_features = self.view_fusion(stacked_features, view_mask)  # (batch_size, num_views, feature_dim)
        
        # Aggregate features for body parameter prediction
        # Use mean pooling over valid views
        if view_mask is not None:
            # Mask invalid views before pooling
            mask_expanded = view_mask.unsqueeze(-1).float()  # (batch_size, num_views, 1)
            masked_features = fused_features * mask_expanded
            aggregated_features = masked_features.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            aggregated_features = fused_features.mean(dim=1)  # (batch_size, feature_dim)
        
        # Apply body aggregator
        body_features = self.body_aggregator(aggregated_features)
        
        # Predict shared body parameters using parent's head
        body_params = self._predict_body_params(body_features, batch_size)
        
        # Predict per-view camera parameters
        per_view_cam_params = self._predict_camera_params_per_view(
            fused_features, camera_indices, view_mask, num_views
        )
        
        # Combine into output dictionary
        output = {**body_params}
        
        # Add per-view camera params
        output['fov_per_view'] = per_view_cam_params['fov']  # List of (batch_size, 1)
        output['cam_rot_per_view'] = per_view_cam_params['cam_rot']  # List of (batch_size, 3, 3)
        output['cam_trans_per_view'] = per_view_cam_params['cam_trans']  # List of (batch_size, 3)
        output['num_views'] = num_views
        output['view_mask'] = view_mask
        
        return output
    
    def _predict_body_params(self, features: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Predict shared body parameters from aggregated features.
        
        Uses the parent class's regression head for body parameters only.
        """
        if self.head_type == 'mlp':
            # Pass through regression head
            x = F.relu(self.ln1(self.fc1(features)))
            x = self.dropout(x)
            x = F.relu(self.ln2(self.fc2(x)))
            x = self.dropout(x)
            x = F.relu(self.ln3(self.fc3(x)))
            x = self.dropout(x)
            
            output = self.regressor(x)
            
            # Parse body parameters only (exclude camera params)
            params = {}
            idx = 0
            
            # Global rotation
            params['global_rot'] = output[:, idx:idx + self.global_rot_dim]
            idx += self.global_rot_dim
            
            # Joint rotations
            joint_rot_flat = output[:, idx:idx + self.joint_rot_dim]
            if self.rotation_representation == '6d':
                params['joint_rot'] = joint_rot_flat.view(batch_size, config.N_POSE, 6)
            else:
                params['joint_rot'] = joint_rot_flat.view(batch_size, config.N_POSE, 3)
            idx += self.joint_rot_dim
            
            # Shape parameters
            params['betas'] = output[:, idx:idx + self.betas_dim]
            idx += self.betas_dim
            
            # Translation
            params['trans'] = output[:, idx:idx + self.trans_dim]
            idx += self.trans_dim
            
            # Skip camera params (fov, cam_rot, cam_trans) - they come from per-view heads
            idx += self.fov_dim + self.cam_rot_dim + self.cam_trans_dim
            
            # Joint scales (if available)
            if self.scales_dim > 0:
                if self.scale_trans_mode == 'separate':
                    # In separate mode, these are PCA weights, keep as 1D
                    params['log_beta_scales'] = output[:, idx:idx + self.scales_dim]  # (batch_size, N_BETAS)
                else:
                    # In other modes, reshape to per-joint values
                    scales_flat = output[:, idx:idx + self.scales_dim]
                    n_joints = self.scales_dim // 3
                    params['log_beta_scales'] = scales_flat.view(batch_size, n_joints, 3)
                idx += self.scales_dim
            
            # Joint translations (if available)
            if self.joint_trans_dim > 0:
                if self.scale_trans_mode == 'separate':
                    # In separate mode, these are PCA weights, keep as 1D
                    params['betas_trans'] = output[:, idx:idx + self.joint_trans_dim]  # (batch_size, N_BETAS)
                else:
                    # In other modes, reshape to per-joint values
                    trans_flat = output[:, idx:idx + self.joint_trans_dim]
                    n_joints = self.joint_trans_dim // 3
                    params['betas_trans'] = trans_flat.view(batch_size, n_joints, 3)
            
            return params
            
        elif self.head_type == 'transformer_decoder':
            # Use transformer decoder for body params
            params = self.transformer_head(features, None)
            
            # Handle scale/trans mode
            if self.scale_trans_mode == 'entangled_with_betas':
                if 'betas' in params:
                    log_beta_scales, betas_trans = self._transform_betas_to_joint_values(params['betas'])
                    params['log_beta_scales'] = log_beta_scales
                    params['betas_trans'] = betas_trans
            
            return params
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")
    
    def _predict_camera_params_per_view(self, fused_features: torch.Tensor,
                                         camera_indices: torch.Tensor,
                                         view_mask: Optional[torch.Tensor],
                                         num_views: int) -> Dict[str, List[torch.Tensor]]:
        """
        Predict camera parameters for each view using the appropriate camera head.
        
        Args:
            fused_features: Fused features of shape (batch_size, num_views, feature_dim)
            camera_indices: Canonical camera index for each view (batch_size, num_views)
            view_mask: Boolean mask for valid views (batch_size, num_views)
            num_views: Number of views
            
        Returns:
            Dictionary with lists of per-view camera parameters
        """
        batch_size = fused_features.shape[0]
        
        fov_list = []
        cam_rot_list = []
        cam_trans_list = []
        
        for v in range(num_views):
            view_features = fused_features[:, v, :]  # (batch_size, feature_dim)
            cam_idx = camera_indices[:, v]  # (batch_size,)
            
            # Predict camera params per sample using appropriate head
            fov_batch = torch.zeros(batch_size, 1, device=self.device)
            cam_rot_batch = torch.zeros(batch_size, 3, 3, device=self.device)
            cam_trans_batch = torch.zeros(batch_size, 3, device=self.device)
            
            # Group samples by camera index for efficient processing
            for head_idx in range(self.num_canonical_cameras):
                mask = (cam_idx == head_idx)
                if mask.any():
                    head_features = view_features[mask]
                    head_output = self.camera_heads[head_idx](head_features)
                    
                    fov_batch[mask] = head_output['fov']
                    cam_rot_batch[mask] = head_output['cam_rot']
                    cam_trans_batch[mask] = head_output['cam_trans']
            
            fov_list.append(fov_batch)
            cam_rot_list.append(cam_rot_batch)
            cam_trans_list.append(cam_trans_batch)
        
        return {
            'fov': fov_list,
            'cam_rot': cam_rot_list,
            'cam_trans': cam_trans_list
        }
    
    def compute_multiview_batch_loss(self, predicted_params: Dict[str, Any],
                                      target_data: List[Dict[str, Any]],
                                      loss_weights: Optional[Dict[str, float]] = None,
                                      return_components: bool = False) -> torch.Tensor:
        """
        Compute multi-view loss.
        
        Body parameter losses are computed once using aggregated predictions.
        2D keypoint losses are computed separately for each view and averaged.
        
        Args:
            predicted_params: Dictionary containing predicted parameters from forward_multiview
            target_data: List of target data dictionaries (one per sample in batch)
            loss_weights: Dictionary of loss weights
            return_components: Whether to return individual loss components
            
        Returns:
            Total loss, or (total_loss, loss_components) if return_components=True
        """
        if loss_weights is None:
            loss_weights = {
                'global_rot': 0.02,
                'joint_rot': 0.02,
                'betas': 0.01,
                'trans': 0.001,
                'log_beta_scales': 0.1,
                'betas_trans': 0.1,
                'keypoint_2d': 1.0,
                'joint_angle_regularization': 0.0001,  # Penalty for large joint angles
            }
        
        batch_size = predicted_params['global_rot'].shape[0]
        num_views = predicted_params['num_views']
        view_mask = predicted_params.get('view_mask', None)
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_components = {}
        eps = 1e-8
        
        # =================== BODY PARAMETER LOSSES (computed once) ===================
        # These losses are for the shared body parameters predicted from fused features
        
        # Collect body parameter targets from first valid view (they should be same across views)
        body_targets = self._collect_body_targets_batch(target_data)
        
        # Global rotation loss
        if body_targets.get('global_rot') is not None and loss_weights.get('global_rot', 0) > 0:
            loss = self._compute_rotation_loss(
                predicted_params['global_rot'], 
                body_targets['global_rot'],
                body_targets.get('global_rot_mask')
            )
            loss_components['global_rot'] = loss
            total_loss = total_loss + loss_weights['global_rot'] * loss
        
        # Joint rotation loss
        if body_targets.get('joint_rot') is not None and loss_weights.get('joint_rot', 0) > 0:
            loss = self._compute_rotation_loss(
                predicted_params['joint_rot'].reshape(batch_size, -1),
                body_targets['joint_rot'].reshape(batch_size, -1),
                body_targets.get('joint_rot_mask')
            )
            loss_components['joint_rot'] = loss
            total_loss = total_loss + loss_weights['joint_rot'] * loss
        
        # Joint angle regularization: penalize deviations from default (0,0,0) angles
        # This helps prevent extreme joint angles and encourages natural poses
        if loss_weights.get('joint_angle_regularization', 0) > 0:
            joint_rot_pred = predicted_params['joint_rot']  # (batch_size, N_POSE, 6) or (batch_size, N_POSE, 3)
            
            # Convert to axis-angle if needed
            if self.rotation_representation == '6d':
                # Convert 6D to axis-angle
                joint_rot_aa = rotation_6d_to_axis_angle(joint_rot_pred)  # (batch_size, N_POSE, 3)
            else:
                # Already in axis-angle format
                joint_rot_aa = joint_rot_pred  # (batch_size, N_POSE, 3)
            
            # Compute L2 norm of joint angles (excluding root joint which is index 0)
            # joint_rot_aa is already excluding root (it's N_POSE joints, not N_POSE+1)
            # So we can use all joints, or if we want to be explicit, we can verify
            joint_angle_norms = torch.norm(joint_rot_aa, dim=-1)  # (batch_size, N_POSE)
            
            # Regularization: penalize large joint angles
            # Use L2 penalty (squared norm) to encourage small angles
            joint_angle_reg = torch.mean(joint_angle_norms ** 2)
            
            loss_components['joint_angle_regularization'] = joint_angle_reg
            total_loss = total_loss + loss_weights['joint_angle_regularization'] * joint_angle_reg
        
        # Betas loss
        if body_targets.get('betas') is not None and loss_weights.get('betas', 0) > 0:
            mask = body_targets.get('betas_mask')
            if mask is not None and mask.any():
                loss = F.mse_loss(
                    predicted_params['betas'][mask],
                    body_targets['betas'][mask]
                )
            else:
                loss = torch.tensor(eps, device=self.device, requires_grad=True)
            loss_components['betas'] = loss
            total_loss = total_loss + loss_weights['betas'] * loss
        
        # Translation loss
        if body_targets.get('trans') is not None and loss_weights.get('trans', 0) > 0:
            mask = body_targets.get('trans_mask')
            if mask is not None and mask.any():
                loss = F.mse_loss(
                    predicted_params['trans'][mask],
                    body_targets['trans'][mask]
                )
            else:
                loss = torch.tensor(eps, device=self.device, requires_grad=True)
            loss_components['trans'] = loss
            total_loss = total_loss + loss_weights['trans'] * loss
        
        # Scale/trans losses if available
        if 'log_beta_scales' in predicted_params and loss_weights.get('log_beta_scales', 0) > 0:
            if body_targets.get('log_beta_scales') is not None:
                loss = F.mse_loss(predicted_params['log_beta_scales'], body_targets['log_beta_scales'])
                loss_components['log_beta_scales'] = loss
                total_loss = total_loss + loss_weights['log_beta_scales'] * loss
        
        if 'betas_trans' in predicted_params and loss_weights.get('betas_trans', 0) > 0:
            if body_targets.get('betas_trans') is not None:
                loss = F.mse_loss(predicted_params['betas_trans'], body_targets['betas_trans'])
                loss_components['betas_trans'] = loss
                total_loss = total_loss + loss_weights['betas_trans'] * loss
        
        # =================== PER-VIEW 2D KEYPOINT LOSS ===================
        # This is the core multi-view loss: render 2D keypoints for each view using
        # shared body params + per-view camera params, compare to per-view GT keypoints
        
        if loss_weights.get('keypoint_2d', 0) > 0:
            keypoint_loss_sum = torch.tensor(0.0, device=self.device, requires_grad=True)
            valid_view_count = 0
            
            for v in range(num_views):
                # Get per-view camera parameters
                fov_v = predicted_params['fov_per_view'][v]
                cam_rot_v = predicted_params['cam_rot_per_view'][v]
                cam_trans_v = predicted_params['cam_trans_per_view'][v]
                
                # Render 2D keypoints using body params + this view's camera
                try:
                    rendered_joints = self._render_keypoints_with_camera(
                        predicted_params, fov_v, cam_rot_v, cam_trans_v
                    )
                except Exception as e:
                    print(f"Warning: Failed to render keypoints for view {v}: {e}")
                    continue
                
                # Collect target keypoints for this view
                view_kp_data = self._collect_view_keypoint_data(target_data, v, batch_size)
                
                if view_kp_data is None:
                    continue
                
                # Compute loss for this view with visibility weighting
                view_loss = self._compute_visibility_weighted_keypoint_loss(
                    rendered_joints, 
                    view_kp_data['keypoints_2d'],
                    view_kp_data['visibility'],
                    view_mask[:, v] if view_mask is not None else None
                )
                
                if torch.isfinite(view_loss):
                    keypoint_loss_sum = keypoint_loss_sum + view_loss
                    valid_view_count += 1
            
            if valid_view_count > 0:
                keypoint_loss = keypoint_loss_sum / valid_view_count
                loss_components['keypoint_2d'] = keypoint_loss
                total_loss = total_loss + loss_weights['keypoint_2d'] * keypoint_loss
            else:
                loss_components['keypoint_2d'] = torch.tensor(eps, device=self.device, requires_grad=True)
        
        if return_components:
            return total_loss, loss_components
        return total_loss
    
    def _collect_body_targets_batch(self, target_data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collect body parameter targets from batch."""
        batch_size = len(target_data)
        targets = {}
        
        # Global rotation
        global_rots = []
        global_rot_mask = []
        for td in target_data:
            if td.get('root_rot') is not None:
                global_rots.append(safe_to_tensor(td['root_rot'], device=self.device))
                global_rot_mask.append(True)
            else:
                global_rots.append(torch.zeros(3, device=self.device))
                global_rot_mask.append(False)
        targets['global_rot'] = torch.stack(global_rots)
        targets['global_rot_mask'] = torch.tensor(global_rot_mask, device=self.device)
        
        # Joint rotations
        joint_rots = []
        joint_rot_mask = []
        for td in target_data:
            if td.get('joint_angles') is not None:
                jr = safe_to_tensor(td['joint_angles'], device=self.device)
                joint_rots.append(jr[1:])  # Exclude root
                joint_rot_mask.append(True)
            else:
                joint_rots.append(torch.zeros(config.N_POSE, 3, device=self.device))
                joint_rot_mask.append(False)
        targets['joint_rot'] = torch.stack(joint_rots)
        targets['joint_rot_mask'] = torch.tensor(joint_rot_mask, device=self.device)
        
        # Betas
        betas = []
        betas_mask = []
        for td in target_data:
            if td.get('shape_betas') is not None:
                betas.append(safe_to_tensor(td['shape_betas'], device=self.device))
                betas_mask.append(True)
            else:
                betas.append(torch.zeros(config.N_BETAS, device=self.device))
                betas_mask.append(False)
        targets['betas'] = torch.stack(betas)
        targets['betas_mask'] = torch.tensor(betas_mask, device=self.device)
        
        # Translation
        trans = []
        trans_mask = []
        for td in target_data:
            if td.get('root_loc') is not None:
                trans.append(safe_to_tensor(td['root_loc'], device=self.device))
                trans_mask.append(True)
            else:
                trans.append(torch.zeros(3, device=self.device))
                trans_mask.append(False)
        targets['trans'] = torch.stack(trans)
        targets['trans_mask'] = torch.tensor(trans_mask, device=self.device)
        
        return targets
    
    def _collect_view_keypoint_data(self, target_data: List[Dict[str, Any]], 
                                    view_idx: int, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Collect keypoint data for a specific view across the batch."""
        keypoints = []
        visibility = []
        has_data = False
        
        for td in target_data:
            kp_2d = td.get('keypoints_2d')
            kp_vis = td.get('keypoint_visibility')
            
            if kp_2d is not None and len(kp_2d.shape) >= 2:
                # Check if this is multi-view data (num_views, n_joints, 2)
                if len(kp_2d.shape) == 3 and view_idx < kp_2d.shape[0]:
                    keypoints.append(safe_to_tensor(kp_2d[view_idx], device=self.device))
                    if kp_vis is not None and len(kp_vis.shape) == 2:
                        visibility.append(safe_to_tensor(kp_vis[view_idx], device=self.device))
                    else:
                        # Create all-visible mask
                        visibility.append(torch.ones(kp_2d.shape[1], device=self.device))
                    has_data = True
                elif len(kp_2d.shape) == 2 and view_idx == 0:
                    # Single-view data, only valid for view_idx 0
                    keypoints.append(safe_to_tensor(kp_2d, device=self.device))
                    if kp_vis is not None:
                        visibility.append(safe_to_tensor(kp_vis, device=self.device))
                    else:
                        visibility.append(torch.ones(kp_2d.shape[0], device=self.device))
                    has_data = True
                else:
                    # No data for this view
                    n_joints = kp_2d.shape[-2] if len(kp_2d.shape) >= 2 else len(config.CANONICAL_MODEL_JOINTS)
                    keypoints.append(torch.zeros(n_joints, 2, device=self.device))
                    visibility.append(torch.zeros(n_joints, device=self.device))
            else:
                # No keypoint data
                n_joints = len(config.CANONICAL_MODEL_JOINTS)
                keypoints.append(torch.zeros(n_joints, 2, device=self.device))
                visibility.append(torch.zeros(n_joints, device=self.device))
        
        if not has_data:
            return None
        
        return {
            'keypoints_2d': torch.stack(keypoints),  # (batch_size, n_joints, 2)
            'visibility': torch.stack(visibility)  # (batch_size, n_joints)
        }
    
    def _render_keypoints_with_camera(self, body_params: Dict[str, torch.Tensor],
                                       fov: torch.Tensor,
                                       cam_rot: torch.Tensor,
                                       cam_trans: torch.Tensor) -> torch.Tensor:
        """
        Render 2D keypoints using body parameters and specific camera parameters.
        
        Args:
            body_params: Dictionary with body parameters (global_rot, joint_rot, betas, trans)
            fov: Camera field of view (batch_size, 1)
            cam_rot: Camera rotation matrix (batch_size, 3, 3)
            cam_trans: Camera translation (batch_size, 3)
            
        Returns:
            Rendered 2D keypoints of shape (batch_size, n_joints, 2)
        """
        batch_size = body_params['global_rot'].shape[0]
        
        # Convert rotations to axis-angle format for SMAL model
        if self.rotation_representation == '6d':
            global_rot_aa = rotation_6d_to_axis_angle(body_params['global_rot'])
            joint_rot_aa = rotation_6d_to_axis_angle(body_params['joint_rot'])
        else:
            global_rot_aa = body_params['global_rot']
            joint_rot_aa = body_params['joint_rot']
        
        # Ensure correct shapes for concatenation
        if global_rot_aa.dim() == 2:
            global_rot_aa = global_rot_aa.unsqueeze(1)  # (batch_size, 1, 3)
        if joint_rot_aa.dim() == 2:
            joint_rot_aa = joint_rot_aa.unsqueeze(1)  # Handle edge case
        
        # Build pose tensor: (batch_size, N_POSE+1, 3)
        pose_tensor = torch.cat([global_rot_aa, joint_rot_aa], dim=1)
        
        # Handle scale/trans parameters based on mode
        # In 'separate' mode, predicted values are PCA weights that would need transformation
        # However, for rendering we skip these as they require PCA components that may not be available
        # and their loss weights are typically 0 anyway. The SMAL model handles None gracefully.
        betas_logscale = None
        betas_trans_val = None
        
        if 'log_beta_scales' in body_params and body_params['log_beta_scales'] is not None:
            if self.scale_trans_mode == 'separate':
                # In separate mode, skip scale/trans for rendering (PCA transformation often fails
                # due to missing scaledirs/transdirs, and loss weights are typically 0)
                # Just pass None and let SMAL model use defaults
                pass
            elif self.scale_trans_mode != 'ignore':
                # In other modes (e.g., entangled_with_betas), values are already per-joint
                betas_logscale = body_params['log_beta_scales']
                betas_trans_val = body_params.get('betas_trans', None)
        
        # Run SMAL model
        verts, joints, Rs, v_shaped = self.smal_model(
            body_params['betas'],
            pose_tensor,
            betas_logscale=betas_logscale,
            betas_trans=betas_trans_val,
            propagate_scaling=self.propagate_scaling
        )
        
        # Apply transformation
        if self.use_ue_scaling:
            root_joint = joints[:, 0:1, :]  # (batch_size, 1, 3) - safer indexing
            verts = (verts - root_joint) * 10 + body_params['trans'].unsqueeze(1)
            joints = (joints - root_joint) * 10 + body_params['trans'].unsqueeze(1)
        else:
            verts = verts + body_params['trans'].unsqueeze(1)
            joints = joints + body_params['trans'].unsqueeze(1)
        
        # Get canonical joints - check bounds first
        max_joint_idx = max(config.CANONICAL_MODEL_JOINTS) if config.CANONICAL_MODEL_JOINTS else 0
        if joints.shape[1] <= max_joint_idx:
            print(f"Warning: joints has {joints.shape[1]} joints but CANONICAL_MODEL_JOINTS needs index {max_joint_idx}")
            print(f"  CANONICAL_MODEL_JOINTS: {config.CANONICAL_MODEL_JOINTS}")
            # Return zeros as fallback
            n_canonical = len(config.CANONICAL_MODEL_JOINTS)
            return torch.zeros(batch_size, n_canonical, 2, device=self.device)
        
        canonical_joints = joints[:, config.CANONICAL_MODEL_JOINTS]
        
        # Set camera parameters for rendering
        self.renderer.set_camera_parameters(R=cam_rot, T=cam_trans, fov=fov)
        
        # Render joints
        faces_tensor = self.smal_model.faces
        if faces_tensor.dim() == 2:
            faces_batch = faces_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            faces_batch = faces_tensor
        
        _, rendered_joints_raw = self.renderer(verts, canonical_joints, faces_batch)
        
        # Normalize rendered joints using the actual renderer image size
        # This ensures consistency between training loss and visualization
        rendered_image_size = self.renderer.image_size
        eps = 1e-8
        rendered_joints = rendered_joints_raw / (rendered_image_size + eps)
        rendered_joints = torch.clamp(rendered_joints, 0.0, 1.0)
        
        return rendered_joints
    
    def _compute_visibility_weighted_keypoint_loss(self, 
                                                    rendered_joints: torch.Tensor,
                                                    target_keypoints: torch.Tensor,
                                                    visibility: torch.Tensor,
                                                    sample_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute visibility-weighted 2D keypoint loss.
        
        Args:
            rendered_joints: Rendered 2D joints (batch_size, n_joints, 2)
            target_keypoints: Target 2D keypoints (batch_size, n_joints, 2)
            visibility: Joint visibility (batch_size, n_joints)
            sample_mask: Optional mask for valid samples (batch_size,)
            
        Returns:
            Visibility-weighted keypoint loss
        """
        eps = 1e-8
        
        # Apply sample mask if provided
        if sample_mask is not None:
            if not sample_mask.any():
                return torch.tensor(eps, device=self.device, requires_grad=True)
            rendered_joints = rendered_joints[sample_mask]
            target_keypoints = target_keypoints[sample_mask]
            visibility = visibility[sample_mask]
        
        batch_size = rendered_joints.shape[0]
        
        # Sanitize rendered joints
        rendered_joints = torch.nan_to_num(rendered_joints, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create valid mask: visible and in bounds
        visible_mask = visibility.bool()  # (batch_size, n_joints)
        gt_in_bounds = (target_keypoints >= 0.0) & (target_keypoints <= 1.0)
        gt_in_bounds = gt_in_bounds.all(dim=-1)  # (batch_size, n_joints)
        
        valid_mask = visible_mask & gt_in_bounds
        
        if not valid_mask.any():
            return torch.tensor(eps, device=self.device, requires_grad=True)
        
        # Compute weighted squared error
        diff_squared = (rendered_joints - target_keypoints) ** 2
        
        # Weight by visibility
        weights = valid_mask.float().unsqueeze(-1)  # (batch_size, n_joints, 1)
        weighted_diff = diff_squared * weights
        
        # Sum over joints and coordinates, average over valid joints per sample
        num_valid_per_sample = valid_mask.sum(dim=1).float() * 2 + eps  # *2 for x,y
        sample_losses = weighted_diff.sum(dim=(1, 2)) / num_valid_per_sample
        
        # Average over batch
        return sample_losses.mean() + eps
    
    def _compute_rotation_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute rotation loss with optional masking."""
        eps = 1e-8
        
        if mask is not None and mask.any():
            pred = pred[mask]
            target = target[mask]
        elif mask is not None and not mask.any():
            return torch.tensor(eps, device=self.device, requires_grad=True)
        
        if self.rotation_representation == '6d':
            from pytorch3d.transforms import rotation_6d_to_matrix
            # Reshape for conversion if needed
            orig_shape = pred.shape
            if len(orig_shape) > 2:
                pred_flat = pred.reshape(-1, 6)
                target_flat = target.reshape(-1, 6)
            else:
                pred_flat = pred
                target_flat = target
            
            pred_matrices = rotation_6d_to_matrix(pred_flat)
            target_matrices = rotation_6d_to_matrix(target_flat)
            loss = torch.norm(pred_matrices - target_matrices, p='fro', dim=(-2, -1)).mean()
        else:
            loss = F.mse_loss(pred, target)
        
        return loss + eps
    
    def predict_from_multiview_batch(self, x_data_batch: List[Dict], 
                                      y_data_batch: List[Dict]) -> Tuple[Dict, List[Dict], Dict]:
        """
        Process a multi-view batch for training.
        
        Args:
            x_data_batch: List of x_data dictionaries (one per sample)
            y_data_batch: List of y_data dictionaries (one per sample)
            
        Returns:
            Tuple of (predicted_params, y_data_batch, auxiliary_data)
        """
        batch_size = len(x_data_batch)
        
        # Collect images and camera info from all samples
        all_images_per_view = []
        all_camera_indices = []
        all_view_masks = []
        max_views_in_batch = 0
        
        # First pass: find max views and collect data
        for x_data in x_data_batch:
            num_views = x_data.get('num_active_views', 1)
            max_views_in_batch = max(max_views_in_batch, num_views)
        
        # Initialize storage for each view position
        for v in range(max_views_in_batch):
            all_images_per_view.append([])
        
        # Second pass: organize images by view position
        for sample_idx, x_data in enumerate(x_data_batch):
            images = x_data.get('images', [])
            cam_indices = x_data.get('camera_indices', [])
            num_views = len(images)
            
            sample_view_mask = []
            sample_cam_indices = []
            
            for v in range(max_views_in_batch):
                if v < num_views:
                    # Preprocess image
                    img = images[v]
                    img_tensor = self.preprocess_image(img).squeeze(0)  # Remove batch dim
                    # Ensure tensor is on the correct device
                    img_tensor = img_tensor.to(self.device)
                    all_images_per_view[v].append(img_tensor)
                    
                    # Get camera index
                    if v < len(cam_indices):
                        sample_cam_indices.append(int(cam_indices[v]))
                    else:
                        sample_cam_indices.append(v)  # Default to view index
                    
                    sample_view_mask.append(True)
                else:
                    # Pad with zeros
                    dummy_img = torch.zeros(3, 224, 224, device=self.device)
                    all_images_per_view[v].append(dummy_img)
                    sample_cam_indices.append(0)
                    sample_view_mask.append(False)
            
            all_camera_indices.append(sample_cam_indices)
            all_view_masks.append(sample_view_mask)
        
        # Stack into tensors
        images_per_view = [
            torch.stack(all_images_per_view[v]).to(self.device)
            for v in range(max_views_in_batch)
        ]
        camera_indices = torch.tensor(all_camera_indices, device=self.device)
        view_mask = torch.tensor(all_view_masks, device=self.device)
        
        # Forward pass
        predicted_params = self.forward_multiview(images_per_view, camera_indices, view_mask)
        
        # Build auxiliary data
        auxiliary_data = {
            'is_multiview': True,
            'num_views': max_views_in_batch,
            'view_mask': view_mask,
        }
        
        return predicted_params, y_data_batch, auxiliary_data


def create_multiview_regressor(device, batch_size, shape_family, use_unity_prior,
                               max_views: int = 4,
                               canonical_camera_order: Optional[List[str]] = None,
                               **kwargs) -> MultiViewSMILImageRegressor:
    """
    Factory function to create a multi-view SMIL regressor.
    
    Args:
        device: PyTorch device
        batch_size: Batch size
        shape_family: SMIL shape family
        use_unity_prior: Whether to use unity prior
        max_views: Maximum number of views
        canonical_camera_order: List of canonical camera names
        **kwargs: Additional arguments for SMILImageRegressor
        
    Returns:
        MultiViewSMILImageRegressor instance
    """
    # Create placeholder data batch
    # Use input_resolution if provided, otherwise default to 224
    # This ensures the renderer is initialized with the correct size
    input_resolution = kwargs.get('input_resolution', 224)
    data_batch = torch.zeros(batch_size, 3, input_resolution, input_resolution, device=device)
    
    return MultiViewSMILImageRegressor(
        device=device,
        data_batch=data_batch,
        batch_size=batch_size,
        shape_family=shape_family,
        use_unity_prior=use_unity_prior,
        max_views=max_views,
        canonical_camera_order=canonical_camera_order,
        **kwargs
    )

