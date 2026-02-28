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
from training_config import TrainingConfig
from pytorch3d.renderer import FoVPerspectiveCameras


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
                 default_fov: float = 30.0, fov_range: Tuple[float, float] = (5.0, 120.0),
                 trans_scale: float = 5.0,
                 fov_delta_scale: float = 5.0,
                 trans_delta_scale: float = 0.25,
                 rot_delta_scale: float = 0.1):
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
        self.fov_delta_scale = fov_delta_scale
        self.trans_delta_scale = trans_delta_scale
        self.rot_delta_scale = rot_delta_scale
        
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
        
        # Ensure output is float32
        output = output.float()
        
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
                                   device=output.device, dtype=torch.float32)
        rot_6d = rot_6d + identity_6d
        cam_rot = self._rotation_6d_to_matrix(rot_6d)
        idx += self.cam_rot_dim
        
        # Translation: Scale with tanh to bound range, then scale
        cam_trans_raw = output[:, idx:idx + self.cam_trans_dim]
        cam_trans = torch.tanh(cam_trans_raw) * self.trans_scale
        
        return {
            'fov': fov.float(),
            'cam_rot': cam_rot.float(),
            'cam_trans': cam_trans.float()
        }

    def forward_delta(self, features: torch.Tensor,
                      base_fov: torch.Tensor,
                      base_cam_rot: torch.Tensor,
                      base_cam_trans: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict small corrections (deltas) around provided base camera parameters.
        """
        output = self.layers(features).float()

        idx = 0

        # FOV delta (degrees)
        fov_raw = output[:, idx:idx + self.fov_dim]
        delta_fov = torch.tanh(fov_raw) * self.fov_delta_scale
        fov = base_fov + delta_fov
        fov = torch.clamp(fov, self.fov_min, self.fov_max)
        idx += self.fov_dim

        # Rotation delta as 6D -> matrix
        rot_6d = output[:, idx:idx + self.cam_rot_dim] * self.rot_delta_scale
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                   device=output.device, dtype=torch.float32)
        rot_6d = rot_6d + identity_6d
        delta_rot = self._rotation_6d_to_matrix(rot_6d)
        cam_rot = torch.matmul(delta_rot, base_cam_rot)
        idx += self.cam_rot_dim

        # Translation delta
        cam_trans_raw = output[:, idx:idx + self.cam_trans_dim]
        delta_trans = torch.tanh(cam_trans_raw) * self.trans_delta_scale
        cam_trans = base_cam_trans + delta_trans
        
        return {
            'fov': fov.float(),
            'cam_rot': cam_rot.float(),
            'cam_trans': cam_trans.float()
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
                 use_gt_camera_init: bool = False,
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
        self.use_gt_camera_init = use_gt_camera_init
        
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
                          view_mask: Optional[torch.Tensor] = None,
                          target_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
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
        
        # =================== OPTIMIZED: Single backbone pass for all views ===================
        # Stack all views: List of (B, C, H, W) -> (B, V, C, H, W) -> (B*V, C, H, W)
        # This gives 5-8x speedup vs sequential backbone calls
        all_images = torch.stack(images_per_view, dim=1)  # (B, V, C, H, W)
        B, V, C, H, W = all_images.shape
        all_images_flat = all_images.view(B * V, C, H, W)  # (B*V, C, H, W)
        
        # Single backbone forward pass for all views at once
        all_features = self.backbone(all_images_flat)  # (B*V, feature_dim)
        
        # Reshape back to (B, V, feature_dim)
        stacked_features = all_features.view(B, V, -1)
        
        # Add view embeddings based on camera indices
        # camera_indices: (batch_size, num_views)
        view_embeds = self.view_embeddings(camera_indices)  # (batch_size, num_views, feature_dim)
        stacked_features = stacked_features + view_embeds
        
        # Apply cross-view attention for feature fusion
        fused_features = self.view_fusion(stacked_features, view_mask)  # (batch_size, num_views, feature_dim)

        # Predict shared body parameters using parent's head
        # For transformer_decoder: cross-attend directly to per-view fused features
        # For mlp: mean-pool first since the MLP needs a flat vector
        if self.head_type == 'transformer_decoder':
            body_params = self._predict_body_params(
                fused_features, batch_size, view_mask=view_mask
            )
        else:
            # MLP path: pool to (B, feature_dim) then pass through aggregator
            if view_mask is not None:
                mask_expanded = view_mask.unsqueeze(-1).float()  # (batch_size, num_views, 1)
                masked_features = fused_features * mask_expanded
                aggregated_features = masked_features.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
            else:
                aggregated_features = fused_features.mean(dim=1)  # (batch_size, feature_dim)
            body_features = self.body_aggregator(aggregated_features)
            body_params = self._predict_body_params(body_features, batch_size)
        
        # Predict per-view camera parameters
        per_view_cam_params = self._predict_camera_params_per_view(
            fused_features, camera_indices, view_mask, num_views, target_data=target_data
        )
        
        # Combine into output dictionary
        output = {**body_params}
        
        # Add per-view camera params
        output['fov_per_view'] = per_view_cam_params['fov']  # List of (batch_size, 1)
        output['cam_rot_per_view'] = per_view_cam_params['cam_rot']  # List of (batch_size, 3, 3)
        output['cam_trans_per_view'] = per_view_cam_params['cam_trans']  # List of (batch_size, 3)
        output['num_views'] = num_views
        output['view_mask'] = view_mask
        output['camera_indices'] = camera_indices  # (batch_size, num_views) - canonical camera indices for verification
        
        return output
    
    def _predict_body_params(self, features: torch.Tensor, batch_size: int,
                             view_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predict shared body parameters from features.

        For the MLP head, ``features`` is a pooled vector ``(B, feature_dim)``.
        For the transformer_decoder head, ``features`` can be per-view fused
        features ``(B, V, feature_dim)`` which are used as cross-attention
        context, letting the decoder attend over views.

        Args:
            features: Either ``(B, feature_dim)`` (MLP) or ``(B, V, feature_dim)``
                      (transformer_decoder with multiview context).
            batch_size: Batch size.
            view_mask: Optional ``(B, V)`` boolean mask for valid views.  Only
                       used when features is 3-D (transformer_decoder path).
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
                    # Check if using PCA or per-joint
                    scale_trans_config = TrainingConfig.get_scale_trans_config()
                    use_pca_transformation = scale_trans_config.get('separate', {}).get('use_pca_transformation', True)
                    if use_pca_transformation:
                        # PCA weights - keep as 1D
                        params['log_beta_scales'] = output[:, idx:idx + self.scales_dim]  # (batch_size, N_BETAS)
                    else:
                        # Per-joint values - reshape to (batch_size, n_joints, 3)
                        scales_flat = output[:, idx:idx + self.scales_dim]
                        n_joints = self.scales_dim // 3
                        params['log_beta_scales'] = scales_flat.view(batch_size, n_joints, 3)
                else:
                    # In other modes, reshape to per-joint values
                    scales_flat = output[:, idx:idx + self.scales_dim]
                    n_joints = self.scales_dim // 3
                    params['log_beta_scales'] = scales_flat.view(batch_size, n_joints, 3)
                idx += self.scales_dim
            
            # Joint translations (if available)
            if self.joint_trans_dim > 0:
                if self.scale_trans_mode == 'separate':
                    # Check if using PCA or per-joint
                    scale_trans_config = TrainingConfig.get_scale_trans_config()
                    use_pca_transformation = scale_trans_config.get('separate', {}).get('use_pca_transformation', True)
                    if use_pca_transformation:
                        # PCA weights - keep as 1D
                        params['betas_trans'] = output[:, idx:idx + self.joint_trans_dim]  # (batch_size, N_BETAS)
                    else:
                        # Per-joint values - reshape to (batch_size, n_joints, 3)
                        trans_flat = output[:, idx:idx + self.joint_trans_dim]
                        n_joints = self.joint_trans_dim // 3
                        params['betas_trans'] = trans_flat.view(batch_size, n_joints, 3)
                else:
                    # In other modes, reshape to per-joint values
                    trans_flat = output[:, idx:idx + self.joint_trans_dim]
                    n_joints = self.joint_trans_dim // 3
                    params['betas_trans'] = trans_flat.view(batch_size, n_joints, 3)
            
            return params
            
        elif self.head_type == 'transformer_decoder':
            # Use transformer decoder for body params.
            # When features is 3-D (B, V, D) — multiview path — the per-view
            # fused features serve as the cross-attention context so the decoder
            # can attend over views (analogous to attending over spatial patch
            # tokens in the single-view path).
            if features.dim() == 3:
                # Multiview: features is (B, V, D)
                spatial_feats = features  # (B, V, D) — one token per view

                # Pool for the global feature vector the decoder also receives
                if view_mask is not None:
                    mask_exp = view_mask.unsqueeze(-1).float()  # (B, V, 1)
                    global_feats = (features * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-8)
                else:
                    global_feats = features.mean(dim=1)  # (B, D)
            else:
                # Single-view / already-pooled fallback: (B, D)
                spatial_feats = features.unsqueeze(1)  # (B, 1, D)
                global_feats = features

            params = self.transformer_head(global_feats, spatial_feats)
            
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
                                         num_views: int,
                                         target_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[torch.Tensor]]:
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

            gt_cam = None
            if self.use_gt_camera_init and target_data is not None:
                gt_cam = self._collect_view_camera_data(target_data, v, batch_size)
            
            # Predict camera params per sample using appropriate head
            fov_batch = torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device)
            cam_rot_batch = torch.zeros(batch_size, 3, 3, dtype=torch.float32, device=self.device)
            cam_trans_batch = torch.zeros(batch_size, 3, dtype=torch.float32, device=self.device)
            
            # Group samples by camera index for efficient processing
            for head_idx in range(self.num_canonical_cameras):
                mask = (cam_idx == head_idx)
                if not mask.any():
                    continue

                gt_mask = None
                if gt_cam is not None and gt_cam.get('mask') is not None:
                    gt_mask = mask & gt_cam['mask']

                if gt_mask is not None and gt_mask.any():
                    head_features = view_features[gt_mask]
                    base_fov = gt_cam['fov'][gt_mask]
                    base_rot = gt_cam['cam_rot'][gt_mask]
                    base_trans = gt_cam['cam_trans'][gt_mask]
                    head_output = self.camera_heads[head_idx].forward_delta(
                        head_features, base_fov, base_rot, base_trans
                    )
                    fov_batch[gt_mask] = head_output['fov'].float()
                    cam_rot_batch[gt_mask] = head_output['cam_rot'].float()
                    cam_trans_batch[gt_mask] = head_output['cam_trans'].float()

                abs_mask = mask if gt_mask is None else (mask & ~gt_mask)
                if abs_mask.any():
                    head_features = view_features[abs_mask]
                    head_output = self.camera_heads[head_idx](head_features)
                    fov_batch[abs_mask] = head_output['fov'].float()
                    cam_rot_batch[abs_mask] = head_output['cam_rot'].float()
                    cam_trans_batch[abs_mask] = head_output['cam_trans'].float()
            
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
                'keypoint_3d': 1.0,
                'fov': 0.0,
                'cam_rot': 0.0,
                'cam_trans': 0.0,
                'joint_angle_regularization': 0.0001,  # Penalty for large joint angles
                'limb_scale_regularization': 0.01,      # Penalty for deviations from scale=1 (log_beta_scales)
                'limb_trans_regularization': 0.1,       # Heavy penalty for translation changes (betas_trans)
                'triangulation_consistency': 0.0,       # Triangulate GT 2D keypoints with predicted cameras, compare to predicted 3D
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
        
        # Scale/trans losses if available (supervised losses against targets)
        if 'log_beta_scales' in predicted_params and loss_weights.get('log_beta_scales', 0) > 0:
            if body_targets.get('log_beta_scales') is not None:
                # Only supervise when we have real targets (not all zeros) to allow model to learn scales for keypoint fitting
                target_norm = torch.norm(body_targets['log_beta_scales'], dim=-1).mean()
                if target_norm > 1e-6:  # Only supervise if targets are non-zero (real ground truth)
                    loss = F.mse_loss(predicted_params['log_beta_scales'], body_targets['log_beta_scales'])
                    loss_components['log_beta_scales'] = loss
                    total_loss = total_loss + loss_weights['log_beta_scales'] * loss
                else:
                    # Targets are all zeros - don't supervise, let model learn scales from keypoint losses
                    loss_components['log_beta_scales'] = torch.tensor(eps, device=self.device, requires_grad=True)
        
        if 'betas_trans' in predicted_params and loss_weights.get('betas_trans', 0) > 0:
            if body_targets.get('betas_trans') is not None:
                loss = F.mse_loss(predicted_params['betas_trans'], body_targets['betas_trans'])
                loss_components['betas_trans'] = loss
                total_loss = total_loss + loss_weights['betas_trans'] * loss
        
        # Limb scale regularization: penalize deviations from scale=1 (log_beta_scales close to 0)
        # This encourages the model to keep joint scales close to their original values
        if (loss_weights.get('limb_scale_regularization', 0) > 0 and 
            'log_beta_scales' in predicted_params and 
            self.scale_trans_mode == 'separate'):
            log_scales = predicted_params['log_beta_scales']  # (batch_size, ...)
            
            # Check if using PCA or per-joint
            scale_trans_config = TrainingConfig.get_scale_trans_config()
            use_pca_transformation = scale_trans_config.get('separate', {}).get('use_pca_transformation', True)
            
            if use_pca_transformation:
                # PCA weights - convert to per-joint values for regularization
                try:
                    scale_weights = log_scales
                    trans_weights = predicted_params.get('betas_trans', None)
                    log_beta_scales_joint, _ = self._transform_separate_pca_weights_to_joint_values(
                        scale_weights, trans_weights
                    )
                except Exception as e:
                    print(f"Warning: Failed to convert PCA scale weights to joint values for regularization: {e}")
                    log_beta_scales_joint = torch.zeros_like(log_scales.view(batch_size, -1, 3))
            else:
                # Already per-joint values - use directly
                log_beta_scales_joint = log_scales  # (batch_size, n_joints, 3)
            
            # Penalize squared norm to encourage values close to 0 (log(1) = 0 means scale=1)
            scale_reg = torch.mean(log_beta_scales_joint ** 2)
            loss_components['limb_scale_regularization'] = scale_reg
            total_loss = total_loss + loss_weights['limb_scale_regularization'] * scale_reg
        
        # Limb translation regularization: heavily penalize translation changes (betas_trans)
        # This prevents the model from using translation to "cheat" by dragging bones to desired positions
        # without learning proper joint angles. Translation can lead to odd artifacts.
        if (loss_weights.get('limb_trans_regularization', 0) > 0 and 
            'betas_trans' in predicted_params and 
            self.scale_trans_mode == 'separate'):
            trans_params = predicted_params['betas_trans']  # (batch_size, ...)
            
            # Check if using PCA or per-joint
            scale_trans_config = TrainingConfig.get_scale_trans_config()
            use_pca_transformation = scale_trans_config.get('separate', {}).get('use_pca_transformation', True)
            
            if use_pca_transformation:
                # PCA weights - convert to per-joint values for regularization
                try:
                    scale_weights = predicted_params.get('log_beta_scales', None)
                    trans_weights = trans_params
                    _, betas_trans_joint = self._transform_separate_pca_weights_to_joint_values(
                        scale_weights, trans_weights
                    )
                except Exception as e:
                    print(f"Warning: Failed to convert PCA trans weights to joint values for regularization: {e}")
                    betas_trans_joint = torch.zeros_like(trans_params.view(batch_size, -1, 3))
            else:
                # Already per-joint values - use directly
                betas_trans_joint = trans_params  # (batch_size, n_joints, 3)
            
            # Heavy penalty: squared norm to strongly encourage values close to 0
            trans_reg = torch.mean(betas_trans_joint ** 2)
            loss_components['limb_trans_regularization'] = trans_reg
            total_loss = total_loss + loss_weights['limb_trans_regularization'] * trans_reg
        
        # =================== BATCHED 2D KEYPOINT LOSS (OPTIMIZED) ===================
        # This is the core multi-view loss: project 3D keypoints to each view using
        # batched camera projection (8-10x faster than per-view SMAL forward passes)
        
        # Get joint importance weights (computed once, used for both 2D and 3D losses)
        joint_importance_weights = self._get_joint_importance_weights()
        
        # Pre-compute shared data needed by multiple loss terms
        need_kp = (loss_weights.get('keypoint_2d', 0) > 0
                   or loss_weights.get('triangulation_consistency', 0) > 0
                   or loss_weights.get('ief_intermediate', 0) > 0)
        all_kp_data = None
        joints_3d = None
        aspect_ratio_list = None

        if need_kp:
            all_kp_data = self._collect_all_view_keypoint_data(target_data, num_views)
            if all_kp_data is not None:
                joints_3d = self._compute_world_space_joints(predicted_params)  # (B, J, 3)
                gt_cam_all = self._collect_all_view_camera_data(target_data, num_views)
                if gt_cam_all is not None and gt_cam_all.get('aspect_ratio') is not None:
                    aspect_ratio_list = [gt_cam_all['aspect_ratio'][:, v] for v in range(num_views)]

        if loss_weights.get('keypoint_2d', 0) > 0:
            if all_kp_data is not None and joints_3d is not None:
                # Batch project to all views at once
                try:
                    rendered_joints_all = self._batch_project_joints_to_views(
                        joints_3d,
                        predicted_params['fov_per_view'],
                        predicted_params['cam_rot_per_view'],
                        predicted_params['cam_trans_per_view'],
                        aspect_ratio_per_view=aspect_ratio_list,
                        view_mask=view_mask
                    )  # (B, V, J, 2)

                    target_kps = all_kp_data['keypoints_2d']  # (B, V, J, 2)
                    visibility = all_kp_data['visibility']    # (B, V, J)

                    keypoint_loss = self._compute_batched_keypoint_loss(
                        rendered_joints_all, target_kps, visibility, view_mask,
                        joint_weights=joint_importance_weights
                    )

                    if torch.isfinite(keypoint_loss):
                        loss_components['keypoint_2d'] = keypoint_loss
                        total_loss = total_loss + loss_weights['keypoint_2d'] * keypoint_loss
                    else:
                        loss_components['keypoint_2d'] = torch.tensor(eps, device=self.device, requires_grad=True)

                except Exception as e:
                    print(f"Warning: Batched keypoint projection failed: {e}")
                    loss_components['keypoint_2d'] = torch.tensor(eps, device=self.device, requires_grad=True)
            else:
                loss_components['keypoint_2d'] = torch.tensor(eps, device=self.device, requires_grad=True)

        # =================== TRIANGULATION CONSISTENCY LOSS ===================
        # Triangulate GT 2D keypoints using predicted cameras, compare to
        # predicted 3D joints (detached).  Gradients flow through the
        # differentiable triangulation into the camera heads, enforcing
        # multi-view geometric consistency.  The body model's 3D predictions
        # serve as the stable target (already supervised by keypoint_3d).
        if loss_weights.get('triangulation_consistency', 0) > 0:
            if all_kp_data is not None and joints_3d is not None:
                try:
                    target_kps = all_kp_data['keypoints_2d']  # (B, V, J, 2)
                    visibility = all_kp_data['visibility']    # (B, V, J)

                    triangulated, tri_valid = self._triangulate_joints_dlt(
                        target_kps, visibility,
                        predicted_params['fov_per_view'],
                        predicted_params['cam_rot_per_view'],
                        predicted_params['cam_trans_per_view'],
                        aspect_ratio_per_view=aspect_ratio_list,
                        view_mask=view_mask,
                    )  # (B, J, 3), (B, J)

                    with torch.no_grad():
                        # Reject outlier triangulations (behind camera or extremely far)
                        tri_norm = triangulated.norm(dim=-1)  # (B, J)
                        sane = tri_norm < 50.0  # generous bound
                        tri_valid = tri_valid & sane

                    if tri_valid.any():
                        diff_sq = (triangulated - joints_3d.detach()) ** 2  # (B, J, 3)

                        # Mask to valid joints and apply importance weights
                        mask_weights = tri_valid.float()  # (B, J)
                        if joint_importance_weights is not None:
                            mask_weights = mask_weights * joint_importance_weights.unsqueeze(0)

                        masked_loss = diff_sq * mask_weights.unsqueeze(-1)  # (B, J, 3)

                        if joint_importance_weights is not None:
                            denom = (tri_valid.float() * joint_importance_weights.unsqueeze(0)).sum() * 3 + eps
                        else:
                            denom = tri_valid.sum().float() * 3 + eps

                        tri_loss = masked_loss.sum() / denom

                        if torch.isfinite(tri_loss):
                            loss_components['triangulation_consistency'] = tri_loss
                            total_loss = total_loss + loss_weights['triangulation_consistency'] * tri_loss
                        else:
                            loss_components['triangulation_consistency'] = torch.tensor(eps, device=self.device, requires_grad=True)
                    else:
                        loss_components['triangulation_consistency'] = torch.tensor(eps, device=self.device, requires_grad=True)
                except Exception as e:
                    print(f"Warning: Triangulation consistency loss failed: {e}")
                    loss_components['triangulation_consistency'] = torch.tensor(eps, device=self.device, requires_grad=True)

        # =================== PER-VIEW CAMERA SUPERVISION LOSSES ===================
        # If GT camera parameters are available (e.g., from SLEAP 3D calibration), we can supervise
        # the per-view camera heads directly.
        #
        # IMPORTANT: Each view position v uses the camera head corresponding to camera_indices[batch_idx, v].
        # The GT camera data at view position v should correspond to the same canonical camera.
        # We verify this by checking that GT data is collected for the same view position that was used
        # for prediction (which is correct, since both use view position indexing).
        if num_views > 0:
            # Get camera indices for verification (if available)
            camera_indices = predicted_params.get('camera_indices', None)  # (B, num_views) - canonical camera indices
            
            # FOV loss
            if loss_weights.get('fov', 0) > 0:
                fov_loss_sum = torch.tensor(0.0, device=self.device, requires_grad=True)
                fov_count = 0
                for v in range(num_views):
                    gt_cam = self._collect_view_camera_data(target_data, v, batch_size)
                    if gt_cam is None or gt_cam.get('fov') is None:
                        continue
                    pred_fov = predicted_params['fov_per_view'][v].float()  # (B, 1)
                    gt_fov = gt_cam['fov'].float()
                    if gt_fov.dim() == 1:
                        gt_fov = gt_fov.unsqueeze(1)
                    mask = gt_cam.get('mask', None)
                    if mask is not None and mask.any():
                        # Verify: For each sample in the batch, the predicted camera head should match
                        # the canonical camera index for this view position.
                        # pred_fov[mask] was predicted using camera_heads[camera_indices[mask, v]]
                        # gt_fov[mask] should be the GT for the same canonical cameras.
                        fov_loss_sum = fov_loss_sum + F.mse_loss(pred_fov[mask], gt_fov[mask])
                        fov_count += 1
                if fov_count > 0:
                    fov_loss = fov_loss_sum / fov_count
                    loss_components['fov'] = fov_loss
                    total_loss = total_loss + loss_weights['fov'] * fov_loss

            # Camera rotation loss
            if loss_weights.get('cam_rot', 0) > 0:
                rot_loss_sum = torch.tensor(0.0, device=self.device, requires_grad=True)
                rot_count = 0
                for v in range(num_views):
                    gt_cam = self._collect_view_camera_data(target_data, v, batch_size)
                    if gt_cam is None or gt_cam.get('cam_rot') is None:
                        continue
                    pred_R = predicted_params['cam_rot_per_view'][v].float()  # (B, 3, 3)
                    gt_R = gt_cam['cam_rot'].float()
                    mask = gt_cam.get('mask', None)
                    if mask is not None and mask.any():
                        # Same verification: pred_R[mask] from camera_heads[camera_indices[mask, v]]
                        # should match gt_R[mask] for the same canonical cameras.
                        rot_loss_sum = rot_loss_sum + F.mse_loss(pred_R[mask], gt_R[mask])
                        rot_count += 1
                if rot_count > 0:
                    rot_loss = rot_loss_sum / rot_count
                    loss_components['cam_rot'] = rot_loss
                    total_loss = total_loss + loss_weights['cam_rot'] * rot_loss

            # Camera translation loss
            if loss_weights.get('cam_trans', 0) > 0:
                trans_loss_sum = torch.tensor(0.0, device=self.device, requires_grad=True)
                trans_count = 0
                for v in range(num_views):
                    gt_cam = self._collect_view_camera_data(target_data, v, batch_size)
                    if gt_cam is None or gt_cam.get('cam_trans') is None:
                        continue
                    pred_T = predicted_params['cam_trans_per_view'][v].float()  # (B, 3)
                    gt_T = gt_cam['cam_trans'].float()
                    mask = gt_cam.get('mask', None)
                    if mask is not None and mask.any():
                        # Same verification: pred_T[mask] from camera_heads[camera_indices[mask, v]]
                        # should match gt_T[mask] for the same canonical cameras.
                        trans_loss_sum = trans_loss_sum + F.mse_loss(pred_T[mask], gt_T[mask])
                        trans_count += 1
                if trans_count > 0:
                    t_loss = trans_loss_sum / trans_count
                    loss_components['cam_trans'] = t_loss
                    total_loss = total_loss + loss_weights['cam_trans'] * t_loss

        # =================== SHARED 3D KEYPOINT LOSS ===================
        # Collect GT 3D keypoints (also used by IEF intermediate supervision below)
        gt_3d = self._collect_keypoints_3d_batch(target_data) if (
            loss_weights.get('keypoint_3d', 0) > 0 or loss_weights.get('ief_intermediate', 0) > 0
        ) else None
        if loss_weights.get('keypoint_3d', 0) > 0:
            if gt_3d is not None and gt_3d.get('keypoints_3d') is not None:
                pred_joints_3d = self._predict_canonical_joints_3d(predicted_params)  # (B, J, 3)
                target_joints_3d = gt_3d['keypoints_3d']  # (B, J, 3)
                sample_mask = gt_3d.get('mask', None)

                if pred_joints_3d.shape == target_joints_3d.shape and sample_mask is not None and sample_mask.any():
                    # Create per-joint mask: exclude joints that are all zeros (filtered outliers)
                    # and exclude NaN/inf values (safety check)
                    joint_norms = torch.norm(target_joints_3d, dim=-1)  # (B, J)
                    finite_mask = torch.isfinite(target_joints_3d).all(dim=-1)  # (B, J)
                    valid_joint_mask = (joint_norms > 1e-6) & finite_mask  # (B, J)
                    
                    # Combine sample mask (has 3D data) with joint mask (valid joints)
                    combined_mask = sample_mask.unsqueeze(1) & valid_joint_mask  # (B, J)
                    
                    if combined_mask.any():
                        # Use masked MSE loss
                        diff = pred_joints_3d - target_joints_3d  # (B, J, 3)
                        diff_squared = diff ** 2
                        
                        # Apply mask: only valid joints contribute to loss
                        mask_weights = combined_mask.float()  # (B, J)
                        
                        # Apply joint importance weights if available
                        if joint_importance_weights is not None:
                            # Multiply per-joint mask by importance weights: (B, J) * (J,) -> (B, J)
                            mask_weights = mask_weights * joint_importance_weights.unsqueeze(0)
                        
                        # Expand for xyz: (B, J) -> (B, J, 3)
                        masked_loss = diff_squared * mask_weights.unsqueeze(-1)
                        
                        # Normalize by weighted count
                        if joint_importance_weights is not None:
                            # Weighted normalization for stable loss magnitude
                            weighted_valid_count = (combined_mask.float() * joint_importance_weights.unsqueeze(0)).sum() * 3 + 1e-8
                        else:
                            weighted_valid_count = combined_mask.sum().float() * 3  # *3 for x,y,z
                        
                        if weighted_valid_count > 0:
                            kp3d_loss = masked_loss.sum() / weighted_valid_count
                        else:
                            kp3d_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                        
                        # Check for NaN/inf
                        if torch.isfinite(kp3d_loss):
                            loss_components['keypoint_3d'] = kp3d_loss
                            total_loss = total_loss + loss_weights['keypoint_3d'] * kp3d_loss
                        else:
                            print(f"Warning: Non-finite 3D keypoint loss detected, skipping")
                            loss_components['keypoint_3d'] = torch.tensor(0.0, device=self.device, requires_grad=True)
                    else:
                        # No valid joints in batch
                        loss_components['keypoint_3d'] = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # IEF intermediate keypoint supervision
        # Runs each non-final IEF prediction through the body model to get 3D
        # joints, then computes the same 2D/3D keypoint losses that supervise the
        # final output.  This gives intermediate steps meaningful gradient signal
        # (the dominant keypoint losses) rather than only parameter-level losses
        # which may have tiny weights or missing GT (e.g. no GT joint angles in
        # SLEAP data).  Controlled by 'ief_intermediate' loss weight.
        ief_w = loss_weights.get('ief_intermediate', 0.0)
        if ief_w > 0 and 'iteration_history' in predicted_params:
            hist = predicted_params['iteration_history']
            n_iters = len(hist.get('pose', []))
            global_rot_dim = 6 if self.rotation_representation == '6d' else 3
            rot_per_joint = 6 if self.rotation_representation == '6d' else 3

            for i in range(n_iters - 1):  # skip final iter — already supervised above
                w = ief_w * (i + 1) / n_iters  # linear ramp: earlier iters get lower weight

                # Reconstruct body_params dict from iteration history
                iter_pose = hist['pose'][i]  # (B, total_pose_dim)
                iter_body_params = {
                    'global_rot': iter_pose[:, :global_rot_dim],
                    'joint_rot': iter_pose[:, global_rot_dim:].view(batch_size, config.N_POSE, rot_per_joint),
                    'betas': hist['betas'][i],
                    'trans': hist['trans'][i],
                }
                if 'scales' in hist:
                    # Iteration history stores flat (B, scales_dim); reshape to
                    # match _predict_canonical_joints_3d: per-joint (B, n_joints, 3)
                    # for non-PCA separate mode, or (B, N_BETAS) for PCA mode.
                    raw_scales = hist['scales'][i]
                    if raw_scales.shape[-1] != config.N_BETAS:
                        iter_body_params['log_beta_scales'] = raw_scales.view(batch_size, -1, 3)
                    else:
                        iter_body_params['log_beta_scales'] = raw_scales
                if 'joint_trans' in hist:
                    raw_jtrans = hist['joint_trans'][i]
                    if raw_jtrans.shape[-1] != config.N_BETAS:
                        iter_body_params['betas_trans'] = raw_jtrans.view(batch_size, -1, 3)
                    else:
                        iter_body_params['betas_trans'] = raw_jtrans
                if 'mesh_scale' in hist:
                    iter_body_params['mesh_scale'] = hist['mesh_scale'][i]

                # Forward through body model to get intermediate 3D joints
                try:
                    iter_joints_3d = self._predict_canonical_joints_3d(iter_body_params)  # (B, J, 3)
                except Exception as e:
                    print(f"Warning: IEF intermediate body model forward failed at iter {i}: {e}")
                    continue

                # 3D keypoint loss on intermediate predictions
                if gt_3d is not None and gt_3d.get('keypoints_3d') is not None and loss_weights.get('keypoint_3d', 0) > 0:
                    target_joints_3d = gt_3d['keypoints_3d']
                    sample_mask_3d = gt_3d.get('mask', None)
                    if (iter_joints_3d.shape == target_joints_3d.shape
                            and sample_mask_3d is not None and sample_mask_3d.any()):
                        joint_norms = torch.norm(target_joints_3d, dim=-1)
                        finite_mask = torch.isfinite(target_joints_3d).all(dim=-1)
                        valid_joint_mask = (joint_norms > 1e-6) & finite_mask
                        combined_mask = sample_mask_3d.unsqueeze(1) & valid_joint_mask
                        if combined_mask.any():
                            diff = iter_joints_3d - target_joints_3d
                            mask_weights_3d = combined_mask.float()
                            if joint_importance_weights is not None:
                                mask_weights_3d = mask_weights_3d * joint_importance_weights.unsqueeze(0)
                            masked_loss = (diff ** 2) * mask_weights_3d.unsqueeze(-1)
                            if joint_importance_weights is not None:
                                wvc = (combined_mask.float() * joint_importance_weights.unsqueeze(0)).sum() * 3 + 1e-8
                            else:
                                wvc = combined_mask.sum().float() * 3 + 1e-8
                            ief_kp3d = masked_loss.sum() / wvc
                            if torch.isfinite(ief_kp3d):
                                total_loss = total_loss + w * loss_weights['keypoint_3d'] * ief_kp3d

                # 2D keypoint loss on intermediate predictions (project using final cameras)
                if all_kp_data is not None and loss_weights.get('keypoint_2d', 0) > 0:
                    try:
                        iter_rendered = self._batch_project_joints_to_views(
                            iter_joints_3d,
                            predicted_params['fov_per_view'],
                            predicted_params['cam_rot_per_view'],
                            predicted_params['cam_trans_per_view'],
                            aspect_ratio_per_view=aspect_ratio_list,
                            view_mask=view_mask
                        )
                        ief_kp2d = self._compute_batched_keypoint_loss(
                            iter_rendered,
                            all_kp_data['keypoints_2d'],
                            all_kp_data['visibility'],
                            view_mask,
                            joint_weights=joint_importance_weights
                        )
                        if torch.isfinite(ief_kp2d):
                            total_loss = total_loss + w * loss_weights['keypoint_2d'] * ief_kp2d
                    except Exception as e:
                        pass  # projection can fail for degenerate intermediate cameras

                # Retain parameter-level losses when GT is available (cheap, extra signal)
                iter_joint_rot = iter_body_params['joint_rot']
                iter_betas = iter_body_params['betas']
                iter_trans = iter_body_params['trans']

                if body_targets.get('joint_rot') is not None and loss_weights.get('joint_rot', 0) > 0:
                    ief_loss = self._compute_rotation_loss(
                        iter_joint_rot.reshape(batch_size, -1),
                        body_targets['joint_rot'].reshape(batch_size, -1),
                        body_targets.get('joint_rot_mask')
                    )
                    total_loss = total_loss + w * loss_weights['joint_rot'] * ief_loss

                if body_targets.get('betas') is not None and loss_weights.get('betas', 0) > 0:
                    mask = body_targets.get('betas_mask')
                    if mask is not None and mask.any():
                        ief_loss = F.mse_loss(iter_betas[mask], body_targets['betas'][mask])
                        total_loss = total_loss + w * loss_weights['betas'] * ief_loss

                if body_targets.get('trans') is not None and loss_weights.get('trans', 0) > 0:
                    mask = body_targets.get('trans_mask')
                    if mask is not None and mask.any():
                        ief_loss = F.mse_loss(iter_trans[mask], body_targets['trans'][mask])
                        total_loss = total_loss + w * loss_weights['trans'] * ief_loss

        if return_components:
            return total_loss, loss_components
        return total_loss

    def _collect_view_camera_data(self, target_data: List[Dict[str, Any]], view_idx: int, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Collect GT camera parameters for a given view across the batch.

        Expects `target_data[i]` to optionally contain:
          - cam_fov_per_view: (num_views, 1) or (num_views,) in degrees
          - cam_rot_per_view: (num_views, 3, 3)
          - cam_trans_per_view: (num_views, 3)
        """
        fov_list = []
        rot_list = []
        trans_list = []
        aspect_list = []
        mask_list = []
        has_any = False

        for td in target_data:
            fov_v = None
            rot_v = None
            trans_v = None
            aspect_v = None

            if td.get('cam_fov_per_view') is not None:
                arr = td['cam_fov_per_view']
                if isinstance(arr, np.ndarray):
                    if view_idx < arr.shape[0]:
                        fov_v = torch.from_numpy(arr[view_idx]).to(dtype=torch.float32, device=self.device)
                else:
                    t = safe_to_tensor(arr, device=self.device)
                    if view_idx < t.shape[0]:
                        fov_v = t[view_idx].to(dtype=torch.float32)

            if td.get('cam_rot_per_view') is not None:
                arr = td['cam_rot_per_view']
                if isinstance(arr, np.ndarray):
                    if view_idx < arr.shape[0]:
                        rot_v = torch.from_numpy(arr[view_idx]).to(dtype=torch.float32, device=self.device)
                else:
                    t = safe_to_tensor(arr, device=self.device)
                    if view_idx < t.shape[0]:
                        rot_v = t[view_idx].to(dtype=torch.float32)

            if td.get('cam_trans_per_view') is not None:
                arr = td['cam_trans_per_view']
                if isinstance(arr, np.ndarray):
                    if view_idx < arr.shape[0]:
                        trans_v = torch.from_numpy(arr[view_idx]).to(dtype=torch.float32, device=self.device)
                else:
                    t = safe_to_tensor(arr, device=self.device)
                    if view_idx < t.shape[0]:
                        trans_v = t[view_idx].to(dtype=torch.float32)

            if td.get('cam_aspect_per_view') is not None:
                arr = td['cam_aspect_per_view']
                if isinstance(arr, np.ndarray):
                    if view_idx < arr.shape[0]:
                        aspect_v = torch.from_numpy(np.array(arr[view_idx])).to(dtype=torch.float32, device=self.device).reshape(-1)
                else:
                    t = safe_to_tensor(arr, device=self.device)
                    if view_idx < t.shape[0]:
                        aspect_v = t[view_idx].to(dtype=torch.float32).reshape(-1)

            is_valid = (fov_v is not None) and (rot_v is not None) and (trans_v is not None)
            mask_list.append(bool(is_valid))
            if is_valid:
                has_any = True
                # Normalize shapes
                if fov_v.dim() == 0:
                    fov_v = fov_v.unsqueeze(0)
                fov_list.append(fov_v.reshape(1))
                rot_list.append(rot_v.reshape(3, 3))
                trans_list.append(trans_v.reshape(3))
                if aspect_v is None:
                    aspect_list.append(torch.ones(1, dtype=torch.float32, device=self.device))
                else:
                    aspect_list.append(aspect_v[:1])
            else:
                fov_list.append(torch.zeros(1, dtype=torch.float32, device=self.device))
                rot_list.append(torch.eye(3, dtype=torch.float32, device=self.device))
                trans_list.append(torch.zeros(3, dtype=torch.float32, device=self.device))
                aspect_list.append(torch.ones(1, dtype=torch.float32, device=self.device))

        if not has_any:
            return None

        return {
            'fov': torch.stack(fov_list, dim=0).reshape(batch_size, 1),
            'cam_rot': torch.stack(rot_list, dim=0).reshape(batch_size, 3, 3),
            'cam_trans': torch.stack(trans_list, dim=0).reshape(batch_size, 3),
            'aspect_ratio': torch.stack(aspect_list, dim=0).reshape(batch_size, 1).squeeze(-1),  # (B,)
            'mask': torch.tensor(mask_list, dtype=torch.bool, device=self.device),
        }

    def _collect_keypoints_3d_batch(self, target_data: List[Dict[str, Any]]) -> Optional[Dict[str, torch.Tensor]]:
        """Collect GT 3D keypoints for the batch (if present)."""
        kp_list = []
        mask_list = []
        has_any = False

        for td in target_data:
            if td.get('keypoints_3d') is not None and bool(td.get('has_3d_data', False)):
                kp = td['keypoints_3d']
                kp_t = safe_to_tensor(kp, device=self.device).to(dtype=torch.float32)
                kp_list.append(kp_t)
                mask_list.append(True)
                has_any = True
            else:
                kp_list.append(None)
                mask_list.append(False)

        if not has_any:
            return None

        # Determine joint count from first valid
        first = next(k for k in kp_list if k is not None)
        J = first.shape[0]
        filled = []
        for k in kp_list:
            if k is None:
                filled.append(torch.zeros(J, 3, dtype=torch.float32, device=self.device))
            else:
                filled.append(k.reshape(J, 3))

        return {
            'keypoints_3d': torch.stack(filled, dim=0),  # (B, J, 3)
            'mask': torch.tensor(mask_list, dtype=torch.bool, device=self.device),
        }

    def _predict_canonical_joints_3d(self, body_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict canonical 3D joints (world/model space) for 3D supervision.

        This mirrors the SMAL forward + scaling/trans logic used in `_render_keypoints_with_camera`,
        but returns 3D joints instead of rendering/projecting to 2D.
        """
        batch_size = body_params['global_rot'].shape[0]

        # Convert rotations to axis-angle format for SMAL model
        if self.rotation_representation == '6d':
            global_rot_aa = rotation_6d_to_axis_angle(body_params['global_rot'])
            joint_rot_aa = rotation_6d_to_axis_angle(body_params['joint_rot'])
        else:
            global_rot_aa = body_params['global_rot']
            joint_rot_aa = body_params['joint_rot']

        if global_rot_aa.dim() == 2:
            global_rot_aa = global_rot_aa.unsqueeze(1)  # (B, 1, 3)
        if joint_rot_aa.dim() == 2:
            joint_rot_aa = joint_rot_aa.unsqueeze(1)

        pose_tensor = torch.cat([global_rot_aa, joint_rot_aa], dim=1)  # (B, N_POSE+1, 3)

        # Handle scale/trans parameters - apply when mode is 'separate' or 'entangled_with_betas'
        # This ensures scales and translations are used when computing 2D/3D keypoint losses
        # in modes where they are predicted, allowing the model to learn them implicitly from supervision signals
        betas_logscale = None
        betas_trans_val = None
        if 'log_beta_scales' in body_params and body_params['log_beta_scales'] is not None:
            if self.scale_trans_mode == 'separate':
                # Check if using PCA or per-joint values
                scale_trans_config = TrainingConfig.get_scale_trans_config()
                use_pca_transformation = scale_trans_config.get('separate', {}).get('use_pca_transformation', True)
                
                if use_pca_transformation:
                    # PCA weights - convert to per-joint values
                    try:
                        scale_weights = body_params['log_beta_scales']
                        trans_weights = body_params.get('betas_trans', None)
                        log_beta_scales_joint, betas_trans_joint = self._transform_separate_pca_weights_to_joint_values(
                            scale_weights, trans_weights
                        )
                        betas_logscale = log_beta_scales_joint
                        betas_trans_val = betas_trans_joint
                    except Exception as e:
                        # If conversion fails, log and fall back
                        print(f"Warning: Failed to convert PCA weights to joint values in _predict_canonical_joints_3d: {e}")
                        betas_logscale = None
                        betas_trans_val = None
                else:
                    # Already per-joint values - use directly
                    betas_logscale = body_params['log_beta_scales']  # (batch_size, n_joints, 3)
                    betas_trans_val = body_params.get('betas_trans', None)  # (batch_size, n_joints, 3)
            elif self.scale_trans_mode == 'entangled_with_betas':
                # In entangled mode, values are already per-joint and should be applied
                betas_logscale = body_params['log_beta_scales']
                betas_trans_val = body_params.get('betas_trans', None)
            # If mode is 'ignore', scales/translations are not applied (betas_logscale and betas_trans_val remain None)

        verts, joints, _, _ = self.smal_model(
            body_params['betas'],
            pose_tensor,
            betas_logscale=betas_logscale,
            betas_trans=betas_trans_val,
            propagate_scaling=self.propagate_scaling
        )

        # Apply transformation consistent with `_render_keypoints_with_camera`
        if self.use_ue_scaling:
            root_joint = joints[:, 0:1, :]
            joints = (joints - root_joint) * 10 + body_params['trans'].unsqueeze(1)
        elif self.allow_mesh_scaling and 'mesh_scale' in body_params:
            mesh_scale = body_params['mesh_scale']
            root_joint = joints[:, 0:1, :]
            joints = (joints - root_joint) * mesh_scale.unsqueeze(-1) + body_params['trans'].unsqueeze(1)
        else:
            joints = joints + body_params['trans'].unsqueeze(1)

        canonical_joints = joints[:, config.CANONICAL_MODEL_JOINTS]  # (B, J, 3)
        return canonical_joints.float()

    def _compute_world_space_joints(self, body_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute 3D joints in world space from body parameters.
        
        This is the optimized version that computes joints ONCE for use with
        batched multi-view projection, instead of recomputing for each view.
        
        Args:
            body_params: Dictionary with body parameters
            
        Returns:
            World-space 3D joints of shape (B, J, 3) where J = len(CANONICAL_MODEL_JOINTS)
        """
        # This is essentially the same as _predict_canonical_joints_3d
        # but we make it explicit that this is for caching/reuse
        return self._predict_canonical_joints_3d(body_params)

    def _batch_project_joints_to_views(self, 
                                        joints_3d: torch.Tensor,
                                        fov_per_view: List[torch.Tensor],
                                        cam_rot_per_view: List[torch.Tensor],
                                        cam_trans_per_view: List[torch.Tensor],
                                        aspect_ratio_per_view: Optional[List[torch.Tensor]] = None,
                                        view_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Project 3D joints to 2D for all views in a single batched operation.
        
        This gives 8-10x speedup vs calling _render_keypoints_with_camera per view,
        as we avoid running SMAL model N times and use batched camera projection.
        
        Args:
            joints_3d: World-space 3D joints (B, J, 3)
            fov_per_view: List of FOV tensors, each (B, 1)
            cam_rot_per_view: List of rotation matrices, each (B, 3, 3)
            cam_trans_per_view: List of translation vectors, each (B, 3)
            aspect_ratio_per_view: Optional list of aspect ratios, each (B,)
            view_mask: Optional mask (B, V) for valid views
            
        Returns:
            Projected 2D joints (B, V, J, 2) normalized to [0, 1]
        """
        batch_size = joints_3d.shape[0]
        num_joints = joints_3d.shape[1]
        num_views = len(fov_per_view)
        
        # Stack camera parameters: each is (B,) or (B, 3) or (B, 3, 3) -> (B*V, ...)
        # First reshape each to add view dimension, then flatten
        fov_stacked = torch.stack([f.squeeze(-1) if f.dim() > 1 else f for f in fov_per_view], dim=1)  # (B, V)
        fov_flat = fov_stacked.view(batch_size * num_views)  # (B*V,)
        
        cam_rot_stacked = torch.stack(cam_rot_per_view, dim=1)  # (B, V, 3, 3)
        cam_rot_flat = cam_rot_stacked.view(batch_size * num_views, 3, 3)  # (B*V, 3, 3)
        
        cam_trans_stacked = torch.stack(cam_trans_per_view, dim=1)  # (B, V, 3)
        cam_trans_flat = cam_trans_stacked.view(batch_size * num_views, 3)  # (B*V, 3)
        
        # Handle aspect ratio
        if aspect_ratio_per_view is not None and len(aspect_ratio_per_view) > 0:
            aspect_stacked = torch.stack([
                a.squeeze(-1) if a.dim() > 1 else a 
                for a in aspect_ratio_per_view
            ], dim=1)  # (B, V)
            aspect_flat = aspect_stacked.view(batch_size * num_views)  # (B*V,)
        else:
            aspect_flat = torch.ones(batch_size * num_views, dtype=torch.float32, device=self.device)
        
        # Create batched cameras for all views at once
        cameras = FoVPerspectiveCameras(
            device=self.device,
            R=cam_rot_flat.float(),
            T=cam_trans_flat.float(),
            fov=fov_flat.float(),
            aspect_ratio=aspect_flat.float(),
            znear=self.renderer.DEFAULT_ZNEAR if hasattr(self.renderer, 'DEFAULT_ZNEAR') else 0.001,
            zfar=self.renderer.DEFAULT_ZFAR if hasattr(self.renderer, 'DEFAULT_ZFAR') else 1000.0
        )
        
        # Expand joints to match camera batch: (B, J, 3) -> (B*V, J, 3)
        # Each batch sample's joints are repeated V times for V views
        joints_expanded = joints_3d.unsqueeze(1).expand(-1, num_views, -1, -1)  # (B, V, J, 3)
        joints_flat = joints_expanded.reshape(batch_size * num_views, num_joints, 3)  # (B*V, J, 3)
        
        # Project all joints to all views at once
        screen_size = torch.ones(batch_size * num_views, 2, dtype=torch.float32, device=self.device) * self.renderer.image_size
        proj_points = cameras.transform_points_screen(joints_flat.float(), image_size=screen_size)[:, :, [1, 0]]  # (B*V, J, 2)
        
        # Normalize to [0, 1] range
        rendered_image_size = float(self.renderer.image_size)
        proj_points_normalized = proj_points / (rendered_image_size + 1e-8)
        proj_points_normalized = torch.clamp(proj_points_normalized, 0.0, 1.0)
        
        # Reshape back to (B, V, J, 2)
        proj_points_batched = proj_points_normalized.view(batch_size, num_views, num_joints, 2)

        return proj_points_batched

    def _triangulate_joints_dlt(
        self,
        keypoints_2d: torch.Tensor,
        visibility: torch.Tensor,
        fov_per_view: List[torch.Tensor],
        cam_rot_per_view: List[torch.Tensor],
        cam_trans_per_view: List[torch.Tensor],
        aspect_ratio_per_view: Optional[List[torch.Tensor]] = None,
        view_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Triangulate 3D joint positions from multi-view 2D keypoints using DLT.

        Uses the predicted camera parameters to build projection matrices, then
        solves the linear triangulation system per joint via SVD.

        Args:
            keypoints_2d: GT 2D keypoints ``(B, V, J, 2)`` normalised to [0, 1].
            visibility: Per-joint visibility ``(B, V, J)``.
            fov_per_view: Predicted FOV per view, list of ``(B, 1)``.
            cam_rot_per_view: Predicted rotation per view, list of ``(B, 3, 3)``.
            cam_trans_per_view: Predicted translation per view, list of ``(B, 3)``.
            aspect_ratio_per_view: Optional aspect ratios, list of ``(B,)``.
            view_mask: Optional ``(B, V)`` mask for valid views.

        Returns:
            Tuple of:
            - triangulated_joints ``(B, J, 3)`` in world space.
            - valid_mask ``(B, J)`` boolean — True where >= 2 views contributed.
        """
        B, V, J, _ = keypoints_2d.shape
        device = keypoints_2d.device
        img_size = float(self.renderer.image_size)

        # --- Build (B*V, 4, 4) projection matrices from predicted cameras ---
        fov_stacked = torch.stack(
            [f.squeeze(-1) if f.dim() > 1 else f for f in fov_per_view], dim=1
        )  # (B, V)
        cam_rot_stacked = torch.stack(cam_rot_per_view, dim=1)  # (B, V, 3, 3)
        cam_trans_stacked = torch.stack(cam_trans_per_view, dim=1)  # (B, V, 3)

        if aspect_ratio_per_view is not None and len(aspect_ratio_per_view) > 0:
            aspect_stacked = torch.stack(
                [a.squeeze(-1) if a.dim() > 1 else a for a in aspect_ratio_per_view],
                dim=1,
            )  # (B, V)
        else:
            aspect_stacked = torch.ones(B, V, device=device)

        cameras = FoVPerspectiveCameras(
            device=device,
            R=cam_rot_stacked.reshape(B * V, 3, 3).float(),
            T=cam_trans_stacked.reshape(B * V, 3).float(),
            fov=fov_stacked.reshape(B * V).float(),
            aspect_ratio=aspect_stacked.reshape(B * V).float(),
            znear=self.renderer.DEFAULT_ZNEAR if hasattr(self.renderer, 'DEFAULT_ZNEAR') else 0.001,
            zfar=self.renderer.DEFAULT_ZFAR if hasattr(self.renderer, 'DEFAULT_ZFAR') else 1000.0,
        )

        # Full projection: world -> NDC (4x4 matrices, row-vector convention)
        P = cameras.get_full_projection_transform().get_matrix()  # (B*V, 4, 4)
        P = P.view(B, V, 4, 4)

        # --- Convert normalised [0,1] keypoints to NDC coordinates ---
        # _batch_project_joints_to_views does:
        #   screen = transform_points_screen(...)[:, :, [1, 0]]  (swap x,y)
        #   norm   = screen / img_size
        # So to invert: un-normalise then un-swap.
        kp_screen = keypoints_2d * img_size             # (B, V, J, 2)  in (y, x)
        kp_screen = kp_screen[..., [1, 0]]              # (B, V, J, 2)  in (x, y)

        # PyTorch3D's ndc_to_screen_points_naive uses:
        #   screen = (W - 1) / 2 * (1 - ndc)
        # Approximating (W-1) ≈ W for large image sizes, the inverse is:
        #   ndc = 1 - screen / (W / 2)
        ndc_xy = 1.0 - kp_screen / (img_size / 2.0)     # (B, V, J, 2)

        # --- Build DLT system and solve per joint ---
        # Combined visibility: view must be valid AND joint visible
        joint_vis = visibility.bool()  # (B, V, J)
        if view_mask is not None:
            joint_vis = joint_vis & view_mask.unsqueeze(-1)  # (B, V, J)

        # PyTorch3D uses row-vector convention: ndc_homo = world_homo @ P
        # so columns of P map to output dimensions:
        #   col 0 -> u*w,  col 1 -> v*w,  col 2 -> z*w,  col 3 -> w
        P_col0 = P[:, :, :, 0]  # (B, V, 4) — x output
        P_col1 = P[:, :, :, 1]  # (B, V, 4) — y output
        P_col3 = P[:, :, :, 3]  # (B, V, 4) — w output

        u = ndc_xy[..., 0]  # (B, V, J)
        v = ndc_xy[..., 1]  # (B, V, J)

        # DLT constraints: u * w_col - x_col = 0,  v * w_col - y_col = 0
        # With w=1 normalization: A_xyz @ [X,Y,Z]^T = -a_w
        # row1 = u * P_col3 - P_col0   shape (B, V, J, 4)
        row1 = u.unsqueeze(-1) * P_col3.unsqueeze(2) - P_col0.unsqueeze(2)
        # row2 = v * P_col3 - P_col1   shape (B, V, J, 4)
        row2 = v.unsqueeze(-1) * P_col3.unsqueeze(2) - P_col1.unsqueeze(2)

        # Zero out rows from invisible views
        vis_mask = joint_vis.unsqueeze(-1).float()  # (B, V, J, 1)
        row1 = row1 * vis_mask
        row2 = row2 * vis_mask

        # Stack rows: (B, V, J, 2, 4) -> (B, J, 2V, 4)
        A = torch.stack([row1, row2], dim=3)  # (B, V, J, 2, 4)
        A = A.permute(0, 2, 1, 3, 4)         # (B, J, V, 2, 4)
        A = A.reshape(B, J, V * 2, 4)        # (B, J, 2V, 4)

        # Solve via normal equations with w=1 normalization.
        # The homogeneous system A @ [X,Y,Z,1]^T = 0 becomes:
        #   A_xyz @ p = -a_w   =>   (A_xyz^T A_xyz + λI) @ p = -A_xyz^T a_w
        # The damping λI ensures the system is always full-rank (Tikhonov
        # regularisation), avoiding LAPACK failures on zero-padded rows from
        # invisible joints, and the gradient through torch.linalg.solve is
        # stable (implicit-function theorem, no singular-value gaps).
        A_xyz = A[:, :, :, :3]                # (B, J, 2V, 3)
        a_w = A[:, :, :, 3]                   # (B, J, 2V)

        AtA = torch.einsum('bkni,bknj->bkij', A_xyz, A_xyz)  # (B, J, 3, 3)
        Atb = torch.einsum('bkni,bkn->bki', A_xyz, -a_w)     # (B, J, 3)

        # Damping for numerical stability
        damping = 1e-6 * torch.eye(3, device=device, dtype=AtA.dtype)
        AtA = AtA + damping

        triangulated = torch.linalg.solve(AtA, Atb)  # (B, J, 3)

        # Valid mask: joint must be visible in >= 2 views
        view_count = joint_vis.float().sum(dim=1)  # (B, J)
        valid_mask = view_count >= 2.0

        return triangulated, valid_mask

    def _collect_all_view_keypoint_data(self, target_data: List[Dict[str, Any]],
                                         num_views: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Collect keypoint data for ALL views at once (batched).
        
        This is more efficient than calling _collect_view_keypoint_data per view.
        
        Args:
            target_data: List of target data dictionaries (one per batch sample)
            num_views: Number of views to collect
            
        Returns:
            Dictionary with:
                - 'keypoints_2d': (B, V, J, 2) tensor
                - 'visibility': (B, V, J) tensor
            Or None if no keypoint data available
        """
        batch_size = len(target_data)
        n_joints = len(config.CANONICAL_MODEL_JOINTS)
        
        # Pre-allocate output tensors
        keypoints_all = torch.zeros(batch_size, num_views, n_joints, 2, 
                                    dtype=torch.float32, device=self.device)
        visibility_all = torch.zeros(batch_size, num_views, n_joints, 
                                     dtype=torch.float32, device=self.device)
        has_any_data = False
        
        for b_idx, td in enumerate(target_data):
            kp_2d = td.get('keypoints_2d')
            kp_vis = td.get('keypoint_visibility')
            
            if kp_2d is None:
                continue
                
            kp_2d = np.array(kp_2d) if not isinstance(kp_2d, np.ndarray) else kp_2d
            
            if len(kp_2d.shape) == 3:
                # Multi-view format: (V, J, 2)
                actual_views = min(kp_2d.shape[0], num_views)
                actual_joints = min(kp_2d.shape[1], n_joints)
                
                keypoints_all[b_idx, :actual_views, :actual_joints, :] = torch.from_numpy(
                    kp_2d[:actual_views, :actual_joints, :]
                ).to(dtype=torch.float32, device=self.device)
                
                if kp_vis is not None:
                    kp_vis = np.array(kp_vis) if not isinstance(kp_vis, np.ndarray) else kp_vis
                    if len(kp_vis.shape) == 2:
                        visibility_all[b_idx, :actual_views, :actual_joints] = torch.from_numpy(
                            kp_vis[:actual_views, :actual_joints]
                        ).to(dtype=torch.float32, device=self.device)
                    else:
                        visibility_all[b_idx, :actual_views, :actual_joints] = 1.0
                else:
                    visibility_all[b_idx, :actual_views, :actual_joints] = 1.0
                    
                has_any_data = True
                
            elif len(kp_2d.shape) == 2:
                # Single-view format: (J, 2) - only for view 0
                actual_joints = min(kp_2d.shape[0], n_joints)
                keypoints_all[b_idx, 0, :actual_joints, :] = torch.from_numpy(
                    kp_2d[:actual_joints, :]
                ).to(dtype=torch.float32, device=self.device)
                
                if kp_vis is not None:
                    kp_vis = np.array(kp_vis) if not isinstance(kp_vis, np.ndarray) else kp_vis
                    visibility_all[b_idx, 0, :actual_joints] = torch.from_numpy(
                        kp_vis[:actual_joints]
                    ).to(dtype=torch.float32, device=self.device)
                else:
                    visibility_all[b_idx, 0, :actual_joints] = 1.0
                    
                has_any_data = True
        
        if not has_any_data:
            return None
        
        return {
            'keypoints_2d': keypoints_all,  # (B, V, J, 2)
            'visibility': visibility_all     # (B, V, J)
        }

    def _collect_all_view_camera_data(self, target_data: List[Dict[str, Any]], 
                                       num_views: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Collect GT camera data for ALL views at once (batched).
        
        Args:
            target_data: List of target data dictionaries
            num_views: Number of views
            
        Returns:
            Dictionary with batched camera params or None
        """
        batch_size = len(target_data)
        
        fov_all = torch.zeros(batch_size, num_views, dtype=torch.float32, device=self.device)
        cam_rot_all = torch.eye(3, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_views, -1, -1).clone()
        cam_trans_all = torch.zeros(batch_size, num_views, 3, dtype=torch.float32, device=self.device)
        aspect_all = torch.ones(batch_size, num_views, dtype=torch.float32, device=self.device)
        mask_all = torch.zeros(batch_size, num_views, dtype=torch.bool, device=self.device)
        has_any_data = False
        
        for b_idx, td in enumerate(target_data):
            for v in range(num_views):
                fov_v, rot_v, trans_v, aspect_v = None, None, None, None
                
                if td.get('cam_fov_per_view') is not None:
                    arr = td['cam_fov_per_view']
                    arr = np.array(arr) if not isinstance(arr, np.ndarray) else arr
                    if v < arr.shape[0]:
                        fov_v = float(arr[v].item() if hasattr(arr[v], 'item') else arr[v])
                        
                if td.get('cam_rot_per_view') is not None:
                    arr = td['cam_rot_per_view']
                    arr = np.array(arr) if not isinstance(arr, np.ndarray) else arr
                    if v < arr.shape[0]:
                        rot_v = arr[v]
                        
                if td.get('cam_trans_per_view') is not None:
                    arr = td['cam_trans_per_view']
                    arr = np.array(arr) if not isinstance(arr, np.ndarray) else arr
                    if v < arr.shape[0]:
                        trans_v = arr[v]
                        
                if td.get('cam_aspect_per_view') is not None:
                    arr = td['cam_aspect_per_view']
                    arr = np.array(arr) if not isinstance(arr, np.ndarray) else arr
                    if v < arr.shape[0]:
                        aspect_v = float(arr[v].item() if hasattr(arr[v], 'item') else arr[v])
                
                if fov_v is not None and rot_v is not None and trans_v is not None:
                    fov_all[b_idx, v] = fov_v
                    cam_rot_all[b_idx, v] = torch.from_numpy(np.array(rot_v)).to(dtype=torch.float32, device=self.device)
                    cam_trans_all[b_idx, v] = torch.from_numpy(np.array(trans_v)).to(dtype=torch.float32, device=self.device)
                    if aspect_v is not None:
                        aspect_all[b_idx, v] = aspect_v
                    mask_all[b_idx, v] = True
                    has_any_data = True
        
        if not has_any_data:
            return None
            
        return {
            'fov': fov_all,           # (B, V)
            'cam_rot': cam_rot_all,   # (B, V, 3, 3)
            'cam_trans': cam_trans_all,  # (B, V, 3)
            'aspect_ratio': aspect_all,  # (B, V)
            'mask': mask_all          # (B, V)
        }
    
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
                global_rots.append(torch.zeros(3, dtype=torch.float32, device=self.device))
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
                joint_rots.append(torch.zeros(config.N_POSE, 3, dtype=torch.float32, device=self.device))
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
                betas.append(torch.zeros(config.N_BETAS, dtype=torch.float32, device=self.device))
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
                trans.append(torch.zeros(3, dtype=torch.float32, device=self.device))
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
                        visibility.append(torch.ones(kp_2d.shape[1], dtype=torch.float32, device=self.device))
                    has_data = True
                elif len(kp_2d.shape) == 2 and view_idx == 0:
                    # Single-view data, only valid for view_idx 0
                    keypoints.append(safe_to_tensor(kp_2d, device=self.device))
                    if kp_vis is not None:
                        visibility.append(safe_to_tensor(kp_vis, device=self.device))
                    else:
                        visibility.append(torch.ones(kp_2d.shape[0], dtype=torch.float32, device=self.device))
                    has_data = True
                else:
                    # No data for this view
                    n_joints = kp_2d.shape[-2] if len(kp_2d.shape) >= 2 else len(config.CANONICAL_MODEL_JOINTS)
                    keypoints.append(torch.zeros(n_joints, 2, dtype=torch.float32, device=self.device))
                    visibility.append(torch.zeros(n_joints, dtype=torch.float32, device=self.device))
            else:
                # No keypoint data
                n_joints = len(config.CANONICAL_MODEL_JOINTS)
                keypoints.append(torch.zeros(n_joints, 2, dtype=torch.float32, device=self.device))
                visibility.append(torch.zeros(n_joints, dtype=torch.float32, device=self.device))
        
        if not has_data:
            return None
        
        return {
            'keypoints_2d': torch.stack(keypoints),  # (batch_size, n_joints, 2)
            'visibility': torch.stack(visibility)  # (batch_size, n_joints)
        }
    
    def _render_keypoints_with_camera(self, body_params: Dict[str, torch.Tensor],
                                       fov: torch.Tensor,
                                       cam_rot: torch.Tensor,
                                       cam_trans: torch.Tensor,
                                       aspect_ratio: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        
        # Handle scale/trans parameters - apply when mode is 'separate' or 'entangled_with_betas'
        # This ensures scales and translations are used when computing 2D/3D keypoint losses
        # in modes where they are predicted, allowing the model to learn them implicitly from supervision signals
        betas_logscale = None
        betas_trans_val = None
        
        if 'log_beta_scales' in body_params and body_params['log_beta_scales'] is not None:
            if self.scale_trans_mode == 'separate':
                # Check if using PCA or per-joint values
                scale_trans_config = TrainingConfig.get_scale_trans_config()
                use_pca_transformation = scale_trans_config.get('separate', {}).get('use_pca_transformation', True)
                
                if use_pca_transformation:
                    # PCA weights - convert to per-joint values
                    try:
                        scale_weights = body_params['log_beta_scales']
                        trans_weights = body_params.get('betas_trans', None)
                        log_beta_scales_joint, betas_trans_joint = self._transform_separate_pca_weights_to_joint_values(
                            scale_weights, trans_weights
                        )
                        betas_logscale = log_beta_scales_joint
                        betas_trans_val = betas_trans_joint
                    except Exception as e:
                        # If conversion fails (e.g., missing PCA dirs), log and fall back
                        print(f"Warning: Failed to convert PCA weights to joint values in _render_keypoints_with_camera: {e}")
                        # Fall back to None - SMAL model will use default (no scaling)
                        betas_logscale = None
                        betas_trans_val = None
                else:
                    # Already per-joint values - use directly
                    betas_logscale = body_params['log_beta_scales']  # (batch_size, n_joints, 3)
                    betas_trans_val = body_params.get('betas_trans', None)  # (batch_size, n_joints, 3)
            elif self.scale_trans_mode == 'entangled_with_betas':
                # In entangled mode, values are already per-joint and should be applied
                betas_logscale = body_params['log_beta_scales']
                betas_trans_val = body_params.get('betas_trans', None)
            # If mode is 'ignore', scales/translations are not applied (betas_logscale and betas_trans_val remain None)
        
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
        elif self.allow_mesh_scaling and 'mesh_scale' in body_params:
            # Apply predicted mesh scale - centers at root, scales, then translates
            mesh_scale = body_params['mesh_scale']  # (batch_size, 1)
            root_joint = joints[:, 0:1, :]  # (batch_size, 1, 3)
            verts = (verts - root_joint) * mesh_scale.unsqueeze(-1) + body_params['trans'].unsqueeze(1)
            joints = (joints - root_joint) * mesh_scale.unsqueeze(-1) + body_params['trans'].unsqueeze(1)
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
            return torch.zeros(batch_size, n_canonical, 2, dtype=torch.float32, device=self.device)
        
        canonical_joints = joints[:, config.CANONICAL_MODEL_JOINTS]
        
        # Set camera parameters for rendering
        self.renderer.set_camera_parameters(R=cam_rot, T=cam_trans, fov=fov, aspect_ratio=aspect_ratio)
        
        # Render joints
        faces_tensor = self.smal_model.faces
        if faces_tensor.dim() == 2:
            faces_batch = faces_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            faces_batch = faces_tensor

        # Defensive: SMAL model / buffers can be float64 depending on how it was loaded.
        # Renderer + PyTorch3D ops expect float32.
        verts = verts.float()
        canonical_joints = canonical_joints.float()
        faces_batch = faces_batch.long()

        _, rendered_joints_raw = self.renderer(verts, canonical_joints, faces_batch)
        
        # Normalize rendered joints using the actual renderer image size
        # This ensures consistency between training loss and visualization
        rendered_image_size = float(self.renderer.image_size)
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

    def _compute_batched_keypoint_loss(self,
                                        rendered_joints: torch.Tensor,
                                        target_keypoints: torch.Tensor,
                                        visibility: torch.Tensor,
                                        view_mask: Optional[torch.Tensor] = None,
                                        joint_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute visibility-weighted 2D keypoint loss for all views at once (batched).
        
        This is the optimized batched version of _compute_visibility_weighted_keypoint_loss.
        
        Args:
            rendered_joints: Rendered 2D joints (B, V, J, 2)
            target_keypoints: Target 2D keypoints (B, V, J, 2)
            visibility: Joint visibility (B, V, J)
            view_mask: Optional mask for valid views (B, V)
            joint_weights: Optional per-joint importance weights (J,)
                          Higher values = more importance for that joint
            
        Returns:
            Visibility-weighted keypoint loss averaged over all valid joints and views
        """
        eps = 1e-8
        B, V, J, _ = rendered_joints.shape
        
        # Sanitize rendered joints
        rendered_joints = torch.nan_to_num(rendered_joints, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create valid mask: visible and in bounds
        visible_mask = visibility.bool()  # (B, V, J)
        gt_in_bounds = (target_keypoints >= 0.0) & (target_keypoints <= 1.0)
        gt_in_bounds = gt_in_bounds.all(dim=-1)  # (B, V, J)
        
        valid_joint_mask = visible_mask & gt_in_bounds  # (B, V, J)
        
        # Apply view mask if provided
        if view_mask is not None:
            # Expand view mask to joint level: (B, V) -> (B, V, J)
            valid_joint_mask = valid_joint_mask & view_mask.unsqueeze(-1)
        
        if not valid_joint_mask.any():
            return torch.tensor(eps, device=self.device, requires_grad=True)
        
        # Compute squared error: (B, V, J, 2)
        diff_squared = (rendered_joints - target_keypoints) ** 2
        
        # Weight by validity: (B, V, J, 1)
        weights = valid_joint_mask.float().unsqueeze(-1)
        
        # Apply joint importance weights if provided
        if joint_weights is not None:
            # Expand joint weights: (J,) -> (1, 1, J, 1) for broadcasting
            joint_weights_expanded = joint_weights.view(1, 1, -1, 1)
            weights = weights * joint_weights_expanded
        
        weighted_diff = diff_squared * weights
        
        # Total loss: sum over all dimensions, divide by weighted valid count
        # When using joint weights, normalize by weighted sum to keep loss magnitude stable
        if joint_weights is not None:
            # Weighted normalization: sum of (valid_mask * joint_weights) * 2 for x,y
            weighted_valid_count = (valid_joint_mask.float() * joint_weights.view(1, 1, -1)).sum() * 2 + eps
        else:
            weighted_valid_count = valid_joint_mask.sum().float() * 2 + eps
        
        total_loss = weighted_diff.sum() / weighted_valid_count
        
        return total_loss + eps
    
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
                    dummy_img = torch.zeros(3, 224, 224, dtype=torch.float32, device=self.device)
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
        predicted_params = self.forward_multiview(
            images_per_view, camera_indices, view_mask, target_data=y_data_batch
        )
        
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
    data_batch = torch.zeros(batch_size, 3, input_resolution, input_resolution, dtype=torch.float32, device=device)
    
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

