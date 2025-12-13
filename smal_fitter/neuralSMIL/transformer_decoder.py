"""
Transformer Decoder Head for SMIL Image Regressor

This module implements a transformer decoder-based regression head inspired by AniMer,
providing an alternative to the standard MLP regression head with cross-attention
and iterative error feedback capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import config
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d


class CrossAttention(nn.Module):
    """Cross-attention module for attending to spatial features."""
    
    def __init__(self, dim: int, context_dim: Optional[int] = None, heads: int = 8, 
                 dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        context_dim = context_dim if context_dim is not None else dim
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        context = context if context is not None else x
        
        # Get query, key, value
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(q.shape[0], q.shape[1], self.heads, -1).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.heads, -1).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.heads, -1).transpose(1, 2)
        
        # Compute attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(out.shape[0], out.shape[2], -1)
        
        return self.to_out(out)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with cross-attention."""
    
    def __init__(self, dim: int, context_dim: Optional[int] = None, heads: int = 8,
                 dim_head: int = 64, mlp_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, context_dim, heads, dim_head, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Cross-attention
        x = x + self.cross_attn(self.norm1(x), context)
        
        # Feed-forward
        x = x + self.ff(self.norm2(x))
        
        return x


class SMILTransformerDecoderHead(nn.Module):
    """
    Transformer decoder head for SMIL parameter regression.
    
    This head uses cross-attention to attend to spatial features from the backbone
    and implements iterative error feedback for progressive refinement of predictions.
    """
    
    def __init__(self, feature_dim: int, context_dim: int, hidden_dim: int = 1024,
                 depth: int = 6, heads: int = 8, dim_head: int = 64, 
                 mlp_dim: int = 1024, dropout: float = 0.0, 
                 ief_iters: int = 3, rotation_representation: str = 'axis_angle',
                 scales_scale_factor: float = 0.01, trans_scale_factor: float = 0.01,
                 scale_trans_mode: str = 'separate'):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.ief_iters = ief_iters
        self.rotation_representation = rotation_representation
        self.scales_scale_factor = scales_scale_factor
        self.trans_scale_factor = trans_scale_factor
        self.scale_trans_mode = scale_trans_mode
        
        # Calculate output dimensions for SMIL parameters
        self._calculate_output_dims()
        
        # Token embedding for parameter tokens
        self.token_embedding = nn.Linear(1, hidden_dim)  # Start with zero tokens
        
        # Positional embedding for tokens
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                dim=hidden_dim,
                context_dim=context_dim,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Output heads for different parameter types
        # Note: pose includes both global and joint rotations
        total_pose_dim = self.global_rot_dim + self.joint_rot_dim
        self.pose_head = nn.Linear(hidden_dim, total_pose_dim)
        self.betas_head = nn.Linear(hidden_dim, self.betas_dim)
        self.trans_head = nn.Linear(hidden_dim, self.trans_dim)
        self.fov_head = nn.Linear(hidden_dim, self.fov_dim)
        self.cam_rot_head = nn.Linear(hidden_dim, self.cam_rot_dim)
        self.cam_trans_head = nn.Linear(hidden_dim, self.cam_trans_dim)
        
        # Optional joint scales and translations
        if self.scales_dim > 0:
            self.scales_head = nn.Linear(hidden_dim, self.scales_dim)
        if self.joint_trans_dim > 0:
            self.joint_trans_head = nn.Linear(hidden_dim, self.joint_trans_dim)
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Initialize prediction buffers
        self._initialize_prediction_buffers()
    
    def _calculate_output_dims(self):
        """Calculate output dimensions for SMIL parameters."""
        if self.rotation_representation == '6d':
            self.global_rot_dim = 6
            self.joint_rot_dim = config.N_POSE * 6
        else:  # axis_angle
            self.global_rot_dim = 3
            self.joint_rot_dim = config.N_POSE * 3
        
        self.betas_dim = config.N_BETAS
        self.trans_dim = 3
        self.fov_dim = 1
        self.cam_rot_dim = 9  # 3x3 rotation matrix (flattened)
        self.cam_trans_dim = 3
        
        # Handle scale and translation dimensions based on mode
        self.scales_dim = 0
        self.joint_trans_dim = 0
        
        if self.scale_trans_mode == 'entangled_with_betas':
            # In entangled mode, scales and trans are derived from betas via PCA
            # No separate prediction heads needed
            self.scales_dim = 0
            self.joint_trans_dim = 0
        elif self.scale_trans_mode == 'separate':
            # In separate mode, predict PCA weights directly (same dimension as betas)
            self.scales_dim = config.N_BETAS  # PCA weights for scaling
            self.joint_trans_dim = config.N_BETAS  # PCA weights for translation
        elif self.scale_trans_mode == 'ignore':
            # In ignore mode, no scales or translations
            self.scales_dim = 0
            self.joint_trans_dim = 0
        else:
            # Fallback for legacy code - check config.ignore_hardcoded_body
            if config.ignore_hardcoded_body:
                n_joints = config.N_POSE + 1
                self.scales_dim = n_joints * 3
                self.joint_trans_dim = n_joints * 3
    
    def _initialize_parameters(self):
        """Initialize transformer decoder parameters."""
        # Initialize token embedding
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.constant_(self.token_embedding.bias, 0)
        
        # Initialize positional embedding
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        # Initialize output heads with very small weights for IEF stability
        for head in [self.pose_head, self.betas_head, self.trans_head, 
                    self.fov_head, self.cam_rot_head, self.cam_trans_head]:
            nn.init.xavier_uniform_(head.weight, gain=0.001)  # Even smaller gain
            nn.init.constant_(head.bias, 0)
        
        if self.scales_dim > 0:
            nn.init.xavier_uniform_(self.scales_head.weight, gain=0.001)
            nn.init.constant_(self.scales_head.bias, 0)
        
        if self.joint_trans_dim > 0:
            nn.init.xavier_uniform_(self.joint_trans_head.weight, gain=0.001)
            nn.init.constant_(self.joint_trans_head.bias, 0)
    
    def _initialize_prediction_buffers(self):
        """Initialize prediction buffers for IEF."""
        # Initialize with zeros for IEF
        # Note: pose includes both global and joint rotations
        total_pose_dim = self.global_rot_dim + self.joint_rot_dim
        self.register_buffer('init_pose', torch.zeros(1, total_pose_dim))
        self.register_buffer('init_betas', torch.zeros(1, self.betas_dim))
        self.register_buffer('init_trans', torch.zeros(1, self.trans_dim))
        # Initialize FOV with reasonable value based on ground truth (typically around 8 degrees)
        self.register_buffer('init_fov', torch.tensor([[8.0]]))
        self.register_buffer('init_cam_rot', torch.eye(3).flatten().unsqueeze(0))
        # Initialize camera translation with reasonable values based on ground truth range
        # Ground truth camera translation is typically around [0, 0, 100-150]
        self.register_buffer('init_cam_trans', torch.tensor([[0.0, 0.0, 100.0]]))
        
        if self.scales_dim > 0:
            self.register_buffer('init_scales', torch.zeros(1, self.scales_dim))
        if self.joint_trans_dim > 0:
            self.register_buffer('init_joint_trans', torch.zeros(1, self.joint_trans_dim))
    
    def forward(self, features: torch.Tensor, spatial_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through transformer decoder head.
        
        Args:
            features: Global features from backbone (batch_size, feature_dim)
            spatial_features: Spatial features from backbone (batch_size, seq_len, context_dim)
            
        Returns:
            Dictionary containing predicted SMIL parameters
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Initialize predictions
        pred_pose = self.init_pose.expand(batch_size, -1).to(device)
        pred_betas = self.init_betas.expand(batch_size, -1).to(device)
        pred_trans = self.init_trans.expand(batch_size, -1).to(device)
        pred_fov = self.init_fov.expand(batch_size, -1).to(device)
        pred_cam_rot = self.init_cam_rot.expand(batch_size, -1).to(device)
        pred_cam_trans = self.init_cam_trans.expand(batch_size, -1).to(device)
        
        if self.scales_dim > 0:
            pred_scales = self.init_scales.expand(batch_size, -1).to(device)
        if self.joint_trans_dim > 0:
            pred_joint_trans = self.init_joint_trans.expand(batch_size, -1).to(device)
        
        # Store predictions for each iteration
        pred_pose_list = []
        pred_betas_list = []
        pred_trans_list = []
        pred_fov_list = []
        pred_cam_rot_list = []
        pred_cam_trans_list = []
        
        if self.scales_dim > 0:
            pred_scales_list = []
        if self.joint_trans_dim > 0:
            pred_joint_trans_list = []
        
        # Iterative Error Feedback (IEF)
        for i in range(self.ief_iters):
            # Create parameter token (concatenate all current predictions)
            param_tokens = [pred_pose, pred_betas, pred_trans, pred_fov, pred_cam_rot, pred_cam_trans]
            if self.scales_dim > 0:
                param_tokens.append(pred_scales)
            if self.joint_trans_dim > 0:
                param_tokens.append(pred_joint_trans)
            
            # Use a single token representing the current state
            token = torch.zeros(batch_size, 1, 1).to(device)
            
            # Embed token
            token = self.token_embedding(token)
            token = token + self.pos_embedding
            
            # Pass through transformer decoder layers
            for layer in self.layers:
                token = layer(token, spatial_features)
            
            # Extract predictions (residual updates)
            token_out = token.squeeze(1)  # (batch_size, hidden_dim)
            
            # Apply residual updates
            pred_pose = pred_pose + self.pose_head(token_out)
            pred_betas = pred_betas + self.betas_head(token_out)
            pred_trans = pred_trans + self.trans_head(token_out)
            pred_fov = pred_fov + self.fov_head(token_out)
            pred_cam_rot = pred_cam_rot + self.cam_rot_head(token_out)
            pred_cam_trans = pred_cam_trans + self.cam_trans_head(token_out)
            
            # Check for NaN values after each iteration
            if not torch.isfinite(pred_pose).all():
                print(f"Warning: Non-finite values in pred_pose at iteration {i}: {pred_pose}")
                pred_pose = torch.zeros_like(pred_pose)
            if not torch.isfinite(pred_betas).all():
                print(f"Warning: Non-finite values in pred_betas at iteration {i}: {pred_betas}")
                pred_betas = torch.zeros_like(pred_betas)
            if not torch.isfinite(pred_trans).all():
                print(f"Warning: Non-finite values in pred_trans at iteration {i}: {pred_trans}")
                pred_trans = torch.zeros_like(pred_trans)
            if not torch.isfinite(pred_fov).all():
                print(f"Warning: Non-finite values in pred_fov at iteration {i}: {pred_fov}")
                pred_fov = torch.tensor([[0.9]], device=pred_fov.device, dtype=pred_fov.dtype).expand_as(pred_fov)
            if not torch.isfinite(pred_cam_rot).all():
                print(f"Warning: Non-finite values in pred_cam_rot at iteration {i}: {pred_cam_rot}")
                pred_cam_rot = torch.eye(3, device=pred_cam_rot.device, dtype=pred_cam_rot.dtype).flatten().unsqueeze(0).expand_as(pred_cam_rot)
            if not torch.isfinite(pred_cam_trans).all():
                print(f"Warning: Non-finite values in pred_cam_trans at iteration {i}: {pred_cam_trans}")
                pred_cam_trans = torch.zeros_like(pred_cam_trans)
            
            if self.scales_dim > 0:
                pred_scales = pred_scales + self.scales_head(token_out) * self.scales_scale_factor
                if not torch.isfinite(pred_scales).all():
                    print(f"Warning: Non-finite values in pred_scales at iteration {i}: {pred_scales}")
                    pred_scales = torch.zeros_like(pred_scales)
            if self.joint_trans_dim > 0:
                pred_joint_trans = pred_joint_trans + self.joint_trans_head(token_out) * self.trans_scale_factor
                if not torch.isfinite(pred_joint_trans).all():
                    print(f"Warning: Non-finite values in pred_joint_trans at iteration {i}: {pred_joint_trans}")
                    pred_joint_trans = torch.zeros_like(pred_joint_trans)
            
            # Store predictions for this iteration
            pred_pose_list.append(pred_pose.clone())
            pred_betas_list.append(pred_betas.clone())
            pred_trans_list.append(pred_trans.clone())
            pred_fov_list.append(pred_fov.clone())
            pred_cam_rot_list.append(pred_cam_rot.clone())
            pred_cam_trans_list.append(pred_cam_trans.clone())
            
            if self.scales_dim > 0:
                pred_scales_list.append(pred_scales.clone())
            if self.joint_trans_dim > 0:
                pred_joint_trans_list.append(pred_joint_trans.clone())
        
        # Convert pose predictions to proper format
        if self.rotation_representation == '6d':
            # For 6D representation, keep as 6D vectors (don't convert to matrices)
            global_rot_6d = pred_pose[:, :self.global_rot_dim]
            joint_rot_6d = pred_pose[:, self.global_rot_dim:]
            
            pred_global_rot = global_rot_6d
            pred_joint_rot = joint_rot_6d.view(batch_size, config.N_POSE, 6)
        else:
            # Axis-angle format
            global_rot_aa = pred_pose[:, :self.global_rot_dim]
            joint_rot_aa = pred_pose[:, self.global_rot_dim:].view(batch_size, config.N_POSE, 3)
            
            pred_global_rot = global_rot_aa
            pred_joint_rot = joint_rot_aa
        
        # Reshape camera rotation to 3x3 matrix
        pred_cam_rot_mat = pred_cam_rot.view(batch_size, 3, 3)
        
        # Prepare output dictionary
        output = {
            'global_rot': pred_global_rot,
            'joint_rot': pred_joint_rot,
            'betas': pred_betas,
            'trans': pred_trans,
            'fov': pred_fov,
            'cam_rot': pred_cam_rot_mat,
            'cam_trans': pred_cam_trans,
        }
        
        # Debug: Print tensor shapes and values occasionally
        if hasattr(self, '_debug_shapes') and torch.rand(1).item() < 0.01:
            print(f"DEBUG - Transformer decoder output shapes:")
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                    if key in ['log_beta_scales', 'betas_trans']:
                        print(f"    {key} values: min={value.min().item():.6f}, max={value.max().item():.6f}, mean={value.mean().item():.6f}")
                        print(f"    {key} scale factor: {self.scales_scale_factor if key == 'log_beta_scales' else self.trans_scale_factor}")
        
        if self.scales_dim > 0:
            output['log_beta_scales'] = pred_scales.view(batch_size, -1, 3)
        if self.joint_trans_dim > 0:
            output['betas_trans'] = pred_joint_trans.view(batch_size, -1, 3)
        
        # Store iteration history for analysis
        output['iteration_history'] = {
            'pose': pred_pose_list,
            'betas': pred_betas_list,
            'trans': pred_trans_list,
            'fov': pred_fov_list,
            'cam_rot': pred_cam_rot_list,
            'cam_trans': pred_cam_trans_list,
        }
        
        if self.scales_dim > 0:
            output['iteration_history']['scales'] = pred_scales_list
        if self.joint_trans_dim > 0:
            output['iteration_history']['joint_trans'] = pred_joint_trans_list
        
        return output


def build_smil_transformer_decoder_head(feature_dim: int, context_dim: int, 
                                       hidden_dim: int = 1024, depth: int = 6,
                                       heads: int = 8, dim_head: int = 64,
                                       mlp_dim: int = 1024, dropout: float = 0.0,
                                       ief_iters: int = 3, 
                                       rotation_representation: str = 'axis_angle',
                                       scales_scale_factor: float = 0.01,
                                       trans_scale_factor: float = 0.01,
                                       scale_trans_mode: str = 'separate') -> SMILTransformerDecoderHead:
    """
    Build a SMIL transformer decoder head.
    
    Args:
        feature_dim: Dimension of global features from backbone
        context_dim: Dimension of spatial features from backbone
        hidden_dim: Hidden dimension of transformer
        depth: Number of transformer decoder layers
        heads: Number of attention heads
        dim_head: Dimension per attention head
        mlp_dim: MLP hidden dimension
        dropout: Dropout rate
        ief_iters: Number of iterative error feedback iterations
        rotation_representation: '6d' or 'axis_angle'
        scales_scale_factor: Scaling factor for log_beta_scales predictions (default: 0.01)
        trans_scale_factor: Scaling factor for betas_trans predictions (default: 0.01)
        scale_trans_mode: Mode for handling scale and translation betas ('ignore', 'separate', 'entangled_with_betas')
        
    Returns:
        SMILTransformerDecoderHead instance
    """
    return SMILTransformerDecoderHead(
        feature_dim=feature_dim,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        mlp_dim=mlp_dim,
        dropout=dropout,
        ief_iters=ief_iters,
        rotation_representation=rotation_representation,
        scales_scale_factor=scales_scale_factor,
        trans_scale_factor=trans_scale_factor,
        scale_trans_mode=scale_trans_mode
    )
