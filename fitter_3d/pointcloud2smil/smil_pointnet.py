"""
SMIL PointNet: A lightweight end-to-end mesh registration approach.

This module implements a PointNet-style neural network to estimate SMIL parameters
from a point cloud, enabling direct mesh registration from point cloud data.
The model is trained on randomly sampled SMIL configurations and their resulting point clouds.

References:
- PointNet: https://arxiv.org/pdf/1612.00593
- PointNet implementation: https://github.com/fxia22/pointnet.pytorch
- PointNet++: https://arxiv.org/pdf/1706.02413
- PointNet++ implementation: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, rotation_6d_to_matrix, matrix_to_rotation_6d

# Add the parent directory to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import config
from fitter_3d.pointcloud2smil.sample_smil_model import load_smil_model, generate_random_parameters

# Import PointNet++ utilities
from fitter_3d.pointcloud2smil.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction



# ----------------------- PARAMETER NORMALIZATION UTILS ----------------------- #

def compute_param_stats(parameters_list, keys):
    """
    Compute mean and std for each parameter group from a list of parameter dicts.
    Args:
        parameters_list: List of parameter dicts (from dataset)
        keys: List of parameter keys to compute stats for
    Returns:
        stats: Dict of {key: {'mean': tensor, 'std': tensor}}
    """
    stats = {}
    for key in keys:
        # Stack all values for this key (do NOT flatten)
        values = [p[key] for p in parameters_list]
        values = torch.stack(values)
        stats[key] = {
            'mean': values.mean(dim=0),
            'std': values.std(dim=0) + 1e-8  # avoid div by zero
        }
    return stats

def normalize_params(params, stats):
    """
    Normalize parameter dict using provided stats.
    Args:
        params: Dict of parameter tensors
        stats: Dict of {key: {'mean': tensor, 'std': tensor}}
    Returns:
        Dict of normalized parameter tensors
    """
    normed = {}
    for key in params:
        if key in stats:
            normed[key] = (params[key] - stats[key]['mean'].to(params[key].device)) / stats[key]['std'].to(params[key].device)
        else:
            normed[key] = params[key]
    return normed

def denormalize_params(normed_params, stats):
    """
    Denormalize parameter dict using provided stats.
    Args:
        normed_params: Dict of normalized parameter tensors
        stats: Dict of {key: {'mean': tensor, 'std': tensor}}
    Returns:
        Dict of denormalized parameter tensors
    """
    denormed = {}
    for key in normed_params:
        if key in stats:
            denormed[key] = normed_params[key] * stats[key]['std'].to(normed_params[key].device) + stats[key]['mean'].to(normed_params[key].device)
        else:
            denormed[key] = normed_params[key]
    return denormed


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
    # Ensure the matrix is on the correct device and dtype for eye
    identity = torch.eye(3, device=matrix.device, dtype=matrix.dtype).unsqueeze(0).expand_as(matrix)
    # Check if matrix is close to identity
    # If matrix is identity, axis_angle is (0,0,0). matrix_to_axis_angle might produce NaNs or unstable gradients.
    # A common way to handle this is to return a zero vector for identity matrices.
    # However, for simplicity and to rely on pytorch3d's handling, we'll use it directly.
    # Users should be aware of potential issues if many rotations are identities.
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
    # matrix_to_rotation_6d from PyTorch3D directly performs this.
    return matrix_to_rotation_6d(rotation_matrix)

def rotation_6d_to_axis_angle(rotation_6d):
    """
    Converts 6D rotation representation back to axis-angle.
    Args:
        rotation_6d: Tensor of shape (..., 6)
    Returns:
        Tensor of shape (..., 3)
    """
    # rotation_6d_to_matrix from PyTorch3D handles the Gram-Schmidt process.
    rotation_matrix = rotation_6d_to_matrix(rotation_6d)
    return robust_matrix_to_axis_angle(rotation_matrix)


# ----------------------- POINTNET MODEL IMPLEMENTATION ----------------------- #

class TNet(nn.Module):
    """
    T-Net architecture for spatial transformation, as described in PointNet.
    It predicts a transformation matrix to align the input point cloud.
    """
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size()[0]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k*self.k).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        
        return x


class SMILPointNet(nn.Module):
    """
    PointNet-based architecture for SMIL parameter estimation.
    
    This network takes a 3D point cloud as input and outputs the SMIL parameters
    (joint rotations, shape parameters, and other relevant parameters).
    """
    def __init__(self, num_points=5000, n_betas=20, n_pose=34, include_scales=True, rotation_representation='6d'):
        """
        Initialize the SMILPointNet model.
        
        Args:
            num_points: Number of points in the input point cloud
            n_betas: Number of shape parameters in the SMIL model
            n_pose: Number of pose parameters in the SMIL model (excluding global rotation)
            include_scales: Whether to predict joint scales
            rotation_representation: '6d' or 'axis_angle' for joint rotations
        """
        super(SMILPointNet, self).__init__()
        
        self.num_points = num_points
        self.n_betas = n_betas
        self.n_pose = n_pose
        self.include_scales = include_scales
        self.rotation_representation = rotation_representation
        
        # Feature extraction network
        self.stn = TNet(k=3)  # Spatial transformer network for input point cloud
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Calculate total output size
        # - Global rotation (3 for axis-angle, 6 for 6D)
        # - Joint rotations (n_pose * 6 for 6D, n_pose * 3 for axis_angle)
        # - Shape parameters (n_betas)
        # - Translation (3)
        # - Joint scales (optional, different for each model)
        n_scales = 0  # Will be set later if include_scales=True
        
        if self.rotation_representation == '6d':
            self.global_rot_dim = 6
            self.joint_rot_dim = self.n_pose * 6
        else:  # axis_angle
            self.global_rot_dim = 3
            self.joint_rot_dim = self.n_pose * 3
            
        self.output_size = self.global_rot_dim + self.joint_rot_dim + n_betas + 3 + n_scales
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.output_size) # Initialize fc3
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(p=0.3)
        
    def set_joint_scales_size(self, n_joints):
        """
        Set the number of joint scales parameters based on the SMIL model.
        This must be called before using the model if include_scales=True.
        
        Args:
            n_joints: Number of joints in the SMIL model
        """
        if self.include_scales:
            n_scales = n_joints * 3  # 3 scale parameters per joint
            self.output_size = self.global_rot_dim + self.joint_rot_dim + self.n_betas + 3 + n_scales
            
            # Recreate the final layer with the right output size
            self.fc3 = nn.Linear(256, self.output_size).to(self.fc2.weight.device)

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x: Input point cloud of shape (batch_size, num_points, 3)
            
        Returns:
            SMIL parameters as a dictionary with keys:
                'global_rot', 'joint_rot', 'betas', 'trans', 'log_beta_scales' (if include_scales=True)
        """
        batch_size = x.size()[0]
        
        # Transpose to (batch_size, 3, num_points) for Conv1d layers
        x = x.transpose(2, 1)
        
        # Apply T-Net (input transform)
        trans = self.stn(x)
        x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)
        
        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Parse the output into different parameter groups
        params = {}
        
        # Global rotation (axis-angle or 6D)
        params['global_rot'] = x[:, :self.global_rot_dim]
        
        # Joint rotations (n_pose * 6 values for 6D representation or n_pose * 3 for axis_angle)
        joint_rot_start = self.global_rot_dim
        if self.rotation_representation == '6d':
            joint_rot_end = joint_rot_start + self.n_pose * 6
            params['joint_rot'] = x[:, joint_rot_start:joint_rot_end].reshape(batch_size, self.n_pose, 6)
        else: # axis_angle
            joint_rot_end = joint_rot_start + self.n_pose * 3
            params['joint_rot'] = x[:, joint_rot_start:joint_rot_end].reshape(batch_size, self.n_pose, 3)
        
        # Shape parameters (n_betas values)
        betas_start = joint_rot_end
        betas_end = betas_start + self.n_betas
        params['betas'] = x[:, betas_start:betas_end]
        
        # Translation (3 values)
        trans_start = betas_end
        trans_end = trans_start + 3
        params['trans'] = x[:, trans_start:trans_end]
        
        # Joint scales (optional, n_joints * 3 values)
        if self.include_scales:
            scales_start = trans_end
            n_scales = self.output_size - scales_start
            n_joints = n_scales // 3
            params['log_beta_scales'] = x[:, scales_start:].reshape(batch_size, n_joints, 3)
        
        return params


class SMILPointNet2(nn.Module):
    """
    PointNet++ based architecture for SMIL parameter estimation.
    
    This network takes a 3D point cloud as input and outputs the SMIL parameters
    (joint rotations, shape parameters, and other relevant parameters).
    It uses a PointNet++ (MSG) backbone for feature extraction.
    """
    def __init__(self, num_points=5000, n_betas=20, n_pose=34, include_scales=True, rotation_representation='6d'):
        """
        Initialize the SMILPointNet2 model.
        
        Args:
            num_points: Number of points in the input point cloud (not directly used by PointNet++ backbone, but kept for consistency)
            n_betas: Number of shape parameters in the SMIL model
            n_pose: Number of pose parameters in the SMIL model (excluding global rotation)
            include_scales: Whether to predict joint scales
            rotation_representation: '6d' or 'axis_angle' for joint rotations
        """
        super(SMILPointNet2, self).__init__()
        
        self.n_betas = n_betas
        self.n_pose = n_pose
        self.include_scales = include_scales
        self.rotation_representation = rotation_representation

        # PointNet++ (MSG) backbone
        # Assuming input point cloud has 3 channels (x, y, z) and no other features (e.g., normals)
        # in_channel for sa1 is 0 because we pass None for the 'points' (features) tensor,
        # and PointNetSetAbstraction will use the 3 xyz coordinates.
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], 0, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # Output of sa1 has 64+128+128 = 320 channels
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        # Output of sa2 has 128+256+256 = 640 channels
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        # Output of sa3 is a global feature of 1024 channels
        
        if self.rotation_representation == '6d':
            self.global_rot_dim = 6
            self.joint_rot_dim = self.n_pose * 6
        else: # axis_angle
            self.global_rot_dim = 3
            self.joint_rot_dim = self.n_pose * 3
            
        # Regression head for SMIL parameters (same as SMILPointNet)
        n_scales = 0 # Will be set later if include_scales=True
        self.output_size = self.global_rot_dim + self.joint_rot_dim + self.n_betas + 3 + n_scales # global_rot + joint_rot_dim + betas(n_betas) + trans(3)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.output_size) # Placeholder, will be updated by set_joint_scales_size if needed
        
        self.bn1_reg = nn.BatchNorm1d(512) # Renamed to avoid conflict if TNet bn names were similar
        self.bn2_reg = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(p=0.3)

    def set_joint_scales_size(self, n_joints):
        """
        Set the number of joint scales parameters based on the SMIL model.
        This must be called before using the model if include_scales=True.
        
        Args:
            n_joints: Number of joints in the SMIL model
        """
        if self.include_scales:
            n_scales = n_joints * 3  # 3 scale parameters per joint
            # global_rot + joint_rot_dim + betas(n_betas) + trans(3) + scales(n_scales)
            self.output_size = self.global_rot_dim + self.joint_rot_dim + self.n_betas + 3 + n_scales
            
            # Recreate the final layer with the right output size
            self.fc3 = nn.Linear(256, self.output_size).to(self.fc2.weight.device)

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x: Input point cloud of shape (batch_size, num_points, 3)
            
        Returns:
            SMIL parameters as a dictionary with keys:
                'global_rot', 'joint_rot', 'betas', 'trans', 'log_beta_scales' (if include_scales=True)
        """
        batch_size = x.size()[0]
        
        # Transpose to (batch_size, 3, num_points) for PointNet++ layers
        xyz = x.transpose(2, 1)
        norm = None # Assuming no input normals for SMIL data

        # PointNet++ backbone
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # l3_points is (batch_size, 1024, 1)
        
        # Global feature vector
        feat = l3_points.view(batch_size, 1024)
        
        # Fully connected layers for regression
        feat = F.relu(self.bn1_reg(self.fc1(feat)))
        feat = F.relu(self.bn2_reg(self.fc2(feat)))
        feat = self.dropout(feat)
        output_params = self.fc3(feat)
        
        # Parse the output into different parameter groups
        params = {}
        
        # Global rotation (axis-angle or 6D)
        params['global_rot'] = output_params[:, :self.global_rot_dim]
        
        # Joint rotations (n_pose * 6 or n_pose * 3 values for 6D or axis-angle representation)
        joint_rot_start = self.global_rot_dim
        if self.rotation_representation == '6d':
            joint_rot_end = joint_rot_start + self.n_pose * 6
            params['joint_rot'] = output_params[:, joint_rot_start:joint_rot_end].reshape(batch_size, self.n_pose, 6)
        else: # axis_angle
            joint_rot_end = joint_rot_start + self.n_pose * 3
            params['joint_rot'] = output_params[:, joint_rot_start:joint_rot_end].reshape(batch_size, self.n_pose, 3)
        
        # Shape parameters (n_betas values)
        betas_start = joint_rot_end
        betas_end = betas_start + self.n_betas
        params['betas'] = output_params[:, betas_start:betas_end]
        
        # Translation (3 values)
        trans_start = betas_end
        trans_end = trans_start + 3
        params['trans'] = output_params[:, trans_start:trans_end]
        
        # Joint scales (optional, n_joints * 3 values)
        if self.include_scales:
            scales_start = trans_end
            # Calculate n_scales based on remaining output size
            # This assumes set_joint_scales_size has been called to set self.output_size correctly
            current_n_scales = self.output_size - scales_start 
            if current_n_scales > 0 : # Ensure there are scale parameters
                n_joints = current_n_scales // 3
                if n_joints > 0 : # Ensure n_joints is valid
                    params['log_beta_scales'] = output_params[:, scales_start:].reshape(batch_size, n_joints, 3)
                else: # Handle case where scales are expected but n_joints is 0 (should not happen if configured correctly)
                    params['log_beta_scales'] = torch.empty(batch_size, 0, 3, device=output_params.device)
            else: # Handle case where scales are included but output_size doesn't account for them
                 params['log_beta_scales'] = torch.empty(batch_size, 0, 3, device=output_params.device)

        return params


# ----------------------- DATASET IMPLEMENTATION ----------------------- #

class SMILDataset(Dataset):
    """
    Dataset for SMIL parameter estimation from point clouds.
    
    Generates random SMIL configurations and their corresponding point clouds.
    Supports curriculum learning through customizable randomization scales.
    """
    def __init__(self, num_samples=1000, num_points=5000, device='cuda', shape_family=-1, seed=None, param_stats=None, normalize=True,
                 noise_std=0.01, dropout_prob=0.1, augment=True, rotation_representation='6d', exclude_rot_from_norm=False,
                 shape_scale=2.0, pose_scale=0.25, trans_scale=0.01, scale_scale=0.25, global_rot_scale=0.0):
        """
        Initialize the dataset.
        
        Args:
            num_samples: Number of samples in the dataset
            num_points: Number of points to sample from each mesh
            device: Device to use for computation (primarily for the main process, workers will use CPU for generation)
            shape_family: Shape family ID to use for the SMIL model
            seed: Random seed for reproducibility
            param_stats: Optional parameter statistics for normalization
            normalize: Whether to normalize parameters (default: True)
            noise_std: Standard deviation of Gaussian noise to add (default: 0.01)
            dropout_prob: Probability of dropping points (default: 0.1)
            augment: Whether to apply augmentation during training (default: True)
            rotation_representation: '6d' or 'axis_angle' for joint rotations
            exclude_rot_from_norm: Whether to exclude rotations from normalization (default: False)
            shape_scale: Scale for shape parameter randomization (default: 2.0)
            pose_scale: Scale for joint rotation randomization (default: 0.25)
            trans_scale: Scale for translation randomization (default: 0.01)
            scale_scale: Scale for joint scaling randomization (default: 0.25)
            global_rot_scale: Scale for global rotation randomization (default: 0.0)
        """
        self.num_samples = num_samples
        self.num_points = num_points
        self.device = device # Stored, but smal_fitter for generation will be on CPU
        self.shape_family = shape_family
        self.param_stats = param_stats
        self.normalize = normalize
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.augment = augment
        self.seed = seed
        self.rotation_representation = rotation_representation
        self.exclude_rot_from_norm = exclude_rot_from_norm
        
        # Curriculum learning parameters
        self.shape_scale = shape_scale
        self.pose_scale = pose_scale
        self.trans_scale = trans_scale
        self.scale_scale = scale_scale
        self.global_rot_scale = global_rot_scale
        
        # Load the SMIL model on CPU for data generation to avoid issues with DataLoader workers
        self.smal_fitter = load_smil_model(batch_size=1, device='cpu')
        
        # Store model configuration for later use
        self.n_betas = self.smal_fitter.n_betas
        self.n_pose = config.N_POSE
        self.n_joints = self.smal_fitter.n_joints
        
        # Generate and store dataset
        self.point_clouds = []
        self.parameters = []
        self.normalized_parameters = []
        
        print(f"Generating {self.num_samples} random SMIL configurations and point clouds...")
        print(f"Randomization scales - Shape: {self.shape_scale}, Pose: {self.pose_scale}, Trans: {self.trans_scale}, Scale: {self.scale_scale}, Global Rot: {self.global_rot_scale}")
        self.generate_dataset()

    def update_randomization_scales(self, shape_scale=None, pose_scale=None, trans_scale=None, 
                                   scale_scale=None, global_rot_scale=None):
        """
        Update randomization scales for curriculum learning.
        
        Args:
            shape_scale: New scale for shape parameter randomization
            pose_scale: New scale for joint rotation randomization
            trans_scale: New scale for translation randomization
            scale_scale: New scale for joint scaling randomization
            global_rot_scale: New scale for global rotation randomization
        """
        if shape_scale is not None:
            self.shape_scale = shape_scale
        if pose_scale is not None:
            self.pose_scale = pose_scale
        if trans_scale is not None:
            self.trans_scale = trans_scale
        if scale_scale is not None:
            self.scale_scale = scale_scale
        if global_rot_scale is not None:
            self.global_rot_scale = global_rot_scale
            
        print(f"Updated randomization scales - Shape: {self.shape_scale}, Pose: {self.pose_scale}, Trans: {self.trans_scale}, Scale: {self.scale_scale}, Global Rot: {self.global_rot_scale}")

    def generate_dataset(self, regenerate=False):
        # Clear existing data if regenerating
        if regenerate:
            self.point_clouds = []
            self.parameters = []
            self.normalized_parameters = []
            print("Regenerating dataset...")
        
        # Set random seed if provided
        if self.seed is not None:
            if regenerate:
                print("Regenerating dataset...")
                self.seed += 3 # so the seed is never the same as for the training or validation set
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        print(f"Generating {self.num_samples} samples with scales: "
              f"shape={self.shape_scale}, pose={self.pose_scale}, "
              f"trans={self.trans_scale}, scale={self.scale_scale}, "
              f"global_rot={self.global_rot_scale}")

        for i in tqdm(range(self.num_samples)):
            # Generate random parameters with current scales
            generate_random_parameters(self.smal_fitter, 
                                       seed=self.seed + i if self.seed is not None else None,
                                       random_dist="uniform",
                                       shape_scale=self.shape_scale,
                                       pose_scale=self.pose_scale,
                                       trans_scale=self.trans_scale,
                                       scale_scale=self.scale_scale,
                                       global_rot_scale=self.global_rot_scale)
            
            # Forward pass to get vertices
            with torch.no_grad():
                verts, joints = self.smal_fitter.forward(
                    betas=self.smal_fitter.betas,
                    global_rot=self.smal_fitter.global_rot,
                    joint_rot=self.smal_fitter.joint_rot,
                    trans=self.smal_fitter.trans,
                    log_beta_scales=self.smal_fitter.log_beta_scales,
                    deform_verts=self.smal_fitter.deform_verts,
                    return_joints=True # Explicitly request joints
                )
            
            # Get faces from the model
            faces = self.smal_fitter.faces
            
            # Create a mesh object for sampling
            mesh = Meshes(verts=verts, faces=faces)
            
            # Sample points from the mesh surface
            point_cloud = sample_points_from_meshes(mesh, num_samples=self.num_points)
            
            # Store the point cloud (explicitly detach)
            self.point_clouds.append(point_cloud[0].cpu().detach())
            
            # Store the parameters (explicitly detach each tensor)
            global_rot_aa = self.smal_fitter.global_rot[0].cpu().detach() # Axis-Angle Shape: (1, 3)
            joint_rot_aa = self.smal_fitter.joint_rot[0].cpu().detach() # Axis-Angle Shape: (N_POSE, 3)
            
            if self.rotation_representation == '6d':
                global_rot_param = axis_angle_to_rotation_6d(global_rot_aa) # 6D rot matrix Shape: (1,6)
                joint_rot_param = axis_angle_to_rotation_6d(joint_rot_aa) # 6D rot matrix Shape: (N_POSE, 6)
            else: # axis_angle
                global_rot_param = global_rot_aa
                joint_rot_param = joint_rot_aa
                
            params = {
                'global_rot': global_rot_param, # Store potentially 6D global_rot
                'joint_rot': joint_rot_param,
                'betas': self.smal_fitter.betas[0].cpu().detach(),
                'trans': self.smal_fitter.trans[0].cpu().detach(),
                'log_beta_scales': self.smal_fitter.log_beta_scales[0].cpu().detach(),
                'joints': joints[0].cpu().detach()
            }
            self.parameters.append(params)
        
        # Compute stats if not provided
        if self.param_stats is None:
            keys = ['global_rot', 'joint_rot', 'betas', 'trans', 'log_beta_scales', 'joints']
            if self.exclude_rot_from_norm:
                keys_to_exclude_from_stats = ['global_rot', 'joint_rot']
                keys = [k for k in keys if k not in keys_to_exclude_from_stats]
            self.param_stats = compute_param_stats(self.parameters, keys)
        
        # Normalize all parameters if requested
        if self.normalize:
            for params in self.parameters:
                normed = normalize_params(params, self.param_stats)
                self.normalized_parameters.append(normed)
        else:
            # If not normalizing, just copy the parameters
            for params in self.parameters:
                self.normalized_parameters.append(params)

    def add_gaussian_noise(self, point_cloud):
        """Add Gaussian noise to the point cloud."""
        noise = torch.randn_like(point_cloud) * self.noise_std
        return point_cloud + noise

    def random_point_dropout(self, point_cloud):
        """Randomly drop points from the point cloud."""
        if self.dropout_prob > 0:
            mask = torch.rand(point_cloud.shape[0], device=point_cloud.device) > self.dropout_prob
            return point_cloud[mask]
        return point_cloud
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (point_cloud, parameters)
        """
        point_cloud = self.point_clouds[idx].clone()
        params = self.normalized_parameters[idx]
        
        # Apply augmentation if enabled
        if self.augment:
            # Add Gaussian noise
            point_cloud = self.add_gaussian_noise(point_cloud)
            
            # Random point dropout
            point_cloud = self.random_point_dropout(point_cloud)
            
            # If points were dropped, resample to maintain fixed size
            if point_cloud.shape[0] < self.num_points:
                # Randomly duplicate points to maintain size
                indices = torch.randint(0, point_cloud.shape[0], (self.num_points - point_cloud.shape[0],))
                point_cloud = torch.cat([point_cloud, point_cloud[indices]], dim=0)
            elif point_cloud.shape[0] > self.num_points:
                # Randomly select points to maintain size
                indices = torch.randperm(point_cloud.shape[0])[:self.num_points]
                point_cloud = point_cloud[indices]
        
        # Detach all parameter tensors
        detached_params = {key: value.detach() for key, value in params.items()}
        return point_cloud, detached_params


# ----------------------- CURRICULUM LEARNING UTILS ----------------------- #

def update_curriculum_scales(dataset, epoch, curriculum_schedule):
    """
    Update dataset randomization scales based on curriculum learning schedule.
    
    Args:
        dataset: SMILDataset instance
        epoch: Current training epoch
        curriculum_schedule: Dictionary defining curriculum schedule
    """
    updates = {}
    
    for param_name, schedule in curriculum_schedule.items():
        start_epoch = schedule.get('start_epoch', 0)
        end_epoch = schedule.get('end_epoch', epoch)
        start_value = schedule.get('start_value', 0.0)
        end_value = schedule.get('end_value', 1.0)
        schedule_type = schedule.get('schedule', 'linear')
        
        if epoch < start_epoch:
            # Before curriculum starts, use start value
            current_value = start_value
        elif epoch >= end_epoch:
            # After curriculum ends, use end value
            current_value = end_value
        else:
            # During curriculum, interpolate between start and end values
            progress = (epoch - start_epoch) / (end_epoch - start_epoch)
            
            if schedule_type == 'linear':
                current_value = start_value + progress * (end_value - start_value)
            elif schedule_type == 'exponential':
                # Exponential interpolation (starts slow, accelerates)
                current_value = start_value + (end_value - start_value) * (progress ** 2)
            else:
                # Default to linear
                current_value = start_value + progress * (end_value - start_value)
        
        updates[param_name] = current_value
    
    # Update the dataset with new scales
    if updates:
        dataset.update_randomization_scales(**updates)
        
        # Regenerate dataset with new scales if we have significant changes
        # Only regenerate if any scale changed by more than 10%
        current_scales = {
            'shape_scale': dataset.shape_scale,
            'pose_scale': dataset.pose_scale,
            'trans_scale': dataset.trans_scale,
            'scale_scale': dataset.scale_scale,
            'global_rot_scale': dataset.global_rot_scale
        }
        
        should_regenerate = False
        for param_name, new_value in updates.items():
            if param_name in current_scales:
                old_value = current_scales[param_name]
                if abs(new_value - old_value) / (abs(old_value) + 1e-8) > 0.1:  # 10% threshold
                    should_regenerate = True
                    break
        
        if should_regenerate:
            print(f"Regenerating dataset due to curriculum scale changes at epoch {epoch}")
            dataset.generate_dataset(regenerate=True)


# ----------------------- TRAINING CODE ----------------------- #

def compute_mesh_and_joint_losses(smal_fitter, params, input_pointclouds, gt_joints, num_points=3000, device='cuda'):
    """
    Compute chamfer distance between input point clouds and point clouds generated 
    from predicted SMIL parameters, as well as computing joint location loss.
    
    Args:
        smal_fitter: SMAL3DFitter object
        params: Dictionary of predicted parameters (from SMILPointNet)
        input_pointclouds: Input point clouds tensor of shape (batch_size, num_points, 3)
        gt_joints: Ground truth joint locations tensor of shape (batch_size, num_joints, 3)
        num_points: Number of points to sample from generated meshes
        device: Device to use for computation
        
    Returns:
        chamfer_loss: Scalar tensor
        joint_loc_loss: Scalar tensor
    """
    batch_size = input_pointclouds.shape[0]
    
    predicted_pointclouds_list = []
    predicted_joints_list = []
    
    for i in range(batch_size):
        current_betas = params['betas'][i].unsqueeze(0).to(device)
        current_global_rot_param = params['global_rot'][i].unsqueeze(0).to(device)
        # current_joint_rot is 6D, convert to axis-angle for smal_fitter
        current_joint_rot_param = params['joint_rot'][i].unsqueeze(0).to(device)
        if current_global_rot_param.shape[-1] == 6: # Check if global_rot is 6D
            current_global_rot_aa = rotation_6d_to_axis_angle(current_global_rot_param)
        else: # Assumed to be axis-angle (shape[-1] == 3)
            current_global_rot_aa = current_global_rot_param

        if current_joint_rot_param.shape[-1] == 6: # Check if joint_rot is 6D
            current_joint_rot_aa = rotation_6d_to_axis_angle(current_joint_rot_param)
        else: # Assumed to be axis-angle (shape[-1] == 3)
            current_joint_rot_aa = current_joint_rot_param
            
        current_trans = params['trans'][i].unsqueeze(0).to(device)
        
        current_log_beta_scales = None
        if 'log_beta_scales' in params and params['log_beta_scales'] is not None and params['log_beta_scales'].nelement() > 0:
            if smal_fitter.n_joints == params['log_beta_scales'].shape[1]:
                 current_log_beta_scales = params['log_beta_scales'][i].unsqueeze(0).to(device)
            else:
                print(f"Warning: Mismatch in n_joints for log_beta_scales. Predicted: {params['log_beta_scales'].shape[1]}, Fitter: {smal_fitter.n_joints}. Using fitter's internal scales for sample {i}.")

        current_deform_verts = None

        # SMAL3DFitter.forward now returns verts and joints when return_joints=True
        verts, joints_single_sample = smal_fitter.forward( # New unpacking for 2 return values
            betas=current_betas,
            global_rot=current_global_rot_aa, # Use axis-angle representation
            joint_rot=current_joint_rot_aa, # Use axis-angle representation
            trans=current_trans,
            log_beta_scales=current_log_beta_scales,
            deform_verts=current_deform_verts,
            return_joints=True # Explicitly request joints
        )
        # smal_fitter is batch_size=1, so joints_single_sample is (1, N_JOINTS, 3)
        predicted_joints_list.append(joints_single_sample[0])

        if torch.isnan(verts).any() or torch.isinf(verts).any():
            print(f"Error: NaNs or Infs detected in 'verts' from smal_fitter.forward for sample {i}.")
            print(f"  Verts shape: {verts.shape}, nans: {torch.isnan(verts).sum().item()}, infs: {torch.isinf(verts).sum().item()}")
            # Optionally print parameters that led to this
            print("  Parameters leading to NaN/Inf verts:")
            for p_name, p_val in [("betas", current_betas), ("global_rot", current_global_rot_aa), 
                                  ("joint_rot", current_joint_rot_aa), ("trans", current_trans),
                                  ("log_beta_scales", current_log_beta_scales if current_log_beta_scales is not None else "None (using fitter default)")]:
                if isinstance(p_val, torch.Tensor):
                    print(f"    {p_name}: shape={p_val.shape}, "
                          f"min={p_val.min().item():.4f}, max={p_val.max().item():.4f}, "
                          f"mean={p_val.mean().item():.4f}, nans={torch.isnan(p_val).sum().item()}")
                else:
                    print(f"    {p_name}: {p_val}")
            raise ValueError("Meshes would contain nan or inf due to smal_fitter output.")
        
        faces = smal_fitter.faces
        
        mesh = Meshes(verts=verts, faces=faces)
        
        sampled_points_batch = sample_points_from_meshes(mesh, num_samples=num_points)
        
        predicted_pointclouds_list.append(sampled_points_batch[0])
    
    predicted_pointclouds = torch.stack(predicted_pointclouds_list)
    predicted_joints_batch = torch.stack(predicted_joints_list) # Stack collected joints
    
    # Compute Chamfer distance loss
    chamfer_loss, _ = chamfer_distance(predicted_pointclouds, input_pointclouds)
    
    # Compute Joint location loss relative to the root joint (index 0)
    # Root joint is assumed to be at index 0
    root_joint_pred = predicted_joints_batch[:, 0:1, :]  # Keep dim for broadcasting
    root_joint_gt = gt_joints[:, 0:1, :]              # Keep dim for broadcasting
    
    relative_pred_joints = predicted_joints_batch - root_joint_pred
    relative_gt_joints = gt_joints - root_joint_gt
    
    joint_loc_loss = F.mse_loss(relative_pred_joints, relative_gt_joints)

    return chamfer_loss, joint_loc_loss # Return both losses


def visualize_epoch_results(model, val_loader, smal_fitter, epoch, device, save_dir='plots/epoch_vis', num_points=3000):
    """
    Generate visualization of validation results after an epoch.
    
    Args:
        model: Trained SMILPointNet model
        val_loader: Validation data loader
        smal_fitter: SMAL3DFitter object for visualization
        epoch: Current epoch number
        device: Device to use for computation
        save_dir: Directory to save plots
        num_points: Number of points to sample from predicted mesh
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # Get param_stats and normalization flag from the val_loader's dataset for denormalization
    param_stats = val_loader.dataset.param_stats
    normalize_flag = getattr(val_loader.dataset, 'normalize', True)

    with torch.no_grad():
        # Get the first batch from validation loader
        point_clouds, param_dicts = next(iter(val_loader))
        
        # Use only the first sample
        point_cloud = point_clouds[0].unsqueeze(0).to(device)
        
        # Forward pass to get predicted parameters
        pred_params = model(point_cloud)
        
        # Denormalize predicted parameters if normalization was used
        if normalize_flag:
            denorm_pred_params = {key: pred_params[key].clone() for key in pred_params}
            for key in denorm_pred_params:
                if key in param_stats:
                    denorm_pred_params[key] = denormalize_params({key: denorm_pred_params[key]}, param_stats)[key]
        else:
            denorm_pred_params = pred_params

        # Get ground truth parameters for the first sample and denormalize if needed
        true_params_sample = {key: val[0].unsqueeze(0) for key, val in param_dicts.items()}
        if normalize_flag:
            denorm_true_params_sample = {key: true_params_sample[key].clone() for key in true_params_sample}
            for key in denorm_true_params_sample:
                if key in param_stats:
                    # Ensure tensors are on the correct device for denormalization if stats are on CPU
                    device_for_denorm = denorm_true_params_sample[key].device
                    stats_device = param_stats[key]['mean'].device
                    denorm_true_params_sample[key] = denormalize_params(
                        {key: denorm_true_params_sample[key].to(stats_device)}, 
                        param_stats
                    )[key].to(device_for_denorm)
        else:
            denorm_true_params_sample = true_params_sample

        # Apply predicted parameters to SMAL fitter to generate mesh AND JOINTS
        viz_betas = denorm_pred_params['betas'][0].unsqueeze(0).to(device)
        
        viz_global_rot_param = denorm_pred_params['global_rot'][0].unsqueeze(0).to(device)
        if viz_global_rot_param.shape[-1] == 6: # Check if global_rot is 6D
            viz_global_rot_aa = rotation_6d_to_axis_angle(viz_global_rot_param)
        else: # Assumed axis-angle
            viz_global_rot_aa = viz_global_rot_param
            
        # viz_joint_rot is 6D, convert to axis-angle for smal_fitter
        viz_joint_rot_param = denorm_pred_params['joint_rot'][0].unsqueeze(0).to(device)
        if viz_joint_rot_param.shape[-1] == 6:
            viz_joint_rot_aa = rotation_6d_to_axis_angle(viz_joint_rot_param)
        else: # Assumed axis-angle
            viz_joint_rot_aa = viz_joint_rot_param
            
        viz_trans = denorm_pred_params['trans'][0].unsqueeze(0).to(device)
        
        viz_log_beta_scales = None
        if 'log_beta_scales' in denorm_pred_params and denorm_pred_params['log_beta_scales'] is not None:
            if smal_fitter.n_joints == denorm_pred_params['log_beta_scales'].shape[1]:
                viz_log_beta_scales = denorm_pred_params['log_beta_scales'][0].unsqueeze(0).to(device)

        # No torch.no_grad() here for visualization, but it's okay as it's not part of training backward pass
        # smal_fitter.forward returns verts, joints, Rs, v_shaped -> now verts, joints when return_joints=True
        verts, pred_joints = smal_fitter.forward( # New unpacking for 2 return values
            betas=viz_betas,
            global_rot=viz_global_rot_aa, # Use axis-angle representation
            joint_rot=viz_joint_rot_aa, # Use axis-angle representation
            trans=viz_trans,
            log_beta_scales=viz_log_beta_scales,
            return_joints=True # Explicitly request joints
        )
        
        # Get faces from the model
        faces = smal_fitter.faces
        
        # Create a mesh object
        pred_mesh = Meshes(verts=verts, faces=faces)
        
        # Sample points from the predicted mesh
        pred_point_cloud = sample_points_from_meshes(pred_mesh, num_samples=num_points)
        if isinstance(pred_point_cloud, tuple):
            pred_point_cloud = pred_point_cloud[0]  # Handle case where normals are returned
        
        # Get ground truth joints for visualization (already denormalized if needed)
        # Ensure it's on the correct device and squeezed if batch dim was 1
        gt_joints = denorm_true_params_sample['joints'].squeeze(0).cpu().numpy()
        pred_joints_np = pred_joints[0].cpu().numpy() # pred_joints is (1, num_joints, 3)

        # Convert to numpy for plotting
        target_points = point_cloud[0].cpu().numpy()
        pred_points = pred_point_cloud[0].cpu().numpy()
        
        # Create figure with 3 subplots
        fig = plt.figure(figsize=(18, 6))
        
        # 1. Target point cloud
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
                   c='blue', marker='.', s=1, alpha=0.7)
        ax1.set_title('Target Point Cloud')
        # Plot ground truth joints on target point cloud plot
        ax1.scatter(gt_joints[:, 0], gt_joints[:, 1], gt_joints[:, 2],
                    c='cyan', marker='o', s=50, edgecolors='k', label='GT Joints')
        ax1.legend()
        
        # 2. Predicted point cloud
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                   c='red', marker='.', s=1, alpha=0.7)
        ax2.set_title('Predicted Point Cloud')
        # Plot predicted joints on predicted point cloud plot
        ax2.scatter(pred_joints_np[:, 0], pred_joints_np[:, 1], pred_joints_np[:, 2],
                    c='magenta', marker='o', s=50, edgecolors='k', label='Pred Joints')
        ax2.legend()
        
        # 3. Overlay of both point clouds
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
                   c='blue', marker='.', s=1, alpha=0.5, label='Target PC')
        ax3.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                   c='red', marker='.', s=1, alpha=0.5, label='Predicted PC')
        # Plot both GT and predicted joints on the overlay
        ax3.scatter(gt_joints[:, 0], gt_joints[:, 1], gt_joints[:, 2],
                    c='cyan', marker='o', s=50, edgecolors='k', label='GT Joints')
        ax3.scatter(pred_joints_np[:, 0], pred_joints_np[:, 1], pred_joints_np[:, 2],
                    c='magenta', marker='X', s=50, edgecolors='k', label='Pred Joints') # Use 'X' for predicted here for distinction
        ax3.set_title('Overlay of Point Clouds and Joints')
        ax3.legend()
        
        # Set view parameters for all subplots to be the same
        for ax in [ax1, ax2, ax3]:
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=30, azim=45)
        
        # Add title with epoch information
        fig.suptitle(f'Validation Results at Epoch {epoch}', fontsize=16)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'epoch_{epoch:03d}_prediction.png'), dpi=200)
        plt.close(fig)


def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, device='cuda', 
                weights=None, chamfer_weight=0.1, checkpoint_dir='checkpoints', 
                vis_interval=5, log_interval=10, regenerate_training_every=-1,
                curriculum_learning=False, curriculum_schedule=None):
    """
    Train the SMIL PointNet model.
    
    Args:
        model: SMILPointNet model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs to train
        lr: Learning rate
        device: Device to use for computation
        weights: Dictionary of loss weights for different parameter groups
        chamfer_weight: Weight for the chamfer distance loss
        checkpoint_dir: Directory to save model checkpoints
        vis_interval: Interval (in epochs) for visualizing validation results
        log_interval: Interval for logging training progress
        regenerate_training_every: Whether to regenerate training data every N epochs
        curriculum_learning: Whether to use curriculum learning (default: False)
        curriculum_schedule: Dictionary defining curriculum schedule. Example:
            {
                'pose_scale': {
                    'start_epoch': 10,
                    'end_epoch': 50,
                    'start_value': 0.0,
                    'end_value': 0.25,
                    'schedule': 'linear'  # 'linear' or 'exponential'
                },
                'global_rot_scale': {
                    'start_epoch': 20,
                    'end_epoch': 60,
                    'start_value': 0.0,
                    'end_value': 0.1,
                    'schedule': 'linear'
                }
            }
    Returns:
        Trained model and training history
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Default weights if not provided
    if weights is None:
        weights = {
            'global_rot': 1.0,
            'joint_rot': 1.0,
            'betas': 1.0,
            'trans': 1.0,
            'log_beta_scales': 1.0,
            'joints': 1.0
        }
    # Ensure 'joints' key exists in weights if not None, for consistent param_losses initialization
    elif 'joints' not in weights:
        weights['joints'] = 1.0
    
    # Create a SMAL3DFitter for computing chamfer loss
    smal_fitter = load_smil_model(batch_size=1, device=device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), 
                           lr=lr,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=1e-4)

    # Initialize scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'param_losses': {key: [] for key in weights.keys()}, # This will now include 'joints' if it's in weights.
        'chamfer_loss': [],
        'train_joint_loc_loss': [], # For unweighted average joint location loss during training
        'val_joint_loc_loss': []   # For unweighted average joint location loss during validation
    }
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Curriculum learning: update randomization scales if enabled
        if curriculum_learning and curriculum_schedule is not None:
            update_curriculum_scales(train_loader.dataset, epoch, curriculum_schedule)
        
        model.train()
        train_loss = 0.0
        param_losses = {key: 0.0 for key in weights.keys()} # Initializes all keys from weights, including 'joints'
        chamfer_losses = 0.0
        # joint_location_losses accumulator is for the unweighted sum over the epoch, to be averaged.
        current_epoch_train_joint_loc_loss_sum = 0.0

        if regenerate_training_every > 0 and epoch > 0 and epoch % regenerate_training_every == 0:
            train_loader.dataset.generate_dataset(regenerate=True)
        
        # Training
        for i, (point_clouds, param_dicts) in enumerate(train_loader):
            # Move data to device
            point_clouds = point_clouds.to(device)
            param_tensors = {key: param_dicts[key].to(device) for key in param_dicts}
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred_params = model(point_clouds)
            
            # Compute MSE loss for each parameter group
            current_batch_mse_loss = 0.0 # Renamed from mse_loss to avoid confusion in this scope
            for key in weights.keys():
                if key == 'joints': 
                    continue # Skip 'joints' key here, handled by compute_mesh_and_joint_losses if active
                
                if key in pred_params and key in param_tensors:
                    if model.rotation_representation == '6d' and (key == 'global_rot' or key == 'joint_rot'):
                        # Convert 6D representation to rotation matrices
                        R_pred = rotation_6d_to_matrix(pred_params[key])
                        R_gt = rotation_6d_to_matrix(param_tensors[key])
                        
                        # Compute Frobenius norm of the difference
                        # For global_rot (batch_size, 3, 3) -> norm gives (batch_size)
                        # For joint_rot (batch_size, n_pose, 3, 3) -> norm gives (batch_size, n_pose)
                        matrix_diff_loss = torch.norm(R_pred - R_gt, p='fro', dim=(-2, -1))
                        param_loss = matrix_diff_loss.mean() # Mean over batch and/or poses
                    else:
                        # Use MSE for other parameters or non-6d rotations
                        param_loss = F.mse_loss(pred_params[key], param_tensors[key])
                    
                    weighted_loss = weights[key] * param_loss
                    current_batch_mse_loss += weighted_loss
                    param_losses[key] += weighted_loss.item() # Accumulate weighted loss for history
            
            loss = current_batch_mse_loss # Initial loss is from parameters (MSE or matrix norm)
            SMIL_param_loss_value = current_batch_mse_loss.clone().detach() # Store SMIL parameter loss separately

            # Compute chamfer distance loss and joint location loss
            cd_loss = torch.tensor(0.0, device=device) # Initialize cd_loss
            joint_loc_loss = torch.tensor(0.0, device=device) # Initialize joint_loc_loss

            if chamfer_weight > 0 or weights.get('joints', 0.0) > 0:
                # compute_mesh_and_joint_losses now returns cd_loss and joint_loc_loss
                current_cd_loss, current_joint_loc_loss = compute_mesh_and_joint_losses(
                    smal_fitter, pred_params, point_clouds, param_tensors['joints'], device=device
                )
                
                if chamfer_weight > 0:
                    cd_loss = current_cd_loss
                    chamfer_losses += cd_loss.item()
                    loss += chamfer_weight * cd_loss
                
                if weights.get('joints', 0.0) > 0:
                    joint_loc_loss = current_joint_loc_loss
                    # param_losses['joints'] accumulates the WEIGHTED loss component for 'joints'
                    param_losses['joints'] += (weights['joints'] * joint_loc_loss).item()
                    # current_epoch_train_joint_loc_loss_sum accumulates the UNWEIGHTED loss for averaging
                    current_epoch_train_joint_loc_loss_sum += joint_loc_loss.item()

                    loss += weights['joints'] * joint_loc_loss
            
            # Backward pass and optimization
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Log progress
            if (i + 1) % log_interval == 0:
                log_msg = f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Total Loss: {loss.item():.4f}, SMIL Param Loss (MSE/Matrix): {SMIL_param_loss_value.item():.4f}'
                if chamfer_weight > 0:
                    log_msg += f', Chamfer Loss: {cd_loss.item():.4f}'
                if weights.get('joints', 0.0) > 0:
                    log_msg += f', JointLoc Loss: {joint_loc_loss.item():.4f}'
                print(log_msg)
        
        # Average loss for the epoch
        train_loss /= len(train_loader)
        for key in param_losses:
            param_losses[key] /= len(train_loader)
        if chamfer_weight > 0:
            chamfer_losses /= len(train_loader)
        if weights.get('joints', 0.0) > 0:
            # joint_location_losses /= len(train_loader) # This was averaging the last batch's loss.
            # Instead, average the sum accumulated over the epoch:
            avg_epoch_train_joint_loc_loss = current_epoch_train_joint_loc_loss_sum / len(train_loader)
            history['train_joint_loc_loss'].append(avg_epoch_train_joint_loc_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_chamfer_loss = 0.0
        # val_joint_loc_loss_epoch accumulator is for the unweighted sum over the validation epoch
        current_epoch_val_joint_loc_loss_sum = 0.0
        
        with torch.no_grad():
            for point_clouds, param_dicts in val_loader:
                # Move data to device
                point_clouds = point_clouds.to(device)
                param_tensors = {key: param_dicts[key].to(device) for key in param_dicts}
                
                # Forward pass
                pred_params = model(point_clouds)
                
                # Compute MSE loss
                current_batch_val_mse_loss = 0.0 # Renamed
                for key in weights.keys():
                    if key == 'joints': continue # Skip 'joints' key here, handled separately
                    if key in pred_params and key in param_tensors:
                        if model.rotation_representation == '6d' and (key == 'global_rot' or key == 'joint_rot'):
                            R_pred_val = rotation_6d_to_matrix(pred_params[key])
                            R_gt_val = rotation_6d_to_matrix(param_tensors[key])
                            matrix_diff_loss_val = torch.norm(R_pred_val - R_gt_val, p='fro', dim=(-2,-1))
                            param_loss_val = matrix_diff_loss_val.mean()
                        else:
                            param_loss_val = F.mse_loss(pred_params[key], param_tensors[key])
                        current_batch_val_mse_loss += weights[key] * param_loss_val
                
                current_val_total_loss = current_batch_val_mse_loss

                cd_loss_val, joint_loc_loss_val = compute_mesh_and_joint_losses(
                    smal_fitter, pred_params, point_clouds, param_tensors['joints'], device=device
                )
                if chamfer_weight > 0:
                    cd_loss_val = cd_loss_val
                    val_chamfer_loss += cd_loss_val.item()
                    current_val_total_loss += chamfer_weight * cd_loss_val

                if weights.get('joints', 0.0) > 0:
                    joint_loc_loss_val = joint_loc_loss_val
                    # current_epoch_val_joint_loc_loss_sum accumulates UNWEIGHTED loss
                    current_epoch_val_joint_loc_loss_sum += joint_loc_loss_val.item()
                    current_val_total_loss += weights['joints'] * joint_loc_loss_val
                
                val_loss += current_val_total_loss.item()
        
        # Average validation loss
        val_loss /= len(val_loader)
        if chamfer_weight > 0:
            val_chamfer_loss /= len(val_loader)
        if weights.get('joints', 0.0) > 0:
            # val_joint_loc_loss_epoch /= len(val_loader)
            # Instead, average the sum accumulated over the epoch:
            avg_epoch_val_joint_loc_loss = current_epoch_val_joint_loc_loss_sum / len(val_loader)
            history['val_joint_loc_loss'].append(avg_epoch_val_joint_loc_loss)

        # Update scheduler
        scheduler.step(val_loss)
        
        # Update training history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        for key in param_losses: # param_losses now includes 'joints' if active
            history['param_losses'][key].append(param_losses[key])
        if chamfer_weight > 0:
            history['chamfer_loss'].append(chamfer_losses)

        # Print progress
        print_msg = f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}'
        if chamfer_weight > 0:
            print_msg += f', Train Chamfer: {chamfer_losses:.4f}, Val Chamfer: {val_chamfer_loss:.4f}'
        if weights.get('joints', 0.0) > 0:
            # Use the correctly averaged values from history for printing
            print_msg += f', Train JointLoc: {history["train_joint_loc_loss"][-1]:.4f}, Val JointLoc: {history["val_joint_loc_loss"][-1]:.4f}'
        print(print_msg)
        
        # Visualize validation results at specified intervals
        if (epoch + 1) % vis_interval == 0 or epoch == 0 or epoch == num_epochs - 1:
            visualize_epoch_results(model, val_loader, smal_fitter, epoch + 1, device)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'n_betas': model.n_betas,
                'n_pose': model.n_pose,
                'include_scales': model.include_scales
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'n_betas': model.n_betas,
                'n_pose': model.n_pose,
                'include_scales': model.include_scales
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    return model, history


def plot_training_history(history, save_dir='plots'):
    """
    Plot training and validation loss history.
    
    Args:
        history: Training history dictionary
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot overall loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'overall_loss.png'))
    plt.close()
    
    # Plot individual parameter losses
    plt.figure(figsize=(12, 8))
    for key, losses in history['param_losses'].items():
        plt.plot(losses, label=f'{key} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Parameter-Specific Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'parameter_losses.png'))
    plt.close()
    
    # Plot chamfer loss if available
    if 'chamfer_loss' in history and history['chamfer_loss']:
        plt.figure(figsize=(10, 6))
        plt.plot(history['chamfer_loss'], label='Chamfer Distance Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Chamfer Distance Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'chamfer_loss.png'))
        plt.close()

    # Plot joint location loss if available
    if history.get('train_joint_loc_loss') and history.get('val_joint_loc_loss'): # Check if keys exist and have data
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_joint_loc_loss'], label='Train Joint Location Loss')
        if history.get('val_joint_loc_loss'): # Plot val if available
             plt.plot(history['val_joint_loc_loss'], label='Validation Joint Location Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Joint Location Loss (Unweighted)')
        plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'joint_location_loss.png'))
    plt.close()


# ----------------------- MODEL EVALUATION AND VISUALIZATION ----------------------- #

def visualize_predictions(model, test_loader, device, smal_fitter, num_samples=5):
    """
    Visualize model predictions by comparing the predicted and ground truth SMIL models.
    Denormalizes parameters before applying to the SMIL fitter.
    Args:
        model: Trained SMILPointNet model
        test_loader: DataLoader for test data
        device: Device to use for computation
        smal_fitter: SMAL3DFitter object for visualization
        num_samples: Number of samples to visualize
    """
    model.eval()

    # Get param_stats and normalization flag from the test_loader's dataset
    param_stats = test_loader.dataset.param_stats
    normalize = getattr(test_loader.dataset, 'normalize', True)
    dataset_rotation_representation = test_loader.dataset.rotation_representation
    model_rotation_representation = model.rotation_representation

    with torch.no_grad():
        # Get a batch from the test loader
        point_clouds, true_params = next(iter(test_loader))
        # Only use the specified number of samples
        point_clouds = point_clouds[:num_samples].to(device)
        # Forward pass
        pred_params = model(point_clouds)
        # Denormalize predicted and true parameters only if normalization was enabled
        if normalize:
            for key in pred_params:
                if key in param_stats:
                    pred_params[key] = denormalize_params({key: pred_params[key]}, param_stats)[key]
            for key in true_params:
                if key in param_stats:
                    true_params[key] = denormalize_params({key: true_params[key]}, param_stats)[key]
        # Visualize each sample
        for i in range(num_samples):
            # Create a figure with two subplots side by side
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.set_title('Ground Truth SMIL Model')
            
            # Apply true parameters to SMAL fitter
            true_betas_i = true_params['betas'][i].unsqueeze(0).to(device)
            true_global_rot_param_i = true_params['global_rot'][i].unsqueeze(0).to(device)
            if dataset_rotation_representation == '6d': # Check dataset's global_rot format
                true_global_rot_aa_i = rotation_6d_to_axis_angle(true_global_rot_param_i)
            else: # axis_angle
                true_global_rot_aa_i = true_global_rot_param_i
                
            true_joint_rot_param_i = true_params['joint_rot'][i].unsqueeze(0).to(device)
            if dataset_rotation_representation == '6d':
                true_joint_rot_aa_i = rotation_6d_to_axis_angle(true_joint_rot_param_i)
            else: # axis_angle
                true_joint_rot_aa_i = true_joint_rot_param_i
                
            true_trans_i = true_params['trans'][i].unsqueeze(0).to(device)
            true_log_beta_scales_i = None
            if 'log_beta_scales' in true_params and true_params['log_beta_scales'] is not None:
                 if smal_fitter.n_joints == true_params['log_beta_scales'].shape[1]:
                    true_log_beta_scales_i = true_params['log_beta_scales'][i].unsqueeze(0).to(device)

            with torch.no_grad(): # Keep no_grad for visualization/evaluation if not training
                true_verts = smal_fitter.forward(
                    betas=true_betas_i,
                    global_rot=true_global_rot_aa_i, # Use axis-angle
                    joint_rot=true_joint_rot_aa_i, # Use axis-angle
                    trans=true_trans_i,
                    log_beta_scales=true_log_beta_scales_i
                )
            
            # Get faces from the model
            faces = smal_fitter.faces
            
            # Create a mesh object
            from pytorch3d.structures import Meshes
            true_mesh = Meshes(verts=true_verts, faces=faces)
            
            # Plot the mesh
            true_points = true_mesh.verts_packed().cpu().numpy()
            ax1.scatter(true_points[:, 0], true_points[:, 1], true_points[:, 2], c='b', marker='.', alpha=0.5)
            
            # Predicted mesh
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.set_title('Predicted SMIL Model')
            
            # Apply predicted parameters to SMAL fitter
            pred_betas_i = pred_params['betas'][i].unsqueeze(0).to(device)
            pred_global_rot_param_i = pred_params['global_rot'][i].unsqueeze(0).to(device)
            if model_rotation_representation == '6d': # Check model's global_rot format
                pred_global_rot_aa_i = rotation_6d_to_axis_angle(pred_global_rot_param_i)
            else: # axis_angle
                pred_global_rot_aa_i = pred_global_rot_param_i
                
            pred_joint_rot_param_i = pred_params['joint_rot'][i].unsqueeze(0).to(device)
            if model_rotation_representation == '6d':
                pred_joint_rot_aa_i = rotation_6d_to_axis_angle(pred_joint_rot_param_i)
            else: # axis_angle
                pred_joint_rot_aa_i = pred_joint_rot_param_i
                
            pred_trans_i = pred_params['trans'][i].unsqueeze(0).to(device)
            pred_log_beta_scales_i = None
            if 'log_beta_scales' in pred_params and pred_params['log_beta_scales'] is not None:
                 if smal_fitter.n_joints == pred_params['log_beta_scales'].shape[1]:
                    pred_log_beta_scales_i = pred_params['log_beta_scales'][i].unsqueeze(0).to(device)
            
            with torch.no_grad(): # Keep no_grad for visualization/evaluation if not training
                pred_verts = smal_fitter.forward(
                    betas=pred_betas_i,
                    global_rot=pred_global_rot_aa_i, # Use axis-angle
                    joint_rot=pred_joint_rot_aa_i, # Use axis-angle
                    trans=pred_trans_i,
                    log_beta_scales=pred_log_beta_scales_i
                )
            
            # Create a mesh object
            pred_mesh = Meshes(verts=pred_verts, faces=faces)
            
            # Plot the mesh
            pred_points = pred_mesh.verts_packed().cpu().numpy()
            ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], c='r', marker='.', alpha=0.5)
            
            # Adjust view for better visualization
            for ax in [ax1, ax2]:
                ax.set_box_aspect([1, 1, 1])
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.view_init(elev=30, azim=45)
            
            plt.tight_layout()
            plt.savefig(f'sample_{i+1}_comparison.png')
            plt.close()


def print_model_stats(model, input_size=(1, 3000, 3)):
    """
    Print model architecture and computational statistics.
    
    Args:
        model: The PyTorch model
        input_size: Tuple of (batch_size, num_points, num_features)
    """
    print("\n" + "="*50)
    print("Model Architecture and Statistics")
    print("="*50)
    
    # Print model summary
    print("\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nParameter Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Print parameter distribution by layer
    print("\nParameter Distribution by Layer:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,} parameters")
    
    # Print input/output shapes
    print("\nInput/Output Shapes:")
    print(f"Input shape: {input_size}")
    
    # Create a sample input on the same device as the model
    device = next(model.parameters()).device
    sample_input = torch.randn(input_size, device=device)
    
    # Get output shape
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        output = model(sample_input)
    model.train()  # Set model back to training mode
    
    print("\nOutput shapes:")
    for key, value in output.items():
        print(f"{key}: {value.shape}")
    
    print("\n" + "="*50 + "\n")


# ----------------------- MAIN FUNCTION ----------------------- #

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train SMIL PointNet model')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train (default: 100)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of worker processes for DataLoader (default: 4, 0 for no multiprocessing)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--train-samples', type=int, default=10000,
                       help='Number of training samples (default: 1000)')
    parser.add_argument('--val-samples', type=int, default=100,
                       help='Number of validation samples (default: 100)')
    parser.add_argument('--num-points', type=int, default=2048,
                       help='Number of points in each point cloud (default: 2048)')
    parser.add_argument('--vis-interval', type=int, default=10,
                       help='Visualization interval in epochs (default: 10)')
    parser.add_argument('--no-multiprocessing', action='store_true',
                       help='Disable multiprocessing in DataLoader (equivalent to --num-workers=0)')
    parser.add_argument('--no-normalization', action='store_true',
                       help='Disable parameter normalization (default: False, i.e., normalization is enabled)')
    parser.add_argument('--device', type=str, default=None,
                       help="Device to use for training (e.g., 'cpu', 'cuda', 'cuda:0'). Default: 'cuda' if available, else 'cpu'.")
    parser.add_argument('--noise-std', type=float, default=0.01,
                       help='Standard deviation of Gaussian noise for point cloud augmentation (default: 0.01)')
    parser.add_argument('--dropout-prob', type=float, default=0.05,
                       help='Probability of dropping points during augmentation (default: 0.1)')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable point cloud augmentation (default: False, i.e., augmentation is enabled)')
    parser.add_argument('--regenerate_training_every', type=int, default=-1,
                       help='Regenerate training data every N epochs (default: -1, i.e., never)')
    parser.add_argument('--rotation-representation', type=str, default='6d', choices=['6d', 'axis_angle'],
                        help='Rotation representation for global and joint rotations (default: 6d)')
    parser.add_argument('--exclude-rot-from-norm', action='store_true', default=True,
                        help='Exclude rotations (global_rot, joint_rot) from parameter normalization (default: False)')
    parser.add_argument('--curriculum-learning', action='store_true', default=False,
                        help='Enable curriculum learning (default: False)')
    parser.add_argument('--curriculum-pose-start', type=int, default=10,
                        help='Epoch to start pose curriculum (default: 10)')
    parser.add_argument('--curriculum-pose-end', type=int, default=50,
                        help='Epoch to end pose curriculum (default: 50)')
    parser.add_argument('--curriculum-pose-start-value', type=float, default=0.0,
                        help='Starting pose scale value (default: 0.0)')
    parser.add_argument('--curriculum-pose-end-value', type=float, default=0.25,
                        help='Ending pose scale value (default: 0.25)')
    return parser.parse_args()

def main():
    """
    Main function to train and evaluate the SMIL PointNet model.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Parameters from command-line arguments
    num_train_samples = args.train_samples
    num_val_samples = args.val_samples
    num_points = args.num_points
    batch_size = args.batch_size
    num_epochs = args.epochs
    vis_interval = args.vis_interval
    
    # Number of worker processes for DataLoader
    num_workers = 0 if args.no_multiprocessing else args.num_workers
    print(f"Using {num_workers} worker processes for DataLoader")
    
    # Normalization flag
    normalize = not args.no_normalization
    print(f"Parameter normalization: {'enabled' if normalize else 'disabled'}")
    
    # Augmentation parameters
    noise_std = args.noise_std
    dropout_prob = args.dropout_prob
    augment = not args.no_augment
    print(f"Point cloud augmentation: {'enabled' if augment else 'disabled'}")
    if augment:
        print(f"  - Gaussian noise std: {noise_std}")
        print(f"  - Point dropout probability: {dropout_prob}")
    
    # Rotation normalization flag
    exclude_rot_from_norm = args.exclude_rot_from_norm
    print(f"Exclude rotations from normalization: {exclude_rot_from_norm}")

    # Create datasets
    train_dataset = SMILDataset(num_samples=num_train_samples, num_points=num_points, 
                                device=device, seed=seed, normalize=normalize,
                                noise_std=noise_std, dropout_prob=dropout_prob, augment=augment,
                                rotation_representation=args.rotation_representation,
                                exclude_rot_from_norm=True) # Changed to True
    val_dataset = SMILDataset(num_samples=num_val_samples, num_points=num_points, 
                              device=device, seed=seed+1, normalize=normalize,
                              noise_std=noise_std, dropout_prob=dropout_prob, augment=False,
                              rotation_representation=args.rotation_representation,
                              exclude_rot_from_norm=True)  # Changed to True, no augmentation for validation

    # Get model configuration
    n_betas = train_dataset.n_betas
    n_pose = train_dataset.n_pose
    n_joints = train_dataset.n_joints
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    
    # Create model
    # Choose one of the models below:
    
    # Original SMILPointNet (PointNet-based)
    # model = SMILPointNet(num_points=num_points, n_betas=n_betas, n_pose=n_pose, 
    #                      include_scales=config.ALLOW_LIMB_SCALING, rotation_representation=args.rotation_representation).to(device)
    
    # New SMILPointNet2 (PointNet++ based)
    model = SMILPointNet2(num_points=num_points, n_betas=n_betas, n_pose=n_pose, 
                         include_scales=config.ALLOW_LIMB_SCALING, rotation_representation=args.rotation_representation).to(device)
    
    # Set joint scales size if using scales
    if config.ALLOW_LIMB_SCALING:
        model.set_joint_scales_size(n_joints)
    
    # Print model architecture and statistics
    print_model_stats(model, input_size=(1, num_points, 3))
    
    # Define loss weights
    loss_weights = {
        'global_rot': 0.01,  
        'joint_rot': 0.5, 
        'betas': 0.2,
        'trans': 0.1,  #
        'log_beta_scales': 0.1 if config.ALLOW_LIMB_SCALING else 0.0,
        'joints': 0.5 
    }
    
    # Set chamfer loss weight
    chamfer_weight = 1.0
    
    # Curriculum learning configuration
    curriculum_learning = args.curriculum_learning
    curriculum_schedule = None
    
    if curriculum_learning:
        curriculum_schedule = {
            'pose_scale': {
                'start_epoch': args.curriculum_pose_start,
                'end_epoch': args.curriculum_pose_end,
                'start_value': args.curriculum_pose_start_value,
                'end_value': args.curriculum_pose_end_value,
                'schedule': 'linear'
            }
        }
        
        print("Curriculum Learning Configuration:")
        for param_name, schedule in curriculum_schedule.items():
            print(f"  {param_name}: {schedule['start_value']} -> {schedule['end_value']} "
                  f"(epochs {schedule['start_epoch']}-{schedule['end_epoch']}, {schedule['schedule']})")
        
        # Initialize training dataset with curriculum starting values
        train_dataset.update_randomization_scales(
            pose_scale=curriculum_schedule['pose_scale']['start_value'],
        )
        train_dataset.generate_dataset(regenerate=True)
    
    # Train the model
    trained_model, history = train_model(model, train_loader, val_loader, num_epochs=num_epochs, 
                                          lr=0.0005, device=device, weights=loss_weights,
                                          chamfer_weight=chamfer_weight, vis_interval=vis_interval,
                                          regenerate_training_every=args.regenerate_training_every,
                                          curriculum_learning=curriculum_learning,
                                          curriculum_schedule=curriculum_schedule)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    smal_fitter = load_smil_model(batch_size=1, device=device)
    # ATTENTION: THIS RUNS ON THE TRAINING DATASET FOR DEBUGGING PURPOSES
    visualize_predictions(trained_model, train_loader, device, smal_fitter)
    
    print("Training and evaluation completed.")


if __name__ == "__main__":
    main() 