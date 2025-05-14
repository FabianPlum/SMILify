"""
SMIL PointNet: A lightweight end-to-end mesh registration approach.

This module implements a PointNet-style neural network to estimate SMIL parameters
from a point cloud, enabling direct mesh registration from point cloud data.
The model is trained on randomly sampled SMIL configurations and their resulting point clouds.

References:
- PointNet: https://arxiv.org/pdf/1612.00593
- PointNet implementation: https://github.com/fxia22/pointnet.pytorch
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
import matplotlib.pyplot as plt
from pytorch3d.loss import chamfer_distance
import argparse
import pickle

# Add the parent directory to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import config
from fitter_3d.trainer import SMAL3DFitter
from fitter_3d.pointcloud2smil.sample_smil_model import load_smil_model, generate_random_parameters
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes


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
    def __init__(self, num_points=5000, n_betas=20, n_pose=34, include_scales=True):
        """
        Initialize the SMILPointNet model.
        
        Args:
            num_points: Number of points in the input point cloud
            n_betas: Number of shape parameters in the SMIL model
            n_pose: Number of pose parameters in the SMIL model (excluding global rotation)
            include_scales: Whether to predict joint scales
        """
        super(SMILPointNet, self).__init__()
        
        self.num_points = num_points
        self.n_betas = n_betas
        self.n_pose = n_pose
        self.include_scales = include_scales
        
        # Feature extraction network
        self.stn = TNet(k=3)  # Spatial transformer network for input point cloud
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Regression head for SMIL parameters
        
        # Calculate total output size
        # - Global rotation (3)
        # - Joint rotations (n_pose * 3)
        # - Shape parameters (n_betas)
        # - Translation (3)
        # - Joint scales (optional, different for each model)
        n_scales = 0  # Will be set later if include_scales=True
        self.output_size = 3 + n_pose * 3 + n_betas + 3 + n_scales
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.output_size)
        
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
            self.output_size = 3 + self.n_pose * 3 + self.n_betas + 3 + n_scales
            
            # Recreate the final layer with the right output size
            self.fc3 = nn.Linear(256, self.output_size).to(self.fc1.weight.device)

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
        
        # Global rotation (axis-angle, 3 values)
        params['global_rot'] = x[:, :3]
        
        # Joint rotations (n_pose * 3 values)
        joint_rot_start = 3
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


# ----------------------- DATASET IMPLEMENTATION ----------------------- #

class SMILDataset(Dataset):
    """
    Dataset for SMIL parameter estimation from point clouds.
    
    Generates random SMIL configurations and their corresponding point clouds.
    """
    def __init__(self, num_samples=1000, num_points=5000, device='cuda', shape_family=-1, seed=None, param_stats=None):
        """
        Initialize the dataset.
        
        Args:
            num_samples: Number of samples in the dataset
            num_points: Number of points to sample from each mesh
            device: Device to use for computation
            shape_family: Shape family ID to use for the SMIL model
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.num_points = num_points
        self.device = device
        self.shape_family = shape_family
        self.param_stats = param_stats
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Load the SMIL model
        self.smal_fitter = load_smil_model(batch_size=1, device=device)
        
        # Store model configuration for later use
        self.n_betas = self.smal_fitter.n_betas
        self.n_pose = config.N_POSE
        self.n_joints = self.smal_fitter.n_joints
        
        # Generate and store dataset
        self.point_clouds = []
        self.parameters = []
        self.normalized_parameters = []
        
        print(f"Generating {num_samples} random SMIL configurations and point clouds...")
        for i in tqdm(range(num_samples)):
            # Generate random parameters
            generate_random_parameters(self.smal_fitter, seed=seed + i if seed is not None else None)
            
            # Forward pass to get vertices
            with torch.no_grad():
                verts = self.smal_fitter()
            
            # Get faces from the model
            faces = self.smal_fitter.faces
            
            # Create a mesh object for sampling
            mesh = Meshes(verts=verts, faces=faces)
            
            # Sample points from the mesh surface
            point_cloud = sample_points_from_meshes(mesh, num_samples=num_points)
            
            # Store the point cloud (explicitly detach)
            self.point_clouds.append(point_cloud[0].cpu().detach())
            
            # Store the parameters (explicitly detach each tensor)
            params = {
                'global_rot': self.smal_fitter.global_rot[0].cpu().detach(),
                'joint_rot': self.smal_fitter.joint_rot[0].cpu().detach(),
                'betas': self.smal_fitter.betas[0].cpu().detach(),
                'trans': self.smal_fitter.trans[0].cpu().detach(),
                'log_beta_scales': self.smal_fitter.log_beta_scales[0].cpu().detach()
            }
            self.parameters.append(params)
        
        # Compute stats if not provided
        if self.param_stats is None:
            keys = ['global_rot', 'joint_rot', 'betas', 'trans', 'log_beta_scales']
            self.param_stats = compute_param_stats(self.parameters, keys)
        
        # Normalize all parameters
        for params in self.parameters:
            normed = normalize_params(params, self.param_stats)
            self.normalized_parameters.append(normed)
    
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
        point_cloud = self.point_clouds[idx]
        normed_params = self.normalized_parameters[idx]
        
        # Detach all parameter tensors
        detached_params = {key: value.detach() for key, value in normed_params.items()}
        
        return point_cloud, detached_params


# ----------------------- TRAINING CODE ----------------------- #

def compute_chamfer_loss(smal_fitter, params, input_pointclouds, num_points=3000, device='cuda'):
    """
    Compute chamfer distance between input point clouds and point clouds generated 
    from predicted SMIL parameters.
    
    Args:
        smal_fitter: SMAL3DFitter object
        params: Dictionary of predicted parameters
        input_pointclouds: Input point clouds tensor of shape (batch_size, num_points, 3)
        num_points: Number of points to sample from generated meshes
        device: Device to use for computation
        
    Returns:
        Chamfer distance loss
    """
    batch_size = input_pointclouds.shape[0]
    
    # Store the original parameters to restore later
    original_params = {}
    for key in params:
        if hasattr(smal_fitter, key):
            original_params[key] = getattr(smal_fitter, key).clone()
    
    # Create container for sampled point clouds from predicted parameters
    predicted_pointclouds = []
    
    # Process each item in the batch individually to save memory
    for i in range(batch_size):
        # Apply predicted parameters to SMAL fitter
        for key in params:
            if hasattr(smal_fitter, key):
                param_tensor = getattr(smal_fitter, key)
                # Set the first item in the batch to the current prediction
                # Since SMAL fitter only needs to generate one mesh at a time
                param_tensor.data[0] = params[key][i].to(device)
        
        # Forward pass to get vertices
        with torch.no_grad():
            verts = smal_fitter()
        
        # Get faces from the model
        faces = smal_fitter.faces
        
        # Create a mesh object
        mesh = Meshes(verts=verts, faces=faces)
        
        # Sample points from the mesh surface
        sampled_points = sample_points_from_meshes(mesh, num_samples=num_points)
        
        # Add to list of predicted point clouds
        predicted_pointclouds.append(sampled_points[0])
    
    # Stack all predicted point clouds into a batch
    predicted_pointclouds = torch.stack(predicted_pointclouds)
    
    # Compute chamfer distance between predicted and input point clouds
    chamfer_loss, _ = chamfer_distance(predicted_pointclouds, input_pointclouds)
    
    # Restore original parameters
    for key, value in original_params.items():
        param_tensor = getattr(smal_fitter, key)
        param_tensor.data.copy_(value)
    
    return chamfer_loss


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
    
    with torch.no_grad():
        # Get the first batch from validation loader
        point_clouds, param_dicts = next(iter(val_loader))
        
        # Use only the first sample
        point_cloud = point_clouds[0].unsqueeze(0).to(device)
        
        # Forward pass to get predicted parameters
        pred_params = model(point_cloud)
        
        # Apply predicted parameters to SMAL fitter to generate mesh
        for key in pred_params:
            if hasattr(smal_fitter, key):
                param_tensor = getattr(smal_fitter, key)
                param_tensor.data[0] = pred_params[key][0].to(device)
        
        # Forward pass to get vertices
        with torch.no_grad():
            verts = smal_fitter()
        
        # Get faces from the model
        faces = smal_fitter.faces
        
        # Create a mesh object
        from pytorch3d.structures import Meshes
        pred_mesh = Meshes(verts=verts, faces=faces)
        
        # Sample points from the predicted mesh
        pred_point_cloud = sample_points_from_meshes(pred_mesh, num_samples=num_points)
        if isinstance(pred_point_cloud, tuple):
            pred_point_cloud = pred_point_cloud[0]  # Handle case where normals are returned
        
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
        
        # 2. Predicted point cloud
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                   c='red', marker='.', s=1, alpha=0.7)
        ax2.set_title('Predicted Point Cloud')
        
        # 3. Overlay of both point clouds
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
                   c='blue', marker='.', s=1, alpha=0.5, label='Target')
        ax3.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                   c='red', marker='.', s=1, alpha=0.5, label='Predicted')
        ax3.set_title('Overlay of Point Clouds')
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
                vis_interval=5, log_interval=10):
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
            'log_beta_scales': 1.0
        }
    
    # Create a SMAL3DFitter for computing chamfer loss
    smal_fitter = load_smil_model(batch_size=1, device=device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'param_losses': {key: [] for key in weights.keys()},
        'chamfer_loss': []
    }
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        param_losses = {key: 0.0 for key in weights.keys()}
        chamfer_losses = 0.0
        
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
            mse_loss = 0.0
            for key in weights.keys():
                if key in pred_params and key in param_tensors:
                    param_loss = F.mse_loss(pred_params[key], param_tensors[key])
                    weighted_loss = weights[key] * param_loss
                    mse_loss += weighted_loss
                    param_losses[key] += weighted_loss.item()
            
            # Compute chamfer distance loss between predicted and input point clouds
            if chamfer_weight > 0:
                cd_loss = compute_chamfer_loss(smal_fitter, pred_params, point_clouds, device=device)
                chamfer_losses += cd_loss.item()
                
                # Combine losses
                loss = mse_loss + chamfer_weight * cd_loss
            else:
                loss = mse_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Log progress
            if (i + 1) % log_interval == 0:
                if chamfer_weight > 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                          f'MSE Loss: {mse_loss.item():.4f}, Chamfer Loss: {cd_loss.item():.4f}, '
                          f'Total Loss: {loss.item():.4f}')
                else:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                          f'Loss: {loss.item():.4f}')
        
        # Average loss for the epoch
        train_loss /= len(train_loader)
        for key in param_losses:
            param_losses[key] /= len(train_loader)
        if chamfer_weight > 0:
            chamfer_losses /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_chamfer_loss = 0.0
        
        with torch.no_grad():
            for point_clouds, param_dicts in val_loader:
                # Move data to device
                point_clouds = point_clouds.to(device)
                param_tensors = {key: param_dicts[key].to(device) for key in param_dicts}
                
                # Forward pass
                pred_params = model(point_clouds)
                
                # Compute MSE loss
                mse_loss = 0.0
                for key in weights.keys():
                    if key in pred_params and key in param_tensors:
                        param_loss = F.mse_loss(pred_params[key], param_tensors[key])
                        mse_loss += weights[key] * param_loss
                
                # Compute chamfer distance loss
                if chamfer_weight > 0:
                    cd_loss = compute_chamfer_loss(smal_fitter, pred_params, point_clouds, device=device)
                    val_chamfer_loss += cd_loss.item()
                    
                    # Combine losses
                    loss = mse_loss + chamfer_weight * cd_loss
                else:
                    loss = mse_loss
                
                val_loss += loss.item()
        
        # Average validation loss
        val_loss /= len(val_loader)
        if chamfer_weight > 0:
            val_chamfer_loss /= len(val_loader)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Update training history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        for key in param_losses:
            history['param_losses'][key].append(param_losses[key])
        if chamfer_weight > 0:
            history['chamfer_loss'].append(chamfer_losses)
        
        # Print progress
        if chamfer_weight > 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
                  f'Train Chamfer Loss: {chamfer_losses:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Val Chamfer Loss: {val_chamfer_loss:.4f}')
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
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


# ----------------------- MODEL EVALUATION AND VISUALIZATION ----------------------- #

def visualize_predictions(model, test_loader, device, smal_fitter, num_samples=5):
    """
    Visualize model predictions by comparing the predicted and ground truth SMIL models.
    
    Args:
        model: Trained SMILPointNet model
        test_loader: DataLoader for test data
        device: Device to use for computation
        smal_fitter: SMAL3DFitter object for visualization
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    with torch.no_grad():
        # Get a batch from the test loader
        point_clouds, true_params = next(iter(test_loader))
        
        # Only use the specified number of samples
        point_clouds = point_clouds[:num_samples].to(device)
        
        # Forward pass
        pred_params = model(point_clouds)
        
        # Visualize each sample
        for i in range(num_samples):
            # Create a figure with two subplots side by side
            fig = plt.figure(figsize=(12, 6))
            
            # Ground truth mesh
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.set_title('Ground Truth SMIL Model')
            
            # Apply true parameters to SMAL fitter
            for key in true_params:
                param_tensor = getattr(smal_fitter, key)
                param_tensor.data[0] = true_params[key][i].to(device)
            
            # Forward pass to get vertices
            with torch.no_grad():
                true_verts = smal_fitter()
            
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
            for key in pred_params:
                param_tensor = getattr(smal_fitter, key)
                if key == 'log_beta_scales' and model.include_scales:
                    # Make sure dimensions match
                    pred_shape = pred_params[key][i].shape
                    if pred_shape[0] != param_tensor.shape[1]:
                        # Skip if dimensions don't match
                        continue
                param_tensor.data[0] = pred_params[key][i].to(device)
            
            # Forward pass to get vertices
            with torch.no_grad():
                pred_verts = smal_fitter()
            
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


# ----------------------- MAIN FUNCTION ----------------------- #

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train SMIL PointNet model')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs to train (default: 50)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of worker processes for DataLoader (default: 4, 0 for no multiprocessing)')
    parser.add_argument('--seed', type=int, default=13,
                       help='Random seed (default: 0)')
    parser.add_argument('--train-samples', type=int, default=10000,
                       help='Number of training samples (default: 1000)')
    parser.add_argument('--val-samples', type=int, default=1000,
                       help='Number of validation samples (default: 100)')
    parser.add_argument('--test-samples', type=int, default=1000,
                       help='Number of test samples (default: 100)')
    parser.add_argument('--num-points', type=int, default=3000,
                       help='Number of points in each point cloud (default: 3000)')
    parser.add_argument('--vis-interval', type=int, default=5,
                       help='Visualization interval in epochs (default: 5)')
    parser.add_argument('--no-multiprocessing', action='store_true',
                       help='Disable multiprocessing in DataLoader (equivalent to --num-workers=0)')
    return parser.parse_args()

def main():
    """
    Main function to train and evaluate the SMIL PointNet model.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Parameters from command-line arguments
    num_train_samples = args.train_samples
    num_val_samples = args.val_samples
    num_test_samples = args.test_samples
    num_points = args.num_points
    batch_size = args.batch_size
    num_epochs = args.epochs
    vis_interval = args.vis_interval
    
    # Number of worker processes for DataLoader
    num_workers = 0 if args.no_multiprocessing else args.num_workers
    print(f"Using {num_workers} worker processes for DataLoader")
    
    # Create datasets
    train_dataset = SMILDataset(num_samples=num_train_samples, num_points=num_points, 
                                device=device, seed=seed)
    val_dataset = SMILDataset(num_samples=num_val_samples, num_points=num_points, 
                              device=device, seed=seed+1)
    test_dataset = SMILDataset(num_samples=num_test_samples, num_points=num_points, 
                               device=device, seed=seed+2)
    
    # Get model configuration
    n_betas = train_dataset.n_betas
    n_pose = train_dataset.n_pose
    n_joints = train_dataset.n_joints
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    
    # Create model
    model = SMILPointNet(num_points=num_points, n_betas=n_betas, n_pose=n_pose, 
                        include_scales=config.ALLOW_LIMB_SCALING).to(device)
    
    # Set joint scales size if using scales
    if config.ALLOW_LIMB_SCALING:
        model.set_joint_scales_size(n_joints)
    
    # Define loss weights
    loss_weights = {
        'global_rot': 0.1,
        'joint_rot': 0.5,
        'betas': 1,
        'trans': 0.1,
        'log_beta_scales': 0.5 if config.ALLOW_LIMB_SCALING else 0.0
    }
    
    # Set chamfer loss weight
    chamfer_weight = 10.0  # Adjust this weight as needed
    
    # Train the model
    trained_model, history = train_model(model, train_loader, val_loader, num_epochs=num_epochs, 
                                          lr=0.001, device=device, weights=loss_weights,
                                          chamfer_weight=chamfer_weight, vis_interval=vis_interval)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    smal_fitter = load_smil_model(batch_size=1, device=device)
    visualize_predictions(trained_model, test_loader, device, smal_fitter)
    
    print("Training and evaluation completed.")


if __name__ == "__main__":
    main() 