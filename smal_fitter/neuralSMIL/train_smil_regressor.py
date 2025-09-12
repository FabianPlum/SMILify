"""
Training script for SMIL Image Regressor

This script demonstrates how to train the SMILImageRegressor network
to predict SMIL parameters from input images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import imageio
import trimesh

# Add the parent directories to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smil_image_regressor import SMILImageRegressor
from smil_datasets import replicAntSMILDataset
from smal_fitter import SMALFitter
from Unreal2Pytorch3D import return_placeholder_data
import config


class ImageExporter():
    """Simple image exporter for visualization during training."""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces, img_idx=0):
        """Export visualization image."""
        imageio.imsave(os.path.join(self.output_dir, f"img_{img_idx:04d}_epoch_{global_id:04d}.png"), collage_np)


def create_placeholder_data_batch(batch_size=1, image_size=512):
    """
    Create placeholder data batch for SMALFitter initialization.
    
    Args:
        batch_size: Batch size
        image_size: Image size (assumed square)
        
    Returns:
        RGB tensor for rgb_only mode
    """
    # For rgb_only=True, we only need the RGB tensor
    rgb = torch.zeros((batch_size, 3, image_size, image_size))
    return rgb


def safe_to_tensor(data, dtype=torch.float32, device='cpu'):
    """
    Safely convert data to PyTorch tensor, handling both numpy arrays and existing tensors.
    
    Args:
        data: Input data (numpy array, tensor, or scalar)
        dtype: Target dtype
        device: Target device
        
    Returns:
        PyTorch tensor
    """
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype, device=device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data.astype(np.float32 if dtype == torch.float32 else data.dtype)).to(dtype=dtype, device=device)
    else:
        return torch.tensor(data, dtype=dtype, device=device)

def extract_target_parameters(y_data, device):
    """
    Extract target SMIL parameters from dataset y_data.
    
    Args:
        y_data: Dictionary containing SMIL data from dataset
        device: PyTorch device
        
    Returns:
        Dictionary containing target parameters as tensors
    """
    targets = {}
    
    # Global rotation (root rotation)
    targets['global_rot'] = safe_to_tensor(y_data['root_rot'], device=device).unsqueeze(0)
    
    # Joint rotations (excluding root joint)
    joint_angles = safe_to_tensor(y_data['joint_angles'], device=device)
    targets['joint_rot'] = joint_angles[1:].unsqueeze(0)  # Exclude root joint
    
    # Shape parameters
    targets['betas'] = safe_to_tensor(y_data['shape_betas'], device=device).unsqueeze(0)
    
    # Translation (root location)
    targets['trans'] = safe_to_tensor(y_data['root_loc'], device=device).unsqueeze(0)
    
    # Camera FOV
    fov_value = y_data['cam_fov']
    if isinstance(fov_value, list):
        fov_value = fov_value[0]  # Take first element if it's a list
    targets['fov'] = torch.tensor([fov_value], dtype=torch.float32).to(device)  # Keep torch.tensor for scalar construction
    
    # Camera rotation (in model space) - preserve as rotation matrix for FoVPerspectiveCamera compatibility
    cam_rot_matrix = y_data['cam_rot']
    # Ensure it's a 3x3 rotation matrix format
    if hasattr(cam_rot_matrix, 'shape') and len(cam_rot_matrix.shape) == 2 and cam_rot_matrix.shape == (3, 3):
        # It's already a 3x3 rotation matrix - keep as-is
        targets['cam_rot'] = safe_to_tensor(cam_rot_matrix, device=device).unsqueeze(0)
    else:
        # If it's axis-angle or other format, convert to rotation matrix
        if hasattr(cam_rot_matrix, 'shape') and cam_rot_matrix.shape == (3,):
            # It's axis-angle, convert to rotation matrix
            r = Rotation.from_rotvec(cam_rot_matrix)
            cam_rot_matrix = r.as_matrix()
            targets['cam_rot'] = safe_to_tensor(cam_rot_matrix, device=device).unsqueeze(0)
        else:
            # Unknown format - assume it needs to be reshaped to 3x3
            targets['cam_rot'] = safe_to_tensor(cam_rot_matrix, device=device).unsqueeze(0)
    
    # Camera translation (in model space)
    targets['cam_trans'] = safe_to_tensor(y_data['cam_trans'], device=device).unsqueeze(0)
    
    # Joint scales and translations (if available)
    if y_data['scale_weights'] is not None and y_data['trans_weights'] is not None:
        # These would need to be converted from PCA weights to actual scales/translations
        # For now, we'll use placeholder values
        n_joints = config.N_POSE + 1
        targets['log_beta_scales'] = torch.zeros(1, n_joints, 3).to(device)
        targets['betas_trans'] = torch.zeros(1, n_joints, 3).to(device)
    
    return targets


def custom_collate_fn(batch):
    """
    Custom collate function to handle the dataset format.
    
    Args:
        batch: List of (x_data, y_data) tuples
        
    Returns:
        Tuple of (x_data_batch, y_data_batch)
    """
    x_data_batch = []
    y_data_batch = []
    
    for x_data, y_data in batch:
        x_data_batch.append(x_data)
        y_data_batch.append(y_data)
    
    return x_data_batch, y_data_batch


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    Train the model for one epoch.
    
    Args:
        model: SMILImageRegressor model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: PyTorch device
        epoch: Current epoch number
        
    Returns:
        Tuple of (average training loss, average parameter errors)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Track parameter-specific errors
    param_errors = {}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (x_data_batch, y_data_batch) in enumerate(pbar):
        try:
            # Process each sample in the batch
            batch_loss = 0.0
            batch_param_errors = {}
            valid_samples = 0
            accumulated_loss = None
            
            # Zero gradients at the start of each batch
            optimizer.zero_grad()
            
            for i, (x_data, y_data) in enumerate(zip(x_data_batch, y_data_batch)):
                # Extract image data
                if x_data['input_image_data'] is not None:
                    image_data = x_data['input_image_data']
                else:
                    # Skip if no image data
                    continue
                
                # Preprocess image (single image)
                image_tensor = model.preprocess_image(image_data).to(device)
                
                # Extract target parameters
                target_params = extract_target_parameters(y_data, device)
                
                # Forward pass
                predicted_params = model(image_tensor)
                
                # Prepare keypoint data for loss computation
                keypoint_data = None
                if 'keypoints_2d' in y_data and 'keypoint_visibility' in y_data:
                    keypoint_data = {
                        'keypoints_2d': y_data['keypoints_2d'],
                        'keypoint_visibility': y_data['keypoint_visibility']
                    }
                
                # Compute loss with components (including keypoint loss if keypoint_data is available)
                loss, loss_components = model.compute_prediction_loss(predicted_params, target_params, pose_data=keypoint_data, return_components=True)
                batch_loss += loss.item()
                valid_samples += 1
                
                # Accumulate gradients (divide by batch size to get average)
                loss = loss / len(x_data_batch)  # Normalize by batch size
                if accumulated_loss is None:
                    accumulated_loss = loss
                else:
                    accumulated_loss += loss
                
                # Accumulate parameter errors
                for param_name, param_loss in loss_components.items():
                    if param_name not in batch_param_errors:
                        batch_param_errors[param_name] = 0.0
                    batch_param_errors[param_name] += param_loss.item()
            
            if valid_samples > 0 and accumulated_loss is not None:
                # Average parameter errors over valid samples
                for param_name in batch_param_errors:
                    batch_param_errors[param_name] /= valid_samples
                    if param_name not in param_errors:
                        param_errors[param_name] = 0.0
                    param_errors[param_name] += batch_param_errors[param_name]
                
                # Backward pass on accumulated loss
                accumulated_loss.backward()
                
                # Gradient clipping to prevent instability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Average loss over valid samples
                avg_loss = batch_loss / valid_samples
                total_loss += avg_loss
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{avg_loss:.6f}'})
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Average parameter errors over all batches
    avg_param_errors = {}
    for param_name in param_errors:
        avg_param_errors[param_name] = param_errors[param_name] / max(num_batches, 1)
    
    return total_loss / max(num_batches, 1), avg_param_errors


def validate_epoch(model, val_loader, criterion, device, epoch):
    """
    Validate the model for one epoch.
    
    Args:
        model: SMILImageRegressor model
        val_loader: Validation data loader
        criterion: Loss function
        device: PyTorch device
        epoch: Current epoch number
        
    Returns:
        Tuple of (average validation loss, average parameter errors)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Track parameter-specific errors
    param_errors = {}
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Validation {epoch}')
        for batch_idx, (x_data_batch, y_data_batch) in enumerate(pbar):
            try:
                # Process each sample in the batch
                batch_loss = 0.0
                batch_param_errors = {}
                valid_samples = 0
                
                for i, (x_data, y_data) in enumerate(zip(x_data_batch, y_data_batch)):
                    # Extract image data
                    if x_data['input_image_data'] is not None:
                        image_data = x_data['input_image_data']
                    else:
                        # Skip if no image data
                        continue
                    
                    # Preprocess image (single image)
                    image_tensor = model.preprocess_image(image_data).to(device)
                    
                    # Extract target parameters
                    target_params = extract_target_parameters(y_data, device)
                    
                    # Forward pass
                    predicted_params = model(image_tensor)
                    
                    # Prepare keypoint data for loss computation
                    keypoint_data = None
                    if 'keypoints_2d' in y_data and 'keypoint_visibility' in y_data:
                        keypoint_data = {
                            'keypoints_2d': y_data['keypoints_2d'],
                            'keypoint_visibility': y_data['keypoint_visibility']
                        }
                    
                    # Compute loss with components (including keypoint loss if keypoint_data is available)
                    loss, loss_components = model.compute_prediction_loss(predicted_params, target_params, pose_data=keypoint_data, return_components=True)
                    batch_loss += loss.item()
                    valid_samples += 1
                    
                    # Accumulate parameter errors
                    for param_name, param_loss in loss_components.items():
                        if param_name not in batch_param_errors:
                            batch_param_errors[param_name] = 0.0
                        batch_param_errors[param_name] += param_loss.item()
                
                if valid_samples > 0:
                    # Average loss over valid samples
                    avg_loss = batch_loss / valid_samples
                    
                    # Average parameter errors over valid samples
                    for param_name in batch_param_errors:
                        batch_param_errors[param_name] /= valid_samples
                        if param_name not in param_errors:
                            param_errors[param_name] = 0.0
                        param_errors[param_name] += batch_param_errors[param_name]
                    
                    total_loss += avg_loss
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({'Loss': f'{avg_loss:.6f}'})
                
            except Exception as e:
                print(f"Error processing validation batch {batch_idx}: {e}")
                continue
    
    # Average parameter errors over all batches
    avg_param_errors = {}
    for param_name in param_errors:
        avg_param_errors[param_name] = param_errors[param_name] / max(num_batches, 1)
    
    return total_loss / max(num_batches, 1), avg_param_errors


def plot_training_history(train_losses, val_losses, save_path='training_history.png'):
    """
    Plot training and validation loss history.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def visualize_training_progress(model, val_loader, device, epoch, output_dir='visualizations', num_samples=5):
    """
    Visualize training progress by rendering the first few validation samples.
    
    Args:
        model: SMILImageRegressor model
        val_loader: Validation data loader
        device: PyTorch device
        epoch: Current epoch number
        output_dir: Directory to save visualization images
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Create output directory for this epoch
    epoch_dir = os.path.join(output_dir, f'epoch_{epoch:03d}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Create image exporter
    image_exporter = ImageExporter(epoch_dir)
    
    with torch.no_grad():
        sample_count = 0
        for batch_idx, (x_data_batch, y_data_batch) in enumerate(val_loader):
            if sample_count >= num_samples:
                break
                
            for i, (x_data, y_data) in enumerate(zip(x_data_batch, y_data_batch)):
                if sample_count >= num_samples:
                    break
                    
                # Extract image data
                if x_data['input_image_data'] is not None:
                    image_data = x_data['input_image_data']
                else:
                    continue
                
                # Preprocess image
                image_tensor = model.preprocess_image(image_data).to(device)
                
                # Get prediction from model
                predicted_params = model(image_tensor)
                
                # Create placeholder data for SMALFitter
                rgb = image_tensor.cpu()

                if x_data["input_image_mask"] is not None:
                    temp_batch, filenames = return_placeholder_data(
                        input_image=x_data["input_image"],
                        num_joints=len(y_data["joint_angles"]),
                        keypoints_2d=y_data["keypoints_2d"],
                        keypoint_visibility=y_data["keypoint_visibility"],
                        silhouette=x_data["input_image_mask"]
                    )
                    temp_fitter = SMALFitter(
                        device=device,
                        data_batch=temp_batch,  # For rgb_only=False, use the temp_batch
                        batch_size=1,
                        shape_family=config.SHAPE_FAMILY,
                        use_unity_prior=False,
                        rgb_only=False
                    )
                else:
                    temp_fitter = SMALFitter(
                        device=device,
                        data_batch=rgb,  # For rgb_only=True, just pass the RGB tensor
                        batch_size=1,
                        shape_family=config.SHAPE_FAMILY,
                        use_unity_prior=False,
                        rgb_only=True
                    )
                
                # Set proper target joints and visibility for visualization
                # Convert normalized keypoints back to pixel coordinates for visualization
                if 'keypoints_2d' in y_data and 'keypoint_visibility' in y_data:
                    # Get image dimensions (assuming square image matching the tensor size)
                    image_height, image_width = image_tensor.shape[-2:]
                    
                    # Convert normalized [0,1] coordinates to pixel coordinates
                    keypoints_2d = y_data['keypoints_2d']  # Shape: (num_joints, 2), already in [y_norm, x_norm] format
                    keypoint_visibility = y_data['keypoint_visibility']  # Shape: (num_joints,)
                    
                    # Convert to pixel coordinates (already in [y, x] format expected by draw_joints)
                    pixel_coords = keypoints_2d.copy()
                    pixel_coords[:, 0] = pixel_coords[:, 0] * image_height  # y to pixels
                    pixel_coords[:, 1] = pixel_coords[:, 1] * image_width   # x to pixels
                    
                    temp_fitter.target_joints = torch.tensor(pixel_coords, dtype=torch.float32, device=device).unsqueeze(0)
                    temp_fitter.target_visibility = torch.tensor(keypoint_visibility, dtype=torch.float32, device=device).unsqueeze(0)
                else:
                    # Fallback to zeros if no keypoint data
                    temp_fitter.target_joints = torch.zeros((1, config.N_POSE, 2), device=device)
                    temp_fitter.target_visibility = torch.zeros((1, config.N_POSE), device=device)
                
                # Set the predicted parameters to the SMALFitter
                temp_fitter.global_rotation.data = predicted_params['global_rot'][0:1]
                temp_fitter.joint_rotations.data = predicted_params['joint_rot'][0:1]
                temp_fitter.betas.data = predicted_params['betas'][0:1]
                temp_fitter.trans.data = predicted_params['trans'][0:1]
                temp_fitter.fov.data = predicted_params['fov'][0:1]
                
                # Set joint scales and translations if available
                if 'log_beta_scales' in predicted_params:
                    temp_fitter.log_beta_scales.data = predicted_params['log_beta_scales'][0:1]
                if 'betas_trans' in predicted_params:
                    temp_fitter.betas_trans.data = predicted_params['betas_trans'][0:1]
                
                # Set camera parameters from ground truth
                if 'cam_rot' in y_data and 'cam_trans' in y_data:
                    # Handle both tensor and numpy array cases
                    cam_rot = y_data['cam_rot']
                    cam_trans = y_data['cam_trans']
                    
                    if hasattr(cam_rot, 'cpu'):
                        cam_rot = cam_rot.cpu().numpy()
                    if hasattr(cam_trans, 'cpu'):
                        cam_trans = cam_trans.cpu().numpy()
                    
                    # Ensure correct dimensions for camera parameters
                    # R should be (1, 3, 3) and T should be (1, 3)
                    if cam_rot.ndim == 2:
                        cam_rot = cam_rot[np.newaxis, ...]  # Add batch dimension
                    if cam_trans.ndim == 1:
                        cam_trans = cam_trans[np.newaxis, ...]  # Add batch dimension
                    
                    # Convert to tensors
                    cam_rot_tensor = safe_to_tensor(cam_rot, device=device)
                    cam_trans_tensor = safe_to_tensor(cam_trans, device=device)
                    fov_tensor = torch.tensor([y_data['cam_fov'][0] if isinstance(y_data['cam_fov'], list) else y_data['cam_fov']], dtype=torch.float32, device=device)
                    
                    temp_fitter.renderer.set_camera_parameters(
                        R=cam_rot_tensor,
                        T=cam_trans_tensor,
                        fov=fov_tensor
                    )
                
                # Generate visualization
                temp_fitter.generate_visualization(image_exporter, apply_UE_transform=True, img_idx=sample_count)
                
                sample_count += 1
    
    print(f"Generated {sample_count} visualization images for epoch {epoch} in {epoch_dir}")


def plot_parameter_errors(train_param_errors, val_param_errors, save_path='parameter_errors.png'):
    """
    Plot parameter-specific error history.
    
    Args:
        train_param_errors: List of training parameter error dictionaries
        val_param_errors: List of validation parameter error dictionaries
        save_path: Path to save the plot
    """
    if not train_param_errors or not val_param_errors:
        return
    
    # Get all parameter names
    all_params = set()
    for epoch_errors in train_param_errors:
        all_params.update(epoch_errors.keys())
    for epoch_errors in val_param_errors:
        all_params.update(epoch_errors.keys())
    
    # Create subplots for each parameter
    n_params = len(all_params)
    if n_params == 0:
        return
    
    # Calculate subplot layout
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, param_name in enumerate(sorted(all_params)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Extract parameter errors over epochs
        train_errors = []
        val_errors = []
        
        for epoch_errors in train_param_errors:
            train_errors.append(epoch_errors.get(param_name, 0.0))
        
        for epoch_errors in val_param_errors:
            val_errors.append(epoch_errors.get(param_name, 0.0))
        
        # Plot with log scale
        epochs = range(len(train_errors))
        ax.plot(epochs, train_errors, label='Train', color='blue', alpha=0.7)
        ax.plot(epochs, val_errors, label='Val', color='red', alpha=0.7)
        ax.set_title(f'{param_name}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error (log scale)')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_params, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """
    Main training function.
    """
    # Set device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_IDS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Dataset parameters
    data_path = "/media/fabi/Data/replicAnt-x-SMIL-OmniAnt-Masked"
    #data_path = "data/replicAnt_trials/replicAnt-x-SMIL-TEX"
    batch_size = 32
    num_epochs = 500
    learning_rate = 0.001
    
    # Create dataset
    print("Loading dataset...")
    dataset = replicAntSMILDataset(data_path)
    print(f"Dataset size: {len(dataset)}")
    
    # Split dataset
    test_size = 0.1
    val_size = 0.1
    test_amount = int(len(dataset) * test_size)
    val_amount = int(len(dataset) * val_size)
    train_amount = len(dataset) - test_amount - val_amount
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_amount, val_amount, test_amount]
    )
    
    print(f"Train set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")
    print(f"Test set: {len(test_set)} samples")
    
    # Create data loaders (disable multiprocessing to avoid issues)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    # Create placeholder data for SMALFitter initialization
    placeholder_data = create_placeholder_data_batch(batch_size)
    
    # Initialize model
    print("Initializing model...")
    model = SMILImageRegressor(
        device=device,
        data_batch=placeholder_data,
        batch_size=batch_size,
        shape_family=config.SHAPE_FAMILY,
        use_unity_prior=False,
        rgb_only=True,
        freeze_backbone=True,
        hidden_dim=512
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.get_trainable_parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training history
    train_losses = []
    val_losses = []
    train_param_errors = []
    val_param_errors = []
    
    # Create output directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_param_err = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        train_losses.append(train_loss)
        train_param_errors.append(train_param_err)
        
        # Validate
        val_loss, val_param_err = validate_epoch(model, val_loader, criterion, device, epoch)
        val_losses.append(val_loss)
        val_param_errors.append(val_param_err)
        
        # Print epoch summary with parameter errors
        print(f'\nEpoch {epoch}:')
        print(f'  Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
        print('  Parameter Errors (Train / Val):')
        
        # Get all parameter names from both train and val
        all_params = set(train_param_err.keys()) | set(val_param_err.keys())
        for param_name in sorted(all_params):
            train_err = train_param_err.get(param_name, 0.0)
            val_err = val_param_err.get(param_name, 0.0)
            print(f'    {param_name:15s}: {train_err:.6f} / {val_err:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_param_errors': train_param_err,
                'val_param_errors': val_param_err,
            }, 'checkpoints/best_model.pth')
            print(f'  New best model saved with validation loss: {val_loss:.6f}')
        
        # Generate visualizations every epoch
        visualize_training_progress(model, val_loader, device, epoch, output_dir='visualizations', num_samples=5)
        
        # Plot training history every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_training_history(train_losses, val_losses, f'plots/training_history_epoch_{epoch}.png')
            plot_parameter_errors(train_param_errors, val_param_errors, f'plots/parameter_errors_epoch_{epoch}.png')
    
    print("Training completed!")
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    test_loss, test_param_err = validate_epoch(model, test_loader, criterion, device, 'test')
    print(f'Test Loss: {test_loss:.6f}')
    print('Test Parameter Errors:')
    for param_name, param_err in sorted(test_param_err.items()):
        print(f'  {param_name:15s}: {param_err:.6f}')
    
    # Save final training history
    plot_training_history(train_losses, val_losses, 'plots/final_training_history.png')
    plot_parameter_errors(train_param_errors, val_param_errors, 'plots/final_parameter_errors.png')


if __name__ == "__main__":
    main()
