"""
Training script for SMIL Image Regressor

This script demonstrates how to train the SMILImageRegressor network
to predict SMIL parameters from input images.
"""

# Set matplotlib backend BEFORE any other imports to prevent tkinter issues
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent tkinter issues in multiprocessing
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import random
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import imageio

# Add the parent directories to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smil_image_regressor import SMILImageRegressor, rotation_6d_to_axis_angle
from smil_datasets import replicAntSMILDataset
from smal_fitter import SMALFitter
from Unreal2Pytorch3D import return_placeholder_data
import config
from training_config import TrainingConfig


def set_random_seeds(seed=0):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value (default: 0)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, loss_weights):
    """
    Train the model for one epoch using batch processing.
    
    Args:
        model: SMILImageRegressor model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: PyTorch device
        epoch: Current epoch number
        loss_weights: Dictionary of loss weights for different components
        
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
            # Zero gradients at the start of each batch
            optimizer.zero_grad()
            
            # Process batch
            result = model.predict_from_batch(x_data_batch, y_data_batch)
            
            if result[0] is None:  # No valid samples in batch
                continue

            # Extract results from batch
            predicted_params, target_params_batch, auxiliary_data = result
            
            # Compute batch loss 
            loss, loss_components = model.compute_batch_loss(
                predicted_params, target_params_batch, auxiliary_data, 
                return_components=True, loss_weights=loss_weights
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Record loss and parameter errors
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            
            # Accumulate parameter errors
            for param_name, param_loss in loss_components.items():
                if param_name not in param_errors:
                    param_errors[param_name] = 0.0
                param_errors[param_name] += param_loss.item()
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{batch_loss:.6f}'})
            
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


def validate_epoch(model, val_loader, criterion, device, epoch, loss_weights):
    """
    Validate the model for one epoch
    
    Args:
        model: SMILImageRegressor model
        val_loader: Validation data loader
        criterion: Loss function
        device: PyTorch device
        epoch: Current epoch number
        loss_weights: Dictionary of loss weights for different components
        
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
                # Process batch
                result = model.predict_from_batch(x_data_batch, y_data_batch)
                
                if result[0] is None:  # No valid samples in batch
                    continue
                    
                predicted_params, target_params_batch, auxiliary_data = result
                
                # Compute batch loss
                loss, loss_components = model.compute_batch_loss(
                    predicted_params, target_params_batch, auxiliary_data, 
                    return_components=True, loss_weights=loss_weights
                )
                
                # Record loss and parameter errors
                batch_loss = loss.item()
                total_loss += batch_loss
                num_batches += 1
                
                # Accumulate parameter errors
                for param_name, param_loss in loss_components.items():
                    if param_name not in param_errors:
                        param_errors[param_name] = 0.0
                    param_errors[param_name] += param_loss.item()
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{batch_loss:.6f}'})
                
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
                # Convert rotations to axis-angle if they're in 6D representation
                if model.rotation_representation == '6d':
                    # Convert 6D rotations to axis-angle for SMALFitter
                    global_rot_aa = rotation_6d_to_axis_angle(predicted_params['global_rot'][0:1])
                    joint_rot_aa = rotation_6d_to_axis_angle(predicted_params['joint_rot'][0:1])
                else:
                    # Already in axis-angle format
                    global_rot_aa = predicted_params['global_rot'][0:1]
                    joint_rot_aa = predicted_params['joint_rot'][0:1]
                
                temp_fitter.global_rotation.data = global_rot_aa
                temp_fitter.joint_rotations.data = joint_rot_aa
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


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """
    Load checkpoint and restore model, optimizer state and training history.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        model: SMILImageRegressor model to load state into
        optimizer: Optimizer to load state into
        device: PyTorch device
        
    Returns:
        Tuple of (start_epoch, train_losses, val_losses, train_param_errors, val_param_errors, best_val_loss)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded successfully")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded successfully")
        
        # Update learning rate to match curriculum for the resumed epoch
        resumed_lr = TrainingConfig.get_learning_rate_for_epoch(start_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = resumed_lr
        print(f"Learning rate set to {resumed_lr} for resumed epoch {start_epoch}")
        
        # Get epoch information
        start_epoch = checkpoint.get('epoch', 0) + 1  # Resume from next epoch
        
        # Initialize training history lists
        train_losses = []
        val_losses = []
        train_param_errors = []
        val_param_errors = []
        best_val_loss = float('inf')
        
        # Try to load training history if available
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            print(f"Loaded training history with {len(train_losses)} epochs")
        
        if 'val_losses' in checkpoint:
            val_losses = checkpoint['val_losses']
            print(f"Loaded validation history with {len(val_losses)} epochs")
        
        if 'train_param_errors_history' in checkpoint:
            train_param_errors = checkpoint['train_param_errors_history']
            print(f"Loaded training parameter error history with {len(train_param_errors)} epochs")
        
        if 'val_param_errors_history' in checkpoint:
            val_param_errors = checkpoint['val_param_errors_history']
            print(f"Loaded validation parameter error history with {len(val_param_errors)} epochs")
        
        # Get best validation loss
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        elif 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
        elif val_losses:
            best_val_loss = min(val_losses)
        
        print(f"Resuming training from epoch {start_epoch}")
        print(f"Best validation loss so far: {best_val_loss:.6f}")
        
        return start_epoch, train_losses, val_losses, train_param_errors, val_param_errors, best_val_loss
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting training from scratch...")
        return 0, [], [], [], [], float('inf')


def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, train_param_err, val_param_err,
                   train_losses, val_losses, train_param_errors, val_param_errors, best_val_loss, 
                   checkpoint_path):
    """
    Save training checkpoint with complete state.
    
    Args:
        epoch: Current epoch number
        model: SMILImageRegressor model
        optimizer: Optimizer
        train_loss: Current training loss
        val_loss: Current validation loss
        train_param_err: Current training parameter errors
        val_param_err: Current validation parameter errors
        train_losses: Full training loss history
        val_losses: Full validation loss history
        train_param_errors: Full training parameter error history
        val_param_errors: Full validation parameter error history
        best_val_loss: Best validation loss so far
        checkpoint_path: Path to save checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_param_errors': train_param_err,
        'val_param_errors': val_param_err,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_param_errors_history': train_param_errors,
        'val_param_errors_history': val_param_errors,
        'best_val_loss': best_val_loss,
    }, checkpoint_path)


def main(dataset_name=None, checkpoint_path=None, config_override=None):
    """
    Main training function.
    
    Args:
        dataset_name (str): Name of the dataset to use (default: uses TrainingConfig.DEFAULT_DATASET)
        checkpoint_path (str): Path to checkpoint file to resume training from (default: None)
        config_override (dict): Dictionary to override specific config values (default: None)
    """
    # Load training configuration
    training_config = TrainingConfig.get_all_config(dataset_name)
    
    # Apply any config overrides
    if config_override:
        for key, value in config_override.items():
            if key in training_config:
                if isinstance(training_config[key], dict):
                    training_config[key].update(value)
                else:
                    training_config[key] = value
    
    # Print configuration summary
    TrainingConfig.print_config_summary(dataset_name)
    
    # Extract configuration values
    data_path = training_config['data_path']
    training_params = training_config['training_params']
    model_config = training_config['model_config']
    output_config = training_config['output_config']
    
    # Use checkpoint from config if not provided as argument
    if checkpoint_path is None:
        checkpoint_path = training_params.get('resume_checkpoint')
    
    # Handle test mode - disable checkpoint loading and use temp directories
    is_test_mode = (checkpoint_path == 'DISABLE_CHECKPOINT_LOADING' or 
                   os.environ.get('PYTEST_TEMP_DIR') is not None)
    
    if is_test_mode:
        checkpoint_path = None  # Disable checkpoint loading in test mode
        # Use temporary directory for outputs if in test mode
        if 'PYTEST_TEMP_DIR' in os.environ:
            temp_dir = os.environ['PYTEST_TEMP_DIR']
            output_config['checkpoint_dir'] = os.path.join(temp_dir, 'checkpoints')
            output_config['plots_dir'] = os.path.join(temp_dir, 'plots')
            output_config['visualizations_dir'] = os.path.join(temp_dir, 'visualizations')
            print(f"Test mode: Using temporary directory: {temp_dir}")
    
    # Set random seeds for reproducibility
    seed = training_params['seed']
    set_random_seeds(seed)
    print(f"Random seed set to: {seed}")
    
    # Set device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_IDS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Dataset parameters
    batch_size = training_params['batch_size']
    num_epochs = training_params['num_epochs']
    learning_rate = training_params['learning_rate']
    rotation_representation = training_params['rotation_representation']
    
    # Create dataset
    print("Loading dataset...")
    print(f"Using rotation representation: {rotation_representation}")
    dataset = replicAntSMILDataset(data_path, rotation_representation=rotation_representation)
    print(f"Dataset size: {len(dataset)}")
    
    # Split dataset using configuration
    train_amount, val_amount, test_amount = TrainingConfig.get_train_val_test_sizes(len(dataset))
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_amount, val_amount, test_amount]
    )
    
    print(f"Train set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")
    print(f"Test set: {len(test_set)} samples")
    
    # Create data loaders with optimized multiprocessing
    # Use configuration parameters for data loading optimization
    num_workers = training_params.get('num_workers', min(8, os.cpu_count() or 4))
    pin_memory = training_params.get('pin_memory', True)
    prefetch_factor = training_params.get('prefetch_factor', 2)
    
    # Ensure matplotlib backend is set for multiprocessing safety
    import matplotlib
    matplotlib.use('Agg')
    
    print(f"Data loading configuration:")
    print(f"  Workers: {num_workers}")
    print(f"  Pin memory: {pin_memory}")
    print(f"  Prefetch factor: {prefetch_factor}")
    print(f"  Matplotlib backend: {matplotlib.get_backend()}")
    
    # Try to create data loaders with multiprocessing, fallback to single-threaded if issues
    try:
        train_loader = DataLoader(
            train_set, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            collate_fn=custom_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,  # Only use persistent workers if multiprocessing
            prefetch_factor=prefetch_factor
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            collate_fn=custom_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor
        )
        test_loader = DataLoader(
            test_set, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            collate_fn=custom_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor
        )
        print(f"Successfully created data loaders with {num_workers} workers")
    except Exception as e:
        print(f"Warning: Failed to create multiprocessing data loaders: {e}")
        print("Falling back to single-threaded data loading...")
        num_workers = 0
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
        use_unity_prior=model_config['use_unity_prior'],
        rgb_only=model_config['rgb_only'],
        freeze_backbone=model_config['freeze_backbone'],
        hidden_dim=model_config['hidden_dim'],
        use_ue_scaling=dataset.get_ue_scaling_flag(),
        rotation_representation=rotation_representation,
        input_resolution=dataset.get_input_resolution()
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize optimizer and loss function
    # Use the learning rate from curriculum (base learning rate for epoch 0)
    initial_lr = TrainingConfig.get_learning_rate_for_epoch(0)
    optimizer = optim.Adam(model.get_trainable_parameters(), lr=initial_lr)
    criterion = nn.MSELoss()
    
    print(f"Initial learning rate set to: {initial_lr}")
    
    # Create output directories
    os.makedirs(output_config['checkpoint_dir'], exist_ok=True)
    os.makedirs(output_config['plots_dir'], exist_ok=True)
    os.makedirs(output_config['visualizations_dir'], exist_ok=True)
    
    # Initialize training state
    start_epoch = 0
    train_losses = []
    val_losses = []
    train_param_errors = []
    val_param_errors = []
    best_val_loss = float('inf')
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        start_epoch, train_losses, val_losses, train_param_errors, val_param_errors, best_val_loss = load_checkpoint(
            checkpoint_path, model, optimizer, device
        )
    
    print("Starting training...")
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Training from scratch")
    
    for epoch in range(start_epoch, num_epochs):
        # Get loss weights and learning rate for current epoch from configuration
        loss_weights = TrainingConfig.get_loss_weights_for_epoch(epoch)
        current_lr = TrainingConfig.get_learning_rate_for_epoch(epoch)
        
        # Update learning rate if it has changed
        for param_group in optimizer.param_groups:
            if param_group['lr'] != current_lr:
                param_group['lr'] = current_lr
                print(f"Learning rate updated to {current_lr} at epoch {epoch}")
        
        # Train
        train_loss, train_param_err = train_epoch(model, train_loader, optimizer, criterion, device, epoch, loss_weights)
        train_losses.append(train_loss)
        train_param_errors.append(train_param_err)
        
        # Validate
        val_loss, val_param_err = validate_epoch(model, val_loader, criterion, device, epoch, loss_weights)
        val_losses.append(val_loss)
        val_param_errors.append(val_param_err)
        
        # Print epoch summary with parameter errors
        print(f'\nEpoch {epoch}:')
        print(f'  Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {current_lr:.2e}')
        print('  Parameter Errors (Train / Val):')
        
        # Get all parameter names from both train and val
        all_params = set(train_param_err.keys()) | set(val_param_err.keys())
        for param_name in sorted(all_params):
            train_err = train_param_err.get(param_name, 0.0)
            val_err = val_param_err.get(param_name, 0.0)
            print(f'    {param_name:15s}: {train_err:.6f} / {val_err:.6f}')
        
        # Save best model (unless in test mode)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not is_test_mode:
                best_model_path = os.path.join(output_config['checkpoint_dir'], 'best_model.pth')
                save_checkpoint(epoch, model, optimizer, train_loss, val_loss, train_param_err, val_param_err,
                              train_losses, val_losses, train_param_errors, val_param_errors, best_val_loss,
                              best_model_path)
                print(f'  New best model saved with validation loss: {val_loss:.6f}')
            else:
                print(f'  New best validation loss: {val_loss:.6f} (checkpoint saving disabled in test mode)')
        
        # Save regular checkpoint (unless in test mode)
        if (epoch + 1) % output_config['save_checkpoint_every'] == 0:
            if not is_test_mode:
                checkpoint_path_save = os.path.join(output_config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
                save_checkpoint(epoch, model, optimizer, train_loss, val_loss, train_param_err, val_param_err,
                              train_losses, val_losses, train_param_errors, val_param_errors, best_val_loss,
                              checkpoint_path_save)
                print(f'  Checkpoint saved at epoch {epoch}')
            else:
                print(f'  Checkpoint saving disabled in test mode (epoch {epoch})')
        
        # Generate visualizations
        if (epoch + 1) % output_config['generate_visualizations_every'] == 0:
            visualize_training_progress(model, val_loader, device, epoch, 
                                      output_dir=output_config['visualizations_dir'], 
                                      num_samples=output_config['num_visualization_samples'])
        
        # Plot training history (unless in test mode)
        if (epoch + 1) % output_config['plot_history_every'] == 0:
            if not is_test_mode:
                history_path = os.path.join(output_config['plots_dir'], f'training_history_epoch_{epoch}.png')
                param_errors_path = os.path.join(output_config['plots_dir'], f'parameter_errors_epoch_{epoch}.png')
                plot_training_history(train_losses, val_losses, history_path)
                plot_parameter_errors(train_param_errors, val_param_errors, param_errors_path)
    
    print("Training completed!")
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    # Use final epoch loss weights for test evaluation
    final_loss_weights = TrainingConfig.get_loss_weights_for_epoch(num_epochs - 1)
    test_loss, test_param_err = validate_epoch(model, test_loader, criterion, device, 'test', final_loss_weights)
    print(f'Test Loss: {test_loss:.6f}')
    print('Test Parameter Errors:')
    for param_name, param_err in sorted(test_param_err.items()):
        print(f'  {param_name:15s}: {param_err:.6f}')
    
    # Save final training history (unless in test mode)
    if not is_test_mode:
        final_history_path = os.path.join(output_config['plots_dir'], 'final_training_history.png')
        final_param_errors_path = os.path.join(output_config['plots_dir'], 'final_parameter_errors.png')
        plot_training_history(train_losses, val_losses, final_history_path)
        plot_parameter_errors(train_param_errors, val_param_errors, final_param_errors_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SMIL Image Regressor')
    parser.add_argument('--dataset', type=str, default=None, 
                       choices=['masked_simple', 'pose_only_simple', 'test_textured'],
                       help='Dataset to use (default: uses TrainingConfig.DEFAULT_DATASET)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file to resume training from (overrides config setting, default: None)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (overrides config default)')
    parser.add_argument('--rotation-representation', type=str, default=None, 
                       choices=['6d', 'axis_angle'],
                       help='Rotation representation for joint rotations (overrides config default)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config default)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (overrides config default)')
    parser.add_argument('--num-epochs', type=int, default=None,
                       help='Number of epochs (overrides config default)')
    args = parser.parse_args()
    
    # Create config override dictionary for any provided arguments
    config_override = {}
    if args.seed is not None or args.rotation_representation is not None or \
       args.batch_size is not None or args.learning_rate is not None or args.num_epochs is not None:
        config_override['training_params'] = {}
        if args.seed is not None:
            config_override['training_params']['seed'] = args.seed
        if args.rotation_representation is not None:
            config_override['training_params']['rotation_representation'] = args.rotation_representation
        if args.batch_size is not None:
            config_override['training_params']['batch_size'] = args.batch_size
        if args.learning_rate is not None:
            config_override['training_params']['learning_rate'] = args.learning_rate
        if args.num_epochs is not None:
            config_override['training_params']['num_epochs'] = args.num_epochs
    
    main(dataset_name=args.dataset, checkpoint_path=args.checkpoint, config_override=config_override or None)
