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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
from smil_datasets import UnifiedSMILDataset
from optimized_dataset import OptimizedSMILDataset
from smal_fitter import SMALFitter
from Unreal2Pytorch3D import return_placeholder_data
import config
from training_config import TrainingConfig
from configs import SingleViewConfig, load_config, save_config_json, apply_smal_file_override, ConfigurationError
from memory_optimization import MemoryOptimizer, recommend_training_config, MixedPrecisionTrainer

# Import SLEAP dataset to enable patching of UnifiedSMILDataset
# Note: This import happens in each worker process due to multiprocessing data loaders
try:
    from sleap_data.sleap_dataset import SLEAPDataset, _patch_unified_dataset
    # Apply the patch to enable SLEAP dataset support (silently to avoid spam from workers)
    _patch_unified_dataset()
except ImportError:
    # SLEAP dataset not available, continue without it
    pass


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


def is_torchrun_launched():
    """
    Check if the script was launched via torchrun/torch.distributed.launch.
    
    When launched via torchrun, environment variables RANK, LOCAL_RANK, and WORLD_SIZE
    are set automatically.

    NOTE: Other distributed training frameworks (e.g. SLURM) may also set these environment variables.
    Thus, this function may or may not work for other frameworks as well. Good luck.
    
    Returns:
        bool: True if launched via torchrun, False otherwise
    """
    return all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])


def setup_ddp(rank, world_size, port=None, local_rank=None):
    """
    Initialize DDP environment.
    
    Args:
        rank: Current process rank (global rank across all nodes)
        world_size: Total number of processes
        port: Master port for communication (default: 12345, ignored if MASTER_PORT env var is set)
        local_rank: Local rank within the node (for GPU assignment). If None, uses rank.
    """
    # Only set MASTER_ADDR/PORT if not already set (e.g., by torchrun or SLURM)
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = port or '12345'
    
    # Initialize process group if not already initialized
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Use local_rank for GPU assignment (important for multi-node setups)
    gpu_rank = local_rank if local_rank is not None else rank
    torch.cuda.set_device(gpu_rank)


def cleanup_ddp():
    """Clean up DDP environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def gather_validation_metrics(val_loss, param_errors, world_size):
    """
    Gather validation metrics from all processes for accurate reporting.
    
    Args:
        val_loss: Validation loss from current process
        param_errors: Parameter errors dict from current process  
        world_size: Total number of processes
        
    Returns:
        Tuple of (averaged validation loss, averaged parameter errors)
    """
    if world_size > 1 and dist.is_initialized():
        # Gather losses from all GPUs
        loss_tensor = torch.tensor(val_loss, device=torch.cuda.current_device())
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = loss_tensor.item() / world_size
        
        # Gather parameter errors
        averaged_param_errors = {}
        for param_name, error_value in param_errors.items():
            error_tensor = torch.tensor(error_value, device=torch.cuda.current_device())
            dist.all_reduce(error_tensor, op=dist.ReduceOp.SUM)
            averaged_param_errors[param_name] = error_tensor.item() / world_size
            
        return avg_val_loss, averaged_param_errors
    else:
        return val_loss, param_errors


def ddp_main(rank, world_size, dataset_name, checkpoint_path, config_override, master_port):
    """
    DDP wrapper around existing main() function.
    
    Supports two launch modes:
    1. mp.spawn (single-node): rank is passed by spawn, local_rank == rank
    2. torchrun/SLURM (multi-node): environment variables are auto-detected and used
    
    When torchrun (or other distributed training frameworks) environment is detected, the environment variables take precedence
    over the passed arguments for rank/world_size/local_rank.
    
    Args:
        rank: Current process rank (may be overridden by env vars if torchrun detected)
        world_size: Total number of processes (may be overridden by env vars)
        dataset_name: Dataset name to use
        checkpoint_path: Path to checkpoint file
        config_override: Configuration overrides
        master_port: Master port for DDP communication (ignored if MASTER_PORT env var is set)
    """
    # Check if running under torchrun/SLURM - if so, use environment variables
    if is_torchrun_launched():
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu_rank = local_rank
    else:
        # mp.spawn mode (single-node) - local_rank == rank
        gpu_rank = rank
    
    setup_ddp(rank, world_size, master_port, local_rank=gpu_rank)
    
    # Modify config for distributed training
    config_override = config_override or {}
    config_override['device_override'] = f"cuda:{gpu_rank}"
    config_override['is_distributed'] = True
    config_override['rank'] = rank
    config_override['world_size'] = world_size
    
    try:
        # Call existing main() with minimal modifications
        main(dataset_name, checkpoint_path, config_override)
    finally:
        cleanup_ddp()


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

def extract_target_parameters(y_data, device, scale_trans_mode='separate'):
    """
    Extract target SMIL parameters from dataset y_data.
    
    Args:
        y_data: Dictionary containing SMIL data from dataset
        device: PyTorch device
        scale_trans_mode: Mode for handling scale and translation betas ('ignore', 'separate', 'entangled_with_betas')
        
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
    
    # Handle scale and translation parameters based on mode
    if scale_trans_mode == 'ignore':
        # Set to zeros - no scaling or translation
        # Use PCA weights (5 parameters) set to zero to match predicted dimensions
        targets['log_beta_scales'] = torch.zeros(1, config.N_BETAS).to(device)
        targets['betas_trans'] = torch.zeros(1, config.N_BETAS).to(device)
        
    elif scale_trans_mode == 'separate':
        # In separate mode, use PCA weights directly as targets
        if y_data['scale_weights'] is not None and y_data['trans_weights'] is not None:
            # Use the PCA weights directly (5 parameters each)
            targets['log_beta_scales'] = torch.from_numpy(y_data['scale_weights']).unsqueeze(0).float().to(device)
            targets['betas_trans'] = torch.from_numpy(y_data['trans_weights']).unsqueeze(0).float().to(device)
        else:
            # No PCA weights available, use zeros
            targets['log_beta_scales'] = torch.zeros(1, config.N_BETAS).to(device)
            targets['betas_trans'] = torch.zeros(1, config.N_BETAS).to(device)
            
    elif scale_trans_mode == 'entangled_with_betas':
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
            targets['log_beta_scales'] = torch.from_numpy(np.log(np.maximum(scale_out, 1e-8))).unsqueeze(0).float().to(device)
            targets['betas_trans'] = torch.from_numpy(translation_out * y_data['translation_factor']).unsqueeze(0).float().to(device)
        else:
            n_joints = len(config.dd["J_names"])
            targets['log_beta_scales'] = torch.zeros(1, n_joints, 3).to(device)
            targets['betas_trans'] = torch.zeros(1, n_joints, 3).to(device)
    
    return targets


def custom_collate_fn(batch):
    """
    Custom collate function to handle the dataset format.
    Preserves metadata fields for multi-dataset training (dataset_source, available_labels).
    
    Args:
        batch: List of (x_data, y_data) tuples
        
    Returns:
        Tuple of (x_data_batch, y_data_batch)
        - x_data contains 'dataset_source' and 'available_labels' for multi-dataset training
        - y_data may contain None values for unavailable labels
    """
    x_data_batch = []
    y_data_batch = []
    
    for x_data, y_data in batch:
        x_data_batch.append(x_data)
        y_data_batch.append(y_data)
    
    # Debug: Print metadata for first sample in first few batches (multi-dataset mode)
    # Only print on main process to avoid duplicate output in multi-GPU training
    should_print = True
    if dist.is_initialized():
        should_print = dist.get_rank() == 0
    
    if should_print:
        if hasattr(custom_collate_fn, '_batch_count'):
            custom_collate_fn._batch_count += 1
        else:
            custom_collate_fn._batch_count = 1
        
        if custom_collate_fn._batch_count <= 3 and len(x_data_batch) > 0:
            first_sample = x_data_batch[0]
            if 'dataset_source' in first_sample:
                # Multi-dataset mode - show batch composition
                sources = [x.get('dataset_source', 'unknown') for x in x_data_batch]
                source_counts = {}
                for src in sources:
                    source_counts[src] = source_counts.get(src, 0) + 1
                print(f"\nBatch {custom_collate_fn._batch_count} composition: {source_counts}")
    
    return x_data_batch, y_data_batch


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, loss_weights, is_distributed=False, rank=0):
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
    
    # Only show progress bar on main process to avoid conflicts
    if is_distributed and rank != 0:
        pbar = train_loader  # No progress bar for non-main processes
    else:
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (x_data_batch, y_data_batch) in enumerate(pbar):
        try:
            # Zero gradients at the start of each batch
            optimizer.zero_grad()
            
            # Process batch (handle DDP model)
            base_model = model.module if is_distributed else model
            result = base_model.predict_from_batch(x_data_batch, y_data_batch)
            
            if result[0] is None:  # No valid samples in batch
                continue

            # Extract results from batch
            predicted_params, target_params_batch, auxiliary_data = result
            
            # Compute batch loss (handle DDP model)
            loss, loss_components = base_model.compute_batch_loss(
                predicted_params, target_params_batch, auxiliary_data, 
                return_components=True, loss_weights=loss_weights
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent instability
            # Use very aggressive clipping for transformer decoder to prevent gradient explosion
            head_type = base_model.head_type if is_distributed else model.head_type
            max_norm = 0.1 if head_type == 'transformer_decoder' else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            
            # Update parameters
            optimizer.step()
            
            # Note: DDP automatically synchronizes gradients during backward pass
            # No need for manual barrier here - it would severely slow down training
            
            # Record loss and parameter errors
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            
            # Accumulate parameter errors
            for param_name, param_loss in loss_components.items():
                if param_name not in param_errors:
                    param_errors[param_name] = 0.0
                param_errors[param_name] += param_loss.item()
            
            # Update progress bar (only on main process)
            if not (is_distributed and rank != 0):
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


def validate_epoch(model, val_loader, criterion, device, epoch, loss_weights, is_distributed=False, rank=0):
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
        # Only show progress bar on main process to avoid conflicts
        if is_distributed and rank != 0:
            pbar = val_loader  # No progress bar for non-main processes
        else:
            pbar = tqdm(val_loader, desc=f'Validation {epoch}')
        
        for batch_idx, (x_data_batch, y_data_batch) in enumerate(pbar):
            try:
                # Process batch (handle DDP model)
                base_model = model.module if is_distributed else model
                result = base_model.predict_from_batch(x_data_batch, y_data_batch)
                
                if result[0] is None:  # No valid samples in batch
                    continue
                    
                predicted_params, target_params_batch, auxiliary_data = result
                
                # Compute batch loss (handle DDP model)
                loss, loss_components = base_model.compute_batch_loss(
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
                
                # Update progress bar (only on main process)
                if not (is_distributed and rank != 0):
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


def visualize_training_progress(model, val_loader, device, epoch, model_config, dataset, output_dir='visualizations', num_samples=5):
    """
    Visualize training progress by rendering samples stratified across datasets.
    
    Uses a fixed random seed (42) for sample selection to ensure the same samples
    are visualized across all epochs, enabling consistent progress tracking.
    
    Args:
        model: SMILImageRegressor model
        val_loader: Validation data loader
        device: PyTorch device
        epoch: Current epoch number
        output_dir: Directory to save visualization images
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Save current random states to restore after visualization
    # This ensures visualization doesn't affect training randomness
    torch_rng_state = torch.get_rng_state()
    numpy_rng_state = np.random.get_state()
    python_rng_state = random.getstate()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    
    # Set fixed seed for deterministic sample selection across epochs
    # This ensures the same samples are visualized every epoch for consistent progress tracking
    visualization_seed = 42
    torch.manual_seed(visualization_seed)
    np.random.seed(visualization_seed)
    random.seed(visualization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(visualization_seed)
    
    try:
        # Create output directory for this epoch
        epoch_dir = os.path.join(output_dir, f'epoch_{epoch:03d}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Create image exporter
        image_exporter = ImageExporter(epoch_dir)
        
        # First pass: collect samples and group by dataset source for stratified sampling
        samples_by_dataset = {}
        samples_per_dataset_target = max(5, num_samples * 2)  # Collect at least 5 samples per dataset
        max_batches_to_process = 200  # Safety limit to avoid processing entire dataset
        
        with torch.no_grad():
            for batch_idx, (x_data_batch, y_data_batch) in enumerate(val_loader):
                if batch_idx >= max_batches_to_process:
                    break
                    
                for i, (x_data, y_data) in enumerate(zip(x_data_batch, y_data_batch)):
                    # Skip if no image data
                    if x_data.get('input_image_data') is None:
                        continue
                    
                    # Get dataset source (defaults to 'unknown' if not available)
                    dataset_source = x_data.get('dataset_source', 'unknown')
                    if isinstance(dataset_source, bytes):
                        dataset_source = dataset_source.decode('utf-8')
                    
                    # Group samples by dataset source
                    if dataset_source not in samples_by_dataset:
                        samples_by_dataset[dataset_source] = []
                    
                    # Only add if we haven't reached the target for this dataset
                    if len(samples_by_dataset[dataset_source]) < samples_per_dataset_target:
                        samples_by_dataset[dataset_source].append((x_data, y_data))
                
                # Check if we have enough samples from all datasets
                if len(samples_by_dataset) > 0:
                    min_samples = min(len(samples) for samples in samples_by_dataset.values())
                    # Continue until all discovered datasets have at least the target number
                    all_have_enough = all(len(samples) >= samples_per_dataset_target for samples in samples_by_dataset.values())
                    if all_have_enough and min_samples >= num_samples:
                        break
        
        # Select samples proportionally from each dataset
        selected_samples = []
        num_datasets = len(samples_by_dataset)
        
        if num_datasets == 0:
            print(f"Warning: No samples collected for visualization")
            return
        
        # Calculate samples per dataset (aim for equal representation)
        samples_per_dataset = max(1, num_samples // num_datasets)
        remaining_samples = num_samples - (samples_per_dataset * num_datasets)
        
        for dataset_idx, (dataset_name, samples) in enumerate(samples_by_dataset.items()):
            # Take samples_per_dataset from this dataset, plus one extra if we have remaining
            num_to_take = samples_per_dataset
            if dataset_idx < remaining_samples:
                num_to_take += 1
            
            num_to_take = min(num_to_take, len(samples))
            selected_samples.extend(samples[:num_to_take])
            print(f"Visualizing {num_to_take} samples from dataset: {dataset_name}")
        
        # Now process the selected samples for visualization
        sample_count = 0
        for x_data, y_data in selected_samples:
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
            # For visualization, we need the original image data, not the preprocessed tensor
            
            # Check if this is an HDF5 dataset by looking for 'input_image_data' field
            is_hdf5_dataset = 'input_image_data' in x_data and x_data['input_image_data'] is not None
            
            if is_hdf5_dataset or x_data.get("is_sleap_dataset", False):
                # For optimized dataset or SLEAP dataset, use the original image data and correct image size
                original_image = x_data['input_image_data']  # This is already in RGB format [0,1]
                
                # Convert RGB to BGR for visualization compatibility
                if isinstance(original_image, np.ndarray):
                    # Swap RGB to BGR channels
                    original_image_bgr = original_image[:, :, [2, 1, 0]]  # RGB -> BGR
                    rgb = torch.from_numpy(original_image_bgr).permute(2, 0, 1).unsqueeze(0)
                else:
                    # Swap RGB to BGR channels for tensor
                    original_image_bgr = original_image[:, :, [2, 1, 0]]  # RGB -> BGR
                    rgb = original_image_bgr.permute(2, 0, 1).unsqueeze(0) if len(original_image_bgr.shape) == 3 else original_image_bgr
                
                # Get the actual image size from the original image
                image_height, image_width = original_image.shape[:2]
            else:
                # For original dataset, use preprocessed tensor (no channel swap needed)
                rgb = image_tensor.cpu()
                # Image size will be determined by return_placeholder_data from the file

            if x_data["input_image_mask"] is not None:
                if is_hdf5_dataset or x_data.get("is_sleap_dataset", False):
                    # Optimized dataset or SLEAP dataset - manually create placeholder data with correct image size
                    # Create silhouette tensor
                    sil = torch.FloatTensor(x_data["input_image_mask"])[None, None, ...]
                    
                    # Convert keypoints to pixel coordinates using backbone image size
                    if model_config['backbone_name'].startswith('vit'):
                        target_size = 224
                    else:
                        target_size = 512
                        
                    pixel_coords = y_data["keypoints_2d"].copy()
                    pixel_coords[:, 0] = pixel_coords[:, 0] * target_size  # y coordinates  
                    pixel_coords[:, 1] = pixel_coords[:, 1] * target_size  # x coordinates
                    
                    num_joints = len(y_data["keypoints_2d"])
                    joints = torch.tensor(pixel_coords.reshape(1, num_joints, 2), dtype=torch.float32)
                    visibility = torch.tensor(y_data["keypoint_visibility"].reshape(1, num_joints), dtype=torch.float32)
                    
                    temp_batch = (rgb, sil, joints, visibility)
                    if x_data.get("is_sleap_dataset", False):
                        filenames = [f"sleap_sample_{sample_count}"]
                    else:
                        filenames = [f"optimized_sample_{sample_count}"]
                else:
                    # Original dataset - use file path
                    num_joints = len(y_data["keypoints_2d"]) if y_data["joint_angles"] is None else len(y_data["joint_angles"])
                    temp_batch, filenames = return_placeholder_data(
                        input_image=x_data["input_image"],
                        num_joints=num_joints,
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
                # CRITICAL: Match propagate_scaling to the training model's setting.
                # The model learns scales with propagate_scaling=True (set in SMILImageRegressor.__init__),
                # so visualization must also use propagate_scaling=True for consistent geometry.
                temp_fitter.propagate_scaling = model.propagate_scaling
            else:
                temp_fitter = SMALFitter(
                    device=device,
                    data_batch=rgb,  # For rgb_only=True, just pass the RGB tensor
                    batch_size=1,
                    shape_family=config.SHAPE_FAMILY,
                    use_unity_prior=False,
                    rgb_only=True
                )
                # CRITICAL: Match propagate_scaling to the training model's setting.
                # The model learns scales with propagate_scaling=True (set in SMILImageRegressor.__init__),
                # so visualization must also use propagate_scaling=True for consistent geometry.
                temp_fitter.propagate_scaling = model.propagate_scaling
            
            # Set proper target joints and visibility for visualization
            # Convert normalized keypoints back to pixel coordinates for visualization
            if 'keypoints_2d' in y_data and 'keypoint_visibility' in y_data:
                # Use backbone-specific image size for keypoint conversion
                if model_config['backbone_name'].startswith('vit'):
                    target_height, target_width = 224, 224  # ViT uses 224x224
                else:
                    target_height, target_width = 512, 512  # ResNet uses 512x512
                
                # Convert normalized [0,1] coordinates to pixel coordinates
                keypoints_2d = y_data['keypoints_2d']  # Shape: (num_joints, 2), already in [y_norm, x_norm] format
                keypoint_visibility = y_data['keypoint_visibility']  # Shape: (num_joints,)
                
                # Convert to pixel coordinates using rendered image size
                # Note: keypoints are normalized [0,1] and need to be scaled to rendered resolution
                pixel_coords = keypoints_2d.copy()
                pixel_coords[:, 0] = pixel_coords[:, 0] * target_height  # y to pixels
                pixel_coords[:, 1] = pixel_coords[:, 1] * target_width   # x to pixels
                
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
            if 'log_beta_scales' in predicted_params and 'betas_trans' in predicted_params:
                # Transform PCA weights to per-joint values for visualization
                if model.scale_trans_mode in ['separate', 'ignore']:
                    # In separate/ignore modes, predicted_params contain PCA weights (5 parameters each)
                    # Transform them to per-joint values for the SMALFitter
                    scale_weights = predicted_params['log_beta_scales'][0:1]  # (1, 5)
                    trans_weights = predicted_params['betas_trans'][0:1]      # (1, 5)
                    
                    # Transform to per-joint values using the helper method
                    log_beta_scales_joint, betas_trans_joint = model._transform_separate_pca_weights_to_joint_values(
                        scale_weights, trans_weights
                    )
                    
                    temp_fitter.log_beta_scales.data = log_beta_scales_joint
                    temp_fitter.betas_trans.data = betas_trans_joint
                else:
                    # In other modes, the values are already per-joint
                    temp_fitter.log_beta_scales.data = predicted_params['log_beta_scales'][0:1]
                    temp_fitter.betas_trans.data = predicted_params['betas_trans'][0:1]
            
            # Set camera parameters from PREDICTED values (not ground truth!)
            # This ensures visualization matches the loss computation
            if 'cam_rot' in predicted_params and 'cam_trans' in predicted_params:
                # Use predicted camera parameters for consistent visualization
                temp_fitter.renderer.set_camera_parameters(
                    R=predicted_params['cam_rot'][0:1],  # Predicted camera rotation
                    T=predicted_params['cam_trans'][0:1],  # Predicted camera translation
                    fov=predicted_params['fov'][0:1]  # Predicted FOV
                )
            else:
                # Fallback to ground truth if predicted camera params not available
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
            
            # Generate visualization - match model's UE scaling setting
            temp_fitter.generate_visualization(image_exporter, apply_UE_transform=model.use_ue_scaling, img_idx=sample_count)
            
            sample_count += 1
        
        print(f"Generated {sample_count} visualization images for epoch {epoch} in {epoch_dir}")
    
    finally:
        # Always restore random states to ensure visualization doesn't affect training
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(numpy_rng_state)
        random.setstate(python_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)


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


def load_checkpoint(checkpoint_path, model, optimizer, device, is_distributed=False, rank=0):
    """
    Load checkpoint and restore model, optimizer state and training history.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        model: SMILImageRegressor model to load state into
        optimizer: Optimizer to load state into
        device: PyTorch device
        is_distributed: Whether running in distributed mode
        rank: Current process rank (for print gating)
        
    Returns:
        Tuple of (start_epoch, train_losses, val_losses, train_param_errors, val_param_errors, best_val_loss)
    """
    # Helper to print only on main process
    def log(msg):
        if not is_distributed or rank == 0:
            print(msg)
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        if checkpoint_path:  # Only raise error if a path was actually provided
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        else:
            # No checkpoint provided, return default values
            return 0, [], [], [], [], float('inf')
    
    log(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state (handle DDP compatibility)
        model_state_dict = checkpoint['model_state_dict']
        
        # Check if we're loading a DDP checkpoint into a non-DDP model or vice versa
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(model_state_dict.keys())
        
        # Check if checkpoint has 'module.' prefix but model doesn't (loading DDP checkpoint into non-DDP model)
        if any(key.startswith('module.') for key in checkpoint_keys) and not any(key.startswith('module.') for key in model_keys):
            log("Converting DDP checkpoint to non-DDP model...")
            new_state_dict = {}
            for key, value in model_state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            model_state_dict = new_state_dict
        
        # Check if model has 'module.' prefix but checkpoint doesn't (loading non-DDP checkpoint into DDP model)
        elif any(key.startswith('module.') for key in model_keys) and not any(key.startswith('module.') for key in checkpoint_keys):
            log("Converting non-DDP checkpoint to DDP model...")
            new_state_dict = {}
            for key, value in model_state_dict.items():
                new_key = f'module.{key}'  # Add 'module.' prefix
                new_state_dict[new_key] = value
            model_state_dict = new_state_dict
        
        model.load_state_dict(model_state_dict, strict=False)
        log("Model state loaded successfully")
        
        # Get epoch information first (before trying to load optimizer)
        start_epoch = checkpoint.get('epoch', 0) + 1  # Resume from next epoch
        
        # Try to load optimizer state (may fail if optimizer configuration changed)
        optimizer_loaded = False
        try:
            if 'optimizer_state_dict' in checkpoint:
                # Debug: Compare parameter groups before loading
                checkpoint_optimizer_state = checkpoint['optimizer_state_dict']
                current_param_groups = optimizer.param_groups
                checkpoint_param_groups = checkpoint_optimizer_state.get('param_groups', [])
                
                log(f"\nOptimizer parameter group comparison:")
                log(f"  Current optimizer has {len(current_param_groups)} parameter group(s)")
                log(f"  Checkpoint has {len(checkpoint_param_groups)} parameter group(s)")
                
                # Show details of each parameter group
                for i, (curr_group, ckpt_group) in enumerate(zip(current_param_groups, checkpoint_param_groups)):
                    curr_num_params = len(curr_group['params'])
                    ckpt_num_params = len(ckpt_group['params'])
                    curr_lr = curr_group.get('lr', 'N/A')
                    ckpt_lr = ckpt_group.get('lr', 'N/A')
                    
                    log(f"  Group {i}:")
                    log(f"    Current: {curr_num_params} parameters, lr={curr_lr}")
                    log(f"    Checkpoint: {ckpt_num_params} parameters, lr={ckpt_lr}")
                    
                    if curr_num_params != ckpt_num_params:
                        log(f"    ⚠️  MISMATCH: Parameter count differs!")
                        
                        # Get parameter names from model state dict to identify the differences
                        # We'll compare the parameter names from both state dicts
                        current_param_names = set(model.state_dict().keys())
                        checkpoint_param_names = set(checkpoint['model_state_dict'].keys())
                        
                        # Find parameters that are in current model but not in checkpoint
                        new_params = current_param_names - checkpoint_param_names
                        # Find parameters that are in checkpoint but not in current model
                        removed_params = checkpoint_param_names - current_param_names
                        
                        if new_params:
                            log(f"\n    Parameters in CURRENT model but NOT in checkpoint ({len(new_params)}):")
                            for param_name in sorted(new_params):
                                param_shape = model.state_dict()[param_name].shape
                                log(f"      + {param_name}: {tuple(param_shape)}")
                        
                        if removed_params:
                            log(f"\n    Parameters in CHECKPOINT but NOT in current model ({len(removed_params)}):")
                            for param_name in sorted(removed_params):
                                param_shape = checkpoint['model_state_dict'][param_name].shape
                                log(f"      - {param_name}: {tuple(param_shape)}")
                        
                        # Also check for shape mismatches in common parameters
                        common_params = current_param_names & checkpoint_param_names
                        shape_mismatches = []
                        for param_name in common_params:
                            curr_shape = model.state_dict()[param_name].shape
                            ckpt_shape = checkpoint['model_state_dict'][param_name].shape
                            if curr_shape != ckpt_shape:
                                shape_mismatches.append((param_name, curr_shape, ckpt_shape))
                        
                        if shape_mismatches:
                            log(f"\n    Parameters with SHAPE mismatches ({len(shape_mismatches)}):")
                            for param_name, curr_shape, ckpt_shape in sorted(shape_mismatches):
                                log(f"      ~ {param_name}: current={tuple(curr_shape)}, checkpoint={tuple(ckpt_shape)}")
                
                # If group counts differ, show that too
                if len(current_param_groups) != len(checkpoint_param_groups):
                    log(f"  ⚠️  MISMATCH: Number of parameter groups differs!")
                    log(f"     This usually happens when the model architecture or optimizer")
                    log(f"     configuration has changed between checkpoint save and resume.")
                
                # Try to load anyway
                optimizer.load_state_dict(checkpoint_optimizer_state)
                log("\nOptimizer state loaded successfully")
                optimizer_loaded = True
        except Exception as opt_error:
            log(f"\n⚠️  Warning: Could not load optimizer state: {opt_error}")
            log("Optimizer will be reinitialized (learning rate and momentum will be reset)")
            log("This is usually safe - training will continue from the model weights,")
            log("just without the optimizer's momentum/state from the checkpoint.")
        
        # Update learning rate to match curriculum for the resumed epoch
        resumed_lr = TrainingConfig.get_learning_rate_for_epoch(start_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = resumed_lr
        
        if optimizer_loaded:
            log(f"Learning rate updated to {resumed_lr} for resumed epoch {start_epoch}")
        else:
            log(f"Learning rate set to {resumed_lr} for resumed epoch {start_epoch} (optimizer reinitialized)")
        
        # Initialize training history lists
        train_losses = []
        val_losses = []
        train_param_errors = []
        val_param_errors = []
        best_val_loss = float('inf')
        
        # Try to load training history if available
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            log(f"Loaded training history with {len(train_losses)} epochs")
        
        if 'val_losses' in checkpoint:
            val_losses = checkpoint['val_losses']
            log(f"Loaded validation history with {len(val_losses)} epochs")
        
        if 'train_param_errors_history' in checkpoint:
            train_param_errors = checkpoint['train_param_errors_history']
            log(f"Loaded training parameter error history with {len(train_param_errors)} epochs")
        
        if 'val_param_errors_history' in checkpoint:
            val_param_errors = checkpoint['val_param_errors_history']
            log(f"Loaded validation parameter error history with {len(val_param_errors)} epochs")
        
        # Get best validation loss
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        elif 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
        elif val_losses:
            best_val_loss = min(val_losses)
        
        log(f"Resuming training from epoch {start_epoch}")
        log(f"Best validation loss so far: {best_val_loss:.6f}")
        
        return start_epoch, train_losses, val_losses, train_param_errors, val_param_errors, best_val_loss
        
    except Exception as e:
        log(f"Error loading checkpoint: {e}")
        log("Starting training from scratch...")
        return 0, [], [], [], [], float('inf')


def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, train_param_err, val_param_err,
                   train_losses, val_losses, train_param_errors, val_param_errors, best_val_loss,
                   checkpoint_path, checkpoint_config=None):
    """
    Save training checkpoint with complete state.

    When checkpoint_config is provided, it is saved so run_inference can load
    model config, scale_trans, and shape_family from the checkpoint.

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
        checkpoint_config: Optional dict with 'model_config', 'training_params' (at least
            'rotation_representation'), 'scale_trans_mode', 'scale_trans_config', 'shape_family'.
    """
    state = {
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
    }
    if checkpoint_config is not None:
        state['config'] = checkpoint_config
    torch.save(state, checkpoint_path)


def main(dataset_name=None, checkpoint_path=None, config_override=None):
    """
    Main training function.
    
    Args:
        dataset_name (str): Name of the dataset to use (default: uses TrainingConfig.DEFAULT_DATASET)
        checkpoint_path (str): Path to checkpoint file to resume training from (default: None)
        config_override (dict): Dictionary to override specific config values (default: None)
    """
    # Re-apply SMAL model override in this process.
    # When using mp.spawn, each worker is a fresh Python process that imports
    # config.py with its defaults.  The parent process may have already called
    # apply_smal_file_override(), but that only patched the parent's globals.
    # We must re-apply here so config.N_POSE, config.N_BETAS, config.dd, etc.
    # match the checkpoint / JSON config in every worker.
    if config_override:
        _smal_file = config_override.get('smal_file')
        if _smal_file:
            apply_smal_file_override(
                _smal_file,
                shape_family=config_override.get('shape_family'),
            )

    # Load training configuration
    training_config = TrainingConfig.get_all_config(dataset_name)

    # Extract DDP configuration from config_override
    is_distributed = config_override.get('is_distributed', False) if config_override else False
    rank = config_override.get('rank', 0) if config_override else 0
    world_size = config_override.get('world_size', 1) if config_override else 1
    device_override = config_override.get('device_override') if config_override else None
    
    # Apply any config overrides
    if config_override:
        for key, value in config_override.items():
            # Skip DDP-specific keys that aren't part of training config
            if key in ['is_distributed', 'rank', 'world_size', 'device_override']:
                continue
            if key in training_config:
                if isinstance(training_config[key], dict):
                    training_config[key].update(value)
                else:
                    training_config[key] = value
    
    # Print configuration summary (only on main process to avoid duplicate output)
    if not is_distributed or rank == 0:
        TrainingConfig.print_config_summary(dataset_name)
    
    # Extract configuration values
    data_path = training_config['data_path']
    training_params = training_config['training_params']
    model_config = training_config['model_config']
    output_config = training_config['output_config']

    # Build config to save in checkpoints so run_inference can load without training_config
    checkpoint_config = {
        'model_config': model_config.copy(),
        'training_params': {
            'rotation_representation': training_params.get('rotation_representation', '6d'),
        },
        'scale_trans_mode': TrainingConfig.get_scale_trans_mode(),
        'scale_trans_config': TrainingConfig.get_scale_trans_config(),
        'shape_family': config.SHAPE_FAMILY,
        'smal_file': getattr(config, 'SMAL_FILE', None),
    }
    
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
            if not is_distributed or rank == 0:
                print(f"Test mode: Using temporary directory: {temp_dir}")
    
    # Set random seeds for reproducibility
    seed = training_params['seed']
    set_random_seeds(seed)
    if not is_distributed or rank == 0:
        print(f"Random seed set to: {seed}")
    
    # Set device (use device_override for distributed training)
    if device_override:
        device = device_override
        if not is_distributed or rank == 0:
            print(f"Using device override: {device}")
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_IDS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not is_distributed or rank == 0:
            print(f"Using device: {device}")
    
    # Dataset parameters
    batch_size = training_params['batch_size']
    num_epochs = training_params['num_epochs']
    learning_rate = training_params['learning_rate']
    rotation_representation = training_params['rotation_representation']
    
    # Create dataset (multi-dataset or single dataset)
    if not is_distributed or rank == 0:
        print("Loading dataset...")
        print(f"Using rotation representation: {rotation_representation}")
        print(f"Using backbone: {model_config['backbone_name']}")
    
    # Check if multi-dataset training is enabled
    use_multi_dataset = TrainingConfig.is_multi_dataset_enabled()
    
    if use_multi_dataset:
        # Multi-dataset training mode
        if not is_distributed or rank == 0:
            print("\n" + "="*70)
            print("MULTI-DATASET TRAINING MODE ENABLED")
            print("="*70)
        
        from combined_dataset import CombinedSMILDataset
        
        # Get enabled dataset configurations
        dataset_configs = TrainingConfig.get_enabled_datasets()
        
        if not dataset_configs:
            raise ValueError("Multi-dataset mode enabled but no datasets are configured!")
        
        if not is_distributed or rank == 0:
            print(f"Loading {len(dataset_configs)} datasets:")
            for ds_config in dataset_configs:
                print(f"  - {ds_config['name']}: {ds_config['path']} (weight: {ds_config['weight']})")
        
        # Create combined dataset
        dataset = CombinedSMILDataset(
            dataset_configs=dataset_configs,
            rotation_representation=rotation_representation,
            backbone_name=model_config['backbone_name']
        )
        
        # Print statistics
        if not is_distributed or rank == 0:
            dataset.print_statistics()
        
        # Split dataset using per-dataset strategy
        split_strategy = TrainingConfig.get_validation_split_strategy()
        if not is_distributed or rank == 0:
            print(f"\nUsing '{split_strategy}' split strategy")
        
        train_set, val_set, test_set = dataset.split_datasets(
            train_size=1.0 - training_config['split_config']['test_size'] - training_config['split_config']['val_size'],
            val_size=training_config['split_config']['val_size'],
            test_size=training_config['split_config']['test_size'],
            seed=seed
        )
        
        # Create weighted sampler for mixed batches
        dataset_weights = TrainingConfig.get_dataset_weights()
        train_sampler_weighted = dataset.create_weighted_sampler(
            weights=dataset_weights,
            train_indices=dataset.train_indices,
            num_samples=len(train_set)
        )
        
        if not is_distributed or rank == 0:
            print(f"\nCreated weighted sampler with weights: {dataset_weights}")
        
    else:
        # Single dataset training mode (legacy)
        if not is_distributed or rank == 0:
            print("\n" + "="*70)
            print("SINGLE DATASET TRAINING MODE (Legacy)")
            print("="*70)
        
        # Check if SLEAP dataset support is available
        try:
            from sleap_data.sleap_dataset import SLEAPDataset
            sleap_available = True
        except ImportError:
            sleap_available = False
        
        if not is_distributed or rank == 0:
            if sleap_available:
                print("SLEAP dataset support: enabled")
            else:
                print("SLEAP dataset support: not available")
        
        dataset = UnifiedSMILDataset.from_path(
            data_path, 
            rotation_representation=rotation_representation,
            backbone_name=model_config['backbone_name']
        )
        if not is_distributed or rank == 0:
            print(f"Dataset size: {len(dataset)}")
            print(f"Original resolution: {dataset.get_input_resolution()}")
            print(f"Target resolution: {dataset.get_target_resolution()}")
        
        # Split dataset using configuration
        train_amount, val_amount, test_amount = TrainingConfig.get_train_val_test_sizes(len(dataset))
        
        train_set, val_set, test_set = torch.utils.data.random_split(
            dataset, [train_amount, val_amount, test_amount]
        )
        
        train_sampler_weighted = None  # No weighted sampler for single dataset
    
    if not is_distributed or rank == 0:
        print(f"Train set: {len(train_set)} samples")
        print(f"Validation set: {len(val_set)} samples")
        print(f"Test set: {len(test_set)} samples")
    
    # Create data loaders with optimized multiprocessing
    # Use configuration parameters for data loading optimization
    num_workers = training_params.get('num_workers', min(8, os.cpu_count() or 4))
    pin_memory = training_params.get('pin_memory', True)
    prefetch_factor = training_params.get('prefetch_factor', 2)
    
    
    if not is_distributed or rank == 0:
        print(f"Data loading configuration:")
        print(f"  Workers: {num_workers}")
        print(f"  Pin memory: {pin_memory}")
        print(f"  Prefetch factor: {prefetch_factor}")
        print(f"  Matplotlib backend: {matplotlib.get_backend()}")
    
    # Create data loaders with distributed samplers if using DDP
    if is_distributed:
        # Create distributed samplers
        # Note: If using weighted sampler for multi-dataset, we can't use DistributedSampler directly
        # Would need to implement DistributedWeightedSampler (TODO for future)
        if use_multi_dataset and train_sampler_weighted is not None and rank == 0:
            print("WARNING: Distributed training with weighted sampling not yet implemented")
            print("  Falling back to standard distributed sampler")
            print("  Batch composition may not follow dataset weights correctly")
        
        train_sampler = DistributedSampler(train_set, rank=rank, num_replicas=world_size, shuffle=True)
        val_sampler = DistributedSampler(val_set, rank=rank, num_replicas=world_size, shuffle=False)
        test_sampler = DistributedSampler(test_set, rank=rank, num_replicas=world_size, shuffle=False)
        
        if rank == 0:
            print(f"Using distributed samplers for {world_size} GPUs")
            print(f"Effective batch size per GPU: {batch_size}")
            print(f"Total effective batch size: {batch_size * world_size}")
            
        # Debug: Print sampler information (only from rank 0 to reduce noise)
        if rank == 0:
            print(f"Rank {rank}: Train sampler size: {len(train_sampler)}, Val sampler size: {len(val_sampler)}")
    else:
        # Use weighted sampler for training if available (multi-dataset mode)
        if use_multi_dataset and train_sampler_weighted is not None:
            train_sampler = train_sampler_weighted
            if not is_distributed or rank == 0:
                print("Using weighted random sampler for mixed-batch training")
        else:
            train_sampler = None
        
        val_sampler = test_sampler = None
    
    # Try to create data loaders with multiprocessing, fallback to single-threaded if issues
    try:
        train_loader = DataLoader(
            train_set, 
            batch_size=batch_size, 
            shuffle=(not is_distributed and train_sampler is None),  # Don't shuffle when using sampler
            sampler=train_sampler,
            num_workers=num_workers, 
            collate_fn=custom_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,  # Only use persistent workers if multiprocessing
            prefetch_factor=prefetch_factor
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=batch_size, 
            shuffle=False,  # Never shuffle validation
            sampler=val_sampler,
            num_workers=num_workers, 
            collate_fn=custom_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor
        )
        test_loader = DataLoader(
            test_set, 
            batch_size=batch_size, 
            shuffle=False,  # Never shuffle test
            sampler=test_sampler,
            num_workers=num_workers, 
            collate_fn=custom_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor
        )
        if not is_distributed or rank == 0:
            print(f"Successfully created data loaders with {num_workers} workers")
    except Exception as e:
        if not is_distributed or rank == 0:
            print(f"Warning: Failed to create multiprocessing data loaders: {e}")
            print("Falling back to single-threaded data loading...")
        num_workers = 0
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=(not is_distributed and train_sampler is None), sampler=train_sampler, num_workers=0, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=0, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=0, collate_fn=custom_collate_fn)
    
    # Create placeholder data for SMALFitter initialization
    placeholder_data = create_placeholder_data_batch(batch_size)
    
    # Memory optimization setup
    memory_optimizer = MemoryOptimizer(device=device, target_memory_gb=20.0)
    if not is_distributed or rank == 0:
        memory_optimizer.monitor.print_memory_status("Before model initialization")
    
    # Get memory recommendations
    memory_config = recommend_training_config(model_config['backbone_name'], 24.0)
    if not is_distributed or rank == 0:
        print(f"Memory recommendations for {model_config['backbone_name']}:")
        for key, value in memory_config.items():
            if key != 'memory_optimizations':
                print(f"  {key}: {value}")
    
    # Initialize model
    if not is_distributed or rank == 0:
        print("Initializing model...")
        print(f"Using backbone: {model_config['backbone_name']}")
    
    # Determine appropriate input resolution based on backbone
    if model_config['backbone_name'].startswith('vit'):
        # Vision Transformers expect 224x224 input
        input_resolution = 224
        if not is_distributed or rank == 0:
            print(f"Using ViT input resolution: {input_resolution}x{input_resolution}")
    else:
        # ResNet can handle higher resolutions
        input_resolution = dataset.get_input_resolution()
        if not is_distributed or rank == 0:
            print(f"Using ResNet input resolution: {input_resolution}x{input_resolution}")
    
    model = SMILImageRegressor(
        device=device,
        data_batch=placeholder_data,
        batch_size=batch_size,
        shape_family=config.SHAPE_FAMILY,
        use_unity_prior=model_config['use_unity_prior'],
        rgb_only=model_config['rgb_only'],
        freeze_backbone=model_config['freeze_backbone'],
        hidden_dim=model_config['hidden_dim'],
        #use_ue_scaling=dataset.get_ue_scaling_flag(),
        use_ue_scaling=True,  # Always apply UE scaling so training matches visualization behaviour
        rotation_representation=rotation_representation,
        input_resolution=input_resolution,
        backbone_name=model_config['backbone_name'],
        head_type=model_config.get('head_type', 'mlp'),
        transformer_config=model_config.get('transformer_config', {}),
        scale_trans_mode=TrainingConfig.get_scale_trans_mode()
    ).to(device)
    
    # Print model configuration
    if not is_distributed or rank == 0:
        print(f"Model created with head type: {model.head_type}")
        print(f"Scale/Translation mode: {TrainingConfig.get_scale_trans_mode()}")
        if model.head_type == 'transformer_decoder':
            print(f"Transformer decoder config: {model.transformer_config}")
            if 'trans_scale_factor' in model.transformer_config:
                print(f"  Trans scale factor: {model.transformer_config['trans_scale_factor']}")
    
    # Apply memory optimizations
    memory_optimizer.optimize_model_for_memory(
        model, model_config['backbone_name'], batch_size,
        use_mixed_precision=memory_config['use_mixed_precision'],
        use_gradient_checkpointing=memory_config['use_gradient_checkpointing']
    )
    
    if not is_distributed or rank == 0:
        memory_optimizer.monitor.print_memory_status("After model initialization")
    
    # Wrap model with DDP if using distributed training
    if is_distributed:
        # find_unused_parameters=True needed due to complex SMAL model structure
        # where some parameters might not be used in every forward pass
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        if rank == 0:
            print(f"Model wrapped with DistributedDataParallel on {world_size} GPUs")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not is_distributed or rank == 0:
        print(f"Model parameters: {total_params:,}")
    
    # Print parameter breakdown by component (only on main process)
    if not is_distributed or rank == 0:
        # Access the underlying model for DDP
        base_model = model.module if is_distributed else model
        
        if base_model.head_type == 'transformer_decoder':
            backbone_params = sum(p.numel() for p in base_model.backbone.parameters() if p.requires_grad)
            head_params = sum(p.numel() for p in base_model.transformer_head.parameters() if p.requires_grad)
            print(f"  Backbone parameters: {backbone_params:,}")
            print(f"  Transformer decoder parameters: {head_params:,}")
        else:
            backbone_params = sum(p.numel() for p in base_model.backbone.parameters() if p.requires_grad)
            head_params = sum(p.numel() for p in [base_model.fc1, base_model.fc2, base_model.fc3, base_model.regressor, base_model.ln1, base_model.ln2, base_model.ln3] for p in p.parameters() if p.requires_grad)
            print(f"  Backbone parameters: {backbone_params:,}")
            print(f"  MLP head parameters: {head_params:,}")
    
    # Initialize optimizer and loss function (AniMer-style AdamW)
    # Use the learning rate from curriculum (base learning rate for epoch 0)
    initial_lr = TrainingConfig.get_learning_rate_for_epoch(0)
    weight_decay = training_params.get('weight_decay', 1e-4)
    
    # Get trainable parameters with different learning rates for different components
    # Access the underlying model for DDP
    base_model = model.module if is_distributed else model
    trainable_params = base_model.get_trainable_parameters()
    
    # For transformer decoder, use a more conservative learning rate
    if base_model.head_type == 'transformer_decoder':
        # Use much lower learning rate for transformer decoder head (AniMer-style)
        transformer_lr = initial_lr * 0.01  # 100x smaller learning rate to prevent gradient explosion
        if not is_distributed or rank == 0:
            print(f"Using conservative learning rate for transformer decoder: {transformer_lr}")
        
        # Create separate parameter groups with different learning rates
        backbone_params = []
        transformer_params = []
        
        for param_group in trainable_params:
            # Use the 'name' field to identify parameter groups
            group_name = param_group.get('name', '')
            if group_name == 'transformer_head':
                # Lower learning rate for transformer head
                transformer_params.append({
                    'params': param_group['params'], 
                    'lr': transformer_lr, 
                    'weight_decay': weight_decay,
                    'name': group_name  # Preserve name for debugging
                })
            else:
                # Normal learning rate for backbone and other components
                backbone_params.append({
                    'params': param_group['params'], 
                    'lr': initial_lr, 
                    'weight_decay': weight_decay,
                    'name': group_name  # Preserve name for debugging
                })
        
        # Combine parameter groups
        trainable_params = backbone_params + transformer_params
        
        if not is_distributed or rank == 0:
            print(f"Optimizer parameter groups created:")
            print(f"  Backbone/other groups: {len(backbone_params)}")
            for i, g in enumerate(backbone_params):
                print(f"    - {g.get('name', 'unnamed')} (lr={g['lr']})")
            print(f"  Transformer groups: {len(transformer_params)}")
            for i, g in enumerate(transformer_params):
                print(f"    - {g.get('name', 'unnamed')} (lr={g['lr']})")
            print(f"  Total groups: {len(trainable_params)}")
    else:
        # For MLP head, add weight decay to all parameters
        trainable_params = [
            {
                'params': param_group['params'], 
                'lr': initial_lr, 
                'weight_decay': weight_decay,
                'name': param_group.get('name', f'group_{i}')  # Preserve name for debugging
            } 
            for i, param_group in enumerate(trainable_params)
        ]
        if not is_distributed or rank == 0:
            print(f"Optimizer parameter groups created: {len(trainable_params)}")
            for i, g in enumerate(trainable_params):
                print(f"  - {g.get('name', 'unnamed')} (lr={g['lr']})")
    
    # Use AdamW optimizer (AniMer-style)
    optimizer = optim.AdamW(trainable_params, lr=initial_lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Show optimizer configuration
    if not is_distributed or rank == 0:
        print(f"\nOptimizer configuration:")
        print(f"  Type: AdamW")
        print(f"  Initial learning rate: {initial_lr}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Number of parameter groups: {len(optimizer.param_groups)}")
        for i, group in enumerate(optimizer.param_groups):
            num_params = len(group['params'])
            total_params = sum(p.numel() for p in group['params'])
            print(f"  Group {i}: {num_params} tensors, {total_params:,} total parameters, lr={group['lr']}")
    
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
            checkpoint_path, model, optimizer, device, is_distributed=is_distributed, rank=rank
        )
    
    if not is_distributed or rank == 0:
        print("Starting training...")
        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch}")
        else:
            print("Training from scratch")
    
    for epoch in range(start_epoch, num_epochs):
        # Set epoch for distributed sampler (important for proper shuffling)
        if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Get loss weights and learning rate for current epoch from configuration
        loss_weights = TrainingConfig.get_loss_weights_for_epoch(epoch)
        current_lr = TrainingConfig.get_learning_rate_for_epoch(epoch)
        
        # Update learning rate if it has changed
        for param_group in optimizer.param_groups:
            if param_group['lr'] != current_lr:
                param_group['lr'] = current_lr
                if not is_distributed or rank == 0:
                    print(f"Learning rate updated to {current_lr} at epoch {epoch}")
        
        # Train
        train_loss, train_param_err = train_epoch(model, train_loader, optimizer, criterion, device, epoch, loss_weights, is_distributed, rank)
        train_losses.append(train_loss)
        train_param_errors.append(train_param_err)
        
        # Validate
        val_loss, val_param_err = validate_epoch(model, val_loader, criterion, device, epoch, loss_weights, is_distributed, rank)
        
        # Gather validation metrics from all processes for accurate reporting
        if is_distributed:
            val_loss, val_param_err = gather_validation_metrics(val_loss, val_param_err, world_size)
            # Ensure all processes complete validation before proceeding
            torch.distributed.barrier()
        
        val_losses.append(val_loss)
        val_param_errors.append(val_param_err)
        
        # Print epoch summary with parameter errors (only on main process)
        if not is_distributed or rank == 0:
            print(f'\nEpoch {epoch}:')
            print(f'  Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {current_lr:.2e}')
            print('  Parameter Errors (Train / Val):')
            
            # Get all parameter names from both train and val
            all_params = set(train_param_err.keys()) | set(val_param_err.keys())
            for param_name in sorted(all_params):
                train_err = train_param_err.get(param_name, 0.0)
                val_err = val_param_err.get(param_name, 0.0)
                print(f'    {param_name:15s}: {train_err:.6f} / {val_err:.6f}')
        
        # Save best model (unless in test mode) - only on main process
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if (not is_distributed or rank == 0) and not is_test_mode:
                best_model_path = os.path.join(output_config['checkpoint_dir'], 'best_model.pth')
                # For DDP, save the underlying model state
                model_to_save = model.module if is_distributed else model
                save_checkpoint(epoch, model_to_save, optimizer, train_loss, val_loss, train_param_err, val_param_err,
                              train_losses, val_losses, train_param_errors, val_param_errors, best_val_loss,
                              best_model_path, checkpoint_config=checkpoint_config)
                print(f'  New best model saved with validation loss: {val_loss:.6f}')
            elif not is_distributed or rank == 0:
                print(f'  New best validation loss: {val_loss:.6f} (checkpoint saving disabled in test mode)')
        
        # Save regular checkpoint (unless in test mode) - only on main process
        if ((epoch + 1) % output_config['save_checkpoint_every'] == 0) and (not is_distributed or rank == 0):
            if not is_test_mode:
                checkpoint_path_save = os.path.join(output_config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
                # For DDP, save the underlying model state
                model_to_save = model.module if is_distributed else model
                save_checkpoint(epoch, model_to_save, optimizer, train_loss, val_loss, train_param_err, val_param_err,
                              train_losses, val_losses, train_param_errors, val_param_errors, best_val_loss,
                              checkpoint_path_save, checkpoint_config=checkpoint_config)
                print(f'  Checkpoint saved at epoch {epoch}')
            else:
                print(f'  Checkpoint saving disabled in test mode (epoch {epoch})')
        
        # Generate visualizations - only on main process
        if ((epoch + 1) % output_config['generate_visualizations_every'] == 0) and (not is_distributed or rank == 0):
            # For DDP, pass the underlying model for visualization
            model_for_viz = model.module if is_distributed else model
            visualize_training_progress(model_for_viz, val_loader, device, epoch, model_config, dataset,
                                      output_dir=output_config['visualizations_dir'], 
                                      num_samples=output_config['num_visualization_samples'])
            # Also visualize the train set to see if the model is overfitting
            visualize_training_progress(model_for_viz, train_loader, device, epoch, model_config, dataset,
                                      output_dir=output_config['train_visualizations_dir'], 
                                      num_samples=output_config['num_visualization_samples'])
        
        # Plot training history (unless in test mode) - only on main process
        if ((epoch + 1) % output_config['plot_history_every'] == 0) and (not is_distributed or rank == 0):
            if not is_test_mode:
                history_path = os.path.join(output_config['plots_dir'], f'training_history_epoch_{epoch}.png')
                param_errors_path = os.path.join(output_config['plots_dir'], f'parameter_errors_epoch_{epoch}.png')
                plot_training_history(train_losses, val_losses, history_path)
                plot_parameter_errors(train_param_errors, val_param_errors, param_errors_path)
    
    if not is_distributed or rank == 0:
        print("Training completed!")
    
    # Final evaluation on test set
    if not is_distributed or rank == 0:
        print("Evaluating on test set...")
    # Use final epoch loss weights for test evaluation
    final_loss_weights = TrainingConfig.get_loss_weights_for_epoch(num_epochs - 1)
    # For DDP, pass the wrapped model for evaluation (validate_epoch handles unwrapping internally)
    test_loss, test_param_err = validate_epoch(model, test_loader, criterion, device, 'test', final_loss_weights, is_distributed, rank)
    
    # Gather test metrics from all processes
    if is_distributed:
        test_loss, test_param_err = gather_validation_metrics(test_loss, test_param_err, world_size)
        # Ensure all processes complete test evaluation before proceeding
        torch.distributed.barrier()
    
    if not is_distributed or rank == 0:
        print(f'Test Loss: {test_loss:.6f}')
        print('Test Parameter Errors:')
        for param_name, param_err in sorted(test_param_err.items()):
            print(f'  {param_name:15s}: {param_err:.6f}')
    
    # Save final training history (unless in test mode) - only on main process
    if (not is_distributed or rank == 0) and not is_test_mode:
        final_history_path = os.path.join(output_config['plots_dir'], 'final_training_history.png')
        final_param_errors_path = os.path.join(output_config['plots_dir'], 'final_parameter_errors.png')
        plot_training_history(train_losses, val_losses, final_history_path)
        plot_parameter_errors(train_param_errors, val_param_errors, final_param_errors_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SMIL Image Regressor')
    parser.add_argument('--dataset', type=str, default=None, 
                       choices=['masked_simple', 'pose_only_simple', 'test_textured', 'simple'],
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
    parser.add_argument('--backbone', type=str, default=None,
                       choices=['resnet50', 'resnet101', 'resnet152', 'vit_base_patch16_224', 'vit_large_patch16_224'],
                       help='Backbone network to use (overrides config default)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to dataset (overrides config default). Can be a directory or HDF5 file.')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to JSON config file (must include "mode": "singleview"). '
                            'See configs/examples/singleview_baseline.json for a template.')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='Number of GPUs to use for training (default: 1, ignored when using torchrun)')
    parser.add_argument('--master-port', type=str, default=None,
                       help='Master port for distributed training (default: from MASTER_PORT env var or 12345)')
    parser.add_argument('--scale-trans-mode', type=str, default=None,
                       choices=['ignore', 'separate', 'entangled_with_betas'],
                       help='Scale/translation beta mode (overrides config default)')
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Configuration loading: new JSON config system or legacy CLI args
    # ---------------------------------------------------------------
    config_override = {}

    if args.config is not None:
        # New config system: load from JSON, apply CLI overrides
        cli_overrides = {}
        if args.seed is not None:
            cli_overrides['training'] = cli_overrides.get('training', {})
            cli_overrides['training']['seed'] = args.seed
        if args.rotation_representation is not None:
            cli_overrides['training'] = cli_overrides.get('training', {})
            cli_overrides['training']['rotation_representation'] = args.rotation_representation
        if args.batch_size is not None:
            cli_overrides['training'] = cli_overrides.get('training', {})
            cli_overrides['training']['batch_size'] = args.batch_size
        if args.learning_rate is not None:
            cli_overrides['optimizer'] = cli_overrides.get('optimizer', {})
            cli_overrides['optimizer']['learning_rate'] = args.learning_rate
        if args.num_epochs is not None:
            cli_overrides['training'] = cli_overrides.get('training', {})
            cli_overrides['training']['num_epochs'] = args.num_epochs
        if args.backbone is not None:
            cli_overrides['model'] = cli_overrides.get('model', {})
            cli_overrides['model']['backbone_name'] = args.backbone
        if args.data_path is not None:
            cli_overrides['dataset'] = cli_overrides.get('dataset', {})
            cli_overrides['dataset']['data_path'] = args.data_path
        if args.scale_trans_mode is not None:
            cli_overrides['scale_trans_beta'] = cli_overrides.get('scale_trans_beta', {})
            cli_overrides['scale_trans_beta']['mode'] = args.scale_trans_mode

        new_config = load_config(
            config_file=args.config,
            cli_overrides=cli_overrides,
            expected_mode='singleview',
        )

        # Apply smal_model overrides (SMAL_FILE / SHAPE_FAMILY).
        # apply_smal_file_override re-reads the pickle and patches config.dd,
        # config.N_POSE, config.N_BETAS, config.joint_names, etc.
        if getattr(new_config, "smal_model", None) is not None:
            if new_config.smal_model.smal_file:
                apply_smal_file_override(
                    new_config.smal_model.smal_file,
                    shape_family=new_config.smal_model.shape_family,
                )
            elif new_config.smal_model.shape_family is not None:
                config.SHAPE_FAMILY = int(new_config.smal_model.shape_family)

        # Sync scale_trans_mode to legacy TrainingConfig (still read by some code paths)
        TrainingConfig.SCALE_TRANS_BETA_CONFIG['mode'] = new_config.scale_trans_beta.mode

        # Convert to legacy dict format for existing main()
        config_override = new_config.to_legacy_dict()

        # Save resolved config for reproducibility
        os.makedirs(new_config.output.checkpoint_dir, exist_ok=True)
        save_config_json(new_config, os.path.join(new_config.output.checkpoint_dir, 'config.json'))

        print(f"Loaded config from: {args.config}")
        print(f"Resolved config saved to: {os.path.join(new_config.output.checkpoint_dir, 'config.json')}")

    else:
        # Legacy config flow: build config_override from CLI args
        if args.seed is not None or args.rotation_representation is not None or \
           args.batch_size is not None or args.learning_rate is not None or args.num_epochs is not None or args.backbone is not None or \
           args.data_path is not None:
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
            if args.backbone is not None:
                config_override['model_config'] = {'backbone_name': args.backbone}
            if args.data_path is not None:
                config_override['data_path'] = args.data_path

        # Apply scale_trans_mode override directly to TrainingConfig
        if args.scale_trans_mode is not None:
            TrainingConfig.SCALE_TRANS_BETA_CONFIG['mode'] = args.scale_trans_mode

    # Get master port from args or environment variable
    master_port = args.master_port or os.environ.get('MASTER_PORT', '12345')
    
    # Check if launched via torchrun/torch.distributed.launch (HPC environment)
    # This is detected by the presence of RANK, LOCAL_RANK, and WORLD_SIZE env vars
    if is_torchrun_launched():
        # Launched via torchrun - processes are already spawned by the launcher
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # Only print from rank 0 to avoid duplicate output
        if rank == 0:
            local_rank = int(os.environ['LOCAL_RANK'])
            print(f"Detected torchrun/HPC launch environment:")
            print(f"  Global rank: {rank}")
            print(f"  Local rank (GPU): {local_rank}")
            print(f"  World size: {world_size}")
            print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
            print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")
        
        # Call ddp_main directly - it will read LOCAL_RANK from env internally
        ddp_main(
            rank,
            world_size,
            args.dataset,
            args.checkpoint,
            config_override or None,
            master_port
        )
    
    elif args.num_gpus > 1:
        # Manual multi-GPU launch using mp.spawn
        if not torch.cuda.is_available():
            print("ERROR: Multi-GPU training requested but CUDA is not available!")
            exit(1)
        available_gpus = torch.cuda.device_count()
        if args.num_gpus > available_gpus:
            print(f"ERROR: Requested {args.num_gpus} GPUs but only {available_gpus} available!")
            exit(1)
            
        print(f"Launching multi-GPU training on {args.num_gpus} GPUs (using mp.spawn)...")
        print(f"Master port: {master_port}")
        print(f"Batch size per GPU: {args.batch_size if args.batch_size else 'from config'}")
        print(f"Total effective batch size: {args.batch_size * args.num_gpus if args.batch_size else 'config_batch_size * num_gpus'}")
        
        # Launch multi-GPU training using spawn
        mp.spawn(ddp_main, 
                args=(args.num_gpus, args.dataset, args.checkpoint, config_override or None, master_port),
                nprocs=args.num_gpus,
                join=True)
    else:
        # Single GPU training (existing path)
        print("Launching single-GPU training...")
        main(dataset_name=args.dataset, checkpoint_path=args.checkpoint, config_override=config_override or None)
