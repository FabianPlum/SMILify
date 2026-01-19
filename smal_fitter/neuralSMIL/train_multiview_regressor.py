"""
Multi-View SMIL Image Regressor Training Script

This script trains the MultiViewSMILImageRegressor network to predict SMIL parameters
from multiple synchronized camera views.

Key Features:
- Cross-attention between views for feature fusion
- Separate camera heads per canonical view position
- Shared body parameter prediction
- Per-view 2D keypoint loss with visibility weighting
"""

# Set matplotlib backend BEFORE any other imports
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting (imported for side effects)

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
import json
from tqdm import tqdm
from datetime import datetime
import argparse
import imageio
from typing import Optional

# Add parent directories to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multiview_smil_regressor import MultiViewSMILImageRegressor, create_multiview_regressor
from smil_image_regressor import rotation_6d_to_axis_angle
from sleap_data.sleap_multiview_dataset import SLEAPMultiViewDataset, multiview_collate_fn
from smal_fitter import SMALFitter
import config
from training_config import TrainingConfig


def set_random_seeds(seed: int = 0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_torchrun_launched():
    """Check if launched via torchrun."""
    return all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])


def setup_ddp(rank: int, world_size: int, port: str = '12345', local_rank: int = None):
    """Initialize DDP environment."""
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = port
    
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    gpu_rank = local_rank if local_rank is not None else rank
    torch.cuda.set_device(gpu_rank)


def cleanup_ddp():
    """Clean up DDP environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _print_component_metrics(train_components: dict, val_components: dict, indent: str = "    "):
    """
    Print a stable, readable table of per-loss component metrics.

    Matches the spirit of the reporting in `train_smil_regressor.py`: show all tracked
    components (including camera + 3D keypoints) for train and validation.
    """
    train_components = train_components or {}
    val_components = val_components or {}
    all_keys = sorted(set(train_components.keys()) | set(val_components.keys()))
    if not all_keys:
        print(f"{indent}(no loss components reported)")
        return

    print(f"{indent}Loss Components (Train / Val):")
    for k in all_keys:
        t = float(train_components.get(k, 0.0))
        v = float(val_components.get(k, 0.0))
        print(f"{indent}  {k:20s}: {t:.6f} / {v:.6f}")


class MultiViewTrainingConfig:
    """
    Configuration for multi-view training.
    
    Extends the base TrainingConfig with multi-view specific settings while
    inheriting model, training, and loss configurations from the shared config.
    """
    
    # Multi-view specific defaults (everything else comes from TrainingConfig)
    MULTIVIEW_DEFAULTS = {
        # Multi-view specific data settings
        'dataset_path': None,  # Required - path to multi-view HDF5
        'num_views_to_use': None,  # None = use all available views
        'min_views_per_sample': 2,
        
        # Cross-attention settings (multi-view specific)
        'cross_attention_layers': 2,
        'cross_attention_heads': 8,
        'cross_attention_dropout': 0.1,
        
        # Output directories (separate from single-view)
        'checkpoint_dir': 'multiview_checkpoints',
        'visualizations_dir': 'multiview_visualizations',
        'singleview_visualizations_dir': 'multiview_singleview_renders',
        
        # Validation/save frequency
        'save_every_n_epochs': 10,
        'validate_every_n_epochs': 1,
        'visualize_every_n_epochs': 10,
        'num_visualization_samples': 3,
        
        # Split ratios
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        
        # Mesh scaling - allows network to predict global mesh scale
        # (useful when 3D ground truth has different scale than model)
        'allow_mesh_scaling': False,
        'mesh_scale_init': 1.0,
    }
    
    @classmethod
    def get_config(cls, dataset_name: str = None) -> dict:
        """
        Get full configuration by merging TrainingConfig with multi-view defaults.
        
        Args:
            dataset_name: Optional dataset name to get base config for
            
        Returns:
            Dictionary with all configuration parameters
        """
        # Get base config from TrainingConfig
        base_config = TrainingConfig.get_all_config(dataset_name)
        
        # Build merged config
        merged = {}
        
        # Training params from TrainingConfig
        training_params = base_config['training_params']
        merged['batch_size'] = training_params['batch_size']
        merged['num_epochs'] = training_params['num_epochs']
        merged['learning_rate'] = training_params['learning_rate']
        merged['weight_decay'] = training_params.get('weight_decay', 1e-4)
        merged['seed'] = training_params['seed']
        merged['rotation_representation'] = training_params['rotation_representation']
        merged['resume_checkpoint'] = training_params.get('resume_checkpoint')
        merged['num_workers'] = training_params.get('num_workers', 4)
        merged['pin_memory'] = training_params.get('pin_memory', True)
        
        # Model config from TrainingConfig
        model_config = base_config['model_config']
        merged['backbone_name'] = model_config['backbone_name']
        merged['freeze_backbone'] = model_config['freeze_backbone']
        merged['head_type'] = model_config['head_type']
        merged['hidden_dim'] = model_config['hidden_dim']
        merged['transformer_config'] = model_config.get('transformer_config', {})
        merged['use_unity_prior'] = model_config.get('use_unity_prior', False)
        
        # Scale/trans mode from TrainingConfig
        merged['scale_trans_mode'] = TrainingConfig.get_scale_trans_mode()
        
        # Loss weights from TrainingConfig (epoch 0 base weights)
        merged['loss_weights'] = TrainingConfig.get_loss_weights_for_epoch(0)
        
        # Add multi-view specific defaults
        merged.update(cls.MULTIVIEW_DEFAULTS)
        
        # Shape family from global config
        merged['shape_family'] = config.SHAPE_FAMILY
        
        # Mesh scaling config from TrainingConfig (if available)
        mesh_scaling_config = TrainingConfig.get_mesh_scaling_config()
        merged['allow_mesh_scaling'] = mesh_scaling_config.get('allow_mesh_scaling', False)
        merged['mesh_scale_init'] = mesh_scaling_config.get('init_mesh_scale', 1.0)
        
        return merged
    
    @classmethod
    def from_args(cls, args) -> dict:
        """
        Create config from command line arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Configuration dictionary
        """
        # Start with merged config from TrainingConfig + multiview defaults
        merged_config = cls.get_config()
        
        # Override with command line args
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                merged_config[key] = value
        
        return merged_config
    
    @classmethod
    def from_file(cls, config_path: str) -> dict:
        """
        Load config from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        
        # Start with merged config
        merged_config = cls.get_config()
        
        # Override with file config
        merged_config.update(file_config)
        
        return merged_config
    
    @classmethod
    def get_loss_weights_for_epoch(cls, epoch: int, base_weights: dict = None) -> dict:
        """
        Get loss weights for a specific epoch using TrainingConfig curriculum.
        
        Args:
            epoch: Current training epoch
            base_weights: Optional base weights to use instead of TrainingConfig
            
        Returns:
            Dictionary of loss weights
        """
        if base_weights is not None:
            # Start with provided base weights
            weights = base_weights.copy()
            # Apply curriculum stages from TrainingConfig
            for epoch_threshold, weight_updates in TrainingConfig.LOSS_CURRICULUM['curriculum_stages']:
                if epoch >= epoch_threshold:
                    weights.update(weight_updates)
            return weights
        else:
            # Use TrainingConfig directly
            return TrainingConfig.get_loss_weights_for_epoch(epoch)
    
    @classmethod
    def get_learning_rate_for_epoch(cls, epoch: int) -> float:
        """
        Get learning rate for a specific epoch using TrainingConfig curriculum.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Learning rate for the epoch
        """
        return TrainingConfig.get_learning_rate_for_epoch(epoch)


def train_epoch(model: MultiViewSMILImageRegressor,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: str,
                epoch: int,
                training_config: dict,
                scaler=None,
                is_distributed: bool = False,
                rank: int = 0) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: MultiViewSMILImageRegressor
        train_loader: DataLoader
        optimizer: Optimizer
        device: Device string
        epoch: Current epoch
        training_config: Training config dictionary
        scaler: GradScaler for mixed precision (optional)
        is_distributed: Whether using DDP
        rank: Process rank
        
    Returns:
        Dictionary with training metrics
    """
    model.train()

    # If wrapped in DDP, call custom helper methods on the underlying module.
    # NOTE: DDP does not automatically expose arbitrary custom methods.
    base_model = model.module if hasattr(model, "module") else model
    
    # Get epoch-specific loss weights from curriculum
    loss_weights = MultiViewTrainingConfig.get_loss_weights_for_epoch(
        epoch, training_config.get('loss_weights')
    )
    
    total_loss = 0.0
    loss_components_sum = {}
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(rank != 0))
    
    for batch_idx, (x_data_batch, y_data_batch) in enumerate(progress_bar):
        optimizer.zero_grad()
        
        try:
            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    predicted_params, _, auxiliary_data = base_model.predict_from_multiview_batch(
                        x_data_batch, y_data_batch
                    )
                    
                    if predicted_params is None:
                        continue
                    
                    loss, loss_components = base_model.compute_multiview_batch_loss(
                        predicted_params, y_data_batch,
                        loss_weights=loss_weights,
                        return_components=True
                    )
                
                # Backward with scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if training_config.get('gradient_clip_norm', 0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        training_config['gradient_clip_norm']
                    )
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward/backward
                predicted_params, _, auxiliary_data = base_model.predict_from_multiview_batch(
                    x_data_batch, y_data_batch
                )
                
                if predicted_params is None:
                    continue
                
                loss, loss_components = base_model.compute_multiview_batch_loss(
                    predicted_params, y_data_batch,
                    loss_weights=loss_weights,
                    return_components=True
                )
                
                loss.backward()
                
                # Gradient clipping
                if training_config.get('gradient_clip_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        training_config['gradient_clip_norm']
                    )
                
                optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            num_batches += 1
            
            for key, value in loss_components.items():
                if key not in loss_components_sum:
                    loss_components_sum[key] = 0.0
                loss_components_sum[key] += value.item() if torch.is_tensor(value) else value
            
            # Update progress bar
            if rank == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'kp2d': f'{loss_components.get("keypoint_2d", torch.tensor(0.0)).item():.4f}'
                })
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute averages
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components_sum.items()}
    else:
        avg_loss = 0.0
        avg_components = {}
    
    return {
        'avg_loss': avg_loss,
        'loss_components': avg_components,
        'num_batches': num_batches
    }


def validate(model: MultiViewSMILImageRegressor,
             val_loader: DataLoader,
             device: str,
             training_config: dict,
             epoch: int = 0,
             rank: int = 0) -> dict:
    """
    Validate the model.
    
    Args:
        model: MultiViewSMILImageRegressor
        val_loader: DataLoader
        device: Device string
        training_config: Training configuration dictionary
        epoch: Current epoch (for loss curriculum)
        rank: Process rank
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()

    # If wrapped in DDP, call custom helper methods on the underlying module.
    base_model = model.module if hasattr(model, "module") else model
    
    # Get epoch-specific loss weights from curriculum
    loss_weights = MultiViewTrainingConfig.get_loss_weights_for_epoch(
        epoch, training_config.get('loss_weights')
    )
    
    total_loss = 0.0
    loss_components_sum = {}
    num_batches = 0
    
    with torch.no_grad():
        for x_data_batch, y_data_batch in tqdm(val_loader, desc="Validating", disable=(rank != 0)):
            try:
                predicted_params, _, auxiliary_data = base_model.predict_from_multiview_batch(
                    x_data_batch, y_data_batch
                )
                
                if predicted_params is None:
                    continue
                
                loss, loss_components = base_model.compute_multiview_batch_loss(
                    predicted_params, y_data_batch,
                    loss_weights=loss_weights,
                    return_components=True
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                for key, value in loss_components.items():
                    if key not in loss_components_sum:
                        loss_components_sum[key] = 0.0
                    loss_components_sum[key] += value.item() if torch.is_tensor(value) else value
                    
            except Exception as e:
                print(f"Validation error: {e}")
                continue
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components_sum.items()}
    else:
        avg_loss = float('inf')
        avg_components = {}
    
    return {
        'avg_loss': avg_loss,
        'loss_components': avg_components,
        'num_batches': num_batches
    }


def visualize_multiview_training_progress(model: MultiViewSMILImageRegressor,
                                           val_loader: DataLoader,
                                           device: str,
                                           epoch: int,
                                           training_config: dict,
                                           output_dir: str = 'multiview_visualizations',
                                           num_samples: int = 3,
                                           rank: int = 0):
    """
    Visualize multi-view training progress by creating grid visualizations.
    
    Creates a visualization showing:
    - Top row: Input images from each camera view
    - Bottom row: Rendered mesh using unified body params + per-view camera params
    - Overlays GT keypoints (circles) and predicted keypoints (crosses)
    
    Uses a fixed random seed (42) for sample selection to ensure the same samples
    are visualized across all epochs, enabling consistent progress tracking.
    
    Args:
        model: MultiViewSMILImageRegressor model
        val_loader: Validation data loader
        device: PyTorch device string
        epoch: Current epoch number
        training_config: Training configuration dictionary
        output_dir: Directory to save visualization images
        num_samples: Number of samples to visualize
        rank: Process rank (only rank 0 generates visualizations)
    """
    if rank != 0:
        return  # Only main process generates visualizations
    
    model.eval()
    
    # Save current random states to restore after visualization
    torch_rng_state = torch.get_rng_state()
    numpy_rng_state = np.random.get_state()
    python_rng_state = random.getstate()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    
    # Set fixed seed for deterministic sample selection
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
        
        # Collect samples for visualization
        collected_samples = []
        max_batches_to_check = 50  # Limit to avoid processing entire dataset
        
        with torch.no_grad():
            for batch_idx, (x_data_batch, y_data_batch) in enumerate(val_loader):
                if batch_idx >= max_batches_to_check or len(collected_samples) >= num_samples:
                    break
                
                # Collect individual samples from the batch
                for i, (x_data, y_data) in enumerate(zip(x_data_batch, y_data_batch)):
                    if len(collected_samples) >= num_samples:
                        break
                    
                    # Skip if no valid images
                    if not x_data.get('images') or len(x_data.get('images', [])) == 0:
                        continue
                    
                    collected_samples.append((x_data, y_data))
        
        if len(collected_samples) == 0:
            print(f"Warning: No samples collected for visualization at epoch {epoch}")
            return
        
        # Get the base model (unwrap DDP if needed)
        base_model = model.module if hasattr(model, 'module') else model
        
        # Process each sample
        for sample_idx, (x_data, y_data) in enumerate(collected_samples):
            try:
                visualization = create_multiview_visualization(
                    base_model, x_data, y_data, device, training_config
                )
                
                if visualization is not None:
                    # Save visualization
                    output_path = os.path.join(epoch_dir, f'sample_{sample_idx:03d}_epoch_{epoch:03d}.png')
                    imageio.imsave(output_path, visualization)
                    
            except Exception as e:
                print(f"Warning: Failed to create visualization for sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Generated {len(collected_samples)} multi-view visualizations for epoch {epoch} in {epoch_dir}")
        
    finally:
        # Restore random states
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(numpy_rng_state)
        random.setstate(python_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)


def create_multiview_visualization(model: MultiViewSMILImageRegressor,
                                    x_data: dict,
                                    y_data: dict,
                                    device: str,
                                    training_config: dict) -> np.ndarray:
    """
    Create a grid visualization for a single multi-view sample.
    
    Layout:
    ┌────────────┬────────────┬────────────┬─────┐
    │  Input V0  │  Input V1  │  Input V2  │ ... │  <- Input images
    ├────────────┼────────────┼────────────┼─────┤
    │ Render V0  │ Render V1  │ Render V2  │ ... │  <- Rendered with keypoints
    └────────────┴────────────┴────────────┴─────┘
    
    Args:
        model: MultiViewSMILImageRegressor
        x_data: Input data dictionary for one sample
        y_data: Target data dictionary for one sample
        device: PyTorch device
        training_config: Training configuration
        
    Returns:
        Numpy array of the visualization image (H, W, 3) uint8
    """
    images = x_data.get('images', [])
    num_views = len(images)
    
    if num_views == 0:
        return None
    
    # Get camera indices
    cam_indices = x_data.get('camera_indices', list(range(num_views)))
    if isinstance(cam_indices, np.ndarray):
        cam_indices = cam_indices.tolist()
    
    # Preprocess images and prepare batch
    images_per_view = []
    for img in images:
        img_tensor = model.preprocess_image(img).to(device)  # (1, 3, H, W)
        images_per_view.append(img_tensor.squeeze(0))  # (3, H, W)
    
    # Stack into tensors for model
    images_tensors = [img.unsqueeze(0) for img in images_per_view]  # List of (1, 3, H, W)
    camera_indices_tensor = torch.tensor([cam_indices], device=device)  # (1, num_views)
    view_mask = torch.ones(1, num_views, dtype=torch.bool, device=device)  # (1, num_views)
    
    # Forward pass through model (no gradient needed for visualization)
    with torch.no_grad():
        predicted_params = model.forward_multiview(images_tensors, camera_indices_tensor, view_mask)
    
    # Visualization parameters
    img_size = 224  # Standard visualization size
    margin = 5
    
    # Create visualization grid
    grid_width = num_views * img_size + (num_views + 1) * margin
    grid_height = 2 * img_size + 3 * margin  # 2 rows: input + rendered
    
    # Create canvas with dark background
    canvas = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 40
    
    # Get target keypoints for each view
    target_keypoints = y_data.get('keypoints_2d', None)
    target_visibility = y_data.get('keypoint_visibility', None)
    
    for v in range(num_views):
        x_offset = margin + v * (img_size + margin)
        
        # === TOP ROW: Input images ===
        input_img = images[v]
        if isinstance(input_img, np.ndarray):
            # Resize to visualization size if needed
            if input_img.shape[0] != img_size or input_img.shape[1] != img_size:
                from PIL import Image
                pil_img = Image.fromarray((input_img * 255).astype(np.uint8) if input_img.max() <= 1 else input_img.astype(np.uint8))
                pil_img = pil_img.resize((img_size, img_size), Image.BILINEAR)
                input_img = np.array(pil_img)
            
            # Ensure uint8 format
            if input_img.max() <= 1.0:
                input_img = (input_img * 255).astype(np.uint8)
            else:
                input_img = input_img.astype(np.uint8)
            
            # Ensure RGB (3 channels)
            if len(input_img.shape) == 2:
                input_img = np.stack([input_img] * 3, axis=-1)
            elif input_img.shape[-1] == 4:
                input_img = input_img[:, :, :3]
            
            canvas[margin:margin + img_size, x_offset:x_offset + img_size] = input_img
        
        # === BOTTOM ROW: Rendered mesh with keypoints ===
        try:
            # Create rendered image with keypoint overlays
            rendered_img = create_rendered_view_with_keypoints(
                model, predicted_params, v, 
                target_keypoints, target_visibility,
                device, img_size
            )
            
            canvas[2 * margin + img_size:2 * margin + 2 * img_size, 
                   x_offset:x_offset + img_size] = rendered_img
            
        except Exception as e:
            print(f"Warning: Could not render view {v}: {e}")
            # Fill with placeholder
            placeholder = np.ones((img_size, img_size, 3), dtype=np.uint8) * 128
            canvas[2 * margin + img_size:2 * margin + 2 * img_size,
                   x_offset:x_offset + img_size] = placeholder
    
    # Add row labels
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_canvas)
        
        # Try to use a basic font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Add labels
        draw.text((5, margin + img_size // 2 - 6), "Input", fill=(255, 255, 255), font=font)
        draw.text((5, 2 * margin + img_size + img_size // 2 - 6), "Pred", fill=(255, 255, 255), font=font)
        
        # Add view labels
        for v in range(num_views):
            x_pos = margin + v * (img_size + margin) + img_size // 2 - 10
            cam_name = x_data.get('camera_names', [f'V{v}'])[v] if v < len(x_data.get('camera_names', [])) else f'V{v}'
            draw.text((x_pos, 2), str(cam_name)[:8], fill=(255, 255, 255), font=font)
        
        canvas = np.array(pil_canvas)
    except ImportError:
        pass  # PIL not available for text, skip labels
    
    return canvas


def create_rendered_view_with_keypoints(model: MultiViewSMILImageRegressor,
                                         predicted_params: dict,
                                         view_idx: int,
                                         target_keypoints: np.ndarray,
                                         target_visibility: np.ndarray,
                                         device: str,
                                         img_size: int) -> np.ndarray:
    """
    Create a rendered view with keypoint overlays.
    
    Args:
        model: The multi-view model
        predicted_params: Output from forward_multiview
        view_idx: Which view to render
        target_keypoints: Ground truth keypoints (num_views, n_joints, 2) or (n_joints, 2)
        target_visibility: Ground truth visibility (num_views, n_joints) or (n_joints,)
        device: PyTorch device
        img_size: Output image size
        
    Returns:
        Rendered image with keypoint overlays (img_size, img_size, 3) uint8
    """
    from smil_image_regressor import rotation_6d_to_axis_angle
    
    # Get camera parameters for this view
    fov = predicted_params['fov_per_view'][view_idx]  # (batch_size, 1)
    cam_rot = predicted_params['cam_rot_per_view'][view_idx]  # (batch_size, 3, 3)
    cam_trans = predicted_params['cam_trans_per_view'][view_idx]  # (batch_size, 3)
    
    # Render 2D keypoints using body params + view camera
    try:
        with torch.no_grad():
            rendered_joints = model._render_keypoints_with_camera(
                predicted_params, fov, cam_rot, cam_trans
            )  # (batch_size, n_joints, 2)
        
        # Convert to numpy for visualization
        pred_kps = rendered_joints[0].detach().cpu().numpy()  # (n_joints, 2)
        pred_kps = pred_kps * img_size  # Scale to image coordinates
        
    except Exception as e:
        print(f"Keypoint rendering failed: {e}")
        pred_kps = None
    
    # Create base image (gray background for now, could be rendered mesh later)
    # For now, use a simple gradient to indicate the view
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 60
    
    # Add a subtle gradient to differentiate views
    for i in range(img_size):
        img[i, :, 0] = min(255, 60 + view_idx * 30)  # Slightly different tint per view
    
    # Get target keypoints for this view
    gt_kps = None
    gt_vis = None
    if target_keypoints is not None:
        if len(target_keypoints.shape) == 3:
            # Multi-view format: (num_views, n_joints, 2)
            if view_idx < target_keypoints.shape[0]:
                gt_kps = target_keypoints[view_idx] * img_size  # (n_joints, 2)
                if target_visibility is not None and view_idx < target_visibility.shape[0]:
                    gt_vis = target_visibility[view_idx]  # (n_joints,)
        elif len(target_keypoints.shape) == 2 and view_idx == 0:
            # Single-view format: (n_joints, 2)
            gt_kps = target_keypoints * img_size
            gt_vis = target_visibility
    
    # Draw keypoints on image
    from PIL import Image, ImageDraw
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Draw ground truth keypoints (green circles)
    if gt_kps is not None:
        for j, (y, x) in enumerate(gt_kps):
            if gt_vis is None or gt_vis[j] > 0.5:
                # Keypoints are in (y, x) format, need (x, y) for PIL
                x, y = float(x), float(y)
                if 0 <= x < img_size and 0 <= y < img_size:
                    draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline='green', width=2)
    
    # Draw predicted keypoints (red crosses)
    if pred_kps is not None:
        for j, (y, x) in enumerate(pred_kps):
            x, y = float(x), float(y)
            if 0 <= x < img_size and 0 <= y < img_size:
                # Draw cross
                draw.line([x - 4, y, x + 4, y], fill='red', width=2)
                draw.line([x, y - 4, x, y + 4], fill='red', width=2)
    
    # Add legend
    try:
        from PIL import ImageFont
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = ImageFont.load_default()
        draw.text((5, img_size - 25), "○ GT", fill='green', font=font)
        draw.text((5, img_size - 12), "+ Pred", fill='red', font=font)
    except:
        pass
    
    return np.array(pil_img)


def draw_keypoints_on_image(image_rgb: np.ndarray,
                            gt_kps: Optional[np.ndarray] = None,
                            gt_vis: Optional[np.ndarray] = None,
                            pred_kps: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Draw GT (green circles) and "pred" (red crosses) keypoints on top of an RGB image.

    Notes:
      - Keypoints are expected in (y, x) pixel coordinates.
      - `image_rgb` can be float in [0,1] or uint8 in [0,255].
    """
    from PIL import Image, ImageDraw

    img = image_rgb
    if img is None:
        return None

    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # GT keypoints (green circles)
    if gt_kps is not None:
        for j, (y, x) in enumerate(gt_kps):
            if gt_vis is None or gt_vis[j] > 0.5:
                x, y = float(x), float(y)
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline='green', width=2)

    # Pred keypoints (red crosses)
    if pred_kps is not None:
        for (y, x) in pred_kps:
            x, y = float(x), float(y)
            draw.line([x - 4, y, x + 4, y], fill='red', width=2)
            draw.line([x, y - 4, x, y + 4], fill='red', width=2)

    return np.array(pil_img)


class SingleViewImageExporter:
    """Image exporter for single-view rendered visualizations."""
    def __init__(self, output_dir: str, sample_idx: int, view_idx: int, epoch: int):
        self.output_dir = output_dir
        self.sample_idx = sample_idx
        self.view_idx = view_idx
        self.epoch = epoch
        os.makedirs(output_dir, exist_ok=True)
    
    def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces, img_idx=0):
        """Export visualization image with sample and view info in filename."""
        filename = f"sample_{self.sample_idx:03d}_view_{self.view_idx:02d}_epoch_{self.epoch:03d}.png"
        imageio.imsave(os.path.join(self.output_dir, filename), collage_np)


def export_mesh_as_obj(vertices: torch.Tensor, faces: torch.Tensor, output_path: str):
    """
    Export mesh vertices and faces as OBJ file.
    
    Args:
        vertices: Vertex positions tensor of shape (N, 3) or (1, N, 3)
        faces: Face indices tensor of shape (F, 3) or (1, F, 3)
        output_path: Path to save OBJ file
    """
    # Handle batch dimension
    if vertices.dim() == 3:
        vertices = vertices[0]  # (N, 3)
    if faces.dim() == 3:
        faces = faces[0]  # (F, 3)
    
    # Convert to numpy
    verts_np = vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    
    # OBJ files use 1-based indexing for faces
    faces_np = faces_np + 1
    
    # Write OBJ file
    with open(output_path, 'w') as f:
        # Write vertices
        for v in verts_np:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces
        for face in faces_np:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


def visualize_singleview_renders(model: MultiViewSMILImageRegressor,
                                  val_loader: DataLoader,
                                  device: str,
                                  epoch: int,
                                  training_config: dict,
                                  output_dir: str = 'singleview_visualizations',
                                  num_samples: int = 3,
                                  rank: int = 0):
    """
    Generate single-view rendered mesh visualizations for multi-view samples.
    
    This function renders the SMIL mesh using predicted parameters and camera
    settings for each view, producing SMALFitter-style collages that show:
    - Original input image
    - Rendered mesh overlay
    - Keypoint projections
    
    Args:
        model: MultiViewSMILImageRegressor model
        val_loader: Validation data loader
        device: PyTorch device
        epoch: Current epoch number
        training_config: Training configuration dictionary
        output_dir: Base directory to save visualizations
        num_samples: Number of samples to visualize
        rank: Process rank for distributed training
    """
    if rank != 0:
        return
    
    model.eval()
    
    # Save random states
    torch_rng_state = torch.get_rng_state()
    numpy_rng_state = np.random.get_state()
    python_rng_state = random.getstate()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    
    # Set fixed seed for deterministic sample selection
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
        
        # Collect samples
        collected_samples = []
        max_batches_to_check = 50
        
        with torch.no_grad():
            for batch_idx, (x_data_batch, y_data_batch) in enumerate(val_loader):
                if batch_idx >= max_batches_to_check or len(collected_samples) >= num_samples:
                    break
                
                for i, (x_data, y_data) in enumerate(zip(x_data_batch, y_data_batch)):
                    if len(collected_samples) >= num_samples:
                        break
                    if not x_data.get('images') or len(x_data.get('images', [])) == 0:
                        continue
                    collected_samples.append((x_data, y_data))
        
        if len(collected_samples) == 0:
            print(f"Warning: No samples collected for single-view visualization at epoch {epoch}")
            return
        
        # Get base model
        base_model = model.module if hasattr(model, 'module') else model
        
        # Process each sample
        num_rendered = 0
        for sample_idx, (x_data, y_data) in enumerate(collected_samples):
            try:
                views_rendered = render_singleview_for_sample(
                    base_model, x_data, y_data, device, training_config,
                    epoch_dir, sample_idx, epoch
                )
                if views_rendered > 0:
                    num_rendered += 1
            except Exception as e:
                print(f"Warning: Failed to render single-view for sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Generated single-view renders for {num_rendered} samples (epoch {epoch}) in {epoch_dir}")
    
    finally:
        # Restore random states
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(numpy_rng_state)
        random.setstate(python_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)


def render_singleview_for_sample(model: MultiViewSMILImageRegressor,
                                  x_data: dict,
                                  y_data: dict,
                                  device: str,
                                  training_config: dict,
                                  output_dir: str,
                                  sample_idx: int,
                                  epoch: int) -> int:
    """
    Render single-view mesh visualizations for each view in a multi-view sample.
    
    Args:
        model: MultiViewSMILImageRegressor model
        x_data: Input data dictionary
        y_data: Target data dictionary
        device: PyTorch device
        training_config: Training configuration
        output_dir: Output directory for visualizations
        sample_idx: Sample index
        epoch: Current epoch
        
    Returns:
        Number of views successfully rendered
    """
    images = x_data.get('images', [])
    num_views = len(images)
    
    if num_views == 0:
        return 0
    
    # Get camera indices
    cam_indices = x_data.get('camera_indices', list(range(num_views)))
    if isinstance(cam_indices, np.ndarray):
        cam_indices = cam_indices.tolist()
    
    # Preprocess images and prepare for model
    images_per_view = []
    for img in images:
        img_tensor = model.preprocess_image(img).to(device)  # (1, 3, H, W)
        images_per_view.append(img_tensor.squeeze(0))  # (3, H, W)
    
    images_tensors = [img.unsqueeze(0) for img in images_per_view]  # List of (1, 3, H, W)
    camera_indices_tensor = torch.tensor([cam_indices], device=device)  # (1, num_views)
    view_mask = torch.ones(1, num_views, dtype=torch.bool, device=device)  # (1, num_views)
    
    # Forward pass to get predictions
    with torch.no_grad():
        predicted_params = model.forward_multiview(images_tensors, camera_indices_tensor, view_mask)
    
    # Render each view separately using SMALFitter
    views_rendered = 0
    
    # Get per-view camera parameters from predictions
    # NOTE: These are LISTS of tensors, one per view, not batched tensors
    fov_per_view = predicted_params.get('fov_per_view', None)  # List of (batch_size, 1)
    cam_rot_per_view = predicted_params.get('cam_rot_per_view', None)  # List of (batch_size, 3, 3)
    cam_trans_per_view = predicted_params.get('cam_trans_per_view', None)  # List of (batch_size, 3)
    
    # IMPORTANT: Use the model's renderer image size, not the backbone input size
    # The model's renderer is initialized with a specific size, and _render_keypoints_with_camera
    # normalizes by this size. We must match this exactly for correct visualization.
    model_renderer_size = model.renderer.image_size
    target_size = model_renderer_size
    
    if sample_idx == 0:
        print(f"\n[Visualization] Using model renderer image size: {target_size}")
        print(f"  Original image size: {images[0].shape if isinstance(images[0], np.ndarray) else 'tensor'}")
        print(f"  Model renderer size: {model_renderer_size}")
        print(f"  This ensures keypoint normalization matches training")
    
    # Debug: Print predicted params summary and input fingerprint for EACH sample
    # This helps diagnose if model receives different inputs but outputs same predictions
    img_fingerprint = images_tensors[0][0, 0, :5, :5].mean().item()  # Small patch mean as fingerprint
    print(f"\n[Sample {sample_idx}] Input fingerprint (img patch mean): {img_fingerprint:.6f}")
    print(f"  trans: [{predicted_params['trans'][0, 0].item():.4f}, {predicted_params['trans'][0, 1].item():.4f}, {predicted_params['trans'][0, 2].item():.4f}]")
    print(f"  global_rot[0:3]: [{predicted_params['global_rot'][0, 0].item():.4f}, {predicted_params['global_rot'][0, 1].item():.4f}, {predicted_params['global_rot'][0, 2].item():.4f}]")
    if fov_per_view is not None:
        print(f"  fov_per_view[0]: {fov_per_view[0][0, 0].item():.4f}")
    if model.allow_mesh_scaling and 'mesh_scale' in predicted_params:
        print(f"  mesh_scale: {predicted_params['mesh_scale'][0].item():.4f}")
    
    for view_idx in range(num_views):
        try:
            # Get original image for this view
            original_image = images[view_idx]  # (H, W, 3) in [0, 1] range
            
            # Resize image to renderer's expected size (target_size)
            # This ensures SMALFitter and renderer use consistent image dimensions
            from PIL import Image
            if isinstance(original_image, np.ndarray):
                # Convert to PIL, resize, then back to numpy
                pil_img = Image.fromarray((original_image * 255).astype(np.uint8))
                pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
                resized_image = np.array(pil_img).astype(np.float32) / 255.0
            else:
                # Tensor case - convert to numpy first, then use PIL for consistency
                img_np = original_image.cpu().numpy() if hasattr(original_image, 'cpu') else original_image.numpy()
                pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
                pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
                resized_image = np.array(pil_img).astype(np.float32) / 255.0
            
            # Prepare RGB tensor for SMALFitter (BGR format for visualization compatibility)
            # Ensure values are in [0, 1] range
            resized_image = np.clip(resized_image, 0.0, 1.0)
            # Swap RGB to BGR channels
            resized_image_bgr = resized_image[:, :, [2, 1, 0]]  # RGB -> BGR
            rgb = torch.from_numpy(resized_image_bgr).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)
            
            # Verify RGB range (SMALFitter expects [0, 1])
            assert rgb.min() >= 0.0 and rgb.max() <= 1.0, f"RGB values out of range: [{rgb.min():.3f}, {rgb.max():.3f}]"
            
            # Get keypoints for this view
            keypoints_2d = y_data.get('keypoints_2d', None)
            visibility = y_data.get('keypoint_visibility', None)
            
            if keypoints_2d is not None:
                # Handle multi-view format (num_views, n_joints, 2) or single-view (n_joints, 2)
                if len(keypoints_2d.shape) == 3:
                    view_keypoints = keypoints_2d[view_idx] if view_idx < keypoints_2d.shape[0] else None
                    view_visibility = visibility[view_idx] if visibility is not None and view_idx < visibility.shape[0] else None
                else:
                    view_keypoints = keypoints_2d if view_idx == 0 else None
                    view_visibility = visibility if view_idx == 0 else None
            else:
                view_keypoints = None
                view_visibility = None
            
            # Create silhouette matching the resized image size
            sil = torch.zeros(1, 1, target_size, target_size)
            
            # Create data batch for SMALFitter
            if view_keypoints is not None and view_visibility is not None:
                # Convert normalized keypoints to pixel coordinates
                pixel_coords = view_keypoints.copy()
                pixel_coords[:, 0] = pixel_coords[:, 0] * target_size  # y coordinates
                pixel_coords[:, 1] = pixel_coords[:, 1] * target_size  # x coordinates
                
                num_joints = len(view_keypoints)
                joints = torch.tensor(pixel_coords.reshape(1, num_joints, 2), dtype=torch.float32)
                vis = torch.tensor(view_visibility.reshape(1, num_joints), dtype=torch.float32)
                
                temp_batch = (rgb, sil, joints, vis)
                rgb_only = False
            else:
                temp_batch = rgb
                rgb_only = True
            
            # Create SMALFitter for this view
            temp_fitter = SMALFitter(
                device=device,
                data_batch=temp_batch,
                batch_size=1,
                shape_family=config.SHAPE_FAMILY,
                use_unity_prior=False,
                rgb_only=rgb_only
            )
            
            # Set target joints for visualization
            if view_keypoints is not None and view_visibility is not None:
                pixel_coords = view_keypoints.copy()
                pixel_coords[:, 0] = pixel_coords[:, 0] * target_size
                pixel_coords[:, 1] = pixel_coords[:, 1] * target_size
                temp_fitter.target_joints = torch.tensor(pixel_coords, dtype=torch.float32, device=device).unsqueeze(0)
                temp_fitter.target_visibility = torch.tensor(view_visibility, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                n_joints = temp_fitter.joint_rotations.shape[1] + 1  # Include root joint
                temp_fitter.target_joints = torch.zeros((1, n_joints, 2), device=device)
                temp_fitter.target_visibility = torch.zeros((1, n_joints), device=device)
            
            # Set predicted body parameters
            # Convert 6D rotation to axis-angle if needed
            if model.rotation_representation == '6d':
                global_rot_aa = rotation_6d_to_axis_angle(predicted_params['global_rot'][0:1].detach())
                joint_rot_aa = rotation_6d_to_axis_angle(predicted_params['joint_rot'][0:1].detach())
            else:
                global_rot_aa = predicted_params['global_rot'][0:1].detach()
                joint_rot_aa = predicted_params['joint_rot'][0:1].detach()
            
            # IMPORTANT: SMALFitter parameters have specific shapes:
            # - global_rotation: (num_images, 3)
            # - joint_rotations: (num_images, N_POSE, 3)
            # - betas: (n_betas,) - 1D, NOT batched!
            # - trans: (num_images, 3)
            # - fov: (num_images,)
            temp_fitter.global_rotation.data = global_rot_aa.to(device)
            temp_fitter.joint_rotations.data = joint_rot_aa.to(device)
            # betas is 1D in SMALFitter, squeeze the batch dimension
            temp_fitter.betas.data = predicted_params['betas'][0].detach().to(device)  # (n_betas,)
            temp_fitter.trans.data = predicted_params['trans'][0:1].detach().to(device)
            
            # Set FOV - fov in SMALFitter is (num_images,) shaped
            if fov_per_view is not None and view_idx < len(fov_per_view):
                # fov_per_view is a List of (batch_size, 1) tensors
                fov_val = fov_per_view[view_idx][0, 0].detach().to(device)  # scalar
                temp_fitter.fov.data = fov_val.unsqueeze(0)  # (1,)
            elif 'fov' in predicted_params:
                temp_fitter.fov.data = predicted_params['fov'][0:1].detach().to(device)  # (1,)
            
            # Set scale and translation parameters if available
            #
            # IMPORTANT:
            # In `separate` mode these are PCA weights; in training we intentionally *do not* apply
            # them in the renderer path (`_render_keypoints_with_camera`) because they require PCA
            # dirs and are typically unsupervised (loss weights often 0).
            #
            # Applying them here in visualization can make the mesh look severely deformed even
            # though training is not actually optimizing those degrees of freedom.
            if 'log_beta_scales' in predicted_params and 'betas_trans' in predicted_params:
                if model.scale_trans_mode in ['separate', 'ignore']:
                    # Mirror training behavior: skip unsupervised per-joint scaling/translation.
                    # This keeps visualizations consistent with what the loss is optimizing.
                    pass
                else:
                    temp_fitter.log_beta_scales.data = predicted_params['log_beta_scales'][0:1].detach().to(device)
                    temp_fitter.betas_trans.data = predicted_params['betas_trans'][0:1].detach().to(device)
            
            # Set per-view camera parameters on the renderer
            # IMPORTANT: This must be done AFTER setting body parameters to ensure
            # the renderer uses the correct camera for this view
            # cam_rot_per_view and cam_trans_per_view are Lists of tensors
            if cam_rot_per_view is not None and cam_trans_per_view is not None:
                if view_idx < len(cam_rot_per_view):
                    cam_rot = cam_rot_per_view[view_idx][0:1].detach().to(device)  # (1, 3, 3)
                    cam_trans = cam_trans_per_view[view_idx][0:1].detach().to(device)  # (1, 3)
                    # fov for renderer should match temp_fitter.fov.data format
                    if fov_per_view is not None and view_idx < len(fov_per_view):
                        # Extract FOV value and ensure it matches the format expected by renderer
                        view_fov_val = fov_per_view[view_idx][0, 0].detach().to(device)  # scalar
                        view_fov = view_fov_val.unsqueeze(0)  # (1,)
                        # Also update temp_fitter.fov to match (for consistency)
                        temp_fitter.fov.data = view_fov
                    else:
                        view_fov = temp_fitter.fov.data
                    
                    # Set camera parameters on renderer - this is critical for correct visualization
                    # IMPORTANT: Must set camera BEFORE calling generate_visualization
                    # If the dataset provides a GT-derived aspect ratio, use it for rendering geometry.
                    # This is critical for calibrated cameras with W!=H and/or fx!=fy.
                    aspect = None
                    try:
                        if y_data.get('cam_aspect_per_view') is not None:
                            aspect = float(np.array(y_data['cam_aspect_per_view'][view_idx]).reshape(-1)[0])
                    except Exception:
                        aspect = None

                    temp_fitter.renderer.set_camera_parameters(
                        R=cam_rot,
                        T=cam_trans,
                        fov=view_fov,
                        aspect_ratio=aspect
                    )
                    
                    # Verify camera was set correctly
                    # The renderer should now use the updated camera
                    assert temp_fitter.renderer.cameras is not None, "Camera not set on renderer"
                    
                    # Debug output for first sample, first view
                    if sample_idx == 0 and view_idx == 0:
                        print(f"\n[Camera Debug] View {view_idx}:")
                        print(f"  FOV: {view_fov_val.item():.2f} degrees")
                        print(f"  cam_trans: {cam_trans[0].cpu().numpy()}")
                        print(f"  cam_rot (first row): {cam_rot[0, 0].cpu().numpy()}")
                        print(f"  Renderer image size: {temp_fitter.renderer.image_size}")
                        print(f"  Body trans: {predicted_params['trans'][0].cpu().numpy()}")
                        print(f"  Betas (first 3): {predicted_params['betas'][0, :3].cpu().numpy()}")
                        print(f"  Camera R shape: {temp_fitter.renderer.cameras.R.shape}")
                        print(f"  Camera T shape: {temp_fitter.renderer.cameras.T.shape}")
                        print(f"  Camera fov shape: {temp_fitter.renderer.cameras.fov.shape}")

                    # ===================== GT CAMERA + GT 3D PROJECTION SANITY CHECK =====================
                    # If the dataset provides 3D keypoints + calibrated cameras, verify that:
                    #   GT 3D keypoints + GT camera -> projected 2D aligns with GT 2D keypoints.
                    #
                    # This catches unit-scale issues (e.g., mm vs m) and convention mismatches in R/T parsing.
                    try:
                        has_3d = bool(y_data.get('has_3d_data', False))
                        kp3d = y_data.get('keypoints_3d', None)
                        gt_cam_R_all = y_data.get('cam_rot_per_view', None)
                        gt_cam_T_all = y_data.get('cam_trans_per_view', None)
                        gt_cam_fov_all = y_data.get('cam_fov_per_view', None)

                        if has_3d and kp3d is not None and gt_cam_R_all is not None and gt_cam_T_all is not None and gt_cam_fov_all is not None:
                            # Extract GT cam params for this view
                            gt_R = torch.from_numpy(np.array(gt_cam_R_all[view_idx])).to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,3,3)
                            gt_T = torch.from_numpy(np.array(gt_cam_T_all[view_idx])).to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,3)
                            gt_fov_arr = np.array(gt_cam_fov_all[view_idx]).reshape(-1)
                            gt_fov = torch.from_numpy(gt_fov_arr).to(device=device, dtype=torch.float32)  # (1,)
                            if gt_fov.numel() != 1:
                                gt_fov = gt_fov[:1]

                            # GT 3D points (world coords, should already be scaled into SMILify world units by dataset)
                            pts3d = torch.from_numpy(np.array(kp3d)).to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,J,3)

                            # Project using GT camera (temporarily swap camera, then restore predicted camera)
                            gt_aspect = None
                            try:
                                if y_data.get('cam_aspect_per_view') is not None:
                                    gt_aspect = float(np.array(y_data['cam_aspect_per_view'][view_idx]).reshape(-1)[0])
                            except Exception:
                                gt_aspect = None
                            temp_fitter.renderer.set_camera_parameters(R=gt_R, T=gt_T, fov=gt_fov, aspect_ratio=gt_aspect)
                            screen_size = torch.ones(1, 2, dtype=torch.float32, device=device) * float(target_size)
                            proj = temp_fitter.renderer.cameras.transform_points_screen(pts3d, image_size=screen_size)[:, :, [1, 0]]  # (1,J,2) (y,x)
                            proj_np = proj[0].detach().cpu().numpy()

                            # Restore predicted camera for SMALFitter visualization
                            temp_fitter.renderer.set_camera_parameters(R=cam_rot, T=cam_trans, fov=view_fov, aspect_ratio=aspect)

                            # Only draw overlay if we have 2D keypoints for this view
                            if view_keypoints is not None and view_visibility is not None:
                                gt2d_px = view_keypoints.copy()
                                gt2d_px[:, 0] = gt2d_px[:, 0] * target_size
                                gt2d_px[:, 1] = gt2d_px[:, 1] * target_size

                                overlay = draw_keypoints_on_image(
                                    resized_image,  # RGB
                                    gt_kps=gt2d_px,
                                    gt_vis=view_visibility,
                                    pred_kps=proj_np,
                                )
                                overlay_name = f"sample_{sample_idx:03d}_view_{view_idx:02d}_epoch_{epoch:03d}_gtcam_gt3dproj.png"
                                imageio.imsave(os.path.join(output_dir, overlay_name), overlay)

                            # Debug stats for first sample/view
                            if sample_idx == 0 and view_idx == 0:
                                tnorm = float(torch.linalg.norm(gt_T[0]).item())
                                kpnorm = float(torch.max(torch.abs(pts3d)).item())
                                print(f"\n[GT Cam+3D Sanity] view {view_idx}:")
                                print(f"  GT fov_y: {float(gt_fov[0].item()):.2f} deg")
                                print(f"  ||GT cam_T||: {tnorm:.3f}  (should be O(0.1-10) after scaling, not O(100-1000))")
                                print(f"  max|GT kp3d|: {kpnorm:.3f} (same unit scale as SMAL verts/joints)")
                                if view_keypoints is not None and view_visibility is not None:
                                    vis_mask = (np.array(view_visibility).reshape(-1) > 0.5)
                                    if vis_mask.any():
                                        gt2d_px = view_keypoints.copy()
                                        gt2d_px[:, 0] = gt2d_px[:, 0] * target_size
                                        gt2d_px[:, 1] = gt2d_px[:, 1] * target_size
                                        diff = proj_np[vis_mask] - gt2d_px[vis_mask]
                                        err_px = float(np.mean(np.linalg.norm(diff, axis=1)))
                                        print(f"  mean reproj err (GT 3D + GT cam -> 2D): {err_px:.2f} px @ {target_size}px")
                    except Exception as _e:
                        # Best-effort debug only; never break training visualization.
                        print(f"Warning: Failed to perform GT Cam+3D Sanity check: {_e}")
                        pass
            else:
                # Fallback: if no per-view camera params, use temp_fitter's default camera
                # This shouldn't happen in multi-view, but handle gracefully
                temp_fitter.renderer.set_camera_parameters(
                    R=temp_fitter.renderer.cameras.R,
                    T=temp_fitter.renderer.cameras.T,
                    fov=temp_fitter.fov.data
                )
            
            # Export mesh geometry as OBJ file for external inspection
            # This uses the same parameters that will be used for rendering
            with torch.no_grad():
                # Get mesh vertices using the same process as generate_visualization
                batch_betas = temp_fitter.betas.expand(1, -1)
                batch_pose = torch.cat([
                    temp_fitter.global_rotation[0:1].unsqueeze(1),
                    temp_fitter.joint_rotations[0:1]
                ], dim=1)
                
                # Get scale/trans parameters
                batch_logscale = None
                batch_trans = None
                if hasattr(temp_fitter, 'log_beta_scales') and temp_fitter.log_beta_scales is not None:
                    batch_logscale = temp_fitter.log_beta_scales[0:1]
                if hasattr(temp_fitter, 'betas_trans') and temp_fitter.betas_trans is not None:
                    batch_trans = temp_fitter.betas_trans[0:1]
                
                # Generate mesh
                mesh_verts, mesh_joints, _, _ = temp_fitter.smal_model(
                    batch_betas, batch_pose,
                    betas_logscale=batch_logscale,
                    betas_trans=batch_trans,
                    propagate_scaling=temp_fitter.propagate_scaling
                )
                
                # Apply transformation to match training rendering logic:
                # - If use_ue_scaling: apply 10x scaling
                # - If allow_mesh_scaling: apply predicted mesh_scale
                # - Otherwise: just translation
                # NOTE: generate_visualization uses apply_UE_transform=False, which only applies translation.
                # However, training rendering (_render_keypoints_with_camera) applies mesh scaling if enabled.
                # We apply the same logic here for OBJ export consistency.
                if model.use_ue_scaling:
                    root_joint = mesh_joints[:, 0:1, :]
                    mesh_verts = (mesh_verts - root_joint) * 10 + temp_fitter.trans[0:1].unsqueeze(1)
                elif model.allow_mesh_scaling and 'mesh_scale' in predicted_params:
                    mesh_scale = predicted_params['mesh_scale'][0:1]  # (1, 1)
                    root_joint = mesh_joints[:, 0:1, :]
                    mesh_verts = (mesh_verts - root_joint) * mesh_scale.unsqueeze(-1) + temp_fitter.trans[0:1].unsqueeze(1)
                else:
                    mesh_verts = mesh_verts + temp_fitter.trans[0:1].unsqueeze(1)
                
                # Get faces
                mesh_faces = temp_fitter.smal_model.faces
                
                # Export OBJ file
                obj_filename = f"sample_{sample_idx:03d}_view_{view_idx:02d}_epoch_{epoch:03d}.obj"
                obj_path = os.path.join(output_dir, obj_filename)
                export_mesh_as_obj(mesh_verts, mesh_faces, obj_path)
                
                if sample_idx == 0 and view_idx == 0:
                    print(f"  Exported mesh to: {obj_path}")
                    print(f"  Mesh vertices: {mesh_verts.shape}, range: [{mesh_verts.min():.3f}, {mesh_verts.max():.3f}]")
                    print(f"  Mesh faces: {mesh_faces.shape}")
            
            # ===================== DEBUG: Print model trans and 3D coordinate stats =====================
            # Compute predicted 3D joints for this sample
            with torch.no_grad():
                pred_joints_3d = model._predict_canonical_joints_3d(predicted_params)  # (1, J, 3)
                pred_joints_mean = pred_joints_3d[0].mean(dim=0).cpu().numpy()  # (3,)
                model_trans = predicted_params['trans'][0].detach().cpu().numpy()  # (3,)
            
            # Get GT 3D keypoints if available
            gt_kp3d_mean = None
            if y_data.get('has_3d_data', False) and y_data.get('keypoints_3d') is not None:
                gt_kp3d = np.array(y_data['keypoints_3d'])  # (J, 3)
                gt_kp3d_mean = gt_kp3d.mean(axis=0)  # (3,)
            
            # Print debug info only for first view of each sample (body params are shared across views)
            if view_idx == 0:
                # Print global_rot (root rotation) to check if it's varying or staying near identity
                global_rot_raw = predicted_params['global_rot'][0].detach().cpu()
                if model.rotation_representation == '6d':
                    # Convert 6D to axis-angle for readability
                    global_rot_aa = rotation_6d_to_axis_angle(global_rot_raw.unsqueeze(0))[0].numpy()
                else:
                    global_rot_aa = global_rot_raw.numpy()
                # Compute rotation magnitude (angle in degrees)
                rot_angle_rad = np.linalg.norm(global_rot_aa)
                rot_angle_deg = np.degrees(rot_angle_rad)
                
                print(f"  Mean pred 3D: [{pred_joints_mean[0]:.4f}, {pred_joints_mean[1]:.4f}, {pred_joints_mean[2]:.4f}]")
                if gt_kp3d_mean is not None:
                    print(f"  Mean GT 3D:  [{gt_kp3d_mean[0]:.4f}, {gt_kp3d_mean[1]:.4f}, {gt_kp3d_mean[2]:.4f}]")
                print(f"  Root rot magnitude: {rot_angle_deg:.2f}°")
            
            # Create image exporter for this view
            image_exporter = SingleViewImageExporter(output_dir, sample_idx, view_idx, epoch)
            
            # Generate visualization - use apply_UE_transform=False to match training
            # (training uses use_ue_scaling=False by default)
            # Pass mesh_scale if allow_mesh_scaling is enabled to match training render path
            vis_mesh_scale = None
            if model.allow_mesh_scaling and 'mesh_scale' in predicted_params:
                vis_mesh_scale = predicted_params['mesh_scale'][0:1].detach()
            temp_fitter.generate_visualization(
                image_exporter, 
                apply_UE_transform=False, 
                img_idx=view_idx,
                mesh_scale=vis_mesh_scale
            )
            
            views_rendered += 1
            
        except Exception as e:
            print(f"Warning: Failed to render view {view_idx} for sample {sample_idx}: {e}")
            continue
    
    # Generate 3D keypoint visualization if 3D data is available
    # This is done once per sample (body params are shared across views)
    if y_data.get('has_3d_data', False) and y_data.get('keypoints_3d') is not None:
        try:
            visualize_3d_keypoints(
                model=model,
                predicted_params=predicted_params,
                y_data=y_data,
                device=device,
                sample_idx=sample_idx,
                epoch=epoch,
                output_dir=output_dir
            )
        except Exception as e:
            print(f"Warning: Failed to create 3D keypoint visualization for sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    return views_rendered


def visualize_3d_keypoints(model: MultiViewSMILImageRegressor,
                            predicted_params: dict,
                            y_data: dict,
                            device: str,
                            sample_idx: int,
                            epoch: int,
                            output_dir: str):
    """
    Create a 3D visualization comparing GT 3D keypoints with predicted 3D joints.
    
    The visualization shows:
    - GT 3D keypoints (circles) in one color per joint
    - Predicted 3D joints (crosses) in matching colors
    - Thin lines connecting GT to predicted for each joint
    
    This helps diagnose alignment issues and understand where 3D errors originate.
    
    Args:
        model: MultiViewSMILImageRegressor model
        predicted_params: Dictionary of predicted parameters
        y_data: Target data dictionary containing GT 3D keypoints
        device: PyTorch device
        sample_idx: Sample index for filename
        epoch: Current epoch number
        output_dir: Directory to save visualization
    """
    # Check if 3D data is available
    if not y_data.get('has_3d_data', False) or y_data.get('keypoints_3d') is None:
        return  # Skip if no 3D data
    
    try:
        # Get GT 3D keypoints
        gt_kp3d = np.array(y_data['keypoints_3d'])  # (J, 3)
        num_joints = gt_kp3d.shape[0]
        
        # Get predicted 3D joints
        base_model = model.module if hasattr(model, 'module') else model
        with torch.no_grad():
            pred_joints_3d = base_model._predict_canonical_joints_3d(predicted_params)  # (1, J, 3)
            pred_joints_3d = pred_joints_3d[0].cpu().numpy()  # (J, 3)
        
        # Ensure same number of joints
        min_joints = min(num_joints, pred_joints_3d.shape[0])
        gt_kp3d = gt_kp3d[:min_joints]
        pred_joints_3d = pred_joints_3d[:min_joints]
        
        # Identify filtered keypoints (NaN or all zeros - these were filtered during preprocessing)
        gt_norms = np.linalg.norm(gt_kp3d, axis=1)
        gt_valid = ~(np.isnan(gt_kp3d).any(axis=1) | np.isinf(gt_kp3d).any(axis=1) | (gt_norms < 1e-6))
        
        # Get joint names if available
        joint_names = None
        try:
            if hasattr(base_model, 'smal_model') and hasattr(base_model.smal_model, 'J_names'):
                joint_names = base_model.smal_model.J_names
            elif hasattr(config, 'joint_names'):
                joint_names = config.joint_names
        except Exception:
            pass
        
        # Track which joints are plotted vs suppressed
        plotted_joint_indices = []
        suppressed_joint_indices = []
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate unique colors for each joint using a colormap
        try:
            # Try new matplotlib API (3.5+)
            cmap = plt.colormaps['tab20']  # 20 distinct colors
            if min_joints > 20:
                # Use a continuous colormap if we have more than 20 joints
                cmap = plt.colormaps['hsv']
        except AttributeError:
            # Fallback to old API
            cmap = plt.cm.get_cmap('tab20')  # 20 distinct colors
            if min_joints > 20:
                # Use a continuous colormap if we have more than 20 joints
                cmap = plt.cm.get_cmap('hsv')
        
        # Plot GT keypoints and predicted joints with connecting lines (only valid joints)
        for j in range(min_joints):
            if not gt_valid[j]:
                # Skip filtered keypoints
                suppressed_joint_indices.append(j)
                continue
            
            plotted_joint_indices.append(j)
            color = cmap(j / max(min_joints - 1, 1))  # Normalize to [0, 1]
            
            # GT keypoint (circle)
            ax.scatter(gt_kp3d[j, 0], gt_kp3d[j, 1], gt_kp3d[j, 2],
                      c=[color], marker='o', s=100, alpha=0.8)
            
            # Predicted joint (cross)
            ax.scatter(pred_joints_3d[j, 0], pred_joints_3d[j, 1], pred_joints_3d[j, 2],
                      c=[color], marker='x', s=150, linewidths=3, alpha=0.8)
            
            # Line connecting GT to predicted
            ax.plot([gt_kp3d[j, 0], pred_joints_3d[j, 0]],
                   [gt_kp3d[j, 1], pred_joints_3d[j, 1]],
                   [gt_kp3d[j, 2], pred_joints_3d[j, 2]],
                   color=color, linewidth=1.5, alpha=0.6, linestyle='--')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Sample {sample_idx:03d} - Epoch {epoch:03d}\n'
                    f'GT 3D Keypoints (circles) vs Predicted 3D Joints (crosses)')
        
        # Set equal aspect ratio for better visualization
        # Compute bounds using only valid (plotted) points
        if len(plotted_joint_indices) > 0:
            valid_gt = gt_kp3d[plotted_joint_indices]
            valid_pred = pred_joints_3d[plotted_joint_indices]
            all_points = np.concatenate([valid_gt, valid_pred], axis=0)
            
            if len(all_points) > 0:
                x_range = all_points[:, 0].max() - all_points[:, 0].min()
                y_range = all_points[:, 1].max() - all_points[:, 1].min()
                z_range = all_points[:, 2].max() - all_points[:, 2].min()
                max_range = max(x_range, y_range, z_range) if max(x_range, y_range, z_range) > 0 else 1.0
                
                x_center = all_points[:, 0].mean()
                y_center = all_points[:, 1].mean()
                z_center = all_points[:, 2].mean()
                
                ax.set_xlim([x_center - max_range/2, x_center + max_range/2])
                ax.set_ylim([y_center - max_range/2, y_center + max_range/2])
                ax.set_zlim([z_center - max_range/2, z_center + max_range/2])
        
        # Create comprehensive legend with plotted and suppressed joints
        from matplotlib.lines import Line2D
        legend_elements = []
        
        # Add marker explanations
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                     markersize=10, label='GT keypoints (○)'))
        legend_elements.append(Line2D([0], [0], marker='x', color='gray', markersize=10, 
                                     linewidth=3, label='Predicted joints (×)'))
        legend_elements.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, 
                                     label='Error vectors'))
        
        # Add separator
        legend_elements.append(Line2D([0], [0], linestyle='', label='---'))
        
        # Add plotted joints
        if len(plotted_joint_indices) > 0:
            plotted_names = []
            for j in plotted_joint_indices:
                if joint_names is not None and j < len(joint_names):
                    plotted_names.append(f"{j}: {joint_names[j]}")
                else:
                    plotted_names.append(f"joint_{j}")
            
            # Group joints for readability (max 10 per line)
            if len(plotted_names) <= 10:
                legend_elements.append(Line2D([0], [0], linestyle='', 
                                             label=f'Plotted ({len(plotted_joint_indices)}): {", ".join(plotted_names)}'))
            else:
                # Split into chunks
                for i in range(0, len(plotted_names), 10):
                    chunk = plotted_names[i:i+10]
                    prefix = 'Plotted:' if i == 0 else ''
                    legend_elements.append(Line2D([0], [0], linestyle='', 
                                                 label=f'{prefix} {", ".join(chunk)}'))
        else:
            legend_elements.append(Line2D([0], [0], linestyle='', label='Plotted: None'))
        
        # Add suppressed joints
        if len(suppressed_joint_indices) > 0:
            suppressed_names = []
            for j in suppressed_joint_indices:
                if joint_names is not None and j < len(joint_names):
                    suppressed_names.append(f"{j}: {joint_names[j]}")
                else:
                    suppressed_names.append(f"joint_{j}")
            
            # Group suppressed joints for readability
            if len(suppressed_names) <= 10:
                legend_elements.append(Line2D([0], [0], linestyle='', 
                                             label=f'Suppressed ({len(suppressed_joint_indices)}): {", ".join(suppressed_names)}'))
            else:
                # Split into chunks
                for i in range(0, len(suppressed_names), 10):
                    chunk = suppressed_names[i:i+10]
                    prefix = 'Suppressed:' if i == 0 else ''
                    legend_elements.append(Line2D([0], [0], linestyle='', 
                                                 label=f'{prefix} {", ".join(chunk)}'))
        else:
            legend_elements.append(Line2D([0], [0], linestyle='', label='Suppressed: None'))
        
        # Add legend
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8, 
                 bbox_to_anchor=(1.05, 1), framealpha=0.9)
        
        # Add text with error statistics (only for valid joints)
        if len(plotted_joint_indices) > 0:
            valid_gt = gt_kp3d[plotted_joint_indices]
            valid_pred = pred_joints_3d[plotted_joint_indices]
            errors = np.linalg.norm(valid_gt - valid_pred, axis=1)
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            textstr = f'Mean error: {mean_error:.4f}\nMax error: {max_error:.4f}\nValid joints: {len(plotted_joint_indices)}/{min_joints}'
        else:
            textstr = 'No valid joints to plot'
        
        ax.text2D(0.02, 0.98, textstr, transform=ax.transAxes, 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save figure
        output_path = os.path.join(output_dir, f'sample_{sample_idx:03d}_epoch_{epoch:03d}_3d_keypoints.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        if sample_idx == 0:
            print(f"  Generated 3D keypoint visualization: {output_path}")
            if len(plotted_joint_indices) > 0:
                valid_gt = gt_kp3d[plotted_joint_indices]
                valid_pred = pred_joints_3d[plotted_joint_indices]
                errors = np.linalg.norm(valid_gt - valid_pred, axis=1)
                mean_error = np.mean(errors)
                max_error = np.max(errors)
                print(f"  Mean 3D error: {mean_error:.4f}, Max error: {max_error:.4f}")
                print(f"  Plotted joints: {len(plotted_joint_indices)}, Suppressed: {len(suppressed_joint_indices)}")
            else:
                print(f"  No valid joints to plot (all filtered)")
        
    except Exception as e:
        print(f"Warning: Failed to create 3D keypoint visualization for sample {sample_idx}: {e}")
        import traceback
        traceback.print_exc()


def save_checkpoint(model, optimizer, scheduler, epoch, config, metrics, filepath):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': config,
        'metrics': metrics,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cuda'):
    """Load training checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def main(config: dict):
    """
    Main training function for multi-view SMIL regressor.
    
    Uses TrainingConfig for:
    - Loss curriculum (base_weights + curriculum_stages)
    - Learning rate curriculum (lr_stages)
    - Model configuration (backbone, head_type, etc.)
    
    Args:
        config: Training configuration dictionary (from MultiViewTrainingConfig)
    """
    # Extract distributed training config
    is_distributed = config.get('is_distributed', False)
    rank = config.get('rank', 0)
    world_size = config.get('world_size', 1)
    device_override = config.get('device_override', None)
    
    # Set random seeds
    set_random_seeds(config['seed'])
    
    # Set device
    if device_override:
        device = device_override
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("MULTI-VIEW SMIL IMAGE REGRESSOR TRAINING")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Distributed: {is_distributed} (world_size={world_size})")
        print(f"Dataset: {config['dataset_path']}")
        print(f"Backbone: {config['backbone_name']}")
        print(f"Num views to use: {config.get('num_views_to_use', 'all')}")
        print(f"Cross-attention layers: {config['cross_attention_layers']}")
        print(f"{'='*60}\n")
    
    # Create output directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['visualizations_dir'], exist_ok=True)
    
    # Load dataset
    if rank == 0:
        print("Loading multi-view dataset...")
    
    dataset = SLEAPMultiViewDataset(
        hdf5_path=config['dataset_path'],
        rotation_representation=config['rotation_representation'],
        num_views_to_use=config.get('num_views_to_use'),
        random_view_sampling=True
    )
    
    if rank == 0:
        dataset.print_dataset_summary()
    
    # Get canonical camera order from dataset
    canonical_camera_order = dataset.get_canonical_camera_order()
    max_views = dataset.get_max_views_in_dataset()
    
    if rank == 0:
        print(f"Max views in dataset: {max_views}")
        print(f"Canonical camera order: {canonical_camera_order}")
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(total_size * config['train_ratio'])
    val_size = int(total_size * config['val_ratio'])
    test_size = total_size - train_size - val_size
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    
    if rank == 0:
        print(f"\nDataset split:")
        print(f"  Train: {len(train_set)}")
        print(f"  Val: {len(val_set)}")
        print(f"  Test: {len(test_set)}")
    
    # Create data loaders
    if is_distributed:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        val_sampler = DistributedSampler(val_set, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        collate_fn=multiview_collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        collate_fn=multiview_collate_fn
    )
    
    # Create model
    if rank == 0:
        print("\nCreating multi-view model...")
    
    # Determine input resolution based on backbone
    # This ensures the renderer is initialized with the correct size
    backbone_name = config['backbone_name']
    if backbone_name.startswith('vit'):
        input_resolution = 224  # ViT uses 224x224
    else:
        input_resolution = 512  # ResNet typically uses 512x512
    
    if rank == 0:
        print(f"Using input resolution: {input_resolution}x{input_resolution} (based on backbone: {backbone_name})")
    
    # Get mesh scaling config
    allow_mesh_scaling = config.get('allow_mesh_scaling', False)
    mesh_scale_init = config.get('mesh_scale_init', 1.0)
    
    if rank == 0 and allow_mesh_scaling:
        print(f"Mesh scaling enabled with init={mesh_scale_init}")
    
    model = create_multiview_regressor(
        device=device,
        batch_size=config['batch_size'],
        shape_family=config.get('shape_family', -1),
        use_unity_prior=config.get('use_unity_prior', False),
        max_views=max_views,
        canonical_camera_order=canonical_camera_order,
        cross_attention_layers=config['cross_attention_layers'],
        cross_attention_heads=config['cross_attention_heads'],
        cross_attention_dropout=config['cross_attention_dropout'],
        backbone_name=backbone_name,
        freeze_backbone=config['freeze_backbone'],
        head_type=config['head_type'],
        hidden_dim=config['hidden_dim'],
        rotation_representation=config['rotation_representation'],
        scale_trans_mode=config['scale_trans_mode'],
        use_ue_scaling=config.get('use_ue_scaling', False),
        input_resolution=input_resolution,
        allow_mesh_scaling=allow_mesh_scaling,
        mesh_scale_init=mesh_scale_init
    )
    
    model = model.to(device)
    
    # Wrap in DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[int(device.split(':')[1])])
    
    # Count parameters
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config['learning_rate'] * 0.01
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.get('use_mixed_precision', False) else None
    
    # Load checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config.get('resume_checkpoint'):
        if os.path.exists(config['resume_checkpoint']):
            if rank == 0:
                print(f"Resuming from checkpoint: {config['resume_checkpoint']}")
            start_epoch, metrics = load_checkpoint(
                config['resume_checkpoint'], model, optimizer, scheduler, device
            )
            best_val_loss = metrics.get('best_val_loss', float('inf'))
    
    # Training loop
    if rank == 0:
        print(f"\nStarting training for {config['num_epochs']} epochs...")
        print(f"Base loss weights: {config['loss_weights']}")
        print(f"Using TrainingConfig loss curriculum: {len(TrainingConfig.LOSS_CURRICULUM['curriculum_stages'])} stages")
        print(f"Using TrainingConfig LR curriculum: {len(TrainingConfig.LEARNING_RATE_CURRICULUM['lr_stages'])} stages\n")
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'loss_components': [],
        'val_loss_components': [],
        'learning_rates': []
    }
    
    for epoch in range(start_epoch, config['num_epochs']):
        # Update sampler for distributed training
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        # Update learning rate based on curriculum from TrainingConfig
        curriculum_lr = MultiViewTrainingConfig.get_learning_rate_for_epoch(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = curriculum_lr
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, config,
            scaler=scaler, is_distributed=is_distributed, rank=rank
        )
        
        training_history['train_loss'].append(train_metrics['avg_loss'])
        training_history['loss_components'].append(train_metrics['loss_components'])
        training_history['learning_rates'].append(curriculum_lr)
        
        # Validate
        if epoch % config['validate_every_n_epochs'] == 0:
            val_metrics = validate(model, val_loader, device, config, epoch=epoch, rank=rank)
            training_history['val_loss'].append(val_metrics['avg_loss'])
            training_history['val_loss_components'].append(val_metrics.get('loss_components', {}))
            
            # Get current epoch loss weights for logging
            current_loss_weights = MultiViewTrainingConfig.get_loss_weights_for_epoch(
                epoch, config.get('loss_weights')
            )
            
            if rank == 0:
                print(f"\nEpoch {epoch} Summary:")
                print(f"  Train Loss: {train_metrics['avg_loss']:.4f}")
                print(f"  Val Loss: {val_metrics['avg_loss']:.4f}")
                _print_component_metrics(
                    train_components=train_metrics.get('loss_components', {}),
                    val_components=val_metrics.get('loss_components', {}),
                    indent="  "
                )
                print(f"  LR (curriculum): {curriculum_lr:.2e}")
                print(f"  Key loss weights: kp2d={current_loss_weights.get('keypoint_2d', 0):.4f}, "
                      f"joint_rot={current_loss_weights.get('joint_rot', 0):.4f}")
            
            # Save best model
            if val_metrics['avg_loss'] < best_val_loss:
                best_val_loss = val_metrics['avg_loss']
                if rank == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, config,
                        {'best_val_loss': best_val_loss, 'val_metrics': val_metrics},
                        os.path.join(config['checkpoint_dir'], 'best_model.pth')
                    )
        
        # Note: We use curriculum LR, so scheduler.step() is commented out
        # scheduler.step()
        
        # Save periodic checkpoint
        if rank == 0 and (epoch + 1) % config['save_every_n_epochs'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, config,
                {'train_loss': train_metrics['avg_loss']},
                os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch:04d}.pth')
            )
        
        # Generate visualizations periodically
        visualize_every = config.get('visualize_every_n_epochs', 10)
        if rank == 0 and visualize_every > 0 and (epoch + 1) % visualize_every == 0:
            # Multi-view grid visualization (keypoint overlay)
            visualize_multiview_training_progress(
                model=model,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                training_config=config,
                output_dir=config['visualizations_dir'],
                num_samples=config.get('num_visualization_samples', 3),
                rank=rank
            )
            
            # Single-view mesh renders (full SMALFitter visualization per view)
            singleview_dir = config.get('singleview_visualizations_dir', 
                                         config['visualizations_dir'].replace('visualizations', 'singleview_renders'))
            visualize_singleview_renders(
                model=model,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                training_config=config,
                output_dir=singleview_dir,
                num_samples=config.get('num_visualization_samples', 3),
                rank=rank
            )
    
    # Save final model
    if rank == 0:
        save_checkpoint(
            model, optimizer, scheduler, config['num_epochs'] - 1, config,
            {'training_history': training_history},
            os.path.join(config['checkpoint_dir'], 'final_model.pth')
        )
        
        # Save training history
        with open(os.path.join(config['checkpoint_dir'], 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {config['checkpoint_dir']}")


def ddp_main(rank, world_size, config, port):
    """DDP wrapper for main training function."""
    if is_torchrun_launched():
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu_rank = local_rank
    else:
        gpu_rank = rank
    
    setup_ddp(rank, world_size, port, local_rank=gpu_rank)
    
    config['device_override'] = f"cuda:{gpu_rank}"
    config['is_distributed'] = True
    config['rank'] = rank
    config['world_size'] = world_size
    
    try:
        main(config)
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Multi-View SMIL Image Regressor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train_multiview_regressor.py --dataset_path multiview_sleap.h5
  
  # With custom configuration
  python train_multiview_regressor.py --dataset_path multiview_sleap.h5 \\
      --batch_size 16 --num_epochs 200 --learning_rate 5e-5
  
  # Distributed training (single node, 4 GPUs)
  torchrun --nproc_per_node=4 train_multiview_regressor.py --dataset_path multiview_sleap.h5
        """
    )
    
    # Required arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to multi-view SLEAP HDF5 dataset")
    
    # Model configuration
    parser.add_argument("--backbone_name", type=str, default='vit_large_patch16_224',
                       help="Backbone network name")
    parser.add_argument("--freeze_backbone", action="store_true",
                       help="Freeze backbone weights")
    parser.add_argument("--head_type", type=str, default='transformer_decoder',
                       choices=['mlp', 'transformer_decoder'],
                       help="Type of regression head")
    parser.add_argument("--hidden_dim", type=int, default=512,
                       help="Hidden dimension for MLP head")
    parser.add_argument("--cross_attention_layers", type=int, default=2,
                       help="Number of cross-attention layers")
    parser.add_argument("--cross_attention_heads", type=int, default=8,
                       help="Number of cross-attention heads")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=200,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0,
                       help="Gradient clipping norm")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Multi-view specific
    parser.add_argument("--num_views_to_use", type=int, default=None,
                       help="Max views to use per sample (None = all)")
    
    # Output configuration
    parser.add_argument("--checkpoint_dir", type=str, default='multiview_checkpoints',
                       help="Checkpoint directory")
    parser.add_argument("--visualizations_dir", type=str, default='multiview_visualizations',
                       help="Visualizations directory")
    parser.add_argument("--save_every_n_epochs", type=int, default=5,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--visualize_every_n_epochs", type=int, default=1,
                       help="Generate visualizations every N epochs (0 to disable)")
    parser.add_argument("--num_visualization_samples", type=int, default=3,
                       help="Number of samples to visualize each time")
    
    # Resume training
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Configuration file
    parser.add_argument("--config", type=str, default=None,
                       help="Path to JSON configuration file")
    
    # Distributed training
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs for distributed training")
    parser.add_argument("--port", type=str, default='12355',
                       help="Port for distributed training")
    
    # Mixed precision
    parser.add_argument("--use_mixed_precision", action="store_true",
                       help="Use mixed precision training")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        training_config = MultiViewTrainingConfig.from_file(args.config)
        # Override with command line args
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                training_config[key] = value
    else:
        training_config = MultiViewTrainingConfig.from_args(args)
    
    # Check for distributed training
    if is_torchrun_launched():
        # Launched via torchrun - use environment variables
        ddp_main(0, 1, training_config, args.port)
    elif args.num_gpus > 1:
        # Multi-GPU single node
        print(f"Launching distributed training with {args.num_gpus} GPUs")
        mp.spawn(
            ddp_main,
            args=(args.num_gpus, training_config, args.port),
            nprocs=args.num_gpus,
            join=True
        )
    else:
        # Single GPU training
        main(training_config)

