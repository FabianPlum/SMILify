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

# Add parent directories to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multiview_smil_regressor import MultiViewSMILImageRegressor, create_multiview_regressor
from sleap_data.sleap_multiview_dataset import SLEAPMultiViewDataset, multiview_collate_fn
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
        
        # Validation/save frequency
        'save_every_n_epochs': 10,
        'validate_every_n_epochs': 1,
        
        # Split ratios
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
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
                    predicted_params, _, auxiliary_data = model.predict_from_multiview_batch(
                        x_data_batch, y_data_batch
                    )
                    
                    if predicted_params is None:
                        continue
                    
                    loss, loss_components = model.compute_multiview_batch_loss(
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
                predicted_params, _, auxiliary_data = model.predict_from_multiview_batch(
                    x_data_batch, y_data_batch
                )
                
                if predicted_params is None:
                    continue
                
                loss, loss_components = model.compute_multiview_batch_loss(
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
                predicted_params, _, auxiliary_data = model.predict_from_multiview_batch(
                    x_data_batch, y_data_batch
                )
                
                if predicted_params is None:
                    continue
                
                loss, loss_components = model.compute_multiview_batch_loss(
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
        backbone_name=config['backbone_name'],
        freeze_backbone=config['freeze_backbone'],
        head_type=config['head_type'],
        hidden_dim=config['hidden_dim'],
        rotation_representation=config['rotation_representation'],
        scale_trans_mode=config['scale_trans_mode'],
        use_ue_scaling=config.get('use_ue_scaling', False)
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
            
            # Get current epoch loss weights for logging
            current_loss_weights = MultiViewTrainingConfig.get_loss_weights_for_epoch(
                epoch, config.get('loss_weights')
            )
            
            if rank == 0:
                print(f"\nEpoch {epoch} Summary:")
                print(f"  Train Loss: {train_metrics['avg_loss']:.4f}")
                print(f"  Val Loss: {val_metrics['avg_loss']:.4f}")
                print(f"  Loss Components: {train_metrics['loss_components']}")
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
    parser.add_argument("--head_type", type=str, default='mlp',
                       choices=['mlp', 'transformer_decoder'],
                       help="Type of regression head")
    parser.add_argument("--hidden_dim", type=int, default=512,
                       help="Hidden dimension for MLP head")
    parser.add_argument("--cross_attention_layers", type=int, default=2,
                       help="Number of cross-attention layers")
    parser.add_argument("--cross_attention_heads", type=int, default=8,
                       help="Number of cross-attention heads")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100,
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
    parser.add_argument("--save_every_n_epochs", type=int, default=10,
                       help="Save checkpoint every N epochs")
    
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

