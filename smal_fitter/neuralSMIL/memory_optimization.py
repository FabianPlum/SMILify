"""
Memory Optimization Utilities for SMIL Training

VIBE CODED, USE WITH CAUTION!

This module provides utilities to optimize memory usage for training on 24GB GPUs,
including mixed precision training, gradient checkpointing, and memory monitoring.
"""

import torch
import torch.nn as nn
import torch.cuda as cuda
from typing import Dict, Any, Optional
import gc
import psutil
import os


class MemoryMonitor:
    """Monitor GPU and system memory usage during training."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.initial_gpu_memory = self.get_gpu_memory_used() if device == 'cuda' else 0
        self.initial_system_memory = self.get_system_memory_used()
    
    def get_gpu_memory_used(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def get_gpu_memory_total(self) -> float:
        """Get total GPU memory in MB."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        return 0.0
    
    def get_system_memory_used(self) -> float:
        """Get current system memory usage in MB."""
        return psutil.virtual_memory().used / 1024 / 1024
    
    def get_system_memory_total(self) -> float:
        """Get total system memory in MB."""
        return psutil.virtual_memory().total / 1024 / 1024
    
    def print_memory_status(self, stage: str = ""):
        """Print current memory status."""
        gpu_used = self.get_gpu_memory_used()
        gpu_total = self.get_gpu_memory_total()
        gpu_percent = (gpu_used / gpu_total) * 100 if gpu_total > 0 else 0
        
        sys_used = self.get_system_memory_used()
        sys_total = self.get_system_memory_total()
        sys_percent = (sys_used / sys_total) * 100 if sys_total > 0 else 0
        
        print(f"Memory Status {stage}:")
        print(f"  GPU: {gpu_used:.1f}MB / {gpu_total:.1f}MB ({gpu_percent:.1f}%)")
        print(f"  System: {sys_used:.1f}MB / {sys_total:.1f}MB ({sys_percent:.1f}%)")
    
    def check_memory_limit(self, limit_gb: float = 20.0) -> bool:
        """Check if memory usage is within limits."""
        gpu_used_gb = self.get_gpu_memory_used() / 1024
        return gpu_used_gb < limit_gb


class MixedPrecisionTrainer:
    """Mixed precision training wrapper for memory optimization."""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 device: str = 'cuda', scaler_init_scale: float = 65536.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(init_scale=scaler_init_scale)
        self.autocast = torch.cuda.amp.autocast if device == 'cuda' else torch.autocast
    
    def train_step(self, loss_fn, *args, **kwargs):
        """Perform a training step with mixed precision."""
        self.optimizer.zero_grad()
        
        with self.autocast():
            loss = loss_fn(*args, **kwargs)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def update_scaler(self):
        """Update the scaler (call after each epoch)."""
        self.scaler.update()


class GradientCheckpointing:
    """Gradient checkpointing utilities for memory optimization."""
    
    @staticmethod
    def enable_checkpointing(model: nn.Module, backbone_name: str):
        """Enable gradient checkpointing for the model."""
        if backbone_name.startswith('vit'):
            # Enable gradient checkpointing for ViT
            if hasattr(model.backbone.backbone, 'blocks'):
                for block in model.backbone.backbone.blocks:
                    if hasattr(block, 'gradient_checkpointing_enable'):
                        block.gradient_checkpointing_enable()
            print(f"Enabled gradient checkpointing for {backbone_name}")
        elif backbone_name.startswith('resnet'):
            # ResNet doesn't typically need gradient checkpointing
            print(f"Gradient checkpointing not needed for {backbone_name}")
    
    @staticmethod
    def disable_checkpointing(model: nn.Module, backbone_name: str):
        """Disable gradient checkpointing for the model."""
        if backbone_name.startswith('vit'):
            if hasattr(model.backbone.backbone, 'blocks'):
                for block in model.backbone.backbone.blocks:
                    if hasattr(block, 'gradient_checkpointing_disable'):
                        block.gradient_checkpointing_disable()
            print(f"Disabled gradient checkpointing for {backbone_name}")


class MemoryOptimizer:
    """Main memory optimization class."""
    
    def __init__(self, device: str = 'cuda', target_memory_gb: float = 20.0):
        self.device = device
        self.target_memory_gb = target_memory_gb
        self.monitor = MemoryMonitor(device)
        self.optimization_enabled = False
    
    def optimize_model_for_memory(self, model: nn.Module, backbone_name: str, 
                                 batch_size: int, use_mixed_precision: bool = True,
                                 use_gradient_checkpointing: bool = True) -> Dict[str, Any]:
        """
        Optimize model for memory usage.
        
        Args:
            model: The model to optimize
            backbone_name: Name of the backbone
            batch_size: Current batch size
            use_mixed_precision: Whether to use mixed precision
            use_gradient_checkpointing: Whether to use gradient checkpointing
            
        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {
            'batch_size': batch_size,
            'use_mixed_precision': use_mixed_precision,
            'use_gradient_checkpointing': use_gradient_checkpointing,
            'memory_optimizations': []
        }
        
        # Check current memory usage
        current_memory_gb = self.monitor.get_gpu_memory_used() / 1024
        
        if current_memory_gb > self.target_memory_gb:
            recommendations['memory_optimizations'].append(f"Current memory usage ({current_memory_gb:.1f}GB) exceeds target ({self.target_memory_gb}GB)")
            
            # Recommend batch size reduction
            if batch_size > 1:
                recommended_batch_size = max(1, batch_size // 2)
                recommendations['batch_size'] = recommended_batch_size
                recommendations['memory_optimizations'].append(f"Reduce batch size to {recommended_batch_size}")
            
            # Enable gradient checkpointing for ViT
            if backbone_name.startswith('vit') and use_gradient_checkpointing:
                GradientCheckpointing.enable_checkpointing(model, backbone_name)
                recommendations['memory_optimizations'].append("Enabled gradient checkpointing for ViT")
        
        # Enable mixed precision if not already enabled
        if use_mixed_precision:
            recommendations['memory_optimizations'].append("Use mixed precision training")
        
        # Memory cleanup recommendations
        recommendations['memory_optimizations'].extend([
            "Clear GPU cache between epochs",
            "Use torch.cuda.empty_cache() after validation",
            "Consider reducing hidden_dim for regression head"
        ])
        
        return recommendations
    
    def cleanup_memory(self):
        """Clean up GPU memory."""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    def get_optimal_batch_size(self, model: nn.Module, input_shape: tuple, 
                              max_batch_size: int = 32) -> int:
        """
        Find optimal batch size for the model.
        
        Args:
            model: The model to test
            input_shape: Input tensor shape (C, H, W)
            max_batch_size: Maximum batch size to test
            
        Returns:
            Optimal batch size
        """
        model.eval()
        optimal_batch_size = 1
        
        for batch_size in range(1, max_batch_size + 1):
            try:
                # Create dummy input
                dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
                
                # Clear cache before test
                self.cleanup_memory()
                
                # Test forward pass
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # Check memory usage
                memory_gb = self.monitor.get_gpu_memory_used() / 1024
                
                if memory_gb < self.target_memory_gb:
                    optimal_batch_size = batch_size
                else:
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise e
        
        # Clean up
        self.cleanup_memory()
        model.train()
        
        return optimal_batch_size


def get_backbone_memory_requirements(backbone_name: str) -> Dict[str, float]:
    """
    Get estimated memory requirements for different backbones.
    
    Args:
        backbone_name: Name of the backbone
        
    Returns:
        Dictionary with memory requirements in GB
    """
    memory_requirements = {
        'resnet50': {'backbone': 1.5, 'total': 3.0},
        'resnet101': {'backbone': 2.0, 'total': 4.0},
        'resnet152': {'backbone': 2.5, 'total': 5.0},
        'vit_base_patch16_224': {'backbone': 3.0, 'total': 6.0},
        'vit_large_patch16_224': {'backbone': 5.0, 'total': 10.0},
    }
    
    return memory_requirements.get(backbone_name, {'backbone': 2.0, 'total': 4.0})


def recommend_training_config(backbone_name: str, gpu_memory_gb: float = 24.0) -> Dict[str, Any]:
    """
    Recommend training configuration based on backbone and GPU memory.
    
    Args:
        backbone_name: Name of the backbone
        gpu_memory_gb: Available GPU memory in GB
        
    Returns:
        Dictionary with recommended configuration
    """
    memory_req = get_backbone_memory_requirements(backbone_name)
    
    config = {
        'backbone_name': backbone_name,
        'freeze_backbone': True,  # Always freeze for memory efficiency
        'use_mixed_precision': True,
        'use_gradient_checkpointing': backbone_name.startswith('vit'),
        'target_memory_gb': min(gpu_memory_gb * 0.8, 20.0),  # Use 80% of available memory
    }
    
    # Estimate batch size based on memory requirements
    available_memory = gpu_memory_gb - memory_req['total']
    if available_memory > 10:
        config['estimated_batch_size'] = 16
    elif available_memory > 5:
        config['estimated_batch_size'] = 8
    else:
        config['estimated_batch_size'] = 4
    
    # Adjust hidden_dim based on backbone
    if backbone_name.startswith('vit'):
        if 'base' in backbone_name:
            config['hidden_dim'] = 768
        elif 'large' in backbone_name:
            config['hidden_dim'] = 1024
    else:
        config['hidden_dim'] = 2048
    
    # Memory optimization recommendations
    config['memory_optimizations'] = []
    if memory_req['total'] > gpu_memory_gb * 0.6:
        config['memory_optimizations'].append("Consider using a smaller backbone")
        config['memory_optimizations'].append("Enable gradient checkpointing")
    
    if backbone_name.startswith('vit'):
        config['memory_optimizations'].append("Use mixed precision training")
        config['memory_optimizations'].append("Consider reducing batch size for ViT")
    
    return config


if __name__ == "__main__":
    # Test memory monitoring
    monitor = MemoryMonitor()
    monitor.print_memory_status("Initial")
    
    # Test recommendations
    for backbone in ['resnet152', 'vit_base_patch16_224', 'vit_large_patch16_224']:
        config = recommend_training_config(backbone, 24.0)
        print(f"\nRecommendations for {backbone}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
