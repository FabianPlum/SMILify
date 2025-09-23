"""
Test script for validating both ResNet and Vision Transformer backbone implementations.

This script tests the backbone factory, memory usage, and forward pass functionality
for both ResNet and ViT backbones.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Dict, Any

# Add the parent directories to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Change to the project root directory to find SMAL files
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
os.chdir(project_root)

from backbone_factory import BackboneFactory, BackboneInterface
from memory_optimization import MemoryMonitor, get_backbone_memory_requirements

# Import config with error handling for missing SMAL files
try:
    import config
    CONFIG_AVAILABLE = True
except FileNotFoundError as e:
    print(f"Warning: Could not import config due to missing SMAL files: {e}")
    print("Some tests will be skipped. Make sure SMAL model files are available for full testing.")
    CONFIG_AVAILABLE = False
    # Create a mock config for basic testing
    class MockConfig:
        SHAPE_FAMILY = 0
        N_POSE = 20
        N_BETAS = 10
    config = MockConfig()


def test_backbone_creation():
    """Test backbone creation through factory."""
    print("=" * 60)
    print("Testing Backbone Creation")
    print("=" * 60)
    
    available_backbones = BackboneFactory.list_available_backbones()
    print(f"Available backbones: {available_backbones}")
    
    for backbone_name in available_backbones:
        try:
            print(f"\nTesting {backbone_name}:")
            
            # Create backbone
            backbone = BackboneFactory.create_backbone(backbone_name, freeze=True)
            print(f"  ✓ Created successfully")
            print(f"  ✓ Feature dimension: {backbone.get_feature_dim()}")
            print(f"  ✓ Frozen: {backbone.freeze_weights_flag}")
            
            # Test info
            info = BackboneFactory.get_backbone_info(backbone_name)
            print(f"  ✓ Type: {info['type']}")
            print(f"  ✓ Memory usage: {info['memory_usage']}")
            
        except ImportError as e:
            print(f"  ⚠ Skipped {backbone_name}: {e}")
        except Exception as e:
            print(f"  ✗ Failed to create {backbone_name}: {e}")


def test_forward_pass():
    """Test forward pass through different backbones."""
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test different input sizes
    test_sizes = [(1, 3, 224, 224), (2, 3, 224, 224), (1, 3, 512, 512)]
    
    # Test available backbones
    available_backbones = BackboneFactory.list_available_backbones()
    test_backbones = [name for name in ['resnet152', 'vit_base_patch16_224'] if name in available_backbones]
    
    for backbone_name in test_backbones:
        print(f"\nTesting {backbone_name}:")
        
        try:
            backbone = BackboneFactory.create_backbone(backbone_name, freeze=True).to(device)
            
            for batch_size, channels, height, width in test_sizes:
                print(f"  Input shape: ({batch_size}, {channels}, {height}, {width})")
                
                # Create dummy input
                dummy_input = torch.randn(batch_size, channels, height, width).to(device)
                
                # Forward pass
                with torch.no_grad():
                    features = backbone(dummy_input)
                
                expected_shape = (batch_size, backbone.get_feature_dim())
                actual_shape = features.shape
                
                print(f"    Expected output shape: {expected_shape}")
                print(f"    Actual output shape: {actual_shape}")
                
                if actual_shape == expected_shape:
                    print(f"    ✓ Shape correct")
                else:
                    print(f"    ✗ Shape mismatch!")
                
                # Check for NaN or inf
                if torch.isfinite(features).all():
                    print(f"    ✓ Output is finite")
                else:
                    print(f"    ✗ Output contains NaN or inf!")
                
                # Print feature statistics
                print(f"    Feature range: [{features.min():.3f}, {features.max():.3f}]")
                print(f"    Feature mean: {features.mean():.3f}, std: {features.std():.3f}")
                
        except ImportError as e:
            print(f"  ⚠ Skipped {backbone_name}: {e}")
        except Exception as e:
            print(f"  ✗ Failed forward pass for {backbone_name}: {e}")


def test_memory_usage():
    """Test memory usage of different backbones."""
    print("\n" + "=" * 60)
    print("Testing Memory Usage")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory tests")
        return
    
    device = 'cuda'
    monitor = MemoryMonitor(device)
    
    # Clear initial memory
    torch.cuda.empty_cache()
    monitor.print_memory_status("Initial")
    
    # Test available backbones
    available_backbones = BackboneFactory.list_available_backbones()
    test_backbones = [name for name in ['resnet152', 'vit_base_patch16_224'] if name in available_backbones]
    
    for backbone_name in test_backbones:
        print(f"\nTesting {backbone_name}:")
        
        try:
            # Clear memory before test
            torch.cuda.empty_cache()
            baseline_memory = monitor.get_gpu_memory_used()
            
            # Create backbone
            backbone = BackboneFactory.create_backbone(backbone_name, freeze=True).to(device)
            backbone_memory = monitor.get_gpu_memory_used()
            backbone_memory_used = backbone_memory - baseline_memory
            
            print(f"  Backbone memory usage: {backbone_memory_used:.1f}MB")
            
            # Test forward pass memory
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            
            with torch.no_grad():
                features = backbone(dummy_input)
            
            forward_memory = monitor.get_gpu_memory_used()
            forward_memory_used = forward_memory - baseline_memory
            
            print(f"  Forward pass memory usage: {forward_memory_used:.1f}MB")
            
            # Compare with estimated requirements
            estimated = get_backbone_memory_requirements(backbone_name)
            print(f"  Estimated backbone memory: {estimated['backbone']:.1f}GB")
            print(f"  Estimated total memory: {estimated['total']:.1f}GB")
            
            # Clean up
            del backbone, dummy_input, features
            torch.cuda.empty_cache()
            
        except ImportError as e:
            print(f"  ⚠ Skipped {backbone_name}: {e}")
        except Exception as e:
            print(f"  ✗ Memory test failed for {backbone_name}: {e}")


def test_smil_integration():
    """Test integration with SMILImageRegressor."""
    print("\n" + "=" * 60)
    print("Testing SMIL Integration")
    print("=" * 60)
    
    if not CONFIG_AVAILABLE:
        print("Skipping SMIL integration test - config not available")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create placeholder data
    placeholder_data = torch.zeros((1, 3, 512, 512))
    
    # Test available backbones
    available_backbones = BackboneFactory.list_available_backbones()
    test_backbones = [name for name in ['resnet152', 'vit_base_patch16_224'] if name in available_backbones]
    
    for backbone_name in test_backbones:
        print(f"\nTesting SMIL integration with {backbone_name}:")
        
        try:
            from smil_image_regressor import SMILImageRegressor
            
            # Initialize model
            model = SMILImageRegressor(
                device=device,
                data_batch=placeholder_data,
                batch_size=1,
                shape_family=config.SHAPE_FAMILY,
                use_unity_prior=False,
                rgb_only=True,
                freeze_backbone=True,
                hidden_dim=512,
                use_ue_scaling=True,
                rotation_representation='axis_angle',
                input_resolution=224 if backbone_name.startswith('vit') else 512,
                backbone_name=backbone_name
            ).to(device)
            
            print(f"  ✓ Model created successfully")
            print(f"  ✓ Backbone: {model.backbone_name}")
            print(f"  ✓ Feature dimension: {model.feature_dim}")
            
            # Test forward pass - use appropriate input size for backbone
            if backbone_name.startswith('vit'):
                input_size = 224  # ViT expects 224x224
            else:
                input_size = 512  # ResNet can handle 512x512
            dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
            
            with torch.no_grad():
                params = model(dummy_input)
            
            print(f"  ✓ Forward pass successful")
            print(f"  ✓ Output parameters: {list(params.keys())}")
            
            # Check parameter shapes
            expected_params = ['global_rot', 'joint_rot', 'betas', 'trans', 'fov', 'cam_rot', 'cam_trans']
            for param_name in expected_params:
                if param_name in params:
                    print(f"    {param_name}: {params[param_name].shape}")
                else:
                    print(f"    ✗ Missing parameter: {param_name}")
            
            # Test trainable parameters
            trainable_params = model.get_trainable_parameters()
            print(f"  ✓ Trainable parameter groups: {len(trainable_params)}")
            
            # Clean up
            del model, dummy_input, params
            if device == 'cuda':
                torch.cuda.empty_cache()
            
        except ImportError as e:
            print(f"  ⚠ Skipped {backbone_name}: {e}")
        except Exception as e:
            print(f"  ✗ SMIL integration failed for {backbone_name}: {e}")
            import traceback
            traceback.print_exc()


def test_training_compatibility():
    """Test compatibility with training pipeline."""
    print("\n" + "=" * 60)
    print("Testing Training Compatibility")
    print("=" * 60)
    
    if not CONFIG_AVAILABLE:
        print("Skipping training compatibility test - config not available")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test available backbones
    available_backbones = BackboneFactory.list_available_backbones()
    test_backbones = [name for name in ['resnet152', 'vit_base_patch16_224'] if name in available_backbones]
    
    for backbone_name in test_backbones:
        print(f"\nTesting training compatibility with {backbone_name}:")
        
        try:
            from smil_image_regressor import SMILImageRegressor
            from memory_optimization import recommend_training_config
            
            # Get training recommendations
            training_config = recommend_training_config(backbone_name, 24.0)
            print(f"  Training config: {training_config}")
            
            # Create model
            placeholder_data = torch.zeros((1, 3, 512, 512))
            model = SMILImageRegressor(
                device=device,
                data_batch=placeholder_data,
                batch_size=1,
                shape_family=config.SHAPE_FAMILY,
                use_unity_prior=False,
                rgb_only=True,
                freeze_backbone=True,
                hidden_dim=training_config['hidden_dim'],
                use_ue_scaling=True,
                rotation_representation='axis_angle',
                input_resolution=224 if backbone_name.startswith('vit') else 512,
                backbone_name=backbone_name
            ).to(device)
            
            # Test optimizer creation
            trainable_params = model.get_trainable_parameters()
            optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
            print(f"  ✓ Optimizer created with {len(trainable_params)} parameter groups")
            
            # Test loss computation (dummy) - use appropriate input size for backbone
            if backbone_name.startswith('vit'):
                input_size = 224  # ViT expects 224x224
            else:
                input_size = 512  # ResNet can handle 512x512
            dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
            dummy_target = {
                'global_rot': torch.randn(1, 3).to(device),
                'joint_rot': torch.randn(1, config.N_POSE, 3).to(device),
                'betas': torch.randn(1, config.N_BETAS).to(device),
                'trans': torch.randn(1, 3).to(device),
                'fov': torch.randn(1, 1).to(device),
                'cam_rot': torch.randn(1, 3, 3).to(device),
                'cam_trans': torch.randn(1, 3).to(device),
            }
            
            # Forward pass
            predicted_params = model(dummy_input)
            
            # Test loss computation
            loss, loss_components = model.compute_prediction_loss(
                predicted_params, dummy_target, return_components=True
            )
            
            print(f"  ✓ Loss computation successful")
            print(f"  ✓ Total loss: {loss.item():.6f}")
            print(f"  ✓ Loss components: {list(loss_components.keys())}")
            
            # Test backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"  ✓ Backward pass successful")
            
            # Clean up
            del model, optimizer, dummy_input, dummy_target, predicted_params, loss
            if device == 'cuda':
                torch.cuda.empty_cache()
            
        except ImportError as e:
            print(f"  ⚠ Skipped {backbone_name}: {e}")
        except Exception as e:
            print(f"  ✗ Training compatibility test failed for {backbone_name}: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Run all tests."""
    print("SMIL Backbone Implementation Tests")
    print("=" * 60)
    
    # Run tests
    test_backbone_creation()
    test_forward_pass()
    test_memory_usage()
    test_smil_integration()
    test_training_compatibility()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
