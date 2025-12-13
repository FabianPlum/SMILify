"""
Example usage of SMIL Image Regressor with both MLP and Transformer Decoder heads.

This script demonstrates how to use the new transformer decoder head inspired by AniMer,
alongside the traditional MLP regression head.
"""

import torch
import numpy as np
from smil_image_regressor import SMILImageRegressor
from training_config import TrainingConfig

def create_dummy_data(batch_size=2, image_size=224):
    """Create dummy data for testing."""
    # Create dummy RGB images in range [0.0, 1.0] as expected by SMALFitter
    images = torch.rand(batch_size, 3, image_size, image_size)
    
    # Create dummy batch data (placeholder for SMALFitter initialization)
    data_batch = images
    
    return data_batch, images

def test_mlp_head():
    """Test the traditional MLP regression head."""
    print("=" * 60)
    print("Testing MLP Regression Head")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy data
    data_batch, images = create_dummy_data(batch_size=2, image_size=224)
    
    # Initialize model with MLP head
    model = SMILImageRegressor(
        device=device,
        data_batch=data_batch,
        batch_size=2,
        shape_family=-1,
        use_unity_prior=False,
        rgb_only=True,
        freeze_backbone=True,
        hidden_dim=512,
        use_ue_scaling=True,
        rotation_representation='axis_angle',
        input_resolution=224,
        backbone_name='vit_base_patch16_224',
        head_type='mlp'  # Use MLP head
    ).to(device)
    
    print(f"Model created with MLP head")
    print(f"Backbone: {model.backbone_name}")
    print(f"Feature dimension: {model.feature_dim}")
    print(f"Head type: {model.head_type}")
    
    # Test forward pass
    with torch.no_grad():
        params = model(images.to(device))
    
    print(f"Forward pass successful!")
    print(f"Output parameters: {list(params.keys())}")
    for key, value in params.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test trainable parameters
    trainable_params = model.get_trainable_parameters()
    print(f"Trainable parameter groups: {len(trainable_params)}")
    
    return model

def test_transformer_decoder_head():
    """Test the new transformer decoder regression head."""
    print("\n" + "=" * 60)
    print("Testing Transformer Decoder Regression Head")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy data
    data_batch, images = create_dummy_data(batch_size=2, image_size=224)
    
    # Transformer decoder configuration
    transformer_config = {
        'hidden_dim': 1024,
        'depth': 6,
        'heads': 8,
        'dim_head': 64,
        'mlp_dim': 1024,
        'dropout': 0.0,
        'ief_iters': 3,  # Iterative Error Feedback iterations
        'scales_scale_factor': 0.01,  # Scale factor for log_beta_scales
        'trans_scale_factor': 0.01,   # Scale factor for betas_trans
    }
    
    # Initialize model with transformer decoder head
    model = SMILImageRegressor(
        device=device,
        data_batch=data_batch,
        batch_size=2,
        shape_family=-1,
        use_unity_prior=False,
        rgb_only=True,
        freeze_backbone=True,
        hidden_dim=1024,
        use_ue_scaling=True,
        rotation_representation='axis_angle',
        input_resolution=224,
        backbone_name='vit_base_patch16_224',
        head_type='transformer_decoder',  # Use transformer decoder head
        transformer_config=transformer_config
    ).to(device)
    
    print(f"Model created with Transformer Decoder head")
    print(f"Backbone: {model.backbone_name}")
    print(f"Feature dimension: {model.feature_dim}")
    print(f"Head type: {model.head_type}")
    print(f"IEF iterations: {transformer_config['ief_iters']}")
    
    # Test forward pass
    with torch.no_grad():
        params = model(images.to(device))
    
    print(f"Forward pass successful!")
    print(f"Output parameters: {list(params.keys())}")
    for key, value in params.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Check if iteration history is available
    if 'iteration_history' in params:
        print(f"Iteration history available for: {list(params['iteration_history'].keys())}")
        for param_name, history in params['iteration_history'].items():
            print(f"  {param_name}: {len(history)} iterations")
    
    # Test trainable parameters
    trainable_params = model.get_trainable_parameters()
    print(f"Trainable parameter groups: {len(trainable_params)}")
    
    return model

def test_with_training_config():
    """Test using the training configuration system."""
    print("\n" + "=" * 60)
    print("Testing with Training Configuration")
    print("=" * 60)
    
    # Get configuration
    config = TrainingConfig.get_all_config()
    
    print("Configuration Summary:")
    print(f"Backbone: {config['model_config']['backbone_name']}")
    print(f"Head type: {config['model_config']['head_type']}")
    print(f"Feature dimension: {config['model_config']['hidden_dim']}")
    
    if config['model_config']['head_type'] == 'transformer_decoder':
        print("Transformer decoder config:")
        for key, value in config['model_config']['transformer_config'].items():
            print(f"  {key}: {value}")
    
    # Create model using config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_batch, images = create_dummy_data(batch_size=1, image_size=224)
    
    model = SMILImageRegressor(
        device=device,
        data_batch=data_batch,
        batch_size=1,
        shape_family=-1,
        use_unity_prior=False,
        rgb_only=True,
        freeze_backbone=config['model_config']['freeze_backbone'],
        hidden_dim=config['model_config']['hidden_dim'],
        use_ue_scaling=True,
        rotation_representation=config['training_params']['rotation_representation'],
        input_resolution=224,
        backbone_name=config['model_config']['backbone_name'],
        head_type=config['model_config']['head_type'],
        transformer_config=config['model_config'].get('transformer_config', {})
    ).to(device)
    
    print(f"Model created successfully using training config!")
    print(f"Head type: {model.head_type}")
    
    # Test forward pass
    with torch.no_grad():
        params = model(images.to(device))
    
    print(f"Forward pass successful with {len(params)} output parameters")
    
    return model

def compare_heads():
    """Compare MLP and Transformer Decoder heads."""
    print("\n" + "=" * 60)
    print("Comparing MLP vs Transformer Decoder Heads")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_batch, images = create_dummy_data(batch_size=1, image_size=224)
    
    # Test MLP head
    print("Creating MLP model...")
    mlp_model = SMILImageRegressor(
        device=device,
        data_batch=data_batch,
        batch_size=1,
        shape_family=-1,
        use_unity_prior=False,
        rgb_only=True,
        freeze_backbone=True,
        hidden_dim=512,
        use_ue_scaling=True,
        rotation_representation='axis_angle',
        input_resolution=224,
        backbone_name='vit_base_patch16_224',
        head_type='mlp'
    ).to(device)
    
    # Test Transformer Decoder head
    print("Creating Transformer Decoder model...")
    transformer_model = SMILImageRegressor(
        device=device,
        data_batch=data_batch,
        batch_size=1,
        shape_family=-1,
        use_unity_prior=False,
        rgb_only=True,
        freeze_backbone=True,
        hidden_dim=1024,
        use_ue_scaling=True,
        rotation_representation='axis_angle',
        input_resolution=224,
        backbone_name='vit_base_patch16_224',
        head_type='transformer_decoder',
        transformer_config={'ief_iters': 3}
    ).to(device)
    
    # Compare parameter counts
    mlp_params = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)
    transformer_params = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)
    
    print(f"\nParameter Comparison:")
    print(f"MLP head parameters: {mlp_params:,}")
    print(f"Transformer decoder parameters: {transformer_params:,}")
    print(f"Ratio: {transformer_params / mlp_params:.2f}x")
    
    # Test forward passes
    with torch.no_grad():
        mlp_output = mlp_model(images.to(device))
        transformer_output = transformer_model(images.to(device))
    
    print(f"\nOutput Comparison:")
    print(f"MLP output keys: {list(mlp_output.keys())}")
    print(f"Transformer output keys: {list(transformer_output.keys())}")
    
    # Check if transformer has iteration history
    if 'iteration_history' in transformer_output:
        print(f"Transformer decoder provides iteration history with {len(transformer_output['iteration_history'])} parameter types")
    
    return mlp_model, transformer_model

def main():
    """Run all tests."""
    print("SMIL Image Regressor - Head Type Comparison")
    print("=" * 60)
    
    try:
        # Test MLP head
        mlp_model = test_mlp_head()
        
        # Test Transformer Decoder head
        transformer_model = test_transformer_decoder_head()
        
        # Test with training config
        config_model = test_with_training_config()
        
        # Compare heads
        mlp_comp, transformer_comp = compare_heads()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        print("\nKey Features:")
        print("✅ MLP Regression Head: Simple, efficient, proven")
        print("✅ Transformer Decoder Head: Advanced, iterative, spatial attention")
        print("✅ Flexible Configuration: Easy switching between head types")
        print("✅ AniMer-inspired: Cross-attention and IEF mechanisms")
        print("✅ Backward Compatible: Existing MLP head unchanged")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
