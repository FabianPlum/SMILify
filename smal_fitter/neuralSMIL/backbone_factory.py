"""
Backbone Factory for SMIL Image Regressor

This module provides a factory pattern for creating different backbone networks
(ResNet, Vision Transformer) for the SMIL parameter regression task.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

# Import timm with error handling
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    print("Warning: timm library not available. Vision Transformer backbones will not work.")
    print("Install with: pip install timm")
    TIMM_AVAILABLE = False
    timm = None


class BackboneInterface(ABC):
    """Abstract interface for backbone networks."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Get the output feature dimension."""
        pass
    
    @abstractmethod
    def freeze_weights(self) -> None:
        """Freeze all backbone parameters."""
        pass
    
    @abstractmethod
    def unfreeze_weights(self) -> None:
        """Unfreeze all backbone parameters."""
        pass
    
    @abstractmethod
    def get_trainable_parameters(self) -> list:
        """Get list of trainable parameter groups."""
        pass
    
    def to(self, device):
        """Move backbone to device."""
        self.backbone = self.backbone.to(device)
        return self
    
    def parameters(self):
        """Get backbone parameters."""
        return self.backbone.parameters()
    
    def eval(self):
        """Set backbone to evaluation mode."""
        self.backbone.eval()
        return self
    
    def train(self, mode=True):
        """Set backbone to training mode."""
        self.backbone.train(mode)
        return self
    
    def __call__(self, x):
        """Make backbone callable."""
        return self.forward(x)


class ResNetBackbone(BackboneInterface):
    """ResNet backbone implementation."""
    
    def __init__(self, model_name: str = 'resnet152', pretrained: bool = True, freeze: bool = True):
        """
        Initialize ResNet backbone.
        
        Args:
            model_name: ResNet variant ('resnet50', 'resnet101', 'resnet152')
            pretrained: Whether to use pretrained weights
            freeze: Whether to freeze backbone weights
        """
        super().__init__()
        self.model_name = model_name
        self.freeze_weights_flag = freeze
        
        # Load pretrained ResNet model
        if model_name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.feature_dim = 2048
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
            self.feature_dim = 2048
        elif model_name == 'resnet152':
            self.backbone = models.resnet152(weights=models.ResNet152_Weights.DEFAULT if pretrained else None)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        if freeze:
            self.freeze_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet backbone."""
        features = self.backbone(x)  # (batch_size, 2048, 1, 1)
        return features.view(x.size(0), -1)  # (batch_size, 2048)
    
    def get_feature_dim(self) -> int:
        """Get ResNet feature dimension."""
        return self.feature_dim
    
    def freeze_weights(self) -> None:
        """Freeze ResNet parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.freeze_weights_flag = True
    
    def unfreeze_weights(self) -> None:
        """Unfreeze ResNet parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_weights_flag = False
    
    def get_trainable_parameters(self) -> list:
        """Get trainable ResNet parameters."""
        if self.freeze_weights_flag:
            return []
        return list(self.backbone.parameters())


class ViTBackbone(BackboneInterface):
    """Vision Transformer backbone implementation."""
    
    def __init__(self, model_name: str = 'vit_base_patch16_224', pretrained: bool = True, freeze: bool = True):
        """
        Initialize ViT backbone.
        
        Args:
            model_name: ViT variant ('vit_base_patch16_224', 'vit_large_patch16_224')
            pretrained: Whether to use pretrained weights
            freeze: Whether to freeze backbone weights
        """
        super().__init__()
        self.model_name = model_name
        self.freeze_weights_flag = freeze
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required for Vision Transformer backbones. Install with: pip install timm")
        
        # Load pretrained ViT model from timm
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='',  # Don't apply global pooling
        )
        
        # Get feature dimension based on model
        if 'base' in model_name:
            self.feature_dim = 768
        elif 'large' in model_name:
            self.feature_dim = 1024
        else:
            raise ValueError(f"Unsupported ViT model: {model_name}")
        
        if freeze:
            self.freeze_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ViT backbone."""
        # ViT outputs (batch_size, num_patches + 1, feature_dim) where +1 is for CLS token
        features = self.backbone.forward_features(x)  # (batch_size, 197, 768) for 224x224 input
        
        # Use CLS token (first token) as global representation
        cls_token = features[:, 0]  # (batch_size, 768, 1024 for large)
        
        return cls_token
    
    def forward_with_spatial(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ViT backbone returning both global and spatial features.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Tuple of (global_features, spatial_features)
            - global_features: CLS token (batch_size, feature_dim)
            - spatial_features: Patch tokens (batch_size, num_patches, feature_dim)
        """
        # ViT outputs (batch_size, num_patches + 1, feature_dim) where +1 is for CLS token
        features = self.backbone.forward_features(x)  # (batch_size, 197, 768) for 224x224 input
        
        # Split CLS token and patch tokens
        cls_token = features[:, 0]  # (batch_size, feature_dim)
        patch_tokens = features[:, 1:]  # (batch_size, num_patches, feature_dim)
        
        return cls_token, patch_tokens
    
    def get_feature_dim(self) -> int:
        """Get ViT feature dimension."""
        return self.feature_dim
    
    def freeze_weights(self) -> None:
        """Freeze ViT parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.freeze_weights_flag = True
    
    def unfreeze_weights(self) -> None:
        """Unfreeze ViT parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_weights_flag = False
    
    def get_trainable_parameters(self) -> list:
        """Get trainable ViT parameters."""
        if self.freeze_weights_flag:
            return []
        return list(self.backbone.parameters())


class BackboneFactory:
    """Factory class for creating backbone networks."""
    
    SUPPORTED_BACKBONES = {
        'resnet50': ResNetBackbone,
        'resnet101': ResNetBackbone,
        'resnet152': ResNetBackbone,
        'vit_base_patch16_224': ViTBackbone,
        'vit_large_patch16_224': ViTBackbone,
    }
    
    @classmethod
    def create_backbone(cls, backbone_name: str, **kwargs) -> BackboneInterface:
        """
        Create a backbone network.
        
        Args:
            backbone_name: Name of the backbone to create
            **kwargs: Additional arguments for backbone initialization
            
        Returns:
            BackboneInterface instance
        """
        if backbone_name not in cls.SUPPORTED_BACKBONES:
            available = list(cls.SUPPORTED_BACKBONES.keys())
            raise ValueError(f"Unsupported backbone '{backbone_name}'. Available: {available}")
        
        # Check if ViT is requested but timm is not available
        if backbone_name.startswith('vit') and not TIMM_AVAILABLE:
            raise ImportError(f"Vision Transformer backbone '{backbone_name}' requires timm library. Install with: pip install timm")
        
        backbone_class = cls.SUPPORTED_BACKBONES[backbone_name]
        
        # Set default arguments based on backbone type
        if backbone_name.startswith('resnet'):
            kwargs.setdefault('model_name', backbone_name)
        elif backbone_name.startswith('vit'):
            kwargs.setdefault('model_name', backbone_name)
        
        return backbone_class(**kwargs)
    
    @classmethod
    def get_backbone_info(cls, backbone_name: str) -> Dict[str, Any]:
        """
        Get information about a backbone.
        
        Args:
            backbone_name: Name of the backbone
            
        Returns:
            Dictionary with backbone information
        """
        if backbone_name not in cls.SUPPORTED_BACKBONES:
            available = list(cls.SUPPORTED_BACKBONES.keys())
            raise ValueError(f"Unsupported backbone '{backbone_name}'. Available: {available}")
        
        info = {
            'name': backbone_name,
            'class': cls.SUPPORTED_BACKBONES[backbone_name].__name__,
        }
        
        # Add specific information based on backbone type
        if backbone_name.startswith('resnet'):
            info.update({
                'type': 'CNN',
                'feature_dim': 2048,
                'input_size': 224,
                'memory_usage': 'Medium',
            })
        elif backbone_name.startswith('vit'):
            if 'base' in backbone_name:
                info.update({
                    'type': 'Transformer',
                    'feature_dim': 768,
                    'input_size': 224,
                    'memory_usage': 'Medium',
                })
            elif 'large' in backbone_name:
                info.update({
                    'type': 'Transformer',
                    'feature_dim': 1024,
                    'input_size': 224,
                    'memory_usage': 'High',
                })
        
        return info
    
    @classmethod
    def list_available_backbones(cls) -> list:
        """List all available backbone names."""
        return list(cls.SUPPORTED_BACKBONES.keys())


# Convenience functions for backward compatibility
def create_resnet_backbone(model_name: str = 'resnet152', pretrained: bool = True, freeze: bool = True) -> ResNetBackbone:
    """Create a ResNet backbone."""
    return BackboneFactory.create_backbone(model_name, pretrained=pretrained, freeze=freeze)


def create_vit_backbone(model_name: str = 'vit_base_patch16_224', pretrained: bool = True, freeze: bool = True) -> ViTBackbone:
    """Create a ViT backbone."""
    return BackboneFactory.create_backbone(model_name, pretrained=pretrained, freeze=freeze)


if __name__ == "__main__":
    # Test the factory
    print("Available backbones:")
    for backbone_name in BackboneFactory.list_available_backbones():
        info = BackboneFactory.get_backbone_info(backbone_name)
        print(f"  {backbone_name}: {info['type']}, {info['feature_dim']}D features, {info['memory_usage']} memory")
    
    # Test creating backbones
    print("\nTesting backbone creation:")
    
    # Test ResNet
    resnet = BackboneFactory.create_backbone('resnet152', freeze=True)
    print(f"ResNet152: {resnet.get_feature_dim()}D features, frozen: {resnet.freeze_weights_flag}")
    
    # Test ViT
    vit = BackboneFactory.create_backbone('vit_base_patch16_224', freeze=True)
    print(f"ViT-B/16: {vit.get_feature_dim()}D features, frozen: {vit.freeze_weights_flag}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        resnet_features = resnet(dummy_input)
        vit_features = vit(dummy_input)
        
    print(f"\nForward pass test:")
    print(f"ResNet output shape: {resnet_features.shape}")
    print(f"ViT output shape: {vit_features.shape}")
