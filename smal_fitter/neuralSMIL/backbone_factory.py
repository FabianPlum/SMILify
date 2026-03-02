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


class BackboneInterface(nn.Module, ABC):
    """Abstract interface for backbone networks.

    Inherits from nn.Module so that backbone parameters are properly
    registered when assigned as attributes of parent modules (e.g.
    SMILImageRegressor.backbone), making them visible to
    model.parameters(), optimizers, and checkpoint saving.
    """

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

    def get_spatial_dim(self) -> int:
        """Spatial feature dimension for patch tokens (same as feature_dim for ViT)."""
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


class UNetDecodeBlock(nn.Module):
    """Single UNet decoder block: bilinear upsample + skip-connection concat + double conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class _UNetModule(nn.Module):
    """Inner nn.Module combining encoder + decoder for BackboneInterface compatibility.

    Storing both as a single nn.Module ensures that BackboneInterface.to(),
    .parameters(), .eval(), and .train() all propagate to the whole UNet.
    """

    def __init__(self, encoder: nn.Module, decode_blocks: list, gap: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decode_blocks = nn.ModuleList(decode_blocks)
        self.gap = gap  # AdaptiveAvgPool2d(1) for bottleneck global features


class UNetBackbone(BackboneInterface):
    """
    UNet backbone for high-resolution spatial feature extraction.

    Uses a pretrained timm encoder (EfficientNet, ResNet, MobileNet…) with a
    lightweight skip-connection decoder. Provides:
      - ``forward(x)``               → global avg-pooled bottleneck: (B, feature_dim)
      - ``forward_with_spatial(x)``  → (global_features, spatial_features)
            global_features: (B, feature_dim)
            spatial_features: (B, H*W, spatial_dim) — flattened decoder feature map

    The decoder is always kept trainable even when the encoder is frozen.
    This follows SLEAP's strategy: the pretrained encoder extracts semantic
    features; the randomly-initialised decoder learns to recover spatial detail
    for joint localisation.
    """

    # Maps registered backbone_name → timm encoder model name
    _ENCODER_MAP: Dict[str, str] = {
        'unet_efficientnet_b0': 'efficientnet_b0',
        'unet_efficientnet_b3': 'efficientnet_b3',
        'unet_resnet34': 'resnet34',
        'unet_mobilenet_v3': 'mobilenetv3_large_100',
    }

    def __init__(self, model_name: str = 'unet_efficientnet_b3',
                 pretrained: bool = True, freeze: bool = False,
                 decoder_channels: Tuple[int, ...] = (256, 128, 64),
                 spatial_level: int = 1):
        """
        Args:
            model_name: Full UNet backbone name, e.g. 'unet_efficientnet_b3'.
            pretrained: Whether to use ImageNet-pretrained encoder weights.
            freeze: Whether to freeze encoder weights (decoder is always trained).
            decoder_channels: Output channels for each decoder block (coarse→fine).
                              Length determines how many upsampling steps are taken.
            spatial_level: Which decoder block's output to expose as spatial context.
                0 = coarsest (fewest tokens, deepest semantics)
                len(decoder_channels)-1 = finest (most tokens)
                Default 1 gives a good balance: ~784 tokens at 512×512 input.
        """
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm library is required for UNet backbones. Install with: pip install timm"
            )

        encoder_name = self._ENCODER_MAP.get(model_name)
        if encoder_name is None:
            raise ValueError(
                f"Unknown UNet backbone '{model_name}'. "
                f"Supported: {list(self._ENCODER_MAP.keys())}"
            )

        self.model_name = model_name
        self.freeze_weights_flag = freeze
        self.spatial_level = spatial_level
        self._decoder_channels = tuple(decoder_channels)

        # ── Encoder ──────────────────────────────────────────────────────────
        # features_only=True returns a list of feature maps at each stride,
        # from finest (stride 2) to coarsest (stride 32).
        encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)

        # Probe encoder to discover per-stage channel counts dynamically.
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, 64, 64)
            _dummy_feats = encoder(_dummy)
        enc_channels = [f.shape[1] for f in _dummy_feats]  # e.g. [24, 32, 48, 136, 384]

        # Global feature dimension = bottleneck channels (deepest encoder stage)
        self.feature_dim = enc_channels[-1]
        self._spatial_dim = int(decoder_channels[spatial_level])

        # ── Decoder ──────────────────────────────────────────────────────────
        # Build one decode block per entry in decoder_channels.
        # decode_blocks[0] takes the bottleneck and concat-skip from enc_channels[-2],
        # decode_blocks[1] takes that output and concat-skip from enc_channels[-3], …
        decode_blocks = []
        prev_ch = self.feature_dim
        n_blocks = len(decoder_channels)
        for i in range(n_blocks):
            skip_idx = -(i + 2)
            skip_ch = enc_channels[skip_idx] if abs(skip_idx) <= len(enc_channels) else 0
            out_ch = int(decoder_channels[i])
            decode_blocks.append(UNetDecodeBlock(prev_ch, skip_ch, out_ch))
            prev_ch = out_ch

        # ── Combined module (for BackboneInterface compatibility) ─────────────
        self.backbone = _UNetModule(encoder, decode_blocks, nn.AdaptiveAvgPool2d(1))

        if freeze:
            self.freeze_weights()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode_decode(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Run full encoder + decoder forward.

        Returns:
            bottleneck: feature map from the deepest encoder stage (B, C, H, W)
            decoder_outputs: list of feature maps from each decode block (coarse→fine)
        """
        enc_feats = self.backbone.encoder(x)  # list: [fine … coarse]
        bottleneck = enc_feats[-1]

        current = bottleneck
        decoder_outputs = []
        for i, block in enumerate(self.backbone.decode_blocks):
            skip_idx = -(i + 2)
            skip = enc_feats[skip_idx] if abs(skip_idx) <= len(enc_feats) else None
            current = block(current, skip)
            decoder_outputs.append(current)

        return bottleneck, decoder_outputs

    # ── BackboneInterface ─────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return global avg-pooled bottleneck features: (B, feature_dim)."""
        bottleneck, _ = self._encode_decode(x)
        return self.backbone.gap(bottleneck).view(x.shape[0], -1)

    def forward_with_spatial(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return global features and spatial decoder features.

        Returns:
            global_features: (B, feature_dim)  — avg-pooled bottleneck
            spatial_features: (B, H*W, spatial_dim) — chosen decoder level, flattened
        """
        bottleneck, decoder_outputs = self._encode_decode(x)
        global_features = self.backbone.gap(bottleneck).view(x.shape[0], -1)

        spatial_map = decoder_outputs[self.spatial_level]  # (B, C, H, W)
        B, C, H, W = spatial_map.shape
        # Permute to (B, H, W, C) then reshape to (B, H*W, C) for attention
        spatial_features = spatial_map.permute(0, 2, 3, 1).reshape(B, H * W, C)

        return global_features, spatial_features

    def get_feature_dim(self) -> int:
        """Bottleneck feature dimension (used as global feature for decoder head)."""
        return self.feature_dim

    def get_spatial_dim(self) -> int:
        """Spatial feature dimension at the chosen decoder level (= context_dim)."""
        return self._spatial_dim

    def freeze_weights(self) -> None:
        """Freeze encoder only; decoder remains trainable."""
        for param in self.backbone.encoder.parameters():
            param.requires_grad = False
        self.freeze_weights_flag = True

    def unfreeze_weights(self) -> None:
        """Unfreeze encoder weights."""
        for param in self.backbone.encoder.parameters():
            param.requires_grad = True
        self.freeze_weights_flag = False

    def get_trainable_parameters(self) -> list:
        """Decoder is always trainable; encoder is trainable only when not frozen."""
        params = list(self.backbone.decode_blocks.parameters())
        if not self.freeze_weights_flag:
            params += list(self.backbone.encoder.parameters())
        return params


class BackboneFactory:
    """Factory class for creating backbone networks."""

    SUPPORTED_BACKBONES = {
        'resnet50': ResNetBackbone,
        'resnet101': ResNetBackbone,
        'resnet152': ResNetBackbone,
        'vit_base_patch16_224': ViTBackbone,
        'vit_large_patch16_224': ViTBackbone,
        # UNet variants: lightweight pretrained encoder + skip-connection decoder
        'unet_efficientnet_b0': UNetBackbone,   # ~8M enc params, spatial_dim=64
        'unet_efficientnet_b3': UNetBackbone,   # ~15M enc params, spatial_dim=128
        'unet_resnet34': UNetBackbone,           # ~25M enc params, spatial_dim=128
        'unet_mobilenet_v3': UNetBackbone,       # ~6M enc params, spatial_dim=64
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
        
        # Check if ViT or UNet is requested but timm is not available
        if (backbone_name.startswith('vit') or backbone_name.startswith('unet_')) and not TIMM_AVAILABLE:
            raise ImportError(f"Backbone '{backbone_name}' requires timm library. Install with: pip install timm")
        
        backbone_class = cls.SUPPORTED_BACKBONES[backbone_name]

        # Set default arguments based on backbone type
        if backbone_name.startswith('resnet'):
            kwargs.setdefault('model_name', backbone_name)
        elif backbone_name.startswith('vit'):
            kwargs.setdefault('model_name', backbone_name)
        elif backbone_name.startswith('unet_'):
            kwargs.setdefault('model_name', backbone_name)
            # Apply sensible per-variant decoder defaults (can be overridden by caller)
            _unet_decoder_defaults: Dict[str, Any] = {
                'unet_efficientnet_b0': {'decoder_channels': (128, 64, 32)},
                'unet_efficientnet_b3': {'decoder_channels': (256, 128, 64)},
                'unet_resnet34':        {'decoder_channels': (256, 128, 64)},
                'unet_mobilenet_v3':    {'decoder_channels': (128, 64, 32)},
            }
            for k, v in _unet_decoder_defaults.get(backbone_name, {}).items():
                kwargs.setdefault(k, v)

        return backbone_class(**kwargs)
    
    @classmethod
    def get_default_input_resolution(cls, backbone_name: str) -> int:
        """Return the recommended input resolution for a given backbone.

        This is the single source of truth for default input resolution.
        ViT models require a fixed 224×224 input.  CNN-based backbones
        (ResNet, UNet) are fully convolutional and default to 512×512,
        but callers may override this via configuration.
        """
        if backbone_name.startswith('vit'):
            return 224
        # ResNet and UNet are fully convolutional → 512 by default
        return 512

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
        
        # Default input resolution from the centralized helper
        default_res = cls.get_default_input_resolution(backbone_name)

        # Add specific information based on backbone type
        if backbone_name.startswith('resnet'):
            info.update({
                'type': 'CNN',
                'feature_dim': 2048,
                'input_size': default_res,
                'memory_usage': 'Medium',
            })
        elif backbone_name.startswith('vit'):
            if 'base' in backbone_name:
                info.update({
                    'type': 'Transformer',
                    'feature_dim': 768,
                    'input_size': default_res,
                    'memory_usage': 'Medium',
                })
            elif 'large' in backbone_name:
                info.update({
                    'type': 'Transformer',
                    'feature_dim': 1024,
                    'input_size': default_res,
                    'memory_usage': 'High',
                })
        elif backbone_name.startswith('unet_'):
            _unet_info: Dict[str, Any] = {
                'unet_efficientnet_b0': {'feature_dim': 320, 'spatial_dim': 64,  'memory_usage': 'Low'},
                'unet_efficientnet_b3': {'feature_dim': 384, 'spatial_dim': 128, 'memory_usage': 'Low'},
                'unet_resnet34':        {'feature_dim': 512, 'spatial_dim': 128, 'memory_usage': 'Low'},
                'unet_mobilenet_v3':    {'feature_dim': 960, 'spatial_dim': 64,  'memory_usage': 'Low'},
            }
            info.update({
                'type': 'UNet (CNN encoder + decoder)',
                'input_size': f'{default_res} (fully convolutional, configurable)',
                'has_spatial_features': True,
                **_unet_info.get(backbone_name, {}),
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
