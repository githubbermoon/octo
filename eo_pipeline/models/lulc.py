"""
Land Use / Land Cover Classification Model

U-Net architecture for semantic segmentation of satellite imagery
into land use and land cover classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import logging

from .base import SegmentationModel

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Double convolution block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_bn: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=not use_bn)
        )
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetEncoder(nn.Module):
    """U-Net encoder with skip connections."""
    
    def __init__(
        self,
        in_channels: int,
        features: List[int] = [64, 128, 256, 512, 1024],
        use_bn: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.features = features
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        prev_channels = in_channels
        for feature in features[:-1]:
            self.encoders.append(ConvBlock(prev_channels, feature, use_bn=use_bn, dropout=dropout))
            self.pools.append(nn.MaxPool2d(2, 2))
            prev_channels = feature
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[-2], features[-1], use_bn=use_bn, dropout=dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        
        Returns:
            bottleneck: Bottleneck features
            skip_connections: List of skip connection features
        """
        skip_connections = []
        
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)
        
        bottleneck = self.bottleneck(x)
        
        return bottleneck, skip_connections


class AttentionGate(nn.Module):
    """Attention gate for skip connections."""
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention.
        
        Args:
            g: Gate signal from decoder
            x: Skip connection
            
        Returns:
            Attended skip connection
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Resize g1 to match x1 if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class UNetDecoder(nn.Module):
    """U-Net decoder with optional attention gates."""
    
    def __init__(
        self,
        features: List[int] = [1024, 512, 256, 128, 64],
        use_bn: bool = True,
        use_attention: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.features = features
        self.use_attention = use_attention
        
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if use_attention else None
        
        for i in range(len(features) - 1):
            self.upconvs.append(
                nn.ConvTranspose2d(features[i], features[i+1], 2, stride=2)
            )
            
            if use_attention:
                self.attention_gates.append(
                    AttentionGate(features[i+1], features[i+1], features[i+1] // 2)
                )
            
            self.decoders.append(
                ConvBlock(features[i+1] * 2, features[i+1], use_bn=use_bn, dropout=dropout)
            )
    
    def forward(
        self,
        x: torch.Tensor,
        skip_connections: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Bottleneck features
            skip_connections: List of encoder skip connections (reverse order)
            
        Returns:
            Decoder output
        """
        skip_connections = skip_connections[::-1]  # Reverse to match decoder order
        
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = skip_connections[i]
            
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            # Apply attention if enabled
            if self.use_attention:
                skip = self.attention_gates[i](x, skip)
            
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        return x


class LULCClassifier(SegmentationModel):
    """
    U-Net model for Land Use / Land Cover classification.
    
    Supports multitemporal input and optional attention mechanisms.
    
    Example:
        >>> model = LULCClassifier(
        ...     in_channels=12,  # Sentinel-2 bands
        ...     num_classes=10,  # LULC classes
        ...     features=[64, 128, 256, 512]
        ... )
        >>> output = model(input_tensor)  # (B, 10, H, W)
    """
    
    # Standard LULC class names (example based on ESA WorldCover)
    CLASS_NAMES = [
        "Tree cover",
        "Shrubland",
        "Grassland",
        "Cropland",
        "Built-up",
        "Bare/sparse vegetation",
        "Snow and ice",
        "Permanent water bodies",
        "Herbaceous wetland",
        "Mangroves"
    ]
    
    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 10,
        features: List[int] = [64, 128, 256, 512, 1024],
        use_attention: bool = True,
        dropout: float = 0.2,
        multitemporal: bool = False,
        time_steps: int = 1
    ):
        """
        Initialize LULC classifier.
        
        Args:
            in_channels: Number of input bands
            num_classes: Number of output classes
            features: Feature dimensions for each encoder level
            use_attention: Use attention gates in decoder
            dropout: Dropout rate
            multitemporal: Enable multitemporal processing
            time_steps: Number of time steps if multitemporal
        """
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            name="LULCClassifier"
        )
        
        self.multitemporal = multitemporal
        self.time_steps = time_steps
        
        # Adjust input channels for multitemporal
        encoder_in_channels = in_channels * time_steps if multitemporal else in_channels
        
        # Build U-Net
        self.encoder = UNetEncoder(
            in_channels=encoder_in_channels,
            features=features,
            dropout=dropout
        )
        
        self.decoder = UNetDecoder(
            features=features[::-1],  # Reverse: bottleneck to output
            use_attention=use_attention,
            dropout=dropout
        )
        
        # Output head
        self.output_conv = nn.Conv2d(features[0], num_classes, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W) or (B, T, C, H, W) for multitemporal
            
        Returns:
            Logits (B, num_classes, H, W)
        """
        # Handle multitemporal input
        if self.multitemporal and x.ndim == 5:
            # (B, T, C, H, W) -> (B, T*C, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B, T * C, H, W)
        
        # Encode
        bottleneck, skip_connections = self.encoder(x)
        
        # Decode
        features = self.decoder(bottleneck, skip_connections)
        
        # Output
        output = self.output_conv(features)
        
        return output
    
    def predict_with_confidence(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class labels with confidence scores.
        
        Args:
            x: Input tensor
            
        Returns:
            predictions: Class predictions (B, H, W)
            confidence: Confidence scores (B, H, W)
        """
        self.eval()
        with torch.no_grad():
            logits = self(x.to(self.device))
            probs = F.softmax(logits, dim=1)
            confidence, predictions = probs.max(dim=1)
        
        return predictions, confidence
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        **kwargs
    ) -> "LULCClassifier":
        """
        Load pretrained model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            **kwargs: Override model parameters
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model config from checkpoint or use defaults
        config = checkpoint.get('config', {})
        config.update(kwargs)
        
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model


class TemporalAttention(nn.Module):
    """Attention mechanism for multitemporal data."""
    
    def __init__(self, channels: int, time_steps: int):
        super().__init__()
        
        self.channels = channels
        self.time_steps = time_steps
        
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal attention.
        
        Args:
            x: (B, T, C, H, W)
            
        Returns:
            Attended features (B, C, H, W)
        """
        B, T, C, H, W = x.shape
        
        # Reshape for attention computation
        x_flat = x.view(B * T, C, H, W)
        
        q = self.query(x_flat).view(B, T, -1, H * W)  # (B, T, C', HW)
        k = self.key(x_flat).view(B, T, -1, H * W)    # (B, T, C', HW)
        v = self.value(x_flat).view(B, T, C, H * W)   # (B, T, C, HW)
        
        # Temporal attention
        attention = torch.einsum('btch,bsch->bts', q.mean(-1), k.mean(-1))  # (B, T, T)
        attention = F.softmax(attention, dim=-1)
        
        # Attend
        out = torch.einsum('bts,bsch->btch', attention, v)  # (B, T, C, HW)
        out = out.view(B, T, C, H, W)
        
        # Aggregate across time
        out = out.mean(dim=1)  # (B, C, H, W)
        
        return x.mean(dim=1) + self.gamma * out
