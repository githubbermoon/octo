"""
Land Surface Temperature Estimation Model

CNN-based regression model for estimating LST from thermal satellite data.
Useful for Urban Heat Island (UHI) analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import logging

from .base import RegressionModel

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual block with optional squeeze-excitation."""
    
    def __init__(
        self,
        channels: int,
        use_se: bool = True,
        se_ratio: int = 16
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Squeeze-Excitation
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // se_ratio, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // se_ratio, channels, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.use_se:
            out = out * self.se(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class LSTEstimator(RegressionModel):
    """
    CNN model for Land Surface Temperature estimation.
    
    Takes thermal and optical bands as input and outputs LST in Kelvin.
    Supports auxiliary inputs like elevation and NDVI.
    
    Example:
        >>> model = LSTEstimator(
        ...     in_channels=7,  # Landsat bands including thermal
        ...     output_range=(250, 330)  # LST range in Kelvin
        ... )
        >>> lst = model(input_tensor)  # (B, 1, H, W)
    """
    
    def __init__(
        self,
        in_channels: int = 7,
        base_features: int = 64,
        num_residual_blocks: int = 4,
        output_range: Tuple[float, float] = (250, 330),
        use_auxiliary: bool = True,
        aux_channels: int = 2,  # e.g., NDVI, elevation
        dropout: float = 0.2
    ):
        """
        Initialize LST estimator.
        
        Args:
            in_channels: Number of input bands
            base_features: Base feature dimension
            num_residual_blocks: Number of residual blocks
            output_range: Output temperature range (Kelvin)
            use_auxiliary: Use auxiliary inputs (NDVI, DEM)
            aux_channels: Number of auxiliary channels
            dropout: Dropout rate
        """
        super().__init__(
            in_channels=in_channels,
            output_range=output_range,
            name="LSTEstimator"
        )
        
        self.use_auxiliary = use_auxiliary
        total_in = in_channels + aux_channels if use_auxiliary else in_channels
        
        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(total_in, base_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(base_features) for _ in range(num_residual_blocks)
        ])
        
        # Multi-scale feature extraction
        self.multi_scale = MultiScaleBlock(base_features)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(base_features * 4, base_features * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(base_features * 2, base_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
        )
        
        # Output head
        self.output = nn.Sequential(
            nn.Conv2d(base_features, 1, 1),
            nn.Sigmoid()  # Normalize to [0, 1], then scale
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        auxiliary: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W) - spectral bands
            auxiliary: Optional auxiliary data (B, aux_C, H, W)
            
        Returns:
            LST prediction (B, 1, H, W) in Kelvin
        """
        # Concatenate auxiliary if provided
        if self.use_auxiliary and auxiliary is not None:
            x = torch.cat([x, auxiliary], dim=1)
        
        # Feature extraction
        features = self.initial(x)
        features = self.residual_blocks(features)
        
        # Multi-scale processing
        multi_scale_features = self.multi_scale(features)
        
        # Decode
        decoded = self.decoder(multi_scale_features)
        
        # Output (normalized)
        output = self.output(decoded)
        
        # Scale to temperature range
        return self.scale_output(output)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        auxiliary: Optional[torch.Tensor] = None,
        n_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation using MC Dropout.
        
        Args:
            x: Input tensor
            auxiliary: Auxiliary data
            n_samples: Number of dropout samples
            
        Returns:
            mean: Mean prediction
            std: Standard deviation (uncertainty)
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self(x.to(self.device), auxiliary)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        self.eval()
        return mean, std
    
    def compute_uhi(
        self,
        lst: torch.Tensor,
        reference_temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute Urban Heat Island intensity.
        
        UHI = LST - reference_temperature
        
        Args:
            lst: Land Surface Temperature (B, 1, H, W)
            reference_temperature: Reference (e.g., rural mean). If None, uses image mean.
            
        Returns:
            UHI intensity (positive = warmer than reference)
        """
        if reference_temperature is None:
            reference_temperature = lst.mean()
        
        return lst - reference_temperature


class MultiScaleBlock(nn.Module):
    """Multi-scale feature extraction using dilated convolutions."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.scale1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=4, dilation=4),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=True),  # No BN to support batch_size=1
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        s4 = F.interpolate(self.scale4(x), size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return torch.cat([s1, s2, s3, s4], dim=1)


class TemperatureCalibration(nn.Module):
    """
    Temperature calibration layer for LST refinement.
    
    Learns site-specific calibration based on auxiliary data.
    """
    
    def __init__(self, aux_channels: int = 5):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(aux_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Scale and bias
        )
    
    def forward(
        self,
        lst: torch.Tensor,
        aux_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Calibrate LST based on auxiliary features.
        
        Args:
            lst: Raw LST prediction (B, 1, H, W)
            aux_features: Auxiliary features (B, aux_C)
            
        Returns:
            Calibrated LST
        """
        params = self.mlp(aux_features)  # (B, 2)
        scale = params[:, 0:1].unsqueeze(-1).unsqueeze(-1)
        bias = params[:, 1:2].unsqueeze(-1).unsqueeze(-1)
        
        return lst * (1 + scale * 0.1) + bias
