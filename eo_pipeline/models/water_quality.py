"""
Water Quality Detection Model

Multi-task model for detecting water pollution, turbidity,
chlorophyll concentration, and floating debris/plastics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

from .base import EOModel

logger = logging.getLogger(__name__)


class WaterQualityDetector(EOModel):
    """
    Multi-head model for water quality assessment.
    
    Outputs:
    - Water mask (binary segmentation)
    - Turbidity index (regression)
    - Chlorophyll concentration (regression)
    - Plastic/debris detection (binary segmentation)
    
    Example:
        >>> model = WaterQualityDetector(in_channels=10)
        >>> outputs = model(input_tensor)
        >>> water_mask = outputs['water_mask']
        >>> turbidity = outputs['turbidity']
    """
    
    OUTPUT_HEADS = ['water_mask', 'turbidity', 'chlorophyll', 'plastic']
    
    def __init__(
        self,
        in_channels: int = 10,
        base_features: int = 64,
        encoder_depth: int = 4,
        use_spectral_indices: bool = True,
        dropout: float = 0.2
    ):
        """
        Initialize water quality detector.
        
        Args:
            in_channels: Number of input bands
            base_features: Base feature dimension
            encoder_depth: Number of encoder levels
            use_spectral_indices: Compute and use spectral indices
            dropout: Dropout rate
        """
        super().__init__(name="WaterQualityDetector")
        
        self.in_channels = in_channels
        self.use_spectral_indices = use_spectral_indices
        
        # Additional channels for spectral indices
        extra_channels = 4 if use_spectral_indices else 0  # NDWI, MNDWI, turbidity, FAI
        total_channels = in_channels + extra_channels
        
        # Shared encoder
        self.encoder = SharedEncoder(
            in_channels=total_channels,
            base_features=base_features,
            depth=encoder_depth,
            dropout=dropout
        )
        
        encoder_out_features = base_features * (2 ** encoder_depth)  # Bottleneck is 2x last encoder
        
        # Skip channel sizes from encoder (reversed for decoder)
        skip_channels = [base_features * (2 ** i) for i in range(encoder_depth)][::-1]
        
        # Task-specific decoders
        self.water_mask_head = SegmentationHead(
            in_features=encoder_out_features,
            out_channels=1,  # Binary
            decoder_features=[256, 128, 64],
            skip_channels=skip_channels[:3]  # Use first 3 skip connections
        )
        
        self.turbidity_head = RegressionHead(
            in_features=encoder_out_features,
            decoder_features=[256, 128],
            output_range=(0, 100)  # NTU scale
        )
        
        self.chlorophyll_head = RegressionHead(
            in_features=encoder_out_features,
            decoder_features=[256, 128],
            output_range=(0, 500)  # μg/L
        )
        
        self.plastic_head = SegmentationHead(
            in_features=encoder_out_features,
            out_channels=1,  # Binary
            decoder_features=[256, 128, 64],
            skip_channels=skip_channels[:3]
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
    
    def compute_spectral_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral indices for water quality.
        
        Assumes standard band ordering: [B, G, R, RE1, RE2, RE3, NIR, SWIR1, SWIR2, ...]
        """
        # Extract bands (adjust indices based on actual band order)
        green = x[:, 1:2]   # Green
        red = x[:, 2:3]     # Red
        nir = x[:, 6:7]     # NIR
        swir = x[:, 7:8]    # SWIR1
        
        eps = 1e-8
        
        # NDWI (water detection)
        ndwi = (green - nir) / (green + nir + eps)
        
        # MNDWI (modified, better for urban water)
        mndwi = (green - swir) / (green + swir + eps)
        
        # Turbidity index (red/green ratio)
        turbidity_idx = red / (green + eps)
        
        # Floating Algae/Debris Index
        fai = nir - (red + (swir - red) * 0.5)
        
        return torch.cat([ndwi, mndwi, turbidity_idx, fai], dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        water_mask_only: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            water_mask_only: Only compute water mask
            
        Returns:
            Dictionary of outputs for each task
        """
        # Compute spectral indices
        if self.use_spectral_indices:
            indices = self.compute_spectral_indices(x)
            x = torch.cat([x, indices], dim=1)
        
        input_size = x.shape[2:]
        
        # Shared encoding
        features, skip_connections = self.encoder(x)
        
        outputs = {}
        
        # Water mask (always computed)
        outputs['water_mask'] = self.water_mask_head(features, skip_connections)
        
        if not water_mask_only:
            # Apply water mask to focus on water pixels
            water_prob = torch.sigmoid(outputs['water_mask'])
            
            # Turbidity (upsample to match water_mask size)
            turb = self.turbidity_head(features)
            turb = F.interpolate(turb, size=outputs['water_mask'].shape[2:], mode='bilinear', align_corners=True)
            outputs['turbidity'] = turb * water_prob
            
            # Chlorophyll
            chl = self.chlorophyll_head(features)
            chl = F.interpolate(chl, size=outputs['water_mask'].shape[2:], mode='bilinear', align_corners=True)
            outputs['chlorophyll'] = chl * water_prob
            
            # Plastic detection
            outputs['plastic'] = self.plastic_head(features, skip_connections) * water_prob
        
        return outputs
    
    def get_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            weights: Task weights
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual losses
        """
        weights = weights or {
            'water_mask': 1.0,
            'turbidity': 0.5,
            'chlorophyll': 0.5,
            'plastic': 1.0
        }
        
        loss_dict = {}
        total_loss = 0
        
        # Water mask (BCE)
        if 'water_mask' in targets:
            loss_dict['water_mask'] = F.binary_cross_entropy_with_logits(
                predictions['water_mask'],
                targets['water_mask'].float()
            )
            total_loss += weights['water_mask'] * loss_dict['water_mask']
        
        # Turbidity (MSE with water mask)
        if 'turbidity' in targets and 'turbidity' in predictions:
            water_mask = targets.get('water_mask', None)
            if water_mask is not None:
                pred_turb = predictions['turbidity'][water_mask > 0.5]
                tgt_turb = targets['turbidity'][water_mask > 0.5]
                if pred_turb.numel() > 0:
                    loss_dict['turbidity'] = F.mse_loss(pred_turb, tgt_turb)
                    total_loss += weights['turbidity'] * loss_dict['turbidity']
        
        # Chlorophyll (MSE)
        if 'chlorophyll' in targets and 'chlorophyll' in predictions:
            water_mask = targets.get('water_mask', None)
            if water_mask is not None:
                pred_chl = predictions['chlorophyll'][water_mask > 0.5]
                tgt_chl = targets['chlorophyll'][water_mask > 0.5]
                if pred_chl.numel() > 0:
                    loss_dict['chlorophyll'] = F.mse_loss(pred_chl, tgt_chl)
                    total_loss += weights['chlorophyll'] * loss_dict['chlorophyll']
        
        # Plastic detection (BCE with focal loss for imbalance)
        if 'plastic' in targets and 'plastic' in predictions:
            loss_dict['plastic'] = self._focal_loss(
                predictions['plastic'],
                targets['plastic'].float()
            )
            total_loss += weights['plastic'] * loss_dict['plastic']
        
        return total_loss, loss_dict
    
    def _focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        gamma: float = 2.0,
        alpha: float = 0.25
    ) -> torch.Tensor:
        """Focal loss for imbalanced plastic detection."""
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = alpha * (1 - pt) ** gamma
        return (focal_weight * bce).mean()
    
    def detect_pollution_hotspots(
        self,
        outputs: Dict[str, torch.Tensor],
        turbidity_threshold: float = 50,
        chlorophyll_threshold: float = 100, 
        plastic_threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Identify pollution hotspots from model outputs.
        
        Returns binary masks for each pollution type.
        """
        hotspots = {}
        
        # High turbidity areas
        if 'turbidity' in outputs:
            hotspots['high_turbidity'] = outputs['turbidity'] > turbidity_threshold
        
        # Algal bloom areas (high chlorophyll)
        if 'chlorophyll' in outputs:
            hotspots['algal_bloom'] = outputs['chlorophyll'] > chlorophyll_threshold
        
        # Plastic debris
        if 'plastic' in outputs:
            hotspots['plastic_debris'] = torch.sigmoid(outputs['plastic']) > plastic_threshold
        
        return hotspots


class SharedEncoder(nn.Module):
    """Shared encoder for multi-task learning."""
    
    def __init__(
        self,
        in_channels: int,
        base_features: int = 64,
        depth: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        features = base_features
        prev_features = in_channels
        
        for i in range(depth):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(prev_features, features, 3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(features, features, 3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout) if i > 0 else nn.Identity()
                )
            )
            self.pools.append(nn.MaxPool2d(2, 2))
            prev_features = features
            features *= 2
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(prev_features, features, 3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns bottleneck features and skip connections."""
        skip_connections = []
        
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)
        
        bottleneck = self.bottleneck(x)
        
        return bottleneck, skip_connections


class SegmentationHead(nn.Module):
    """Decoder head for segmentation tasks."""
    
    def __init__(
        self,
        in_features: int,
        out_channels: int,
        decoder_features: List[int],
        skip_channels: Optional[List[int]] = None
    ):
        super().__init__()
        
        self.upconvs = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.skip_reducers = nn.ModuleList()
        
        # Default skip channels match decoder features
        if skip_channels is None:
            skip_channels = decoder_features
        
        prev = in_features
        for i, feat in enumerate(decoder_features):
            self.upconvs.append(nn.ConvTranspose2d(prev, feat, 2, stride=2))
            
            # Reducer for skip connections if needed
            if i < len(skip_channels) and skip_channels[i] != feat:
                self.skip_reducers.append(
                    nn.Conv2d(skip_channels[i], feat, 1, bias=False)
                )
            else:
                self.skip_reducers.append(nn.Identity())
            
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(feat * 2, feat, 3, padding=1, bias=False),
                    nn.BatchNorm2d(feat),
                    nn.ReLU(inplace=True)
                )
            )
            prev = feat
        
        self.output = nn.Conv2d(decoder_features[-1], out_channels, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        skip_connections: List[torch.Tensor]
    ) -> torch.Tensor:
        skip_connections = skip_connections[::-1]
        
        for i, (upconv, conv, skip_reducer) in enumerate(zip(self.upconvs, self.convs, self.skip_reducers)):
            x = upconv(x)
            if i < len(skip_connections):
                skip = skip_connections[i]
                skip = skip_reducer(skip)  # Reduce channels if needed
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                x = torch.cat([x, skip], dim=1)
            else:
                # No skip connection, pad with zeros
                x = torch.cat([x, torch.zeros_like(x)], dim=1)
            x = conv(x)
        
        return self.output(x)


class RegressionHead(nn.Module):
    """Decoder head for regression tasks (outputs single value per pixel)."""
    
    def __init__(
        self,
        in_features: int,
        decoder_features: List[int],
        output_range: Tuple[float, float] = (0, 1)
    ):
        super().__init__()
        
        self.output_range = output_range
        
        layers = []
        prev = in_features
        for feat in decoder_features:
            layers.extend([
                nn.ConvTranspose2d(prev, feat, 2, stride=2),
                nn.BatchNorm2d(feat),
                nn.ReLU(inplace=True)
            ])
            prev = feat
        
        self.decoder = nn.Sequential(*layers)
        self.output = nn.Sequential(
            nn.Conv2d(decoder_features[-1], 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        x = self.output(x)
        
        # Scale to output range
        min_val, max_val = self.output_range
        return x * (max_val - min_val) + min_val
