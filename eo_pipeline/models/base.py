"""
Base Model Class for Earth Observation

Provides abstract base class for all EO models with common functionality
like device management, checkpointing, and inference utilities.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import logging

from ..core.device import DeviceManager, get_device

logger = logging.getLogger(__name__)


class EOModel(nn.Module, ABC):
    """
    Abstract base class for Earth Observation models.
    
    Provides common functionality:
    - Automatic device placement
    - Checkpoint save/load
    - Inference utilities
    - Model summary
    
    Subclasses must implement:
    - forward(): Forward pass
    - get_loss(): Loss computation
    
    Example:
        >>> class MyModel(EOModel):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.layers = nn.Sequential(...)
        ...     
        ...     def forward(self, x):
        ...         return self.layers(x)
    """
    
    def __init__(self, name: str = "EOModel"):
        """
        Initialize base model.
        
        Args:
            name: Model name for logging and checkpoints
        """
        super().__init__()
        self.name = name
        self._device = None
        
    @property
    def device(self) -> torch.device:
        """Get the device where model parameters are located."""
        if self._device is None:
            # Try to infer from parameters
            try:
                self._device = next(self.parameters()).device
            except StopIteration:
                self._device = get_device()
        return self._device
    
    def to_device(self, device: Optional[torch.device] = None) -> "EOModel":
        """
        Move model to specified device (or best available).
        
        Args:
            device: Target device (None = auto-detect best)
            
        Returns:
            Self for chaining
        """
        if device is None:
            device = get_device()
        self._device = device
        return self.to(device)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute loss for training.
        
        Override in subclass for custom loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments
            
        Returns:
            Loss tensor
        """
        # Default: Cross-entropy for classification
        return nn.functional.cross_entropy(predictions, targets)
    
    def predict(
        self,
        x: torch.Tensor,
        return_probs: bool = False
    ) -> torch.Tensor:
        """
        Make predictions in inference mode.
        
        Args:
            x: Input tensor
            return_probs: Return probabilities instead of class predictions
            
        Returns:
            Predictions
        """
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            output = self(x)
            
            if return_probs:
                return torch.softmax(output, dim=1)
            else:
                return output.argmax(dim=1)
    
    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        **extra_info
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Save path
            optimizer: Optimizer state to save
            epoch: Current epoch
            metrics: Validation metrics
            **extra_info: Additional info to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_name": self.name,
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
            **extra_info
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            path: Checkpoint path
            optimizer: Optimizer to load state into
            strict: Whether to strictly enforce state dict matching
            
        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        logger.info(f"Checkpoint loaded: {path}")
        
        return {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
        }
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.
        
        Args:
            trainable_only: Only count trainable parameters
            
        Returns:
            Parameter count
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def summary(self) -> str:
        """Get model summary string."""
        total_params = self.count_parameters(trainable_only=False)
        trainable_params = self.count_parameters(trainable_only=True)
        
        lines = [
            f"Model: {self.name}",
            f"Total parameters: {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
            f"Non-trainable parameters: {total_params - trainable_params:,}",
            f"Device: {self.device}",
        ]
        return "\n".join(lines)
    
    def freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def freeze_encoder(self) -> None:
        """Freeze encoder/backbone (if applicable)."""
        if hasattr(self, "encoder"):
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen")


class SegmentationModel(EOModel):
    """Base class for semantic segmentation models."""
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        name: str = "SegmentationModel"
    ):
        super().__init__(name=name)
        self.in_channels = in_channels
        self.num_classes = num_classes
    
    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100
    ) -> torch.Tensor:
        """
        Compute segmentation loss (cross-entropy + optional dice).
        
        Args:
            predictions: (B, C, H, W) logits
            targets: (B, H, W) class labels
            class_weights: Optional class weights
            ignore_index: Label to ignore
            
        Returns:
            Loss tensor
        """
        ce_loss = nn.functional.cross_entropy(
            predictions, 
            targets,
            weight=class_weights,
            ignore_index=ignore_index
        )
        
        # Add Dice loss for better boundary learning
        dice_loss = self._dice_loss(predictions, targets, ignore_index)
        
        return ce_loss + dice_loss
    
    def _dice_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
        smooth: float = 1e-6
    ) -> torch.Tensor:
        """Compute Dice loss."""
        probs = torch.softmax(predictions, dim=1)
        
        # One-hot encode targets
        targets_one_hot = torch.zeros_like(probs)
        valid_mask = targets != ignore_index
        targets_masked = targets.clone()
        targets_masked[~valid_mask] = 0
        targets_one_hot.scatter_(1, targets_masked.unsqueeze(1), 1)
        
        # Compute dice per class
        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
        
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()


class RegressionModel(EOModel):
    """Base class for regression models (e.g., LST estimation)."""
    
    def __init__(
        self,
        in_channels: int,
        output_range: Tuple[float, float] = (0, 1),
        name: str = "RegressionModel"
    ):
        super().__init__(name=name)
        self.in_channels = in_channels
        self.output_range = output_range
    
    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute regression loss (MSE + MAE).
        
        Args:
            predictions: Predicted values
            targets: Target values
            mask: Optional validity mask
            
        Returns:
            Loss tensor
        """
        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]
        
        mse = nn.functional.mse_loss(predictions, targets)
        mae = nn.functional.l1_loss(predictions, targets)
        
        return mse + 0.5 * mae
    
    def scale_output(self, x: torch.Tensor) -> torch.Tensor:
        """Scale output to target range."""
        min_val, max_val = self.output_range
        return x * (max_val - min_val) + min_val
