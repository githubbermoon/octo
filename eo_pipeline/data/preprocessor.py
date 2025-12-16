"""
Data Preprocessing Pipeline

Provides normalization, augmentation, and transformation utilities
for satellite imagery preprocessing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NormalizationStats:
    """Statistics for normalization."""
    mean: np.ndarray
    std: np.ndarray
    min_val: Optional[np.ndarray] = None
    max_val: Optional[np.ndarray] = None


# Pre-computed statistics for common satellite data
SENTINEL2_STATS = NormalizationStats(
    mean=np.array([0.1257, 0.1131, 0.1015, 0.0979, 0.1210, 0.2045, 
                   0.2346, 0.2546, 0.2798, 0.1219]),
    std=np.array([0.0379, 0.0353, 0.0415, 0.0604, 0.0494, 0.0641,
                  0.0732, 0.0806, 0.0867, 0.0510])
)

LANDSAT_STATS = NormalizationStats(
    mean=np.array([0.1096, 0.1154, 0.1168, 0.1609, 0.2397, 0.2046, 293.0]),
    std=np.array([0.0458, 0.0533, 0.0658, 0.0867, 0.0865, 0.0726, 12.0])
)


class Preprocessor:
    """
    Preprocessing pipeline for satellite imagery.
    
    Handles normalization, augmentation, and data transforms.
    
    Example:
        >>> preprocessor = Preprocessor(normalize=True, augment=True)
        >>> transformed = preprocessor(image)
        >>> 
        >>> # Or as a transform
        >>> dataset = SatelliteDataset(tiles, loader, transform=preprocessor)
    """
    
    def __init__(
        self,
        normalize: bool = True,
        normalization_type: str = "standardize",  # "standardize", "minmax", "custom"
        stats: Optional[NormalizationStats] = None,
        augment: bool = True,
        augmentation_config: Optional[Dict] = None,
        cloud_mask: bool = False,
        cloud_threshold: float = 0.2,
        target_size: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            normalize: Whether to apply normalization
            normalization_type: Type of normalization
            stats: Pre-computed statistics for normalization
            augment: Whether to apply augmentation
            augmentation_config: Configuration for augmentation
            cloud_mask: Whether to apply cloud masking
            cloud_threshold: Threshold for cloud detection
            target_size: Target image size (height, width)
        """
        self.normalize = normalize
        self.normalization_type = normalization_type
        self.stats = stats or SENTINEL2_STATS
        self.augment = augment
        self.cloud_mask = cloud_mask
        self.cloud_threshold = cloud_threshold
        self.target_size = target_size
        
        # Build augmentation pipeline
        self.augmentation_config = augmentation_config or {
            "horizontal_flip": 0.5,
            "vertical_flip": 0.5,
            "rotation_90": 0.5,
            "brightness": 0.1,
            "contrast": 0.1,
        }
        
    def __call__(
        self, 
        image: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        is_training: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply preprocessing to image and optional mask.
        
        Args:
            image: Input image (C, H, W) or (T, C, H, W)
            mask: Optional mask (H, W)
            is_training: Whether in training mode (enables augmentation)
            
        Returns:
            Preprocessed image (and mask if provided)
        """
        # Convert to numpy if tensor
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        # Handle multitemporal data
        is_multitemporal = image.ndim == 4
        if is_multitemporal:
            # Process each time step
            processed = []
            for t in range(image.shape[0]):
                proc_t = self._process_single(image[t], is_training=is_training)
                processed.append(proc_t)
            image = np.stack(processed, axis=0)
        else:
            image = self._process_single(image, is_training=is_training)
        
        # Apply augmentation (same transform to image and mask)
        if self.augment and is_training:
            image, mask = self._apply_augmentation(image, mask)
        
        # Resize if needed
        if self.target_size is not None:
            image = self._resize(image, self.target_size)
            if mask is not None:
                mask = self._resize(mask, self.target_size, interpolation="nearest")
        
        if mask is not None:
            return image, mask
        return image
    
    def _process_single(
        self,
        image: np.ndarray,
        is_training: bool = True
    ) -> np.ndarray:
        """Process a single image (C, H, W)."""
        
        # Cloud masking (optional)
        if self.cloud_mask:
            image = self._apply_cloud_mask(image)
        
        # Normalization
        if self.normalize:
            image = self._apply_normalization(image)
        
        return image
    
    def _apply_normalization(self, image: np.ndarray) -> np.ndarray:
        """Apply normalization to image."""
        
        if self.normalization_type == "standardize":
            # Z-score normalization: (x - mean) / std
            mean = self.stats.mean[:image.shape[0], None, None]
            std = self.stats.std[:image.shape[0], None, None]
            return (image - mean) / (std + 1e-8)
        
        elif self.normalization_type == "minmax":
            # Min-max normalization to [0, 1]
            if self.stats.min_val is not None and self.stats.max_val is not None:
                min_val = self.stats.min_val[:image.shape[0], None, None]
                max_val = self.stats.max_val[:image.shape[0], None, None]
            else:
                min_val = image.min(axis=(1, 2), keepdims=True)
                max_val = image.max(axis=(1, 2), keepdims=True)
            return (image - min_val) / (max_val - min_val + 1e-8)
        
        elif self.normalization_type == "custom":
            # Custom normalization (override in subclass)
            return image
        
        return image
    
    def _apply_cloud_mask(self, image: np.ndarray) -> np.ndarray:
        """Apply cloud masking to image."""
        # Simple cloud detection using brightness threshold
        # In production, use proper cloud masks (e.g., SCL band)
        brightness = np.mean(image, axis=0)
        cloud_mask = brightness > self.cloud_threshold
        
        # Replace cloudy pixels with interpolated values or NaN
        for c in range(image.shape[0]):
            image[c][cloud_mask] = np.nan
        
        return image
    
    def _apply_augmentation(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply data augmentation."""
        
        # Horizontal flip
        if np.random.random() < self.augmentation_config.get("horizontal_flip", 0):
            image = np.flip(image, axis=-1).copy()
            if mask is not None:
                mask = np.flip(mask, axis=-1).copy()
        
        # Vertical flip
        if np.random.random() < self.augmentation_config.get("vertical_flip", 0):
            image = np.flip(image, axis=-2).copy()
            if mask is not None:
                mask = np.flip(mask, axis=-2).copy()
        
        # 90-degree rotation
        if np.random.random() < self.augmentation_config.get("rotation_90", 0):
            k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
            image = np.rot90(image, k, axes=(-2, -1)).copy()
            if mask is not None:
                mask = np.rot90(mask, k, axes=(-2, -1)).copy()
        
        # Brightness adjustment
        brightness_factor = self.augmentation_config.get("brightness", 0)
        if brightness_factor > 0:
            delta = np.random.uniform(-brightness_factor, brightness_factor)
            image = image + delta
        
        # Contrast adjustment
        contrast_factor = self.augmentation_config.get("contrast", 0)
        if contrast_factor > 0:
            factor = np.random.uniform(1 - contrast_factor, 1 + contrast_factor)
            mean = np.mean(image)
            image = (image - mean) * factor + mean
        
        return image, mask
    
    def _resize(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        interpolation: str = "bilinear"
    ) -> np.ndarray:
        """Resize image to target size."""
        try:
            import cv2
            
            if image.ndim == 3:  # (C, H, W)
                resized = []
                for c in range(image.shape[0]):
                    if interpolation == "nearest":
                        r = cv2.resize(image[c], target_size[::-1], interpolation=cv2.INTER_NEAREST)
                    else:
                        r = cv2.resize(image[c], target_size[::-1], interpolation=cv2.INTER_LINEAR)
                    resized.append(r)
                return np.stack(resized, axis=0)
            else:  # (H, W) - mask
                if interpolation == "nearest":
                    return cv2.resize(image, target_size[::-1], interpolation=cv2.INTER_NEAREST)
                else:
                    return cv2.resize(image, target_size[::-1], interpolation=cv2.INTER_LINEAR)
        except ImportError:
            logger.warning("OpenCV not available, skipping resize")
            return image
    
    def compute_statistics(
        self,
        images: List[np.ndarray]
    ) -> NormalizationStats:
        """
        Compute normalization statistics from a list of images.
        
        Args:
            images: List of images (C, H, W)
            
        Returns:
            Computed statistics
        """
        all_pixels = np.concatenate([img.reshape(img.shape[0], -1) for img in images], axis=1)
        
        return NormalizationStats(
            mean=np.mean(all_pixels, axis=1),
            std=np.std(all_pixels, axis=1),
            min_val=np.min(all_pixels, axis=1),
            max_val=np.max(all_pixels, axis=1)
        )


class ToTensor:
    """Convert numpy array to PyTorch tensor."""
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            return image
        return torch.from_numpy(image.copy()).float()


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
        
    def __call__(self, image, mask=None):
        for t in self.transforms:
            if mask is not None:
                result = t(image, mask)
                if isinstance(result, tuple):
                    image, mask = result
                else:
                    image = result
            else:
                image = t(image)
        
        if mask is not None:
            return image, mask
        return image
