"""
Earth Observation Data Loader

Base classes for loading satellite imagery with support for
multitemporal data and tile-based processing.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import logging

from ..core.tile_grid import Tile, TileGrid, BoundingBox
from ..config.settings import Config, DataConfig

logger = logging.getLogger(__name__)


class SatelliteDataset(Dataset):
    """
    PyTorch Dataset for satellite imagery.
    
    Supports multitemporal data (time series of images) and
    multi-source data (multiple satellite sources).
    
    Example:
        >>> dataset = SatelliteDataset(tiles, data_source, transform=preprocess)
        >>> image, mask = dataset[0]
    """
    
    def __init__(
        self,
        tiles: List[Tile],
        data_source: "EODataLoader",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        time_steps: int = 1,
        return_metadata: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            tiles: List of tiles to load
            data_source: Data loader for fetching satellite data
            transform: Transform to apply to images
            target_transform: Transform to apply to targets/masks
            time_steps: Number of temporal observations per sample
            return_metadata: Whether to return tile metadata
        """
        self.tiles = tiles
        self.data_source = data_source
        self.transform = transform
        self.target_transform = target_transform
        self.time_steps = time_steps
        self.return_metadata = return_metadata
        
    def __len__(self) -> int:
        return len(self.tiles)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        tile = self.tiles[idx]
        
        # Load image data
        image = self.data_source.load_tile(tile, time_steps=self.time_steps)
        
        # Load target/mask if available
        target = self.data_source.load_target(tile)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if target is not None and self.target_transform is not None:
            target = self.target_transform(target)
        
        # Convert to tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if target is not None and isinstance(target, np.ndarray):
            target = torch.from_numpy(target).long()
        
        if self.return_metadata:
            metadata = {
                "tile_id": tile.id,
                "bbox": tile.bbox.to_tuple(),
                "row": tile.row,
                "col": tile.col
            }
            if target is not None:
                return image, target, metadata
            return image, metadata
        
        if target is not None:
            return image, target
        return image


class EODataLoader(ABC):
    """
    Abstract base class for Earth Observation data loading.
    
    Subclasses implement specific API integrations (SentinelHub, GEE, etc.)
    
    Example:
        >>> loader = SentinelHubLoader(config)
        >>> image = loader.load_tile(tile, time_steps=4)
        >>> dataloader = loader.get_dataloader(tiles, batch_size=16)
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize data loader.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def load_tile(
        self, 
        tile: Tile, 
        time_steps: int = 1,
        bands: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Load satellite data for a single tile.
        
        Args:
            tile: Tile to load
            time_steps: Number of temporal observations
            bands: Specific bands to load (None = all)
            
        Returns:
            Image array of shape (time_steps, bands, height, width) or (bands, height, width)
        """
        pass
    
    @abstractmethod
    def load_target(self, tile: Tile) -> Optional[np.ndarray]:
        """
        Load target/label data for a tile.
        
        Args:
            tile: Tile to load target for
            
        Returns:
            Target array or None if not available
        """
        pass
    
    @abstractmethod
    def authenticate(self) -> bool:
        """
        Authenticate with the data source API.
        
        Returns:
            True if authentication successful
        """
        pass
    
    def get_dataloader(
        self,
        tiles: List[Tile],
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 4,
        transform: Optional[Callable] = None,
        **kwargs
    ) -> DataLoader:
        """
        Create PyTorch DataLoader for tiles.
        
        Args:
            tiles: List of tiles to include
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of data loading workers
            transform: Transform to apply
            **kwargs: Additional DataLoader arguments
            
        Returns:
            PyTorch DataLoader
        """
        dataset = SatelliteDataset(
            tiles=tiles,
            data_source=self,
            transform=transform,
            time_steps=self.config.time_steps
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs
        )
    
    def get_cache_path(self, tile: Tile, suffix: str = "") -> Path:
        """Get cache file path for a tile."""
        return self.cache_dir / f"{tile.id}{suffix}.npy"
    
    def load_from_cache(self, tile: Tile, suffix: str = "") -> Optional[np.ndarray]:
        """Load tile data from cache if available."""
        if not self.config.use_cache:
            return None
            
        cache_path = self.get_cache_path(tile, suffix)
        if cache_path.exists():
            logger.debug(f"Loading from cache: {cache_path}")
            return np.load(cache_path)
        return None
    
    def save_to_cache(self, tile: Tile, data: np.ndarray, suffix: str = "") -> None:
        """Save tile data to cache."""
        if not self.config.use_cache:
            return
            
        cache_path = self.get_cache_path(tile, suffix)
        np.save(cache_path, data)
        logger.debug(f"Saved to cache: {cache_path}")


class MultiSourceDataLoader:
    """
    Combines data from multiple satellite sources.
    
    Example:
        >>> sentinel_loader = SentinelHubLoader(config)
        >>> landsat_loader = LandsatLoader(config)
        >>> multi_loader = MultiSourceDataLoader([sentinel_loader, landsat_loader])
    """
    
    def __init__(self, loaders: List[EODataLoader]):
        """
        Initialize multi-source loader.
        
        Args:
            loaders: List of data loaders for different sources
        """
        self.loaders = loaders
        
    def load_tile(
        self, 
        tile: Tile,
        time_steps: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Load data from all sources for a tile.
        
        Returns:
            Dictionary mapping source name to image array
        """
        data = {}
        for loader in self.loaders:
            source_name = loader.__class__.__name__
            try:
                data[source_name] = loader.load_tile(tile, time_steps)
            except Exception as e:
                logger.warning(f"Failed to load from {source_name}: {e}")
                data[source_name] = None
        return data
