"""
Tile Grid System - Spatial Management for Regional Modeling

Provides tile-based spatial approach for processing large satellite imagery
by dividing Areas of Interest (AOI) into manageable tiles.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterator, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Geographic bounding box in WGS84 coordinates."""
    min_lon: float  # West
    min_lat: float  # South
    max_lon: float  # East
    max_lat: float  # North
    
    @property
    def width(self) -> float:
        """Width in degrees."""
        return self.max_lon - self.min_lon
    
    @property
    def height(self) -> float:
        """Height in degrees."""
        return self.max_lat - self.min_lat
    
    @property
    def center(self) -> Tuple[float, float]:
        """Center point (lon, lat)."""
        return (
            (self.min_lon + self.max_lon) / 2,
            (self.min_lat + self.max_lat) / 2
        )
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return as (min_lon, min_lat, max_lon, max_lat)."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)
    
    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this bbox intersects with another."""
        return not (
            self.max_lon < other.min_lon or
            self.min_lon > other.max_lon or
            self.max_lat < other.min_lat or
            self.min_lat > other.max_lat
        )


@dataclass
class Tile:
    """
    Represents a single tile in the grid.
    
    Attributes:
        id: Unique tile identifier (row_col format)
        row: Row index in grid
        col: Column index in grid
        bbox: Geographic bounding box
        pixel_bounds: Pixel coordinates (x_min, y_min, x_max, y_max)
        metadata: Additional tile metadata
    """
    id: str
    row: int
    col: int
    bbox: BoundingBox
    pixel_bounds: Optional[Tuple[int, int, int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Tile center in geographic coordinates."""
        return self.bbox.center


class TileGrid:
    """
    Manages tile-based spatial partitioning for satellite imagery.
    
    Divides an Area of Interest (AOI) into a grid of tiles for efficient
    processing of large satellite images.
    
    Example:
        >>> aoi = BoundingBox(min_lon=72.8, min_lat=18.9, max_lon=73.1, max_lat=19.2)
        >>> grid = TileGrid(aoi, tile_size_meters=5000, overlap=0.1)
        >>> 
        >>> for tile in grid:
        ...     data = load_satellite_data(tile.bbox)
        ...     process(data)
    """
    
    # Approximate meters per degree at equator
    METERS_PER_DEGREE_LAT = 111320
    
    def __init__(
        self,
        aoi: BoundingBox,
        tile_size_meters: float = 5000,
        overlap: float = 0.0,
        crs: str = "EPSG:4326"
    ):
        """
        Initialize tile grid.
        
        Args:
            aoi: Area of Interest bounding box
            tile_size_meters: Size of each tile in meters
            overlap: Overlap between tiles (0.0 to 0.5)
            crs: Coordinate Reference System (default WGS84)
        """
        self.aoi = aoi
        self.tile_size_meters = tile_size_meters
        self.overlap = max(0.0, min(0.5, overlap))  # Clamp to valid range
        self.crs = crs
        
        # Calculate tile sizes in degrees
        self._tile_size_lat = tile_size_meters / self.METERS_PER_DEGREE_LAT
        self._tile_size_lon = tile_size_meters / (
            self.METERS_PER_DEGREE_LAT * math.cos(math.radians(aoi.center[1]))
        )
        
        # Calculate stride (accounting for overlap)
        self._stride_lat = self._tile_size_lat * (1 - self.overlap)
        self._stride_lon = self._tile_size_lon * (1 - self.overlap)
        
        # Calculate grid dimensions
        self.n_rows = max(1, math.ceil(aoi.height / self._stride_lat))
        self.n_cols = max(1, math.ceil(aoi.width / self._stride_lon))
        
        # Generate tiles
        self._tiles: List[Tile] = self._generate_tiles()
        
        logger.info(
            f"Created TileGrid: {self.n_rows}x{self.n_cols} tiles "
            f"({len(self._tiles)} total), tile size: {tile_size_meters}m"
        )
    
    def _generate_tiles(self) -> List[Tile]:
        """Generate all tiles for the AOI."""
        tiles = []
        
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                min_lat = self.aoi.min_lat + row * self._stride_lat
                min_lon = self.aoi.min_lon + col * self._stride_lon
                max_lat = min(min_lat + self._tile_size_lat, self.aoi.max_lat)
                max_lon = min(min_lon + self._tile_size_lon, self.aoi.max_lon)
                
                tile = Tile(
                    id=f"{row}_{col}",
                    row=row,
                    col=col,
                    bbox=BoundingBox(min_lon, min_lat, max_lon, max_lat)
                )
                tiles.append(tile)
        
        return tiles
    
    def __len__(self) -> int:
        """Number of tiles in grid."""
        return len(self._tiles)
    
    def __iter__(self) -> Iterator[Tile]:
        """Iterate over all tiles."""
        return iter(self._tiles)
    
    def __getitem__(self, idx: int) -> Tile:
        """Get tile by index."""
        return self._tiles[idx]
    
    def get_tile(self, row: int, col: int) -> Optional[Tile]:
        """Get tile by row and column indices."""
        if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
            return self._tiles[row * self.n_cols + col]
        return None
    
    def get_tiles_for_point(self, lon: float, lat: float) -> List[Tile]:
        """Get all tiles containing a specific point."""
        point_bbox = BoundingBox(lon, lat, lon, lat)
        return [tile for tile in self._tiles if tile.bbox.intersects(point_bbox)]
    
    def get_tiles_for_bbox(self, bbox: BoundingBox) -> List[Tile]:
        """Get all tiles intersecting with a bounding box."""
        return [tile for tile in self._tiles if tile.bbox.intersects(bbox)]
    
    def set_pixel_bounds(self, image_width: int, image_height: int) -> None:
        """
        Calculate pixel bounds for each tile based on image dimensions.
        
        Args:
            image_width: Width of the full image in pixels
            image_height: Height of the full image in pixels
        """
        pixels_per_degree_lon = image_width / self.aoi.width
        pixels_per_degree_lat = image_height / self.aoi.height
        
        for tile in self._tiles:
            x_min = int((tile.bbox.min_lon - self.aoi.min_lon) * pixels_per_degree_lon)
            y_min = int((self.aoi.max_lat - tile.bbox.max_lat) * pixels_per_degree_lat)
            x_max = int((tile.bbox.max_lon - self.aoi.min_lon) * pixels_per_degree_lon)
            y_max = int((self.aoi.max_lat - tile.bbox.min_lat) * pixels_per_degree_lat)
            
            tile.pixel_bounds = (
                max(0, x_min),
                max(0, y_min),
                min(image_width, x_max),
                min(image_height, y_max)
            )
    
    def to_geojson(self) -> Dict[str, Any]:
        """Export grid as GeoJSON FeatureCollection."""
        features = []
        for tile in self._tiles:
            bbox = tile.bbox
            feature = {
                "type": "Feature",
                "properties": {
                    "tile_id": tile.id,
                    "row": tile.row,
                    "col": tile.col,
                    **tile.metadata
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [bbox.min_lon, bbox.min_lat],
                        [bbox.max_lon, bbox.min_lat],
                        [bbox.max_lon, bbox.max_lat],
                        [bbox.min_lon, bbox.max_lat],
                        [bbox.min_lon, bbox.min_lat]
                    ]]
                }
            }
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    @property
    def info(self) -> Dict[str, Any]:
        """Get grid information summary."""
        return {
            "aoi": self.aoi.to_tuple(),
            "tile_size_meters": self.tile_size_meters,
            "overlap": self.overlap,
            "grid_shape": (self.n_rows, self.n_cols),
            "total_tiles": len(self._tiles),
            "crs": self.crs
        }
