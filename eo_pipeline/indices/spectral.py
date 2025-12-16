"""
Spectral Indices Calculator

Provides calculation of various spectral indices for satellite imagery
including vegetation, water, built-up, and water quality indices.
"""

import torch
import numpy as np
from typing import Union, Dict, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Type alias for array-like inputs
ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass
class BandMapping:
    """Band mapping for different satellite sensors."""
    blue: int
    green: int
    red: int
    nir: int
    swir1: int
    swir2: int
    thermal: Optional[int] = None
    red_edge1: Optional[int] = None
    red_edge2: Optional[int] = None
    red_edge3: Optional[int] = None


# Default band mappings for common configurations
SENTINEL2_BANDS = BandMapping(
    blue=0,      # B02
    green=1,     # B03
    red=2,       # B04
    red_edge1=3, # B05
    red_edge2=4, # B06
    red_edge3=5, # B07
    nir=6,       # B08
    swir1=8,     # B11
    swir2=9,     # B12
)

LANDSAT_BANDS = BandMapping(
    blue=0,      # B2
    green=1,     # B3
    red=2,       # B4
    nir=3,       # B5
    swir1=4,     # B6
    swir2=5,     # B7
    thermal=6,   # B10
)


def _ensure_float(x: ArrayLike) -> ArrayLike:
    """Ensure array is float type."""
    if isinstance(x, torch.Tensor):
        return x.float()
    return x.astype(np.float32)


def _safe_divide(numerator: ArrayLike, denominator: ArrayLike, eps: float = 1e-8) -> ArrayLike:
    """Safe division avoiding divide by zero."""
    if isinstance(numerator, torch.Tensor):
        return numerator / (denominator + eps)
    return numerator / (denominator + eps)


# =============================================================================
# Vegetation Indices
# =============================================================================

def calculate_ndvi(nir: ArrayLike, red: ArrayLike) -> ArrayLike:
    """
    Calculate Normalized Difference Vegetation Index.
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Range: [-1, 1]
    - High values (>0.3): Dense vegetation
    - Low values (<0.1): Bare soil, water, built-up
    
    Args:
        nir: Near-infrared band
        red: Red band
        
    Returns:
        NDVI values
    """
    nir, red = _ensure_float(nir), _ensure_float(red)
    return _safe_divide(nir - red, nir + red)


def calculate_evi(nir: ArrayLike, red: ArrayLike, blue: ArrayLike) -> ArrayLike:
    """
    Calculate Enhanced Vegetation Index.
    
    EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    
    More sensitive in high biomass regions than NDVI.
    
    Args:
        nir: Near-infrared band
        red: Red band
        blue: Blue band
        
    Returns:
        EVI values
    """
    nir, red, blue = _ensure_float(nir), _ensure_float(red), _ensure_float(blue)
    return 2.5 * _safe_divide(
        nir - red,
        nir + 6 * red - 7.5 * blue + 1
    )


def calculate_savi(nir: ArrayLike, red: ArrayLike, L: float = 0.5) -> ArrayLike:
    """
    Calculate Soil Adjusted Vegetation Index.
    
    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    
    Minimizes soil brightness influences.
    
    Args:
        nir: Near-infrared band
        red: Red band
        L: Soil brightness correction factor (0-1)
        
    Returns:
        SAVI values
    """
    nir, red = _ensure_float(nir), _ensure_float(red)
    return _safe_divide(nir - red, nir + red + L) * (1 + L)


# =============================================================================
# Water Indices
# =============================================================================

def calculate_ndwi(green: ArrayLike, nir: ArrayLike) -> ArrayLike:
    """
    Calculate Normalized Difference Water Index (McFeeters).
    
    NDWI = (Green - NIR) / (Green + NIR)
    
    Range: [-1, 1]
    - Positive values: Water bodies
    - Negative values: Vegetation, soil
    
    Args:
        green: Green band
        nir: Near-infrared band
        
    Returns:
        NDWI values
    """
    green, nir = _ensure_float(green), _ensure_float(nir)
    return _safe_divide(green - nir, green + nir)


def calculate_mndwi(green: ArrayLike, swir: ArrayLike) -> ArrayLike:
    """
    Calculate Modified NDWI (Xu).
    
    MNDWI = (Green - SWIR) / (Green + SWIR)
    
    Better for urban areas and mixed pixels.
    
    Args:
        green: Green band
        swir: SWIR band (e.g., Sentinel-2 B11)
        
    Returns:
        MNDWI values
    """
    green, swir = _ensure_float(green), _ensure_float(swir)
    return _safe_divide(green - swir, green + swir)


def calculate_awei(
    green: ArrayLike, 
    nir: ArrayLike, 
    swir1: ArrayLike, 
    swir2: ArrayLike,
    shadow: bool = False
) -> ArrayLike:
    """
    Calculate Automated Water Extraction Index.
    
    AWEInsh = 4*(Green-SWIR1) - (0.25*NIR + 2.75*SWIR2)
    AWEIsh = Green + 2.5*Green - 1.5*(NIR+SWIR1) - 0.25*SWIR2
    
    Args:
        green: Green band
        nir: NIR band
        swir1: SWIR1 band
        swir2: SWIR2 band
        shadow: Use shadow formula if True
        
    Returns:
        AWEI values
    """
    green = _ensure_float(green)
    nir = _ensure_float(nir)
    swir1 = _ensure_float(swir1)
    swir2 = _ensure_float(swir2)
    
    if shadow:
        return green + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2
    else:
        return 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)


# =============================================================================
# Built-up / Urban Indices
# =============================================================================

def calculate_ndbi(swir: ArrayLike, nir: ArrayLike) -> ArrayLike:
    """
    Calculate Normalized Difference Built-up Index.
    
    NDBI = (SWIR - NIR) / (SWIR + NIR)
    
    Range: [-1, 1]
    - High values: Built-up areas
    - Low values: Vegetation
    
    Args:
        swir: SWIR band
        nir: NIR band
        
    Returns:
        NDBI values
    """
    swir, nir = _ensure_float(swir), _ensure_float(nir)
    return _safe_divide(swir - nir, swir + nir)


def calculate_ui(swir2: ArrayLike, nir: ArrayLike) -> ArrayLike:
    """
    Calculate Urban Index.
    
    UI = (SWIR2 - NIR) / (SWIR2 + NIR)
    
    Args:
        swir2: SWIR2 band
        nir: NIR band
        
    Returns:
        UI values
    """
    swir2, nir = _ensure_float(swir2), _ensure_float(nir)
    return _safe_divide(swir2 - nir, swir2 + nir)


# =============================================================================
# Water Quality / Pollution Indices
# =============================================================================

def calculate_turbidity_index(red: ArrayLike, green: ArrayLike) -> ArrayLike:
    """
    Calculate turbidity index for water quality.
    
    Higher values indicate more turbid water.
    
    Args:
        red: Red band
        green: Green band
        
    Returns:
        Turbidity index
    """
    red, green = _ensure_float(red), _ensure_float(green)
    return _safe_divide(red, green)


def calculate_chlorophyll_index(
    red: ArrayLike, 
    red_edge: ArrayLike
) -> ArrayLike:
    """
    Calculate chlorophyll index for algae detection.
    
    CI = (Red Edge / Red) - 1
    
    Useful for detecting algal blooms in water bodies.
    
    Args:
        red: Red band
        red_edge: Red edge band (e.g., Sentinel-2 B05)
        
    Returns:
        Chlorophyll index
    """
    red, red_edge = _ensure_float(red), _ensure_float(red_edge)
    return _safe_divide(red_edge, red) - 1


def calculate_plastic_index(
    nir: ArrayLike, 
    swir1: ArrayLike,
    swir2: ArrayLike
) -> ArrayLike:
    """
    Calculate Floating Debris/Plastic Index.
    
    Uses SWIR bands to detect floating plastics on water surface.
    Based on spectral characteristics of plastics in SWIR region.
    
    FDI ≈ NIR - (SWIR1 + SWIR2)/2 for floating debris
    
    Args:
        nir: NIR band
        swir1: SWIR1 band
        swir2: SWIR2 band
        
    Returns:
        Plastic/debris index
    """
    nir = _ensure_float(nir)
    swir1 = _ensure_float(swir1)
    swir2 = _ensure_float(swir2)
    
    return nir - (swir1 + swir2) / 2


def calculate_fai(
    red: ArrayLike,
    nir: ArrayLike, 
    swir1: ArrayLike
) -> ArrayLike:
    """
    Calculate Floating Algae Index.
    
    FAI = NIR - (Red + (SWIR1 - Red) * (NIR_wavelength - Red_wavelength) / 
                                        (SWIR1_wavelength - Red_wavelength))
    
    Detects floating vegetation, algae, and potentially plastics.
    
    Args:
        red: Red band
        nir: NIR band
        swir1: SWIR1 band
        
    Returns:
        FAI values
    """
    red = _ensure_float(red)
    nir = _ensure_float(nir)
    swir1 = _ensure_float(swir1)
    
    # Wavelength ratios (Sentinel-2 approximate)
    lambda_ratio = (842 - 665) / (1610 - 665)
    
    baseline = red + (swir1 - red) * lambda_ratio
    return nir - baseline


# =============================================================================
# Thermal / LST Indices
# =============================================================================

def calculate_lst(
    thermal: ArrayLike,
    emissivity: Optional[ArrayLike] = None,
    ndvi: Optional[ArrayLike] = None
) -> ArrayLike:
    """
    Calculate Land Surface Temperature from thermal band.
    
    If emissivity not provided, estimates from NDVI.
    
    Args:
        thermal: Brightness temperature (Kelvin)
        emissivity: Surface emissivity (0-1), optional
        ndvi: NDVI for emissivity estimation, optional
        
    Returns:
        LST in Kelvin
    """
    thermal = _ensure_float(thermal)
    
    # Estimate emissivity from NDVI if not provided
    if emissivity is None:
        if ndvi is None:
            # Assume moderate emissivity
            emissivity = 0.97
        else:
            emissivity = estimate_emissivity(ndvi)
    
    # Planck constant ratio
    rho = 1.438e-2  # m*K
    wavelength = 10.9e-6  # Landsat B10 wavelength
    
    if isinstance(thermal, torch.Tensor):
        lst = thermal / (1 + (wavelength * thermal / rho) * torch.log(emissivity))
    else:
        lst = thermal / (1 + (wavelength * thermal / rho) * np.log(emissivity + 1e-8))
    
    return lst


def estimate_emissivity(ndvi: ArrayLike) -> ArrayLike:
    """
    Estimate surface emissivity from NDVI.
    
    Args:
        ndvi: NDVI values
        
    Returns:
        Emissivity values (0-1)
    """
    ndvi = _ensure_float(ndvi)
    
    if isinstance(ndvi, torch.Tensor):
        emissivity = torch.zeros_like(ndvi)
        
        # Water
        emissivity = torch.where(ndvi < 0, torch.tensor(0.991), emissivity)
        # Bare soil
        emissivity = torch.where((ndvi >= 0) & (ndvi < 0.2), torch.tensor(0.966), emissivity)
        # Mixed
        pv = ((ndvi - 0.2) / 0.3) ** 2
        mixed_em = 0.966 + 0.018 * pv
        emissivity = torch.where((ndvi >= 0.2) & (ndvi < 0.5), mixed_em, emissivity)
        # Vegetation
        emissivity = torch.where(ndvi >= 0.5, torch.tensor(0.984), emissivity)
    else:
        emissivity = np.zeros_like(ndvi)
        
        emissivity[ndvi < 0] = 0.991  # Water
        emissivity[(ndvi >= 0) & (ndvi < 0.2)] = 0.966  # Bare soil
        
        mixed_mask = (ndvi >= 0.2) & (ndvi < 0.5)
        pv = ((ndvi[mixed_mask] - 0.2) / 0.3) ** 2
        emissivity[mixed_mask] = 0.966 + 0.018 * pv
        
        emissivity[ndvi >= 0.5] = 0.984  # Vegetation
    
    return emissivity


# =============================================================================
# Main Calculator Class
# =============================================================================

class SpectralIndices:
    """
    Calculator for multiple spectral indices.
    
    Example:
        >>> calc = SpectralIndices(band_mapping=SENTINEL2_BANDS)
        >>> indices = calc.calculate_all(image)
        >>> ndvi = calc.ndvi(image)
    """
    
    def __init__(
        self,
        band_mapping: BandMapping = SENTINEL2_BANDS,
        device: Optional[str] = None
    ):
        """
        Initialize spectral indices calculator.
        
        Args:
            band_mapping: Mapping of band names to indices
            device: Torch device for tensor operations
        """
        self.bands = band_mapping
        self.device = device
        
    def _get_band(self, image: ArrayLike, band_idx: int) -> ArrayLike:
        """Extract a band from image."""
        if image.ndim == 3:  # (C, H, W)
            return image[band_idx]
        elif image.ndim == 4:  # (T, C, H, W) or (B, C, H, W)
            return image[:, band_idx]
        else:
            raise ValueError(f"Unexpected image dimensions: {image.ndim}")
    
    def ndvi(self, image: ArrayLike) -> ArrayLike:
        """Calculate NDVI from multi-band image."""
        nir = self._get_band(image, self.bands.nir)
        red = self._get_band(image, self.bands.red)
        return calculate_ndvi(nir, red)
    
    def evi(self, image: ArrayLike) -> ArrayLike:
        """Calculate EVI from multi-band image."""
        nir = self._get_band(image, self.bands.nir)
        red = self._get_band(image, self.bands.red)
        blue = self._get_band(image, self.bands.blue)
        return calculate_evi(nir, red, blue)
    
    def ndwi(self, image: ArrayLike) -> ArrayLike:
        """Calculate NDWI from multi-band image."""
        green = self._get_band(image, self.bands.green)
        nir = self._get_band(image, self.bands.nir)
        return calculate_ndwi(green, nir)
    
    def mndwi(self, image: ArrayLike) -> ArrayLike:
        """Calculate MNDWI from multi-band image."""
        green = self._get_band(image, self.bands.green)
        swir = self._get_band(image, self.bands.swir1)
        return calculate_mndwi(green, swir)
    
    def ndbi(self, image: ArrayLike) -> ArrayLike:
        """Calculate NDBI from multi-band image."""
        swir = self._get_band(image, self.bands.swir1)
        nir = self._get_band(image, self.bands.nir)
        return calculate_ndbi(swir, nir)
    
    def plastic_index(self, image: ArrayLike) -> ArrayLike:
        """Calculate plastic/debris index."""
        nir = self._get_band(image, self.bands.nir)
        swir1 = self._get_band(image, self.bands.swir1)
        swir2 = self._get_band(image, self.bands.swir2)
        return calculate_plastic_index(nir, swir1, swir2)
    
    def lst(self, image: ArrayLike) -> ArrayLike:
        """Calculate LST (requires thermal band)."""
        if self.bands.thermal is None:
            raise ValueError("Thermal band not available in band mapping")
        thermal = self._get_band(image, self.bands.thermal)
        ndvi = self.ndvi(image)
        return calculate_lst(thermal, ndvi=ndvi)
    
    def calculate_all(
        self,
        image: ArrayLike,
        indices: Optional[List[str]] = None
    ) -> Dict[str, ArrayLike]:
        """
        Calculate multiple indices at once.
        
        Args:
            image: Multi-band image
            indices: List of index names to calculate (default: all available)
            
        Returns:
            Dictionary of index name to values
        """
        available = ["ndvi", "evi", "ndwi", "mndwi", "ndbi"]
        if self.bands.thermal is not None:
            available.append("lst")
        
        indices = indices or available
        
        results = {}
        for idx_name in indices:
            try:
                method = getattr(self, idx_name)
                results[idx_name] = method(image)
            except Exception as e:
                logger.warning(f"Failed to calculate {idx_name}: {e}")
        
        return results
    
    def stack_indices(
        self,
        image: ArrayLike,
        indices: List[str]
    ) -> ArrayLike:
        """
        Calculate and stack indices as additional bands.
        
        Args:
            image: Multi-band image (C, H, W)
            indices: List of indices to stack
            
        Returns:
            Image with indices stacked as new bands
        """
        index_arrays = self.calculate_all(image, indices)
        
        if isinstance(image, torch.Tensor):
            index_stack = torch.stack([index_arrays[i] for i in indices], dim=0)
            return torch.cat([image, index_stack], dim=0)
        else:
            index_stack = np.stack([index_arrays[i] for i in indices], axis=0)
            return np.concatenate([image, index_stack], axis=0)
