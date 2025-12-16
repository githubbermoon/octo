"""
Urban Heat Island Analysis Module
=================================

Analyzes thermal satellite imagery to detect Urban Heat Islands.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
def get_plt():
    import matplotlib.pyplot as plt
    return plt

def get_plotly():
    import plotly.graph_objects as go
    import plotly.express as px
    return go, px


def analyze_uhi(
    thermal_image: np.ndarray,
    threshold_sigma: float = 1.5,
    colormap: str = "hot"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Any]:
    """
    Analyze thermal image for Urban Heat Island detection.
    
    Args:
        thermal_image: Input thermal band image (H, W) or (H, W, C)
        threshold_sigma: Standard deviations above mean for hotspot detection
        colormap: Matplotlib colormap name
        
    Returns:
        heatmap: Colored temperature map
        hotspots: Binary hotspot overlay
        stats: Statistics dictionary
        plot: Temperature distribution plot
    """
    if thermal_image is None:
        return None, None, {}, None
    
    plt = get_plt()
    
    # Convert to grayscale if needed
    if thermal_image.ndim == 3:
        thermal = np.mean(thermal_image, axis=-1)
    else:
        thermal = thermal_image.astype(np.float32)
    
    # Normalize to simulate LST (placeholder - real LST requires calibration)
    # Typical LST range: 280K - 330K (7°C - 57°C)
    thermal_norm = (thermal - thermal.min()) / (thermal.max() - thermal.min() + 1e-8)
    lst_kelvin = 280 + thermal_norm * 50  # Simulated LST in Kelvin
    lst_celsius = lst_kelvin - 273.15
    
    # Calculate statistics
    mean_temp = float(np.mean(lst_celsius))
    std_temp = float(np.std(lst_celsius))
    min_temp = float(np.min(lst_celsius))
    max_temp = float(np.max(lst_celsius))
    
    # Detect hotspots (pixels > mean + threshold * std)
    hotspot_threshold = mean_temp + threshold_sigma * std_temp
    hotspots_mask = lst_celsius > hotspot_threshold
    hotspot_area = float(np.mean(hotspots_mask) * 100)  # Percentage
    
    # UHI intensity (difference from mean)
    uhi_intensity = np.where(hotspots_mask, lst_celsius - mean_temp, 0)
    max_uhi = float(np.max(uhi_intensity))
    
    # Create heatmap visualization
    heatmap = create_uhi_visualization(lst_celsius, colormap)
    
    # Create hotspot overlay
    hotspot_image = create_hotspot_overlay(thermal_image, hotspots_mask, lst_celsius)
    
    # Statistics
    stats = {
        "Mean Temperature": f"{mean_temp:.2f}°C",
        "Max Temperature": f"{max_temp:.2f}°C",
        "Min Temperature": f"{min_temp:.2f}°C",
        "Std Deviation": f"{std_temp:.2f}°C",
        "Hotspot Threshold": f"{hotspot_threshold:.2f}°C",
        "Hotspot Area": f"{hotspot_area:.1f}%",
        "Max UHI Intensity": f"{max_uhi:.2f}°C",
    }
    
    # Create distribution plot
    plot = create_temperature_distribution(lst_celsius, hotspot_threshold)
    
    return heatmap, hotspot_image, stats, plot


def create_uhi_visualization(
    lst_celsius: np.ndarray,
    colormap: str = "hot"
) -> np.ndarray:
    """Create colored temperature heatmap."""
    plt = get_plt()
    
    # Normalize for colormap
    vmin, vmax = np.percentile(lst_celsius, [2, 98])
    normalized = np.clip((lst_celsius - vmin) / (vmax - vmin + 1e-8), 0, 1)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
    
    return colored


def create_hotspot_overlay(
    original: np.ndarray,
    hotspots: np.ndarray,
    lst: np.ndarray
) -> np.ndarray:
    """Create hotspot overlay on original image."""
    
    # Ensure RGB
    if original.ndim == 2:
        base = np.stack([original] * 3, axis=-1)
    else:
        base = original.copy()
    
    # Normalize base image
    base = ((base - base.min()) / (base.max() - base.min() + 1e-8) * 255).astype(np.uint8)
    
    # Create overlay
    overlay = base.copy()
    
    # Color hotspots (red gradient based on intensity)
    if hotspots.any():
        intensity = (lst - lst.min()) / (lst.max() - lst.min() + 1e-8)
        overlay[hotspots, 0] = np.clip(overlay[hotspots, 0] * 0.3 + 255 * intensity[hotspots], 0, 255).astype(np.uint8)
        overlay[hotspots, 1] = (overlay[hotspots, 1] * 0.3).astype(np.uint8)
        overlay[hotspots, 2] = (overlay[hotspots, 2] * 0.3).astype(np.uint8)
    
    return overlay


def create_temperature_distribution(
    lst_celsius: np.ndarray,
    threshold: float
) -> Any:
    """Create temperature distribution plot."""
    go, px = get_plotly()
    
    # Flatten and sample for performance
    flat = lst_celsius.flatten()
    if len(flat) > 10000:
        flat = np.random.choice(flat, 10000, replace=False)
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=flat,
        nbinsx=50,
        name="Temperature",
        marker_color="coral",
        opacity=0.7
    ))
    
    # Threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Hotspot Threshold: {threshold:.1f}°C"
    )
    
    # Mean line
    mean_val = float(np.mean(lst_celsius))
    fig.add_vline(
        x=mean_val,
        line_dash="solid",
        line_color="green",
        annotation_text=f"Mean: {mean_val:.1f}°C"
    )
    
    fig.update_layout(
        title="Temperature Distribution",
        xaxis_title="Temperature (°C)",
        yaxis_title="Frequency",
        template="plotly_white",
        showlegend=False,
        height=300
    )
    
    return fig


def compute_lst_from_landsat(
    band10: np.ndarray,
    band11: Optional[np.ndarray] = None,
    emissivity: float = 0.95
) -> np.ndarray:
    """
    Compute Land Surface Temperature from Landsat-8 thermal bands.
    
    This is a simplified split-window algorithm placeholder.
    Real implementation requires:
    - Radiance calibration using MTL file
    - Atmospheric correction
    - Emissivity estimation from NDVI
    
    Args:
        band10: Landsat-8 Band 10 (TIRS 1)
        band11: Landsat-8 Band 11 (TIRS 2) - optional
        emissivity: Surface emissivity (0.95 default for urban areas)
        
    Returns:
        LST in Kelvin
    """
    # Placeholder: scale DN to approximate brightness temperature
    # Real calculation: BT = K2 / ln(K1/radiance + 1)
    # Where K1=774.8853, K2=1321.0789 for Band 10
    
    if band10.max() > 1:
        bt = 273.15 + (band10 / 65535.0) * 60  # Approximate 0-60°C range
    else:
        bt = 273.15 + band10 * 60
    
    # Simple emissivity correction
    # LST = BT / (1 + (λ * BT / ρ) * ln(ε))
    # Simplified: LST ≈ BT / ε^0.25
    lst = bt / (emissivity ** 0.25)
    
    return lst
