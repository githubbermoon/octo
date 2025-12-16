"""
Floating Plastic Detection Module
=================================

Detects floating debris and plastic pollution in water bodies
using spectral indices.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def detect_plastics(
    image: np.ndarray,
    detection_index: str = "Combined",
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Any]:
    """
    Detect floating plastics in water bodies.
    
    Args:
        image: Input satellite image (H, W, C)
        detection_index: Index to use (NDPI, FAI, FDI, Combined)
        threshold: Detection threshold
        
    Returns:
        water_mask: Binary water mask
        detection_mask: Plastic detection visualization
        overlay: Detection overlay on original
        stats: Detection statistics
        plot: Index distribution plot
    """
    if image is None:
        return None, None, None, {}, None
    
    # Ensure float and normalize
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    
    img = image.astype(np.float32)
    if img.max() > 1:
        img = img / 255.0
    
    h, w = img.shape[:2]
    
    # Extract pseudo-bands (placeholder - real data would have actual bands)
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    
    # Simulate NIR and SWIR from visible (placeholder)
    nir = np.clip(0.5 * (red + green) - 0.2 * blue + np.random.rand(h, w) * 0.1, 0, 1)
    swir = np.clip(0.4 * red + 0.3 * nir - 0.2 * blue + np.random.rand(h, w) * 0.1, 0, 1)
    
    # Detect water
    water_mask = detect_water(red, green, blue, nir)
    
    # Calculate plastic indices
    indices = calculate_plastic_indices(red, green, blue, nir, swir)
    
    # Select index based on method
    if detection_index == "NDPI":
        index_values = indices["ndpi"]
    elif detection_index == "FAI":
        index_values = indices["fai"]
    elif detection_index == "FDI":
        index_values = indices["fdi"]
    else:  # Combined
        index_values = (indices["ndpi"] + indices["fai"] + indices["fdi"]) / 3
    
    # Apply threshold within water areas
    plastic_mask = water_mask & (index_values > threshold)
    
    # Create visualizations
    water_vis = create_water_visualization(water_mask)
    plastic_vis = create_plastic_visualization(plastic_mask, index_values, water_mask)
    overlay = create_plastic_overlay(image, plastic_mask, water_mask)
    
    # Calculate statistics
    stats = calculate_plastic_statistics(water_mask, plastic_mask, index_values)
    
    # Create distribution plot
    plot = create_index_distribution(index_values, water_mask, threshold)
    
    return water_vis, plastic_vis, overlay, stats, plot


def detect_water(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    nir: np.ndarray
) -> np.ndarray:
    """Detect water pixels using NDWI-like index."""
    eps = 1e-8
    
    # Modified NDWI
    ndwi = (green - nir) / (green + nir + eps)
    
    # Water mask (NDWI > 0 typically indicates water)
    water_mask = ndwi > 0
    
    # Additional check: water is usually darker in NIR
    water_mask = water_mask & (nir < 0.3)
    
    return water_mask


def calculate_plastic_indices(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray
) -> Dict[str, np.ndarray]:
    """Calculate various plastic detection indices."""
    eps = 1e-8
    
    # Normalized Difference Plastic Index (NDPI)
    # High values indicate potential floating debris
    ndpi = (nir - blue) / (nir + blue + eps)
    
    # Floating Algae Index (FAI)
    # Also captures floating debris
    red_edge = (red + nir) / 2  # Approximate red edge
    fai = nir - (red + (swir - red) * (0.8 - 0.6) / (1.6 - 0.6))
    fai = (fai - fai.min()) / (fai.max() - fai.min() + eps)  # Normalize
    
    # Floating Debris Index (FDI)
    fdi = nir - (red_edge + swir) / 2
    fdi = (fdi - fdi.min()) / (fdi.max() - fdi.min() + eps)  # Normalize
    
    # Plastic Index (simple ratio)
    pi = (blue - green) / (blue + green + eps)
    
    return {
        "ndpi": ndpi,
        "fai": fai,
        "fdi": fdi,
        "pi": pi,
    }


def create_water_visualization(water_mask: np.ndarray) -> np.ndarray:
    """Create colored water mask."""
    h, w = water_mask.shape
    water_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Water in blue
    water_vis[water_mask, 0] = 0
    water_vis[water_mask, 1] = 100
    water_vis[water_mask, 2] = 200
    
    # Land in green
    water_vis[~water_mask, 0] = 50
    water_vis[~water_mask, 1] = 100
    water_vis[~water_mask, 2] = 50
    
    return water_vis


def create_plastic_visualization(
    plastic_mask: np.ndarray,
    index_values: np.ndarray,
    water_mask: np.ndarray
) -> np.ndarray:
    """Create plastic detection visualization."""
    h, w = plastic_mask.shape
    plastic_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Water background
    plastic_vis[water_mask, 2] = 100
    
    # Normalize index for intensity
    idx_norm = (index_values - index_values.min()) / (index_values.max() - index_values.min() + 1e-8)
    
    # Plastic detections in red-yellow gradient
    if plastic_mask.any():
        plastic_vis[plastic_mask, 0] = 255
        plastic_vis[plastic_mask, 1] = (idx_norm[plastic_mask] * 200).astype(np.uint8)
        plastic_vis[plastic_mask, 2] = 0
    
    # Land
    plastic_vis[~water_mask, 0] = 50
    plastic_vis[~water_mask, 1] = 50
    plastic_vis[~water_mask, 2] = 50
    
    return plastic_vis


def create_plastic_overlay(
    original: np.ndarray,
    plastic_mask: np.ndarray,
    water_mask: np.ndarray
) -> np.ndarray:
    """Create overlay of plastic detections on original image."""
    # Normalize original
    overlay = original.copy().astype(np.float32)
    if overlay.max() > 1:
        overlay = overlay / 255.0
    
    overlay = (overlay * 255).astype(np.uint8)
    
    # Highlight water areas (subtle blue tint)
    overlay[water_mask, 2] = np.clip(overlay[water_mask, 2] * 0.7 + 64, 0, 255).astype(np.uint8)
    
    # Highlight plastic detections (bright red)
    if plastic_mask.any():
        overlay[plastic_mask, 0] = 255
        overlay[plastic_mask, 1] = (overlay[plastic_mask, 1] * 0.3).astype(np.uint8)
        overlay[plastic_mask, 2] = (overlay[plastic_mask, 2] * 0.3).astype(np.uint8)
    
    return overlay


def calculate_plastic_statistics(
    water_mask: np.ndarray,
    plastic_mask: np.ndarray,
    index_values: np.ndarray
) -> Dict[str, Any]:
    """Calculate detection statistics."""
    
    total_pixels = water_mask.size
    water_pixels = int(np.sum(water_mask))
    plastic_pixels = int(np.sum(plastic_mask))
    
    water_pct = water_pixels / total_pixels * 100
    plastic_in_water = plastic_pixels / water_pixels * 100 if water_pixels > 0 else 0
    
    # Index statistics within water
    water_indices = index_values[water_mask] if water_mask.any() else np.array([0])
    
    stats = {
        "Water Coverage": f"{water_pct:.1f}%",
        "Plastic Detections": f"{plastic_pixels:,} pixels",
        "Plastic in Water": f"{plastic_in_water:.2f}%",
        "Mean Index (Water)": f"{float(np.mean(water_indices)):.4f}",
        "Max Index (Water)": f"{float(np.max(water_indices)):.4f}",
        "Detection Confidence": f"{min(plastic_in_water * 5, 100):.0f}%",
    }
    
    return stats


def create_index_distribution(
    index_values: np.ndarray,
    water_mask: np.ndarray,
    threshold: float
):
    """Create index distribution plot."""
    import plotly.graph_objects as go
    
    # Get water pixel values
    water_values = index_values[water_mask]
    
    if len(water_values) > 10000:
        water_values = np.random.choice(water_values, 10000, replace=False)
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=water_values,
        nbinsx=50,
        name="Index Values",
        marker_color="steelblue",
        opacity=0.7
    ))
    
    # Threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {threshold:.2f}"
    )
    
    fig.update_layout(
        title="Plastic Index Distribution (Water Pixels)",
        xaxis_title="Index Value",
        yaxis_title="Frequency",
        template="plotly_white",
        height=300,
        showlegend=False
    )
    
    return fig
