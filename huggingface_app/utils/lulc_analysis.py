"""
LULC Change Detection Module
============================

Detects land use/land cover changes between two time periods.
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Class definitions
LULC_CLASSES = {
    0: {"name": "Background", "color": [0, 0, 0]},
    1: {"name": "Urban/Built-up", "color": [255, 0, 0]},
    2: {"name": "Vegetation", "color": [0, 255, 0]},
    3: {"name": "Water", "color": [0, 0, 255]},
    4: {"name": "Bare Soil", "color": [139, 90, 43]},
    5: {"name": "Agriculture", "color": [255, 255, 0]},
}

CHANGE_COLORS = {
    "no_change": [128, 128, 128],
    "urban_expansion": [255, 0, 0],
    "vegetation_loss": [255, 165, 0],
    "vegetation_gain": [0, 255, 0],
    "water_change": [0, 100, 255],
}


def analyze_lulc_change(
    before_image: np.ndarray,
    after_image: np.ndarray,
    sensitivity: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Any]:
    """
    Detect land use changes between two images.
    
    Args:
        before_image: Earlier time period image
        after_image: Later time period image
        sensitivity: Change detection sensitivity (0-1)
        
    Returns:
        before_seg: Classification for T1
        after_seg: Classification for T2
        change_map: Color-coded change map
        stats: Change statistics
        plot: Land use comparison plot
    """
    if before_image is None or after_image is None:
        return None, None, None, {}, None
    
    # Classify both images (using dummy classifier)
    before_classes = classify_image(before_image)
    after_classes = classify_image(after_image)
    
    # Create colored segmentation maps
    before_seg = colorize_classification(before_classes)
    after_seg = colorize_classification(after_classes)
    
    # Detect changes
    change_map, change_stats = detect_changes(before_classes, after_classes, sensitivity)
    
    # Calculate statistics
    stats = calculate_change_statistics(before_classes, after_classes, change_stats)
    
    # Create comparison plot
    plot = create_comparison_plot(before_classes, after_classes)
    
    return before_seg, after_seg, change_map, stats, plot


def classify_image(image: np.ndarray) -> np.ndarray:
    """
    Classify image into LULC classes.
    
    This is a placeholder using simple color-based rules.
    Real implementation would use the trained LULCClassifier model.
    """
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    
    h, w = image.shape[:2]
    classes = np.zeros((h, w), dtype=np.uint8)
    
    # Normalize
    img = image.astype(np.float32)
    if img.max() > 1:
        img = img / 255.0
    
    # Simple spectral rules (placeholder)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    
    # Vegetation (high green, moderate red)
    ndvi_proxy = (g - r) / (g + r + 1e-8)
    classes[ndvi_proxy > 0.2] = 2  # Vegetation
    
    # Water (high blue, low others)
    water_mask = (b > r + 0.1) & (b > g + 0.1)
    classes[water_mask] = 3  # Water
    
    # Urban (similar RGB, relatively bright)
    gray = np.mean(img, axis=-1)
    urban_mask = (np.std(img, axis=-1) < 0.1) & (gray > 0.4)
    classes[urban_mask] = 1  # Urban
    
    # Bare soil (high red, low green, low blue)
    bare_mask = (r > g + 0.1) & (r > b + 0.1) & (gray < 0.6)
    classes[bare_mask] = 4  # Bare Soil
    
    # Agriculture (moderate green, some variation)
    agri_mask = (ndvi_proxy > 0.1) & (ndvi_proxy < 0.3) & (gray > 0.3)
    classes[agri_mask] = 5  # Agriculture
    
    return classes


def colorize_classification(classes: np.ndarray) -> np.ndarray:
    """Convert classification to colored image."""
    h, w = classes.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, info in LULC_CLASSES.items():
        colored[classes == class_id] = info["color"]
    
    return colored


def detect_changes(
    before: np.ndarray,
    after: np.ndarray,
    sensitivity: float
) -> Tuple[np.ndarray, Dict]:
    """Detect and categorize changes."""
    h, w = before.shape
    change_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Initialize with no change (gray)
    change_map[:, :] = CHANGE_COLORS["no_change"]
    
    # Find changed pixels
    changed = before != after
    
    # Categorize changes
    change_counts = {
        "urban_expansion": 0,
        "vegetation_loss": 0,
        "vegetation_gain": 0,
        "water_change": 0,
        "other_changes": 0,
    }
    
    # Urban expansion (non-urban to urban)
    urban_expansion = changed & (before != 1) & (after == 1)
    change_map[urban_expansion] = CHANGE_COLORS["urban_expansion"]
    change_counts["urban_expansion"] = int(np.sum(urban_expansion))
    
    # Vegetation loss (vegetation to non-vegetation)
    veg_loss = changed & (before == 2) & (after != 2)
    change_map[veg_loss] = CHANGE_COLORS["vegetation_loss"]
    change_counts["vegetation_loss"] = int(np.sum(veg_loss))
    
    # Vegetation gain (non-vegetation to vegetation)
    veg_gain = changed & (before != 2) & (after == 2)
    change_map[veg_gain] = CHANGE_COLORS["vegetation_gain"]
    change_counts["vegetation_gain"] = int(np.sum(veg_gain))
    
    # Water changes
    water_change = changed & ((before == 3) | (after == 3))
    change_map[water_change] = CHANGE_COLORS["water_change"]
    change_counts["water_change"] = int(np.sum(water_change))
    
    return change_map, change_counts


def calculate_change_statistics(
    before: np.ndarray,
    after: np.ndarray,
    change_counts: Dict
) -> Dict[str, Any]:
    """Calculate change statistics."""
    total_pixels = before.size
    
    # Class areas
    before_areas = {}
    after_areas = {}
    
    for class_id, info in LULC_CLASSES.items():
        before_pct = float(np.mean(before == class_id) * 100)
        after_pct = float(np.mean(after == class_id) * 100)
        before_areas[info["name"]] = f"{before_pct:.1f}%"
        after_areas[info["name"]] = f"{after_pct:.1f}%"
    
    # Total change
    total_change = float(np.mean(before != after) * 100)
    
    stats = {
        "Total Changed Area": f"{total_change:.1f}%",
        "Urban Expansion": f"{change_counts['urban_expansion'] / total_pixels * 100:.2f}%",
        "Vegetation Loss": f"{change_counts['vegetation_loss'] / total_pixels * 100:.2f}%",
        "Vegetation Gain": f"{change_counts['vegetation_gain'] / total_pixels * 100:.2f}%",
        "Before Areas": before_areas,
        "After Areas": after_areas,
    }
    
    return stats


def create_comparison_plot(before: np.ndarray, after: np.ndarray):
    """Create land use comparison bar plot."""
    import plotly.graph_objects as go
    
    classes = []
    before_pcts = []
    after_pcts = []
    
    for class_id, info in LULC_CLASSES.items():
        if class_id == 0:
            continue
        classes.append(info["name"])
        before_pcts.append(float(np.mean(before == class_id) * 100))
        after_pcts.append(float(np.mean(after == class_id) * 100))
    
    fig = go.Figure(data=[
        go.Bar(name="Before", x=classes, y=before_pcts, marker_color="lightblue"),
        go.Bar(name="After", x=classes, y=after_pcts, marker_color="coral"),
    ])
    
    fig.update_layout(
        title="Land Use Comparison",
        xaxis_title="Land Use Class",
        yaxis_title="Coverage (%)",
        barmode="group",
        template="plotly_white",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig


def create_change_map(
    before: np.ndarray,
    after: np.ndarray
) -> np.ndarray:
    """Create detailed change transition map."""
    change_map, _ = detect_changes(before, after, 0.3)
    return change_map
