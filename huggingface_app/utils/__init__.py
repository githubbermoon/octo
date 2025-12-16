"""
Utility modules for EO AI Demo
"""

from .uhi_analysis import analyze_uhi, create_uhi_visualization
from .lulc_analysis import analyze_lulc_change, create_change_map
from .plastic_detection import detect_plastics, create_plastic_overlay
from .xai_visualization import (
    generate_shap_explanation,
    generate_lime_explanation,
    generate_gradcam_explanation
)
from .sample_loader import load_sample, get_sample_images

__all__ = [
    "analyze_uhi",
    "create_uhi_visualization",
    "analyze_lulc_change",
    "create_change_map",
    "detect_plastics",
    "create_plastic_overlay",
    "generate_shap_explanation",
    "generate_lime_explanation",
    "generate_gradcam_explanation",
    "load_sample",
    "get_sample_images",
]
