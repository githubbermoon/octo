"""
Earth Observation AI Pipeline
A modular PyTorch-based framework for satellite imagery analysis.

Modules:
    - core: Device management, tile/grid system
    - config: Configuration management
    - data: Data loading and preprocessing
    - indices: Spectral index calculations
    - models: ML models for LULC, LST, water quality
    - training: Model training and evaluation
    - explainability: XAI methods (SHAP, LIME, GradCAM)
    - integrations: MLFlow, DVC, Gradio hooks
"""

__version__ = "0.1.0"
__author__ = "Earth Observation AI Team"

from .config.settings import Config
from .core.device import DeviceManager
