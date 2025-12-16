"""Model architectures for Earth Observation."""

from .base import EOModel
from .lulc import LULCClassifier, UNetEncoder, UNetDecoder
from .lst import LSTEstimator
from .water_quality import WaterQualityDetector

__all__ = [
    "EOModel",
    "LULCClassifier",
    "UNetEncoder",
    "UNetDecoder",
    "LSTEstimator",
    "WaterQualityDetector"
]
