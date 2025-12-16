"""
Device Manager - Hardware Fallback Logic

Provides automatic device selection with priority: CUDA > MPS > CPU
Supports mixed precision training and memory management.
"""

import torch
from typing import Optional, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Information about the selected compute device."""
    device: torch.device
    device_type: Literal["cuda", "mps", "cpu"]
    device_name: str
    supports_amp: bool  # Automatic Mixed Precision
    memory_gb: Optional[float] = None


class DeviceManager:
    """
    Manages compute device selection with automatic fallback.
    
    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
    
    Example:
        >>> device_info = DeviceManager.get_device_info()
        >>> model = model.to(device_info.device)
        >>> 
        >>> # Or simply:
        >>> device = DeviceManager.get_device()
        >>> tensor = tensor.to(device)
    """
    
    _cached_device: Optional[torch.device] = None
    _cached_info: Optional[DeviceInfo] = None
    
    @classmethod
    def get_device(cls, force_cpu: bool = False) -> torch.device:
        """
        Get the best available compute device.
        
        Args:
            force_cpu: If True, always return CPU device.
            
        Returns:
            torch.device: The selected device.
        """
        if force_cpu:
            return torch.device("cpu")
            
        if cls._cached_device is not None:
            return cls._cached_device
            
        cls._cached_device = cls._detect_best_device()
        return cls._cached_device
    
    @classmethod
    def get_device_info(cls, force_cpu: bool = False) -> DeviceInfo:
        """
        Get detailed information about the compute device.
        
        Args:
            force_cpu: If True, return CPU device info.
            
        Returns:
            DeviceInfo: Detailed device information.
        """
        if force_cpu:
            return DeviceInfo(
                device=torch.device("cpu"),
                device_type="cpu",
                device_name="CPU",
                supports_amp=False
            )
            
        if cls._cached_info is not None:
            return cls._cached_info
            
        device = cls.get_device()
        cls._cached_info = cls._build_device_info(device)
        return cls._cached_info
    
    @classmethod
    def _detect_best_device(cls) -> torch.device:
        """Detect the best available device with fallback logic."""
        
        # Priority 1: CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA device: {gpu_name}")
            return device
        
        # Priority 2: MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon)")
            return device
        
        # Priority 3: CPU fallback
        logger.info("Using CPU device (no GPU available)")
        return torch.device("cpu")
    
    @classmethod
    def _build_device_info(cls, device: torch.device) -> DeviceInfo:
        """Build detailed device information."""
        
        if device.type == "cuda":
            return DeviceInfo(
                device=device,
                device_type="cuda",
                device_name=torch.cuda.get_device_name(0),
                supports_amp=True,
                memory_gb=torch.cuda.get_device_properties(0).total_memory / (1024**3)
            )
        elif device.type == "mps":
            return DeviceInfo(
                device=device,
                device_type="mps",
                device_name="Apple Silicon (MPS)",
                supports_amp=True  # MPS supports float16
            )
        else:
            return DeviceInfo(
                device=device,
                device_type="cpu",
                device_name="CPU",
                supports_amp=False
            )
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached device information."""
        cls._cached_device = None
        cls._cached_info = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @classmethod
    def get_memory_stats(cls) -> dict:
        """
        Get memory statistics for CUDA devices.
        
        Returns:
            dict: Memory statistics or empty dict for non-CUDA devices.
        """
        if not torch.cuda.is_available():
            return {}
            
        return {
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        }
    
    @classmethod
    def optimize_for_inference(cls) -> None:
        """Apply optimizations for inference mode."""
        torch.set_grad_enabled(False)
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            
    @classmethod
    def optimize_for_training(cls) -> None:
        """Apply optimizations for training mode."""
        torch.set_grad_enabled(True)
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            # Enable TF32 for faster training on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


# Convenience function
def get_device(force_cpu: bool = False) -> torch.device:
    """Shortcut to get the best available device."""
    return DeviceManager.get_device(force_cpu=force_cpu)
