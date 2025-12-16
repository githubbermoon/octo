"""
Logging Configuration

Provides standardized logging setup for the EO pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "./logs",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the EO pipeline.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name (auto-generated if None)
        log_dir: Directory for log files
        format_string: Custom format string
        
    Returns:
        Root logger
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
        )
    
    # Configure root logger
    logger = logging.getLogger("eo_pipeline")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"eo_pipeline_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_path / log_file)
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized at level {level}")
    logger.info(f"Log file: {log_path / log_file}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the EO pipeline namespace.
    
    Args:
        name: Logger name (will be prefixed with 'eo_pipeline.')
        
    Returns:
        Logger instance
    """
    if not name.startswith("eo_pipeline"):
        name = f"eo_pipeline.{name}"
    return logging.getLogger(name)


class TrainingLogger:
    """
    Specialized logger for training progress.
    
    Example:
        >>> logger = TrainingLogger()
        >>> logger.log_epoch(epoch=1, train_loss=0.5, val_loss=0.4)
        >>> logger.log_batch(batch=10, loss=0.45)
    """
    
    def __init__(self, name: str = "training"):
        self.logger = get_logger(name)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[dict] = None,
        lr: Optional[float] = None
    ) -> None:
        """Log epoch summary."""
        msg = f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}"
        
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.4f}"
        
        if metrics:
            for k, v in metrics.items():
                msg += f" | {k}: {v:.4f}"
        
        if lr is not None:
            msg += f" | LR: {lr:.2e}"
        
        self.logger.info(msg)
    
    def log_batch(
        self,
        batch: int,
        total_batches: int,
        loss: float,
        lr: Optional[float] = None
    ) -> None:
        """Log batch progress."""
        msg = f"Batch {batch:04d}/{total_batches} | Loss: {loss:.4f}"
        
        if lr is not None:
            msg += f" | LR: {lr:.2e}"
        
        self.logger.debug(msg)
    
    def log_metrics(self, metrics: dict, prefix: str = "") -> None:
        """Log multiple metrics."""
        msg = prefix
        for k, v in metrics.items():
            msg += f"{k}: {v:.4f} | "
        self.logger.info(msg.rstrip(" | "))
