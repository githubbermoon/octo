"""
Model Evaluation

Provides metric computation for segmentation and regression tasks,
including per-class metrics, confusion matrices, and error analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SegmentationMetrics:
    """Metrics for semantic segmentation."""
    accuracy: float
    mean_iou: float
    per_class_iou: Dict[int, float]
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    
    def summary(self) -> str:
        """Get summary string."""
        lines = [
            f"Accuracy: {self.accuracy:.4f}",
            f"Mean IoU: {self.mean_iou:.4f}",
            f"Precision: {self.precision:.4f}",
            f"Recall: {self.recall:.4f}",
            f"F1 Score: {self.f1_score:.4f}",
            "Per-class IoU:",
        ]
        for cls, iou in self.per_class_iou.items():
            lines.append(f"  Class {cls}: {iou:.4f}")
        return "\n".join(lines)


@dataclass
class RegressionMetrics:
    """Metrics for regression tasks."""
    mse: float
    rmse: float
    mae: float
    r2: float
    bias: float
    
    def summary(self) -> str:
        """Get summary string."""
        return (
            f"MSE: {self.mse:.4f}\n"
            f"RMSE: {self.rmse:.4f}\n"
            f"MAE: {self.mae:.4f}\n"
            f"R²: {self.r2:.4f}\n"
            f"Bias: {self.bias:.4f}"
        )


class Evaluator:
    """
    Evaluator for Earth Observation models.
    
    Supports segmentation and regression evaluation with
    detailed per-class and spatial analysis.
    
    Example:
        >>> evaluator = Evaluator(num_classes=10)
        >>> predictions = model.predict(test_loader)
        >>> metrics = evaluator.evaluate_segmentation(predictions, targets)
        >>> print(metrics.summary())
    """
    
    def __init__(
        self,
        num_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        ignore_index: int = -100
    ):
        """
        Initialize evaluator.
        
        Args:
            num_classes: Number of classes for segmentation
            class_names: Optional class names for reporting
            ignore_index: Label value to ignore
        """
        self.num_classes = num_classes
        self.class_names = class_names
        self.ignore_index = ignore_index
    
    def evaluate_segmentation(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        per_image: bool = False
    ) -> Union[SegmentationMetrics, List[SegmentationMetrics]]:
        """
        Evaluate segmentation predictions.
        
        Args:
            predictions: Predicted class labels (B, H, W) or (H, W)
            targets: Ground truth labels (B, H, W) or (H, W)
            per_image: Return metrics per image
            
        Returns:
            Segmentation metrics
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Handle single image
        if predictions.ndim == 2:
            predictions = predictions[np.newaxis, ...]
            targets = targets[np.newaxis, ...]
        
        if per_image:
            return [
                self._compute_segmentation_metrics(p, t)
                for p, t in zip(predictions, targets)
            ]
        
        return self._compute_segmentation_metrics(predictions, targets)
    
    def _compute_segmentation_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> SegmentationMetrics:
        """Compute segmentation metrics."""
        # Flatten
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Apply ignore mask
        valid_mask = target_flat != self.ignore_index
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]
        
        # Accuracy
        accuracy = (pred_flat == target_flat).mean()
        
        # Confusion matrix
        num_classes = self.num_classes or max(pred_flat.max(), target_flat.max()) + 1
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for p, t in zip(pred_flat, target_flat):
            confusion[t, p] += 1
        
        # Per-class IoU
        per_class_iou = {}
        for c in range(num_classes):
            intersection = confusion[c, c]
            union = confusion[c, :].sum() + confusion[:, c].sum() - intersection
            if union > 0:
                per_class_iou[c] = intersection / union
            else:
                per_class_iou[c] = 0.0
        
        mean_iou = np.mean(list(per_class_iou.values()))
        
        # Precision, Recall, F1
        tp = np.diag(confusion).sum()
        fp = confusion.sum() - np.diag(confusion).sum()
        fn = confusion.sum() - np.diag(confusion).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        
        return SegmentationMetrics(
            accuracy=float(accuracy),
            mean_iou=float(mean_iou),
            per_class_iou=per_class_iou,
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1_score),
            confusion_matrix=confusion
        )
    
    def evaluate_regression(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        mask: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> RegressionMetrics:
        """
        Evaluate regression predictions.
        
        Args:
            predictions: Predicted values
            targets: Ground truth values
            mask: Optional validity mask
            
        Returns:
            Regression metrics
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Flatten
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Apply mask
        if mask is not None:
            mask_flat = mask.flatten().astype(bool)
            pred_flat = pred_flat[mask_flat]
            target_flat = target_flat[mask_flat]
        
        # Remove NaN/Inf
        valid = np.isfinite(pred_flat) & np.isfinite(target_flat)
        pred_flat = pred_flat[valid]
        target_flat = target_flat[valid]
        
        # Compute metrics
        errors = pred_flat - target_flat
        
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        bias = np.mean(errors)
        
        # R² score
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((target_flat - target_flat.mean()) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return RegressionMetrics(
            mse=float(mse),
            rmse=float(rmse),
            mae=float(mae),
            r2=float(r2),
            bias=float(bias)
        )
    
    def evaluate_lst(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate Land Surface Temperature predictions.
        
        Args:
            predictions: Predicted LST (Kelvin)
            targets: Ground truth LST (Kelvin)
            mask: Optional validity mask
            
        Returns:
            Dictionary of LST-specific metrics
        """
        reg_metrics = self.evaluate_regression(predictions, targets, mask)
        
        # Additional LST-specific analysis
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        if mask is not None:
            mask_flat = mask.flatten().astype(bool)
            pred_flat = pred_flat[mask_flat]
            target_flat = target_flat[mask_flat]
        
        # Temperature error percentiles
        errors = np.abs(pred_flat - target_flat)
        
        return {
            **vars(reg_metrics),
            "error_p50": float(np.percentile(errors, 50)),
            "error_p90": float(np.percentile(errors, 90)),
            "error_p99": float(np.percentile(errors, 99)),
            "within_1K": float(np.mean(errors <= 1.0)),
            "within_2K": float(np.mean(errors <= 2.0)),
            "within_5K": float(np.mean(errors <= 5.0)),
        }
    
    def compute_change_detection_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate land use change detection.
        
        Args:
            predictions: Predicted change map (binary)
            targets: Ground truth change map (binary)
            
        Returns:
            Change detection metrics
        """
        pred_flat = predictions.flatten().astype(bool)
        target_flat = targets.flatten().astype(bool)
        
        tp = np.sum(pred_flat & target_flat)
        fp = np.sum(pred_flat & ~target_flat)
        fn = np.sum(~pred_flat & target_flat)
        tn = np.sum(~pred_flat & ~target_flat)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Kappa coefficient
        po = (tp + tn) / (tp + fp + fn + tn)
        pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / ((tp + fp + fn + tn) ** 2)
        kappa = (po - pe) / (1 - pe + 1e-8)
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "kappa": float(kappa),
            "accuracy": float((tp + tn) / (tp + fp + fn + tn)),
            "false_alarm_rate": float(fp / (fp + tn + 1e-8)),
            "miss_rate": float(fn / (tp + fn + 1e-8))
        }
    
    def analyze_errors(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        num_samples: int = 10
    ) -> Dict[str, List]:
        """
        Analyze prediction errors for debugging.
        
        Returns samples of high-error cases.
        """
        if predictions.ndim > 2:
            # Flatten spatial dimensions but keep batch
            batch_size = predictions.shape[0]
            pred_flat = predictions.reshape(batch_size, -1)
            target_flat = targets.reshape(batch_size, -1)
        else:
            pred_flat = predictions.flatten()
            target_flat = targets.flatten()
            batch_size = 1
        
        # Find high-error samples
        if predictions.dtype in [np.float32, np.float64]:
            # Regression
            errors = np.abs(pred_flat - target_flat).mean(axis=-1) if batch_size > 1 else np.abs(pred_flat - target_flat)
        else:
            # Classification
            errors = (pred_flat != target_flat).mean(axis=-1) if batch_size > 1 else (pred_flat != target_flat).astype(float)
        
        top_error_indices = np.argsort(errors)[-num_samples:][::-1]
        
        return {
            "high_error_indices": top_error_indices.tolist(),
            "high_error_values": errors[top_error_indices].tolist()
        }


class SpatialEvaluator:
    """
    Evaluator with spatial analysis capabilities.
    
    Useful for understanding where models perform poorly.
    """
    
    def __init__(self, tile_grid=None):
        """
        Initialize spatial evaluator.
        
        Args:
            tile_grid: TileGrid for spatial context
        """
        self.tile_grid = tile_grid
        self.tile_metrics: Dict[str, Dict] = {}
    
    def evaluate_per_tile(
        self,
        tile_predictions: Dict[str, np.ndarray],
        tile_targets: Dict[str, np.ndarray],
        evaluator: Evaluator
    ) -> Dict[str, Dict]:
        """
        Evaluate metrics per tile for spatial analysis.
        
        Args:
            tile_predictions: Dict mapping tile_id to predictions
            tile_targets: Dict mapping tile_id to targets
            evaluator: Evaluator instance
            
        Returns:
            Per-tile metrics
        """
        for tile_id in tile_predictions:
            if tile_id in tile_targets:
                pred = tile_predictions[tile_id]
                target = tile_targets[tile_id]
                
                if pred.dtype in [np.float32, np.float64]:
                    metrics = evaluator.evaluate_regression(pred, target)
                else:
                    metrics = evaluator.evaluate_segmentation(pred, target)
                
                self.tile_metrics[tile_id] = vars(metrics) if hasattr(metrics, '__dict__') else metrics
        
        return self.tile_metrics
    
    def get_error_hotspots(
        self,
        metric_name: str = "rmse",
        threshold_percentile: float = 90
    ) -> List[str]:
        """
        Identify tiles with high error.
        
        Args:
            metric_name: Metric to use for hotspot detection
            threshold_percentile: Percentile threshold
            
        Returns:
            List of high-error tile IDs
        """
        if not self.tile_metrics:
            return []
        
        metric_values = [
            (tile_id, m.get(metric_name, 0))
            for tile_id, m in self.tile_metrics.items()
        ]
        
        values = [v for _, v in metric_values]
        threshold = np.percentile(values, threshold_percentile)
        
        return [tile_id for tile_id, v in metric_values if v >= threshold]
