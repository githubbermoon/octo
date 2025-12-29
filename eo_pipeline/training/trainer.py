"""
Model Trainer

Provides training loop with mixed precision, gradient accumulation,
learning rate scheduling, and checkpoint management.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any, Callable, List, Union
from pathlib import Path
from dataclasses import dataclass, field
import time
import logging

from ..core.device import DeviceManager, get_device
from ..config.settings import TrainingConfig
from ..models.base import EOModel

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Tracks training state."""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float('inf')
    best_epoch: int = 0
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    metrics_history: List[Dict[str, float]] = field(default_factory=list)


class ModelTrainer:
    """
    Trainer for Earth Observation models.
    
    Features:
    - Automatic Mixed Precision (AMP)
    - Gradient accumulation
    - Learning rate scheduling with warmup
    - Early stopping
    - Checkpoint management
    - MLFlow integration hooks
    
    Example:
        >>> trainer = ModelTrainer(model, config)
        >>> trainer.fit(train_loader, val_loader)
        >>> trainer.save_checkpoint("best_model.pt")
    """
    
    def __init__(
        self,
        model: EOModel,
        config: TrainingConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            optimizer: Custom optimizer (default: AdamW)
            scheduler: Custom scheduler
            loss_fn: Custom loss function
            device: Training device
        """
        self.config = config
        self.device = device or get_device()
        
        # Model
        self.model = model.to(self.device)
        
        # Optimizer
        self.optimizer = optimizer or self._create_optimizer()
        
        # Scheduler
        self.scheduler = scheduler  # Will be created in fit() with proper total steps
        
        # Loss function
        self.loss_fn = loss_fn
        
        # Mixed precision
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.state = TrainingState()
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Callbacks
        self.callbacks: List[Callable] = []
        
        # MLFlow placeholder
        self._mlflow_run = None
        
        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"AMP enabled: {self.use_amp}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        params = self.model.parameters()
        
        if self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(
        self,
        total_steps: int
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        warmup_steps = self.config.warmup_epochs * (total_steps // self.config.epochs)
        
        if self.config.scheduler.lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler.lower() == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=total_steps // 3,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        return None
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Override epochs from config
            resume_from: Checkpoint path to resume from
            
        Returns:
            Training history
        """
        epochs = epochs or self.config.epochs
        total_steps = epochs * len(train_loader)
        
        # Create scheduler if not provided
        if self.scheduler is None:
            self.scheduler = self._create_scheduler(total_steps)
        
        # Resume from checkpoint
        if resume_from:
            self._load_checkpoint(resume_from)
        
        # Set reproducibility
        torch.manual_seed(self.config.seed)
        
        # Training loop
        logger.info(f"Starting training for {epochs} epochs")
        
        no_improve_count = 0
        
        for epoch in range(self.state.epoch, epochs):
            self.state.epoch = epoch
            
            # Train epoch
            train_loss = self._train_epoch(train_loader, epoch)
            self.state.train_losses.append(train_loss)
            
            # Validate
            val_loss = None
            val_metrics = {}
            if val_loader is not None:
                val_loss, val_metrics = self._validate(val_loader)
                self.state.val_losses.append(val_loss)
                self.state.metrics_history.append(val_metrics)
            
            # Log progress
            self._log_epoch(epoch, train_loss, val_loss, val_metrics)
            
            # Checkpointing
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            # Early stopping
            if val_loss is not None:
                if val_loss < self.state.best_metric - self.config.min_delta:
                    self.state.best_metric = val_loss
                    self.state.best_epoch = epoch
                    self._save_checkpoint("best_model.pt")
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if self.config.early_stopping and no_improve_count >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Run callbacks
            for callback in self.callbacks:
                callback(self, epoch, train_loss, val_loss, val_metrics)
        
        # Final checkpoint
        self._save_checkpoint("final_model.pt")
        
        return {
            "train_losses": self.state.train_losses,
            "val_losses": self.state.val_losses,
            "metrics_history": self.state.metrics_history,
            "best_epoch": self.state.best_epoch,
            "best_metric": self.state.best_metric
        }
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Get data
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None
            
            inputs = inputs.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)
            
            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                
                if self.loss_fn is not None:
                    loss = self.loss_fn(outputs, targets)
                else:
                    loss = self.model.get_loss(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None and not isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step()
            
            total_loss += loss.item()
            self.state.global_step += 1
            
            # Log batch progress
            if (batch_idx + 1) % self.config.log_every_n_steps == 0:
                logger.debug(
                    f"Epoch {epoch + 1}/{self.config.epochs} "
                    f"Batch {batch_idx + 1}/{num_batches} "
                    f"Loss: {loss.item():.4f}"
                )
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def _validate(
        self,
        val_loader: DataLoader
    ) -> tuple[float, Dict[str, float]]:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in val_loader:
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None
            
            inputs = inputs.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            
            if self.loss_fn is not None:
                loss = self.loss_fn(outputs, targets)
            else:
                loss = self.model.get_loss(outputs, targets)
            
            total_loss += loss.item()
            
            # Collect predictions for metrics
            if isinstance(outputs, dict):
                outputs = outputs.get('logits', list(outputs.values())[0])
            all_preds.append(outputs.cpu())
            if targets is not None:
                all_targets.append(targets.cpu())
        
        avg_loss = total_loss / len(val_loader)
        
        # Compute metrics
        metrics = {}
        if all_targets:
            preds = torch.cat(all_preds, dim=0)
            targets = torch.cat(all_targets, dim=0)
            
            # Basic accuracy for segmentation
            if preds.dim() == 4:  # (B, C, H, W)
                pred_classes = preds.argmax(dim=1)
                metrics['accuracy'] = (pred_classes == targets).float().mean().item()
        
        # Update plateau scheduler
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)
        
        return avg_loss, metrics
    
    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        val_metrics: Dict[str, float]
    ) -> None:
        """Log epoch results."""
        log_str = f"Epoch {epoch + 1}/{self.config.epochs} - Train Loss: {train_loss:.4f}"
        
        if val_loss is not None:
            log_str += f" - Val Loss: {val_loss:.4f}"
        
        for name, value in val_metrics.items():
            log_str += f" - {name}: {value:.4f}"
        
        current_lr = self.optimizer.param_groups[0]['lr']
        log_str += f" - LR: {current_lr:.6f}"
        
        logger.info(log_str)
        
        # External tracking hook
        self._log_to_tracker(epoch, train_loss, val_loss, val_metrics)
    
    def _save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        path = self.checkpoint_dir / filename
        
        self.model.save_checkpoint(
            path,
            optimizer=self.optimizer,
            epoch=self.state.epoch,
            metrics={"best_metric": self.state.best_metric}
        )
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _load_checkpoint(self, path: str) -> None:
        """Load checkpoint and resume training."""
        info = self.model.load_checkpoint(path, optimizer=self.optimizer)
        self.state.epoch = info.get('epoch', 0)
        self.state.best_metric = info.get('metrics', {}).get('best_metric', float('inf'))
        logger.info(f"Resumed from epoch {self.state.epoch}")
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        while len(checkpoints) > self.config.keep_n_checkpoints:
            checkpoints[0].unlink()
            checkpoints.pop(0)
    
    def _log_to_tracker(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        val_metrics: Dict[str, float]
    ) -> None:
        """
        Log metrics to external tracker (W&B).
        
        This is handled via callbacks in the current design,
        but we keep this hook for direct integration if needed.
        """
        pass
    
    def add_callback(self, callback: Callable) -> None:
        """Add a callback to be called after each epoch."""
        self.callbacks.append(callback)
    
    @property
    def current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
