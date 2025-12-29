"""
Weights & Biases Integration Hooks

Provides experiment tracking and model versioning via W&B.
"""

from typing import Dict, Any, Optional, List
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class WandBTracker:
    """
    Weights & Biases integration for experiment tracking.
    
    Features:
    - Experiment and run management
    - Metric and parameter logging
    - Model artifact storage
    - System metric monitoring
    
    Example:
        >>> tracker = WandBTracker(project="eo_pipeline", entity="my_org")
        >>> tracker.start_run(name="lulc_experiment", config={"lr": 0.001})
        >>> tracker.log_metrics({"loss": 0.5, "acc": 0.95})
        >>> tracker.log_model(model, "lulc_classifier")
        >>> tracker.end_run()
    """
    
    def __init__(
        self,
        project_name: str,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None
    ):
        """
        Initialize W&B tracker.
        
        Args:
            project_name: W&B project name
            entity: W&B entity (user or team)
            group: Group runs together (e.g., "experiment_1")
            job_type: Type of job (e.g., "train", "eval")
        """
        self.project_name = project_name
        self.entity = entity
        self.group = group
        self.job_type = job_type
        
        self.run = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize W&B API."""
        try:
            import wandb
            self._wandb = wandb
            logger.info("W&B initialized")
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            self._wandb = None
    
    def start_run(
        self,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        resume: bool = False
    ) -> None:
        """
        Start a new W&B run.
        
        Args:
            name: Run name
            config: Hyperparameters/config dict
            tags: List of tags
            resume: Whether to resume run
        """
        if self._wandb is None:
            return
            
        self.run = self._wandb.init(
            project=self.project_name,
            entity=self.entity,
            group=self.group,
            job_type=self.job_type,
            name=name,
            config=config,
            tags=tags,
            resume=resume
        )
        logger.info(f"Started W&B run: {self.run.name}")
    
    def end_run(self) -> None:
        """End the current run."""
        if self.run is not None:
            self.run.finish()
            logger.info("Ended W&B run")
            self.run = None
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        commit: bool = True
    ) -> None:
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number (optional)
            commit: Whether to upload immediately (default: True)
        """
        if self.run is not None:
            self.run.log(metrics, step=step, commit=commit)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters/config to W&B.
        
        Args:
            params: Dictionary of parameters
        """
        if self.run is not None:
            self.run.config.update(params)

    def log_model(
        self,
        model_path: str,
        name: str,
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a model file as an artifact.
        
        Args:
            model_path: Path to model file
            name: Name of the artifact
            aliases: List of aliases (e.g., "latest", "best")
            metadata: Additional metadata
        """
        if self.run is not None:
            artifact = self._wandb.Artifact(
                name=name,
                type="model",
                metadata=metadata
            )
            artifact.add_file(model_path)
            self.run.log_artifact(artifact, aliases=aliases)
            logger.info(f"Logged model artifact: {name}")

    def log_image(
        self,
        key: str,
        image,
        caption: Optional[str] = None
    ) -> None:
        """
        Log an image.
        
        Args:
            key: Metric key
            image: numpy array or PIL Image
            caption: Image caption
        """
        if self.run is not None:
            self.run.log({key: self._wandb.Image(image, caption=caption)})

    def watch_model(self, model, log: str = "gradients", log_freq: int = 100) -> None:
        """
        Watch model architecture and gradients.
        
        Args:
            model: PyTorch model
            log: What to log ("gradients", "parameters", "all")
            log_freq: Logging frequency
        """
        if self.run is not None:
            self.run.watch(model, log=log, log_freq=log_freq)
