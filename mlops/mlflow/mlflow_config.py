"""
MLFlow Configuration and Utilities
===================================

Provides centralized MLFlow configuration and helper functions for
experiment tracking in the EO Pipeline.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MLFlowConfig:
    """MLFlow configuration settings."""
    
    # Connection
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    registry_uri: str = os.getenv("MLFLOW_REGISTRY_URI", "")
    
    # Experiment
    experiment_name: str = "eo-pipeline"
    
    # Artifact storage
    artifact_location: str = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns")
    
    # Autologging
    autolog_pytorch: bool = True
    autolog_disable_system_metrics: bool = False
    
    # Run settings
    nested_runs: bool = True
    log_models: bool = True
    
    @classmethod
    def from_env(cls) -> "MLFlowConfig":
        """Create config from environment variables."""
        return cls(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "eo-pipeline"),
            artifact_location=os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns"),
        )


# =============================================================================
# MLFlow Setup
# =============================================================================

def setup_mlflow(config: Optional[MLFlowConfig] = None) -> None:
    """
    Initialize MLFlow with configuration.
    
    Args:
        config: MLFlow configuration (uses defaults if None)
    """
    try:
        import mlflow
        import mlflow.pytorch
        
        config = config or MLFlowConfig.from_env()
        
        # Set tracking URI
        mlflow.set_tracking_uri(config.tracking_uri)
        logger.info(f"MLFlow tracking URI: {config.tracking_uri}")
        
        # Set or create experiment
        experiment = mlflow.get_experiment_by_name(config.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                config.experiment_name,
                artifact_location=config.artifact_location
            )
            logger.info(f"Created experiment: {config.experiment_name} (ID: {experiment_id})")
        else:
            logger.info(f"Using experiment: {config.experiment_name} (ID: {experiment.experiment_id})")
        
        mlflow.set_experiment(config.experiment_name)
        
        # Enable autologging
        if config.autolog_pytorch:
            mlflow.pytorch.autolog(
                log_models=config.log_models,
                disable=config.autolog_disable_system_metrics
            )
            logger.info("PyTorch autologging enabled")
            
    except ImportError:
        logger.warning("MLFlow not installed. Install with: pip install mlflow")
    except Exception as e:
        logger.warning(f"Failed to setup MLFlow: {e}")


# =============================================================================
# Tracking Utilities
# =============================================================================

class ExperimentTracker:
    """
    High-level experiment tracking wrapper around MLFlow.
    
    Example:
        >>> tracker = ExperimentTracker("lulc_training")
        >>> with tracker.start_run("baseline_unet"):
        ...     tracker.log_params({"lr": 0.001, "epochs": 100})
        ...     for epoch in range(100):
        ...         tracker.log_metrics({"loss": loss, "iou": iou}, step=epoch)
        ...     tracker.log_model(model, "model")
    """
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize tracker.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLFlow server URI
            tags: Default tags for all runs
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.default_tags = tags or {}
        
        self._mlflow = None
        self._run = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize MLFlow connection."""
        try:
            import mlflow
            self._mlflow = mlflow
            
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            
            logger.info(f"Tracker initialized for experiment: {self.experiment_name}")
            
        except ImportError:
            logger.warning("MLFlow not installed")
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ):
        """
        Start an MLFlow run.
        
        Args:
            run_name: Name for this run
            tags: Additional tags
            nested: Allow nested runs
            
        Returns:
            Self for context manager usage
        """
        if self._mlflow is None:
            return self
        
        all_tags = {**self.default_tags, **(tags or {})}
        
        self._run = self._mlflow.start_run(
            run_name=run_name,
            nested=nested,
            tags=all_tags
        )
        
        return self
    
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        if self._mlflow and self._run:
            self._mlflow.end_run(status=status)
            self._run = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FAILED" if exc_type else "FINISHED"
        self.end_run(status)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        if self._mlflow:
            # Flatten nested dicts
            flat_params = self._flatten_dict(params)
            self._mlflow.log_params(flat_params)
    
    def log_param(self, key: str, value: Any) -> None:
        """Log single parameter."""
        if self._mlflow:
            self._mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        if self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log single metric."""
        if self._mlflow:
            self._mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model, artifact_path: str, registered_name: Optional[str] = None) -> None:
        """Log PyTorch model."""
        if self._mlflow:
            import mlflow.pytorch
            mlflow.pytorch.log_model(
                model,
                artifact_path,
                registered_model_name=registered_name
            )
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact file."""
        if self._mlflow:
            self._mlflow.log_artifact(local_path, artifact_path)
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact directory."""
        if self._mlflow:
            self._mlflow.log_artifacts(local_dir, artifact_path)
    
    def log_figure(self, figure, artifact_file: str) -> None:
        """Log matplotlib figure."""
        if self._mlflow:
            self._mlflow.log_figure(figure, artifact_file)
    
    def set_tag(self, key: str, value: str) -> None:
        """Set run tag."""
        if self._mlflow:
            self._mlflow.set_tag(key, value)
    
    def get_run_id(self) -> Optional[str]:
        """Get current run ID."""
        if self._run:
            return self._run.info.run_id
        return None
    
    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ExperimentTracker._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


# =============================================================================
# Model Registry
# =============================================================================

class ModelRegistry:
    """
    Interface to MLFlow Model Registry.
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.register_model("runs:/abc123/model", "lulc-classifier")
        >>> registry.transition_stage("lulc-classifier", "1", "Production")
        >>> model = registry.load_model("lulc-classifier", stage="Production")
    """
    
    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self._client = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize MLFlow client."""
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            self._client = mlflow.tracking.MlflowClient()
        except ImportError:
            logger.warning("MLFlow not installed")
    
    def register_model(self, model_uri: str, name: str) -> Optional[str]:
        """
        Register a model in the registry.
        
        Args:
            model_uri: URI to logged model (e.g., "runs:/run_id/model")
            name: Registered model name
            
        Returns:
            Version number
        """
        if self._client:
            import mlflow
            result = mlflow.register_model(model_uri, name)
            logger.info(f"Registered model {name} version {result.version}")
            return result.version
        return None
    
    def transition_stage(
        self,
        name: str,
        version: str,
        stage: str,  # "Staging", "Production", "Archived"
        archive_existing: bool = True
    ) -> None:
        """Transition model to new stage."""
        if self._client:
            self._client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing
            )
            logger.info(f"Transitioned {name} v{version} to {stage}")
    
    def load_model(self, name: str, stage: str = "Production"):
        """Load model from registry."""
        try:
            import mlflow.pytorch
            model_uri = f"models:/{name}/{stage}"
            return mlflow.pytorch.load_model(model_uri)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        if self._client:
            return [m.name for m in self._client.search_registered_models()]
        return []
    
    def get_latest_version(self, name: str, stages: Optional[List[str]] = None) -> Optional[str]:
        """Get latest model version."""
        if self._client:
            versions = self._client.get_latest_versions(name, stages=stages)
            if versions:
                return versions[0].version
        return None


# =============================================================================
# Convenience Functions
# =============================================================================

def log_training_run(
    model,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    experiment_name: str = "eo-pipeline",
    run_name: Optional[str] = None,
    register_model: Optional[str] = None
) -> Optional[str]:
    """
    Log a complete training run to MLFlow.
    
    Args:
        model: Trained PyTorch model
        params: Training parameters
        metrics: Final metrics
        experiment_name: Experiment name
        run_name: Run name
        register_model: Model name for registry (None = don't register)
        
    Returns:
        Run ID
    """
    tracker = ExperimentTracker(experiment_name)
    
    with tracker.start_run(run_name=run_name):
        tracker.log_params(params)
        tracker.log_metrics(metrics)
        tracker.log_model(model, "model", registered_name=register_model)
        
        return tracker.get_run_id()
