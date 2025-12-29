"""
Configuration Management System

Provides structured configuration for all pipeline components with
YAML/JSON loading support and environment variable overrides.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Data sources
    sentinel_hub_client_id: Optional[str] = None
    sentinel_hub_client_secret: Optional[str] = None
    gee_service_account: Optional[str] = None
    gee_key_file: Optional[str] = None
    
    # Tile settings
    tile_size_meters: float = 5000.0
    tile_overlap: float = 0.1
    
    # Image settings
    image_size: int = 256
    num_bands: int = 12  # Sentinel-2 has 12 bands typically used
    
    # Temporal settings
    time_steps: int = 4  # For multitemporal analysis
    
    # Preprocessing
    normalize: bool = True
    augment: bool = True
    cloud_mask: bool = True
    cloud_threshold: float = 0.2
    
    # Cache
    cache_dir: str = "./cache"
    use_cache: bool = True


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Model type: "lulc", "lst"
    model_type: str = "lulc"
    
    # Architecture
    backbone: str = "resnet34"  # resnet18, resnet34, resnet50, efficientnet
    encoder_weights: str = "imagenet"
    in_channels: int = 12
    num_classes: int = 10  # LULC classes
    
    # U-Net specific
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32, 16])
    decoder_attention: bool = True
    
    # Regression output (for LST)
    regression_output: bool = False
    output_range: tuple = (250, 330)  # Kelvin range for LST


@dataclass  
class TrainingConfig:
    """Configuration for model training."""
    
    # Basic
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Optimizer
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    
    # Mixed precision
    use_amp: bool = True
    
    # Regularization
    dropout: float = 0.2
    label_smoothing: float = 0.1
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 1e-4
    
    # Logging
    log_every_n_steps: int = 10
    
    # Reproducibility
    seed: int = 42


@dataclass
class Config:
    """
    Master configuration for the EO Pipeline.
    
    Example:
        >>> config = Config.from_yaml("config.yaml")
        >>> config.training.learning_rate = 1e-3
        >>> config.save("config_modified.yaml")
    """
    
    # Project info
    project_name: str = "eo_pipeline"
    experiment_name: str = "default"
    
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Paths
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    
    # Integration flags
    use_wandb: bool = False
    use_dvc: bool = False
    
    # WandB settings
    wandb_project: str = "eo_pipeline"
    wandb_entity: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        data_config = DataConfig(**data.pop("data", {}))
        model_config = ModelConfig(**data.pop("model", {}))
        training_config = TrainingConfig(**data.pop("training", {}))
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            **data
        )
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        except ImportError:
            raise ImportError("PyYAML required for YAML config loading. Install with: pip install pyyaml")
    
    def update_from_env(self) -> None:
        """Update config from environment variables."""
        import os
        
        # API credentials from environment
        if os.getenv("SENTINEL_HUB_CLIENT_ID"):
            self.data.sentinel_hub_client_id = os.getenv("SENTINEL_HUB_CLIENT_ID")
        if os.getenv("SENTINEL_HUB_CLIENT_SECRET"):
            self.data.sentinel_hub_client_secret = os.getenv("SENTINEL_HUB_CLIENT_SECRET")
        if os.getenv("GEE_SERVICE_ACCOUNT"):
            self.data.gee_service_account = os.getenv("GEE_SERVICE_ACCOUNT")
        if os.getenv("GEE_KEY_FILE"):
            self.data.gee_key_file = os.getenv("GEE_KEY_FILE")
        
        # WandB from environment
        if os.getenv("WANDB_PROJECT"):
            self.wandb_project = os.getenv("WANDB_PROJECT")
            self.use_wandb = True
        if os.getenv("WANDB_ENTITY"):
            self.wandb_entity = os.getenv("WANDB_ENTITY")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings."""
        warnings = []
        
        if self.training.use_amp and self.training.batch_size < 4:
            warnings.append("AMP works best with batch_size >= 4")
        
        if self.data.tile_overlap >= 0.5:
            warnings.append("tile_overlap >= 0.5 may cause excessive redundant processing")
        
        if self.model.in_channels != self.data.num_bands:
            warnings.append(
                f"Model in_channels ({self.model.in_channels}) != data num_bands ({self.data.num_bands})"
            )
        
        return warnings
