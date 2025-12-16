"""
Training Script Entry Point
============================

Run: python scripts/train.py --model lulc --config configs/train_lulc.yaml
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import json
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train EO Pipeline models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["lulc", "lst", "water_quality"],
        required=True,
        help="Model type to train"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLFlow tracking"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Training {args.model} model")
    logger.info(f"Config: {args.config}")
    
    # Import modules
    from eo_pipeline.core import DeviceManager
    from eo_pipeline.config import Config, TrainingConfig
    from eo_pipeline.training import ModelTrainer
    
    # Setup MLFlow if enabled
    tracker = None
    if args.mlflow:
        from mlops.mlflow.mlflow_config import ExperimentTracker
        tracker = ExperimentTracker(f"eo-pipeline-{args.model}")
        tracker.start_run(run_name=f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        tracker.log_params(config)
    
    # Create model
    device = DeviceManager.get_device()
    logger.info(f"Using device: {device}")
    
    if args.model == "lulc":
        from eo_pipeline.models import LULCClassifier
        model = LULCClassifier(
            in_channels=config.get("model", {}).get("in_channels", 10),
            num_classes=config.get("model", {}).get("num_classes", 10),
            use_attention=config.get("model", {}).get("attention", True)
        )
    elif args.model == "lst":
        from eo_pipeline.models import LSTEstimator
        model = LSTEstimator(
            in_channels=config.get("model", {}).get("in_channels", 7)
        )
    elif args.model == "water_quality":
        from eo_pipeline.models import WaterQualityDetector
        model = WaterQualityDetector(
            in_channels=config.get("model", {}).get("in_channels", 10)
        )
    
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Create output directory
    output_dir = Path(args.output) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TODO: Setup data loaders
    # train_loader = create_dataloader(config["data"], split="train")
    # val_loader = create_dataloader(config["data"], split="val")
    
    # Training config
    training_config = TrainingConfig(
        epochs=config.get("training", {}).get("epochs", 100),
        batch_size=config.get("training", {}).get("batch_size", 16),
        learning_rate=config.get("training", {}).get("learning_rate", 0.001),
        checkpoint_dir=str(output_dir)
    )
    
    # Create trainer
    trainer = ModelTrainer(model, training_config, device=device)
    
    # TODO: Run training
    # history = trainer.fit(train_loader, val_loader, resume_from=args.resume)
    
    logger.info("Training script ready - implement data loading to run actual training")
    
    # Save placeholder metrics
    metrics = {
        "model": args.model,
        "parameters": model.count_parameters(),
        "device": str(device),
        "status": "placeholder"
    }
    
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    if tracker:
        tracker.log_metrics({"parameters": model.count_parameters()})
        tracker.end_run()
    
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
