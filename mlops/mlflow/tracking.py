"""
MLFlow Tracking Example Script
==============================

Demonstrates how to track experiments with MLFlow in the EO Pipeline.
Run: python mlops/mlflow/tracking.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_with_mlflow_tracking():
    """
    Example training script with full MLFlow integration.
    """
    from mlops.mlflow.mlflow_config import ExperimentTracker, setup_mlflow
    from eo_pipeline.models import LULCClassifier
    from eo_pipeline.core import DeviceManager
    
    # -------------------------------------------------------------------------
    # 1. Setup MLFlow
    # -------------------------------------------------------------------------
    setup_mlflow()
    
    # -------------------------------------------------------------------------
    # 2. Define experiment parameters
    # -------------------------------------------------------------------------
    params = {
        "model": {
            "architecture": "unet",
            "encoder": "resnet34",
            "in_channels": 10,
            "num_classes": 8,
            "attention": True,
        },
        "training": {
            "epochs": 50,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
        },
        "data": {
            "tile_size": 256,
            "augmentation": True,
            "dataset": "sentinel2_munich",
        }
    }
    
    # -------------------------------------------------------------------------
    # 3. Initialize tracker and start run
    # -------------------------------------------------------------------------
    tracker = ExperimentTracker(
        experiment_name="lulc-classification",
        tags={
            "model_type": "unet",
            "dataset": "sentinel2",
            "author": "eo-pipeline",
        }
    )
    
    run_name = f"unet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with tracker.start_run(run_name=run_name):
        # Log parameters
        tracker.log_params(params)
        
        # Log system info
        device_info = DeviceManager.get_device_info()
        tracker.set_tag("device", device_info.device_name)
        tracker.set_tag("device_type", device_info.device_type)
        
        # ---------------------------------------------------------------------
        # 4. Create model
        # ---------------------------------------------------------------------
        model = LULCClassifier(
            in_channels=params["model"]["in_channels"],
            num_classes=params["model"]["num_classes"],
            use_attention=params["model"]["attention"]
        )
        
        tracker.log_param("total_params", model.count_parameters())
        logger.info(f"Model created with {model.count_parameters():,} parameters")
        
        # ---------------------------------------------------------------------
        # 5. Simulate training loop
        # ---------------------------------------------------------------------
        epochs = params["training"]["epochs"]
        
        for epoch in range(epochs):
            # Simulated metrics (replace with actual training)
            train_loss = 1.0 / (epoch + 1) + np.random.rand() * 0.1
            val_loss = 1.2 / (epoch + 1) + np.random.rand() * 0.1
            train_iou = 0.5 + 0.4 * (epoch / epochs) + np.random.rand() * 0.05
            val_iou = 0.45 + 0.4 * (epoch / epochs) + np.random.rand() * 0.05
            
            # Log metrics
            tracker.log_metrics({
                "train/loss": train_loss,
                "train/iou": train_iou,
                "val/loss": val_loss,
                "val/iou": val_iou,
                "learning_rate": params["training"]["learning_rate"] * (0.95 ** epoch),
            }, step=epoch)
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}"
                )
        
        # ---------------------------------------------------------------------
        # 6. Log final metrics
        # ---------------------------------------------------------------------
        final_metrics = {
            "final/val_loss": val_loss,
            "final/val_iou": val_iou,
            "final/val_accuracy": 0.85 + np.random.rand() * 0.05,
        }
        tracker.log_metrics(final_metrics)
        
        # ---------------------------------------------------------------------
        # 7. Log model artifact
        # ---------------------------------------------------------------------
        logger.info("Logging model to MLFlow...")
        tracker.log_model(
            model,
            artifact_path="model",
            registered_name="lulc-classifier"  # Register in model registry
        )
        
        # ---------------------------------------------------------------------
        # 8. Log additional artifacts
        # ---------------------------------------------------------------------
        # Save and log config
        config_path = Path("outputs/run_config.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(params, f)
        tracker.log_artifact(str(config_path))
        
        logger.info(f"Run completed: {tracker.get_run_id()}")
        
    return tracker.get_run_id()


def compare_runs():
    """
    Example: Compare multiple runs using MLFlow.
    """
    try:
        import mlflow
        
        # Search for runs
        runs = mlflow.search_runs(
            experiment_names=["lulc-classification"],
            filter_string="metrics.`final/val_iou` > 0.7",
            order_by=["metrics.`final/val_iou` DESC"],
            max_results=10
        )
        
        if len(runs) > 0:
            print("\nTop Runs by Validation IoU:")
            print("=" * 60)
            for _, run in runs.iterrows():
                print(f"Run: {run['run_id'][:8]}... | IoU: {run.get('metrics.final/val_iou', 'N/A'):.4f}")
        else:
            print("No matching runs found")
            
    except ImportError:
        print("MLFlow not installed")


def load_best_model():
    """
    Example: Load the best model from registry.
    """
    from mlops.mlflow.mlflow_config import ModelRegistry
    
    registry = ModelRegistry()
    
    # List registered models
    models = registry.list_models()
    print(f"Registered models: {models}")
    
    # Load production model
    if "lulc-classifier" in models:
        model = registry.load_model("lulc-classifier", stage="Production")
        if model:
            print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
            return model
    
    return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MLFlow Tracking Examples")
    parser.add_argument(
        "--action",
        choices=["train", "compare", "load"],
        default="train",
        help="Action to perform"
    )
    
    args = parser.parse_args()
    
    if args.action == "train":
        run_id = train_with_mlflow_tracking()
        print(f"\n✓ Training run completed: {run_id}")
        print(f"  View at: http://localhost:5000")
        
    elif args.action == "compare":
        compare_runs()
        
    elif args.action == "load":
        model = load_best_model()
        if model:
            print("✓ Model loaded successfully")
