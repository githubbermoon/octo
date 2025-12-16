"""
Earth Observation AI Pipeline - Main Entry Point

Provides CLI interface for running training, evaluation, and inference.
"""

import argparse
import sys
from pathlib import Path

from eo_pipeline.core.device import DeviceManager
from eo_pipeline.config.settings import Config
from eo_pipeline.utils.logging import setup_logging, get_logger


def main():
    """Main entry point for the EO Pipeline."""
    parser = argparse.ArgumentParser(
        description="Earth Observation AI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train LULC classifier
  python -m eo_pipeline train --config config.yaml --model lulc
  
  # Evaluate model
  python -m eo_pipeline evaluate --checkpoint model.pt --data test_data/
  
  # Run inference
  python -m eo_pipeline predict --checkpoint model.pt --input image.tif --output result.tif
  
  # Show system info
  python -m eo_pipeline info
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, required=True, help="Config file path")
    train_parser.add_argument("--model", type=str, choices=["lulc", "lst", "water_quality"], 
                             default="lulc", help="Model type")
    train_parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    train_parser.add_argument("--output", type=str, default="./outputs", help="Output directory")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    eval_parser.add_argument("--data", type=str, required=True, help="Test data directory")
    eval_parser.add_argument("--output", type=str, help="Output directory for results")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    predict_parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    predict_parser.add_argument("--output", type=str, required=True, help="Output path")
    predict_parser.add_argument("--format", type=str, default="tif", help="Output format")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Launch Gradio demo")
    demo_parser.add_argument("--port", type=int, default=7860, help="Port number")
    demo_parser.add_argument("--share", action="store_true", help="Create public link")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Setup logging
    logger = setup_logging(level="INFO")
    
    if args.command == "info":
        show_info()
    elif args.command == "train":
        run_training(args)
    elif args.command == "evaluate":
        run_evaluation(args)
    elif args.command == "predict":
        run_prediction(args)
    elif args.command == "demo":
        run_demo(args)


def show_info():
    """Display system and environment information."""
    print("\n" + "=" * 60)
    print("Earth Observation AI Pipeline - System Information")
    print("=" * 60)
    
    # Device info
    device_info = DeviceManager.get_device_info()
    print(f"\nDevice: {device_info.device_name}")
    print(f"Device Type: {device_info.device_type}")
    print(f"AMP Support: {device_info.supports_amp}")
    if device_info.memory_gb:
        print(f"GPU Memory: {device_info.memory_gb:.2f} GB")
    
    # PyTorch info
    import torch
    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"MPS Available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
    
    # Package versions
    print("\nInstalled Packages:")
    try:
        import numpy
        print(f"  NumPy: {numpy.__version__}")
    except ImportError:
        print("  NumPy: Not installed")
    
    try:
        import rasterio
        print(f"  Rasterio: {rasterio.__version__}")
    except ImportError:
        print("  Rasterio: Not installed")
    
    print("\n" + "=" * 60)


def run_training(args):
    """Run model training."""
    logger = get_logger("main")
    logger.info(f"Starting training with config: {args.config}")
    
    # Load config
    config = Config.from_yaml(args.config) if args.config.endswith('.yaml') else Config.from_json(args.config)
    
    # Select model
    if args.model == "lulc":
        from eo_pipeline.models import LULCClassifier
        model = LULCClassifier(
            in_channels=config.model.in_channels,
            num_classes=config.model.num_classes
        )
    elif args.model == "lst":
        from eo_pipeline.models import LSTEstimator
        model = LSTEstimator(in_channels=config.model.in_channels)
    elif args.model == "water_quality":
        from eo_pipeline.models import WaterQualityDetector
        model = WaterQualityDetector(in_channels=config.model.in_channels)
    
    logger.info(f"Model: {model.name}")
    logger.info(f"Parameters: {model.count_parameters():,}")
    
    # TODO: Setup data loaders and trainer
    # trainer = ModelTrainer(model, config.training)
    # trainer.fit(train_loader, val_loader, resume_from=args.resume)
    
    logger.info("Training placeholder - implement data loading and training loop")


def run_evaluation(args):
    """Run model evaluation."""
    logger = get_logger("main")
    logger.info(f"Evaluating model: {args.checkpoint}")
    
    # TODO: Load model, data, and run evaluation
    # model = LULCClassifier.from_pretrained(args.checkpoint)
    # evaluator = Evaluator(num_classes=10)
    # metrics = evaluator.evaluate_segmentation(predictions, targets)
    
    logger.info("Evaluation placeholder - implement evaluation logic")


def run_prediction(args):
    """Run inference on input data."""
    logger = get_logger("main")
    logger.info(f"Running prediction on: {args.input}")
    
    # TODO: Load model, process input, save output
    # model = LULCClassifier.from_pretrained(args.checkpoint)
    # predictions = model.predict(input_tensor)
    # save_as_geotiff(predictions, args.output)
    
    logger.info("Prediction placeholder - implement inference logic")


def run_demo(args):
    """Launch Gradio demo."""
    logger = get_logger("main")
    logger.info(f"Launching demo on port {args.port}")
    
    from eo_pipeline.integrations import GradioDemo
    
    demo = GradioDemo()
    demo.add_segmentation_interface()
    demo.add_lst_interface()
    demo.add_water_quality_interface()
    demo.add_explainability_interface()
    
    url = demo.launch(share=args.share, server_port=args.port)
    logger.info(f"Demo available at: {url}")


if __name__ == "__main__":
    main()
