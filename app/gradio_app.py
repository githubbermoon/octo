"""
Gradio Web Interface for EO Pipeline
=====================================

Interactive demo for satellite imagery analysis.
Run: python app/gradio_app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import numpy as np
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Loading
# =============================================================================

def load_models():
    """Load production models."""
    models = {}
    
    try:
        from eo_pipeline.models import LULCClassifier, LSTEstimator, WaterQualityDetector
        from eo_pipeline.core import DeviceManager
        
        device = DeviceManager.get_device()
        logger.info(f"Loading models on {device}")
        
        # Try to load from checkpoints, otherwise use placeholder
        model_dir = Path("models/production")
        
        # LULC Classifier
        lulc_path = model_dir / "lulc" / "model.pt"
        if lulc_path.exists():
            models["lulc"] = LULCClassifier.from_pretrained(str(lulc_path))
        else:
            models["lulc"] = LULCClassifier(in_channels=10, num_classes=8)
            logger.warning("Using untrained LULC model")
        
        # LST Estimator
        models["lst"] = LSTEstimator(in_channels=7, use_auxiliary=False)
        
        # Water Quality
        models["water"] = WaterQualityDetector(in_channels=10, use_spectral_indices=False)
        
        logger.info("Models loaded successfully")
        
    except ImportError as e:
        logger.error(f"Failed to load models: {e}")
        
    return models


# =============================================================================
# Prediction Functions
# =============================================================================

def predict_lulc(image: np.ndarray, model) -> Tuple[np.ndarray, dict]:
    """Predict land use/land cover."""
    import torch
    import torch.nn.functional as F
    
    # Preprocess (assume RGB for demo)
    if image is None:
        return None, {}
    
    # Resize to model input size
    h, w = image.shape[:2]
    
    # Create dummy multi-band input (in production, use actual satellite data)
    # For demo, replicate RGB to 10 bands
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    
    multi_band = np.zeros((10, 256, 256), dtype=np.float32)
    for i in range(min(3, image.shape[-1])):
        from scipy.ndimage import zoom
        multi_band[i] = zoom(image[..., i].astype(np.float32), (256/h, 256/w))
    
    # Normalize
    multi_band = multi_band / 255.0
    
    # Predict
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(multi_band).unsqueeze(0)
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        prediction = probs.argmax(dim=1).squeeze().numpy()
        confidence = probs.max(dim=1)[0].squeeze().numpy()
    
    # Resize back
    from scipy.ndimage import zoom
    prediction = zoom(prediction.astype(np.float32), (h/256, w/256), order=0).astype(np.uint8)
    
    # Create colored segmentation map
    class_colors = [
        [34, 139, 34],    # Tree cover - Green
        [139, 90, 43],    # Shrubland - Brown
        [124, 252, 0],    # Grassland - Light green
        [255, 215, 0],    # Cropland - Gold
        [128, 128, 128],  # Built-up - Gray
        [210, 180, 140],  # Bare - Tan
        [255, 255, 255],  # Snow - White
        [0, 0, 255],      # Water - Blue
    ]
    
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(class_colors):
        colored[prediction == i] = color
    
    # Class distribution
    unique, counts = np.unique(prediction, return_counts=True)
    class_names = ["Tree", "Shrub", "Grass", "Crop", "Built", "Bare", "Snow", "Water"]
    distribution = {
        class_names[i]: float(c) / prediction.size 
        for i, c in zip(unique, counts) if i < len(class_names)
    }
    
    return colored, distribution


def predict_lst(thermal_image: np.ndarray, model) -> Tuple[np.ndarray, float]:
    """Estimate land surface temperature."""
    import torch
    
    if thermal_image is None:
        return None, 0.0
    
    h, w = thermal_image.shape[:2]
    
    # Process image
    if thermal_image.ndim == 3:
        thermal = np.mean(thermal_image, axis=-1)
    else:
        thermal = thermal_image
    
    # Create 7-band input (thermal + placeholder)
    from scipy.ndimage import zoom
    multi_band = np.zeros((7, 64, 64), dtype=np.float32)
    multi_band[0] = zoom(thermal.astype(np.float32), (64/h, 64/w)) / 255.0
    
    # Predict
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(multi_band).unsqueeze(0)
        output = model(input_tensor)
        lst = output.squeeze().numpy()
    
    # Resize
    lst = zoom(lst, (h/64, w/64))
    
    # Convert to Celsius for display
    lst_celsius = lst - 273.15
    mean_temp = float(lst_celsius.mean())
    
    # Normalize for display
    lst_display = (lst_celsius - lst_celsius.min()) / (lst_celsius.max() - lst_celsius.min() + 1e-8)
    lst_colored = (plt_colormap(lst_display) * 255).astype(np.uint8)[..., :3]
    
    return lst_colored, mean_temp


def plt_colormap(data: np.ndarray) -> np.ndarray:
    """Apply matplotlib-like colormap."""
    # Simple hot colormap approximation
    r = np.clip(data * 2, 0, 1)
    g = np.clip((data - 0.5) * 2, 0, 1)
    b = np.zeros_like(data)
    return np.stack([r, g, b], axis=-1)


def predict_water_quality(image: np.ndarray, model) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Detect water quality indicators."""
    import torch
    
    if image is None:
        return None, None, {}
    
    h, w = image.shape[:2]
    
    # Create 10-band input
    from scipy.ndimage import zoom
    multi_band = np.zeros((10, 64, 64), dtype=np.float32)
    for i in range(min(3, image.shape[-1] if image.ndim == 3 else 1)):
        if image.ndim == 3:
            multi_band[i] = zoom(image[..., i].astype(np.float32), (64/h, 64/w)) / 255.0
        else:
            multi_band[i] = zoom(image.astype(np.float32), (64/h, 64/w)) / 255.0
    
    # Predict
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(multi_band).unsqueeze(0)
        outputs = model(input_tensor)
    
    # Process outputs
    water_mask = torch.sigmoid(outputs['water_mask']).squeeze().numpy()
    turbidity = outputs['turbidity'].squeeze().numpy()
    
    # Resize
    water_mask = zoom(water_mask, (h/32, w/32)) > 0.5
    turbidity = zoom(turbidity, (h/32, w/32))
    
    # Create visualizations
    water_vis = np.zeros((h, w, 3), dtype=np.uint8)
    water_vis[water_mask] = [0, 128, 255]
    
    turb_vis = (np.clip(turbidity / 50, 0, 1) * 255).astype(np.uint8)
    turb_colored = np.stack([turb_vis, 255 - turb_vis, np.zeros_like(turb_vis)], axis=-1)
    
    metrics = {
        "Water Coverage": f"{water_mask.mean() * 100:.1f}%",
        "Avg Turbidity": f"{turbidity[water_mask].mean():.1f} NTU" if water_mask.any() else "N/A",
    }
    
    return water_vis, turb_colored, metrics


# =============================================================================
# Gradio Interface
# =============================================================================

def create_demo():
    """Create the Gradio demo interface."""
    models = load_models()
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        text-align: center;
        color: #2c3e50;
    }
    """
    
    with gr.Blocks(css=custom_css, title="EO Pipeline Demo") as demo:
        gr.Markdown(
            """
            # 🛰️ Earth Observation AI Pipeline
            
            Interactive demo for satellite imagery analysis. Upload an image to get started.
            
            > **Note**: This demo uses placeholder models. For production, load trained checkpoints.
            """
        )
        
        with gr.Tabs():
            # -----------------------------------------------------------------
            # Tab 1: LULC Classification
            # -----------------------------------------------------------------
            with gr.TabItem("🗺️ Land Use Classification"):
                gr.Markdown("### Classify land use and land cover from satellite imagery")
                
                with gr.Row():
                    lulc_input = gr.Image(label="Input Image", type="numpy")
                    lulc_output = gr.Image(label="Segmentation Map")
                
                lulc_dist = gr.JSON(label="Class Distribution")
                
                lulc_btn = gr.Button("Classify", variant="primary")
                lulc_btn.click(
                    fn=lambda x: predict_lulc(x, models.get("lulc")),
                    inputs=lulc_input,
                    outputs=[lulc_output, lulc_dist]
                )
                
                gr.Examples(
                    examples=[],  # Add example images here
                    inputs=lulc_input,
                )
            
            # -----------------------------------------------------------------
            # Tab 2: LST Estimation
            # -----------------------------------------------------------------
            with gr.TabItem("🌡️ Temperature Estimation"):
                gr.Markdown("### Estimate Land Surface Temperature for UHI analysis")
                
                with gr.Row():
                    lst_input = gr.Image(label="Input Image", type="numpy")
                    lst_output = gr.Image(label="Temperature Map")
                
                lst_temp = gr.Number(label="Mean Temperature (°C)")
                
                lst_btn = gr.Button("Estimate", variant="primary")
                lst_btn.click(
                    fn=lambda x: predict_lst(x, models.get("lst")),
                    inputs=lst_input,
                    outputs=[lst_output, lst_temp]
                )
            
            # -----------------------------------------------------------------
            # Tab 3: Water Quality
            # -----------------------------------------------------------------
            with gr.TabItem("💧 Water Quality"):
                gr.Markdown("### Detect water bodies and assess quality")
                
                with gr.Row():
                    water_input = gr.Image(label="Input Image", type="numpy")
                    with gr.Column():
                        water_mask_output = gr.Image(label="Water Mask")
                        water_turb_output = gr.Image(label="Turbidity Map")
                
                water_metrics = gr.JSON(label="Quality Metrics")
                
                water_btn = gr.Button("Analyze", variant="primary")
                water_btn.click(
                    fn=lambda x: predict_water_quality(x, models.get("water")),
                    inputs=water_input,
                    outputs=[water_mask_output, water_turb_output, water_metrics]
                )
            
            # -----------------------------------------------------------------
            # Tab 4: About
            # -----------------------------------------------------------------
            with gr.TabItem("ℹ️ About"):
                gr.Markdown(
                    """
                    ## About This Demo
                    
                    This is an interactive demonstration of the **Earth Observation AI Pipeline**,
                    a modular PyTorch framework for satellite imagery analysis.
                    
                    ### Features
                    
                    - **LULC Classification**: U-Net based land use/land cover segmentation
                    - **LST Estimation**: CNN-based land surface temperature prediction
                    - **Water Quality**: Multi-task detection of water bodies and pollution
                    
                    ### Data Sources (Placeholder)
                    
                    - Sentinel-2 (optical)
                    - Landsat-8/9 (thermal)
                    
                    ### Links
                    
                    - [GitHub Repository](#)
                    - [Documentation](#)
                    - [MLFlow Experiments](http://localhost:5000)
                    
                    ---
                    
                    *Built with PyTorch, Gradio, and 🛰️ satellite data*
                    """
                )
    
    return demo


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EO Pipeline Gradio Demo")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--auth", type=str, help="Basic auth (user:password)")
    
    args = parser.parse_args()
    
    demo = create_demo()
    
    auth = None
    if args.auth:
        user, password = args.auth.split(":")
        auth = (user, password)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        auth=auth
    )
