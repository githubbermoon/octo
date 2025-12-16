"""
Gradio Interface Builder
========================

Modular interface components for the EO AI Dashboard.
Each tab is a separate component that can be tested independently.
"""

import gradio as gr
from typing import Optional
import numpy as np

# Import analysis modules
from utils.uhi_analysis import analyze_uhi, create_uhi_visualization
from utils.lulc_analysis import analyze_lulc_change, create_change_map
from utils.plastic_detection import detect_plastics, create_plastic_overlay
from utils.xai_visualization import generate_shap_explanation, generate_lime_explanation
from utils.sample_loader import get_sample_images, load_sample


# =============================================================================
# Interface Builder
# =============================================================================

def create_interface(
    title: str,
    description: str,
    theme: Optional[gr.Theme] = None
) -> gr.Blocks:
    """
    Create the main Gradio interface with all tabs.
    
    Args:
        title: Application title
        description: Application description
        theme: Gradio theme
        
    Returns:
        Gradio Blocks application
    """
    
    with gr.Blocks(title=title) as demo:
        # Header
        gr.Markdown(f"# {title}")
        gr.Markdown(description)
        
        # Main tabs
        with gr.Tabs() as tabs:
            # Tab 1: Urban Heat Island
            with gr.TabItem("🌡️ Urban Heat Island", id="uhi"):
                create_uhi_tab()
            
            # Tab 2: Urban Expansion / LULC Change
            with gr.TabItem("🏙️ Urban Expansion", id="lulc"):
                create_lulc_tab()
            
            # Tab 3: Plastic Detection
            with gr.TabItem("🌊 Plastic Detection", id="plastic"):
                create_plastic_tab()
            
            # Tab 4: Explainable AI
            with gr.TabItem("🔍 Explainable AI", id="xai"):
                create_xai_tab()
            
            # Tab 5: About / Documentation
            with gr.TabItem("ℹ️ About", id="about"):
                create_about_tab()
        
        # Footer
        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #666;">
                Built with 🛰️ PyTorch & Gradio | 
                <a href="https://github.com/yourusername/octo">GitHub</a> |
                <a href="https://huggingface.co/spaces/yourusername/eo-ai-demo">HF Spaces</a>
            </div>
            """,
            elem_classes=["footer"]
        )
    
    return demo


# =============================================================================
# Tab: Urban Heat Island Detection
# =============================================================================

def create_uhi_tab():
    """Create the Urban Heat Island analysis tab."""
    
    gr.Markdown(
        """
        ### Urban Heat Island Detection
        
        Analyze thermal satellite imagery to identify urban heat islands.
        Uses Landsat-8 Band 10 (Thermal Infrared) for Land Surface Temperature estimation.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            uhi_input = gr.Image(
                label="Thermal Band Image",
                type="numpy",
                sources=["upload", "clipboard"],
                height=300
            )
            
            with gr.Row():
                uhi_sample_btn = gr.Button("📥 Load Sample", size="sm")
                uhi_analyze_btn = gr.Button("🔥 Analyze", variant="primary")
            
            # Parameters
            with gr.Accordion("⚙️ Parameters", open=False):
                uhi_threshold = gr.Slider(
                    minimum=0.5, maximum=3.0, value=1.5, step=0.1,
                    label="UHI Threshold (σ above mean)"
                )
                uhi_colormap = gr.Dropdown(
                    choices=["hot", "jet", "viridis", "inferno"],
                    value="hot",
                    label="Colormap"
                )
        
        with gr.Column(scale=2):
            # Output section
            with gr.Row():
                uhi_heatmap = gr.Image(label="Temperature Map", height=250)
                uhi_hotspots = gr.Image(label="Heat Island Hotspots", height=250)
            
            uhi_stats = gr.JSON(label="Statistics")
            uhi_plot = gr.Plot(label="Temperature Distribution")
    
    # Event handlers
    uhi_sample_btn.click(
        fn=lambda: load_sample("uhi"),
        outputs=uhi_input
    )
    
    uhi_analyze_btn.click(
        fn=analyze_uhi,
        inputs=[uhi_input, uhi_threshold, uhi_colormap],
        outputs=[uhi_heatmap, uhi_hotspots, uhi_stats, uhi_plot]
    )


# =============================================================================
# Tab: Urban Expansion / LULC Change
# =============================================================================

def create_lulc_tab():
    """Create the Land Use Change analysis tab."""
    
    gr.Markdown(
        """
        ### Urban Expansion Analysis
        
        Compare two time periods to detect land use changes and urban growth.
        Upload Sentinel-2 imagery from different dates to visualize expansion.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Dual image input for before/after
            gr.Markdown("#### Time Period 1 (Earlier)")
            lulc_before = gr.Image(
                label="Before Image",
                type="numpy",
                height=200
            )
            
            gr.Markdown("#### Time Period 2 (Later)")
            lulc_after = gr.Image(
                label="After Image", 
                type="numpy",
                height=200
            )
            
            with gr.Row():
                lulc_sample_btn = gr.Button("📥 Load Samples", size="sm")
                lulc_analyze_btn = gr.Button("🔄 Detect Changes", variant="primary")
            
            # Parameters
            with gr.Accordion("⚙️ Parameters", open=False):
                lulc_sensitivity = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                    label="Change Sensitivity"
                )
        
        with gr.Column(scale=2):
            # Output section
            with gr.Row():
                lulc_before_seg = gr.Image(label="Classification T1", height=200)
                lulc_after_seg = gr.Image(label="Classification T2", height=200)
            
            lulc_change_map = gr.Image(label="Change Detection Map", height=250)
            
            with gr.Row():
                lulc_stats = gr.JSON(label="Change Statistics")
                lulc_plot = gr.Plot(label="Land Use Comparison")
    
    # Event handlers
    lulc_sample_btn.click(
        fn=lambda: load_sample("lulc"),
        outputs=[lulc_before, lulc_after]
    )
    
    lulc_analyze_btn.click(
        fn=analyze_lulc_change,
        inputs=[lulc_before, lulc_after, lulc_sensitivity],
        outputs=[lulc_before_seg, lulc_after_seg, lulc_change_map, lulc_stats, lulc_plot]
    )


# =============================================================================
# Tab: Plastic Detection
# =============================================================================

def create_plastic_tab():
    """Create the Floating Plastic Detection tab."""
    
    gr.Markdown(
        """
        ### Floating Plastic Detection
        
        Detect floating debris and plastic pollution in water bodies using 
        spectral indices like NDPI (Normalized Difference Plastic Index) 
        and FAI (Floating Algae Index).
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input
            plastic_input = gr.Image(
                label="Satellite Image (Water Body)",
                type="numpy",
                height=300
            )
            
            with gr.Row():
                plastic_sample_btn = gr.Button("📥 Load Sample", size="sm")
                plastic_analyze_btn = gr.Button("🔍 Detect Plastics", variant="primary")
            
            # Parameters
            with gr.Accordion("⚙️ Parameters", open=False):
                plastic_index = gr.Dropdown(
                    choices=["NDPI", "FAI", "FDI", "Combined"],
                    value="Combined",
                    label="Detection Index"
                )
                plastic_threshold = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Detection Threshold"
                )
        
        with gr.Column(scale=2):
            # Outputs
            with gr.Row():
                plastic_water_mask = gr.Image(label="Water Mask", height=200)
                plastic_detection = gr.Image(label="Plastic Detection", height=200)
            
            plastic_overlay = gr.Image(label="Detection Overlay", height=250)
            
            with gr.Row():
                plastic_stats = gr.JSON(label="Detection Statistics")
                plastic_plot = gr.Plot(label="Index Distribution")
    
    # Event handlers
    plastic_sample_btn.click(
        fn=lambda: load_sample("plastic"),
        outputs=plastic_input
    )
    
    plastic_analyze_btn.click(
        fn=detect_plastics,
        inputs=[plastic_input, plastic_index, plastic_threshold],
        outputs=[plastic_water_mask, plastic_detection, plastic_overlay, plastic_stats, plastic_plot]
    )


# =============================================================================
# Tab: Explainable AI
# =============================================================================

def create_xai_tab():
    """Create the Explainable AI visualization tab."""
    
    gr.Markdown(
        """
        ### Explainable AI Visualization
        
        Understand model predictions using SHAP and LIME explanations.
        Visualize which regions and spectral bands contribute most to the prediction.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input
            xai_input = gr.Image(
                label="Input Image",
                type="numpy",
                height=250
            )
            
            xai_model = gr.Dropdown(
                choices=["LULC Classifier", "LST Estimator", "Plastic Detector"],
                value="LULC Classifier",
                label="Model"
            )
            
            xai_method = gr.Radio(
                choices=["SHAP", "LIME", "GradCAM"],
                value="GradCAM",
                label="Explanation Method"
            )
            
            xai_target_class = gr.Dropdown(
                choices=["Auto (Top Prediction)", "Urban", "Vegetation", "Water", "Bare Soil"],
                value="Auto (Top Prediction)",
                label="Target Class"
            )
            
            with gr.Row():
                xai_sample_btn = gr.Button("📥 Load Sample", size="sm")
                xai_explain_btn = gr.Button("🧠 Generate Explanation", variant="primary")
        
        with gr.Column(scale=2):
            # Outputs
            with gr.Row():
                xai_original = gr.Image(label="Original", height=200)
                xai_overlay = gr.Image(label="Explanation Overlay", height=200)
            
            xai_heatmap = gr.Image(label="Attention Heatmap", height=250)
            
            with gr.Row():
                xai_importance = gr.BarPlot(
                    title="Feature Importance",
                    x="feature",
                    y="importance",
                    height=200
                )
            
            xai_info = gr.JSON(label="Explanation Details")
    
    # Event handlers
    xai_sample_btn.click(
        fn=lambda: load_sample("xai"),
        outputs=xai_input
    )
    
    xai_explain_btn.click(
        fn=generate_explanation,
        inputs=[xai_input, xai_model, xai_method, xai_target_class],
        outputs=[xai_original, xai_overlay, xai_heatmap, xai_importance, xai_info]
    )


def generate_explanation(image, model_name, method, target_class):
    """Route to appropriate explanation method."""
    if method == "SHAP":
        return generate_shap_explanation(image, model_name, target_class)
    elif method == "LIME":
        return generate_lime_explanation(image, model_name, target_class)
    else:  # GradCAM
        from utils.xai_visualization import generate_gradcam_explanation
        return generate_gradcam_explanation(image, model_name, target_class)


# =============================================================================
# Tab: About
# =============================================================================

def create_about_tab():
    """Create the About/Documentation tab."""
    
    gr.Markdown(
        """
        ## About This Demo
        
        This interactive dashboard demonstrates Earth Observation AI capabilities 
        for environmental monitoring using satellite imagery.
        
        ### Models Used
        
        | Model | Architecture | Input | Task |
        |-------|--------------|-------|------|
        | LULC Classifier | U-Net + Attention | Sentinel-2 (10 bands) | Land Use Classification |
        | LST Estimator | ResNet-CNN | Landsat-8 Thermal | Temperature Mapping |
        | Plastic Detector | Multi-head CNN | Sentinel-2 | Water Pollution Detection |
        
        ### Data Sources
        
        - **Sentinel-2**: 10m resolution optical imagery from ESA
        - **Landsat-8**: 30m resolution including thermal bands from USGS
        
        ### Spectral Indices
        
        - **NDVI**: Normalized Difference Vegetation Index
        - **NDWI**: Normalized Difference Water Index  
        - **NDBI**: Normalized Difference Built-up Index
        - **NDPI**: Normalized Difference Plastic Index
        - **LST**: Land Surface Temperature
        
        ### Explainable AI Methods
        
        - **GradCAM**: Gradient-weighted Class Activation Mapping
        - **SHAP**: SHapley Additive exPlanations
        - **LIME**: Local Interpretable Model-agnostic Explanations
        
        ### Citation
        
        If you use this demo in your research, please cite:
        
        ```bibtex
        @software{eo_ai_pipeline,
            title = {Earth Observation AI Pipeline},
            author = {Your Name},
            year = {2024},
            url = {https://github.com/yourusername/octo}
        }
        ```
        
        ### License
        
        MIT License - Free for academic and commercial use.
        
        ### Contact
        
        - GitHub: [yourusername/octo](https://github.com/yourusername/octo)
        - Email: your.email@example.com
        """
    )


# =============================================================================
# Custom CSS
# =============================================================================

CUSTOM_CSS = """
/* Global styles */
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}

/* Tab styling */
.tab-nav button {
    font-size: 1.1em !important;
    padding: 12px 24px !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

/* Card-like containers */
.gr-box {
    border-radius: 12px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
}

/* Button styling */
.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

/* Footer */
.footer {
    margin-top: 20px;
    padding: 20px;
    border-top: 1px solid #e5e7eb;
}

/* Image containers */
.image-container {
    border-radius: 8px;
    overflow: hidden;
}

/* Stats display */
.stats-box {
    background: #f8fafc;
    padding: 16px;
    border-radius: 8px;
}
"""
