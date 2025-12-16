"""
Earth Observation AI Demo - Hugging Face Spaces
================================================

Main application entry point for the Gradio demo.
Showcases UHI detection, urban expansion, plastic detection, and XAI visualization.

Deploy to Hugging Face Spaces:
    1. Create a new Space at huggingface.co/new-space
    2. Select "Gradio" as SDK
    3. Upload this directory
    4. The Space will automatically build and deploy

Local run:
    python app.py
"""

import os
import sys

# Add parent directory for imports if running locally
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from interface import create_interface

# =============================================================================
# Configuration
# =============================================================================

TITLE = "🛰️ Earth Observation AI Dashboard"
DESCRIPTION = """
<div style="text-align: center;">
    <h2>Satellite Imagery Analysis with Explainable AI</h2>
    <p>Analyze satellite data for Urban Heat Islands, Land Use Changes, and Water Pollution</p>
</div>

### Features
- 🌡️ **Urban Heat Island Detection** - Identify hotspots using Landsat-8 thermal bands
- 🏙️ **Urban Expansion Analysis** - Track land use changes with Sentinel-2 imagery  
- 🌊 **Floating Plastic Detection** - Detect pollution in water bodies
- 🔍 **Explainable AI** - Understand model decisions with SHAP/LIME overlays

### How to Use
1. Select an analysis tab
2. Upload a satellite image or use a sample
3. Click "Analyze" to run the model
4. Explore the results and explanations
"""

# Theme configuration - set to None for default (Gradio 6.x compatibility)
THEME = None  # Use default theme for compatibility

# =============================================================================
# Main Application
# =============================================================================

def main():
    """Create and launch the Gradio application."""
    
    # Create the interface
    demo = create_interface(
        title=TITLE,
        description=DESCRIPTION,
        theme=THEME
    )
    
    # Launch configuration
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": int(os.environ.get("PORT", 7860)),
        "share": os.environ.get("GRADIO_SHARE", "false").lower() == "true",
        "show_error": True,
    }
    
    # For Hugging Face Spaces
    if os.environ.get("SPACE_ID"):
        launch_kwargs["show_api"] = True
    
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
