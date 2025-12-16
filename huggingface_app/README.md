---
title: Earth Observation AI Demo
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.10.0
app_file: app.py
pinned: false
license: mit
tags:
  - earth-observation
  - satellite-imagery
  - urban-heat-island
  - land-use-change
  - plastic-detection
  - explainable-ai
---

# 🛰️ Earth Observation AI Dashboard

Interactive demo for satellite imagery analysis using deep learning and explainable AI.

## Features

- 🌡️ **Urban Heat Island Detection** - Identify thermal hotspots from Landsat-8 data
- 🏙️ **Urban Expansion Analysis** - Track land use changes over time
- 🌊 **Floating Plastic Detection** - Detect water pollution using spectral indices
- 🔍 **Explainable AI** - Understand predictions with GradCAM, SHAP, and LIME

## Quick Start

1. Select an analysis tab
2. Upload a satellite image or click "Load Sample"
3. Adjust parameters if needed
4. Click the analyze button
5. Explore results and explanations

## Models

| Model | Task | Status |
|-------|------|--------|
| LULC Classifier | Land Use Classification | Placeholder |
| LST Estimator | Temperature Mapping | Placeholder |
| Plastic Detector | Pollution Detection | Placeholder |

## Upload Your Own Models

1. Save your trained model: `torch.save(model.state_dict(), "model.pt")`
2. Add to the `models/` directory
3. Update the loading logic in `utils/` modules

## Local Development

```bash
# Clone repository
git clone https://huggingface.co/spaces/yourusername/eo-ai-demo

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

## Citation

```bibtex
@software{eo_ai_demo,
    title = {Earth Observation AI Demo},
    author = {Your Name},
    year = {2024},
    url = {https://huggingface.co/spaces/yourusername/eo-ai-demo}
}
```

## License

MIT License
