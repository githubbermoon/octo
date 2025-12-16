# 🛰️ Earth Observation AI Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MLFlow](https://img.shields.io/badge/MLFlow-2.9+-0194E2.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.30+-945DD6.svg)](https://dvc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **scalable, hardware-agnostic** PyTorch framework for Earth Observation AI, featuring land-use classification, Urban Heat Island analysis, and water quality detection from satellite imagery.

---

## 🌟 Features

- **🔧 Modular Architecture**: Clean, extensible codebase with separate modules for data, models, training, and explainability
- **⚡ Hardware Agnostic**: Automatic device selection (CUDA → MPS → CPU) with mixed precision training
- **📊 Experiment Tracking**: Full MLFlow integration for metrics, parameters, and model versioning
- **📦 Data Versioning**: DVC pipelines for reproducible data processing and model training
- **🐳 Containerized**: Docker support with GPU/CPU switching and multi-service compose
- **🌐 Web Interface**: Gradio demo for interactive model inference
- **🔍 Explainable AI**: SHAP, LIME, and GradCAM hooks for model interpretation

---

## 📁 Project Structure

```
octo/
├── eo_pipeline/           # Core ML package
│   ├── core/              # Device management, tile grid
│   ├── config/            # Configuration system
│   ├── data/              # Data loaders, preprocessors
│   ├── indices/           # Spectral indices (NDVI, NDWI, etc.)
│   ├── models/            # LULC, LST, Water Quality models
│   ├── training/          # Trainer, Evaluator
│   ├── explainability/    # SHAP, LIME, GradCAM
│   └── integrations/      # MLFlow, DVC, Gradio hooks
├── mlops/                 # MLOps infrastructure
│   ├── mlflow/            # Experiment tracking
│   ├── dvc/               # Data versioning guides
│   └── docker/            # Container configs
├── app/                   # Web interface
├── scripts/               # Training/eval entry points
├── configs/               # YAML configurations
├── data/                  # Data directory (DVC tracked)
├── models/                # Model checkpoints (DVC tracked)
├── docker-compose.yml     # Multi-container orchestration
├── dvc.yaml               # DVC pipeline definition
└── params.yaml            # Shared parameters
```

---

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
# Clone repository
git clone https://github.com/yourusername/octo.git
cd octo

# Create conda environment
conda create -n eo-pipeline python=3.12 -y
conda activate eo-pipeline

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### 2. Start MLFlow Server

```bash
# Option A: Local server
mlflow server --host 0.0.0.0 --port 5000

# Option B: Docker Compose (recommended)
docker-compose up -d mlflow minio
```

### 3. Initialize DVC

```bash
# Initialize DVC
dvc init

# Configure remote storage (choose one)
dvc remote add -d storage s3://your-bucket/dvc-storage        # AWS S3
dvc remote add -d storage gs://your-bucket/dvc-storage        # Google Cloud
dvc remote add -d storage gdrive://folder_id                   # Google Drive

# Pull existing data (if available)
dvc pull
```

### 4. Train a Model

```bash
# Train LULC classifier
python scripts/train.py --model lulc --config configs/train_lulc.yaml --mlflow

# Or use DVC pipeline
dvc repro train_lulc
```

### 5. Launch Demo Interface

```bash
# Local
python app/gradio_app.py --port 7860

# Docker
docker-compose up gradio
```

---

## 🧠 Models

| Model | Task | Architecture | Input |
|-------|------|--------------|-------|
| **LULCClassifier** | Land Use Classification | U-Net + Attention | Sentinel-2 (10 bands) |
| **LSTEstimator** | Temperature Estimation | ResNet-style CNN | Landsat-8/9 Thermal |
| **WaterQualityDetector** | Multi-task Detection | Multi-head CNN | Sentinel-2 |

### Usage Example

```python
from eo_pipeline.models import LULCClassifier
from eo_pipeline.core import DeviceManager

# Load model
model = LULCClassifier(in_channels=10, num_classes=10)
model.to_device(DeviceManager.get_device())

# Load checkpoint
model.load_checkpoint("models/checkpoints/lulc/best_model.pt")

# Predict
predictions, confidence = model.predict_with_confidence(input_tensor)
```

---

## 📊 MLFlow Experiment Tracking

```python
from mlops.mlflow.mlflow_config import ExperimentTracker

tracker = ExperimentTracker("lulc-classification")

with tracker.start_run("experiment_v1"):
    tracker.log_params({"lr": 0.001, "epochs": 100})
    
    for epoch in range(100):
        # ... training loop ...
        tracker.log_metrics({"loss": loss, "iou": iou}, step=epoch)
    
    tracker.log_model(model, "model", registered_name="lulc-classifier")
```

**Access MLFlow UI**: http://localhost:5000

---

## 📦 DVC Data Pipelines

```bash
# View pipeline DAG
dvc dag

# Run full pipeline
dvc repro

# Run specific stage
dvc repro train_lulc

# Push data to remote
dvc push

# Pull data from remote
dvc pull
```

See [mlops/dvc/remote_setup.md](mlops/dvc/remote_setup.md) for detailed configuration.

---

## 🐳 Docker Deployment

```bash
# Build images
docker build -t eo-pipeline:cpu .
docker build -t eo-pipeline:gpu --build-arg BASE_IMAGE=nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04 .

# Run with Docker Compose
docker-compose up -d                    # All services
docker-compose --profile gpu up         # With GPU support
docker-compose up mlflow minio gradio   # Specific services

# View logs
docker-compose logs -f training
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| mlflow | 5000 | Experiment tracking |
| minio | 9000/9001 | S3-compatible storage |
| gradio | 7860 | Web interface |
| jupyter | 8888 | Development notebooks |

---

## 🔍 Explainable AI

```python
from eo_pipeline.explainability import GradCAM, SHAPExplainer

# GradCAM
gradcam = GradCAM(model, target_layer="encoder.layer4")
heatmap = gradcam.generate(input_tensor, target_class=5)
overlay = gradcam.overlay_on_image(image, heatmap)

# SHAP
explainer = SHAPExplainer(model, band_names=["B2", "B3", "B4", ...])
shap_values = explainer.explain(input_tensor)
importance = explainer.compute_band_importance(shap_values["shap_values"])
```

---

## 📈 Spectral Indices

```python
from eo_pipeline.indices import SpectralIndices, calculate_ndvi, calculate_ndwi

# Individual index
ndvi = calculate_ndvi(nir_band, red_band)

# Multiple indices
calculator = SpectralIndices()
indices = calculator.compute_all(
    image,
    bands=["B2", "B3", "B4", "B8", "B11"]
)
# Returns: {"ndvi": ..., "ndwi": ..., "ndbi": ..., ...}
```

---

## 🌍 Data Sources (Placeholder)

The framework includes placeholder hooks for:

- **[SentinelHub](https://www.sentinel-hub.com/)**: Sentinel-2 optical imagery
- **[Google Earth Engine](https://earthengine.google.com/)**: Landsat thermal data
- **[Planetary Computer](https://planetarycomputer.microsoft.com/)**: Various datasets

See `eo_pipeline/data/sentinel_hub.py` and `eo_pipeline/data/gee.py` for API integration templates.

---

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=eo_pipeline --cov-report=html

# Type checking
mypy eo_pipeline/
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- PyTorch team for the amazing ML framework
- MLFlow for experiment tracking
- DVC for data versioning
- ESA for Sentinel-2 data
- USGS for Landsat data

---

## 📞 Contact

- GitHub Issues: [Report a bug](https://github.com/yourusername/octo/issues)
- Email: your.email@example.com
