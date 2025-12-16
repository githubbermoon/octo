# Geospatial AI MLOps Project Structure
# =====================================
#
# This document outlines the recommended folder structure for a scalable,
# hardware-agnostic geospatial AI project.
#
# octo/
# ├── .dvc/                          # DVC configuration (auto-generated)
# ├── .git/                          # Git repository
# ├── .github/
# │   └── workflows/
# │       └── ci.yml                 # GitHub Actions for CI/CD
# ├── configs/
# │   ├── train_lulc.yaml            # Training configs (tracked by Git)
# │   ├── train_lst.yaml
# │   └── experiment_template.yaml
# ├── data/
# │   ├── raw/                       # Raw satellite data (tracked by DVC)
# │   │   ├── sentinel2/
# │   │   └── landsat/
# │   ├── processed/                 # Processed tiles (tracked by DVC)
# │   │   ├── tiles/
# │   │   └── labels/
# │   └── external/                  # External datasets (tracked by DVC)
# ├── models/
# │   ├── checkpoints/               # Training checkpoints (tracked by DVC)
# │   └── production/                # Production-ready models
# ├── outputs/
# │   ├── predictions/               # Model predictions
# │   └── visualizations/            # Generated maps/plots
# ├── notebooks/
# │   ├── 01_data_exploration.ipynb
# │   ├── 02_model_training.ipynb
# │   └── 03_inference_demo.ipynb
# ├── scripts/
# │   ├── train.py                   # Training entry point
# │   ├── evaluate.py                # Evaluation entry point
# │   ├── predict.py                 # Inference entry point
# │   └── preprocess.py              # Data preprocessing
# ├── eo_pipeline/                   # Core ML package (already built)
# │   ├── core/
# │   ├── config/
# │   ├── data/
# │   ├── indices/
# │   ├── models/
# │   ├── training/
# │   ├── explainability/
# │   ├── integrations/
# │   └── utils/
# ├── mlops/
# │   ├── mlflow/
# │   │   ├── mlflow_config.py       # MLFlow utilities
# │   │   └── tracking.py            # Experiment tracking
# │   ├── dvc/
# │   │   └── remote_setup.md        # DVC remote configuration guide
# │   └── docker/
# │       ├── Dockerfile.gpu         # GPU-optimized image
# │       ├── Dockerfile.cpu         # CPU-only image
# │       └── entrypoint.sh          # Container entrypoint
# ├── app/
# │   ├── gradio_app.py              # Gradio web interface
# │   └── api_server.py              # FastAPI backend (optional)
# ├── tests/
# │   ├── test_models.py
# │   ├── test_data.py
# │   └── test_integration.py
# ├── docker-compose.yml             # Multi-container orchestration
# ├── Dockerfile                     # Main Dockerfile with GPU/CPU logic
# ├── dvc.yaml                       # DVC pipeline definition
# ├── dvc.lock                       # DVC pipeline lock (auto-generated)
# ├── params.yaml                    # Shared parameters for DVC
# ├── requirements.txt               # Python dependencies
# ├── requirements-dev.txt           # Development dependencies
# ├── pyproject.toml                 # Project metadata
# ├── .env.example                   # Environment variables template
# ├── .gitignore
# ├── .dvcignore
# └── README.md
