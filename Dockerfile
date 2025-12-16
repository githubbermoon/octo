# =============================================================================
# Multi-Stage Dockerfile with GPU/CPU Switching
# =============================================================================
# Build Options:
#   CPU:  docker build -t eo-pipeline:cpu .
#   GPU:  docker build -t eo-pipeline:gpu --build-arg BASE_IMAGE=nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04 .
#
# Run Options:
#   CPU:  docker run -it eo-pipeline:cpu
#   GPU:  docker run --gpus all -it eo-pipeline:gpu
# =============================================================================

# -----------------------------------------------------------------------------
# Build Arguments
# -----------------------------------------------------------------------------
ARG BASE_IMAGE=python:3.12-slim

# =============================================================================
# Stage 1: Builder - Install dependencies
# =============================================================================
FROM ${BASE_IMAGE} AS builder

# Set build-time environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM ${BASE_IMAGE} AS runtime

# Metadata
LABEL maintainer="EO Pipeline Team" \
    version="1.0" \
    description="Earth Observation AI Pipeline"

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:$PYTHONPATH" \
    PATH="/opt/venv/bin:$PATH" \
    # GPU/CPU auto-detection
    TORCH_DEVICE="" \
    # MLFlow
    MLFLOW_TRACKING_URI="http://localhost:5000" \
    # DVC
    DVC_NO_ANALYTICS=1

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal32 \
    libgeos-c1v5 \
    libproj25 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN useradd -m -u 1000 eouser
WORKDIR /app

# Copy application code
COPY --chown=eouser:eouser . /app

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/outputs /app/logs && \
    chown -R eouser:eouser /app

# Switch to non-root user
USER eouser

# Expose ports
EXPOSE 7860 8888 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"mps\" if torch.backends.mps.is_available() else \"cpu\")}')" || exit 1

# Default entrypoint
COPY --chown=eouser:eouser mlops/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default command (can be overridden)
CMD ["python", "-m", "eo_pipeline.main", "info"]

# =============================================================================
# Stage 3: Development image (optional)
# =============================================================================
FROM runtime AS dev

USER root

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Install Jupyter extensions
RUN pip install jupyterlab jupyter_contrib_nbextensions

USER eouser

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
