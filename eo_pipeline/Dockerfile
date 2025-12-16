# Earth Observation AI Pipeline
# Docker configuration for containerized deployment

# Base image with CUDA support (use python:3.12-slim for CPU only)
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 eoapps
USER eoapps
WORKDIR /home/eoapps

# Set up Python virtual environment
RUN python3.12 -m venv /home/eoapps/venv
ENV PATH="/home/eoapps/venv/bin:$PATH"

# Install Python dependencies
COPY --chown=eoapps:eoapps requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY --chown=eoapps:eoapps . /home/eoapps/eo_pipeline

# Set working directory
WORKDIR /home/eoapps/eo_pipeline

# Default command
CMD ["python", "-m", "eo_pipeline.main"]

# --- Build instructions ---
# 
# Build the image:
#   docker build -t eo-pipeline:latest .
#
# Run with GPU:
#   docker run --gpus all -it eo-pipeline:latest
#
# Run with CPU only:
#   docker run -it eo-pipeline:latest
#
# Mount data volume:
#   docker run --gpus all -v /path/to/data:/data -it eo-pipeline:latest
#
# Run Gradio demo:
#   docker run --gpus all -p 7860:7860 eo-pipeline:latest python demo.py
