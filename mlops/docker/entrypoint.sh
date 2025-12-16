#!/bin/bash
# =============================================================================
# Docker Entrypoint Script
# =============================================================================
# Handles GPU/CPU detection and environment setup before running commands.
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Color codes for output
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Print banner
# -----------------------------------------------------------------------------
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         Earth Observation AI Pipeline                            ║"
echo "║         Geospatial Intelligence at Scale                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# -----------------------------------------------------------------------------
# Detect available hardware
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[INFO] Detecting hardware...${NC}"

DEVICE="cpu"

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
        echo -e "${GREEN}[✓] NVIDIA GPU detected: ${GPU_NAME} (${GPU_MEMORY})${NC}"
        DEVICE="cuda"
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    fi
fi

# Check for Apple Silicon MPS
if [ "$DEVICE" = "cpu" ]; then
    python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null && {
        echo -e "${GREEN}[✓] Apple Silicon MPS detected${NC}"
        DEVICE="mps"
    }
fi

# Fallback to CPU
if [ "$DEVICE" = "cpu" ]; then
    echo -e "${YELLOW}[!] No GPU detected, using CPU${NC}"
fi

export TORCH_DEVICE="$DEVICE"
echo -e "${BLUE}[INFO] Using device: ${TORCH_DEVICE}${NC}"

# -----------------------------------------------------------------------------
# Verify MLFlow connection
# -----------------------------------------------------------------------------
if [ -n "$MLFLOW_TRACKING_URI" ]; then
    echo -e "${YELLOW}[INFO] MLFlow URI: ${MLFLOW_TRACKING_URI}${NC}"
    
    # Try to connect (with timeout)
    if curl -s --connect-timeout 5 "${MLFLOW_TRACKING_URI}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}[✓] MLFlow server is reachable${NC}"
    else
        echo -e "${YELLOW}[!] MLFlow server not reachable (will retry on first use)${NC}"
    fi
fi

# -----------------------------------------------------------------------------
# Verify DVC configuration
# -----------------------------------------------------------------------------
if [ -f "dvc.yaml" ]; then
    echo -e "${GREEN}[✓] DVC pipeline found${NC}"
fi

# -----------------------------------------------------------------------------
# Run the command
# -----------------------------------------------------------------------------
echo -e "${BLUE}[INFO] Starting application...${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

exec "$@"
