#!/bin/bash
# ============================================================
# ReasonRank B200 Environment Setup (conda)
# ============================================================
set -e

ENV_NAME="${REASONRANK_ENV_NAME:-reasonrank}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "=========================================="
echo "  ReasonRank B200 Environment Setup"
echo "=========================================="

# ---------- 0. Pre-flight checks ----------
echo "[0/6] Pre-flight checks..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Make sure NVIDIA drivers are installed."
    exit 1
fi
echo "GPU Info:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "unknown")
echo "Detected CUDA Version: ${CUDA_VERSION}"

CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
if [ "$CUDA_MAJOR" -lt 12 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -lt 8 ]); then
    echo "WARNING: B200 (Blackwell) requires CUDA >= 12.8. Detected: ${CUDA_VERSION}"
    echo "Please upgrade your CUDA toolkit before proceeding."
fi

# ---------- 1. Create conda environment ----------
echo "[1/6] Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
if conda info --envs 2>/dev/null | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists, skipping creation."
else
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

# Get the conda env's pip/python path directly to avoid conda run buffering
CONDA_PREFIX=$(conda info --envs | grep "^${ENV_NAME} " | awk '{print $NF}')
if [ -z "${CONDA_PREFIX}" ]; then
    CONDA_PREFIX=$(conda info --envs | grep "^${ENV_NAME} " | awk '{print $2}')
fi
ENV_PIP="${CONDA_PREFIX}/bin/pip"
ENV_PYTHON="${CONDA_PREFIX}/bin/python"

echo "Conda env path: ${CONDA_PREFIX}"
echo "Using pip: ${ENV_PIP}"

# ---------- 2. Install PyTorch (auto-detect CUDA version) ----------
# Map driver CUDA version to the best available PyTorch cu index
# CUDA 13.0 is backward compatible, use the highest available cu build
if [ "$CUDA_MAJOR" -ge 13 ]; then
    # CUDA 13.x: PyTorch nightly/latest may have cu130, fallback to cu128
    # Check if cu130 index exists, otherwise use cu128 (forward compatible)
    PYTORCH_CUDA_TAG="cu128"
    # Try cu130 first
    if ${ENV_PIP} index versions torch --index-url https://download.pytorch.org/whl/cu130 &>/dev/null 2>&1; then
        PYTORCH_CUDA_TAG="cu130"
    fi
elif [ "$CUDA_MAJOR" -eq 12 ]; then
    if [ "$CUDA_MINOR" -ge 8 ]; then
        PYTORCH_CUDA_TAG="cu128"
    elif [ "$CUDA_MINOR" -ge 6 ]; then
        PYTORCH_CUDA_TAG="cu126"
    elif [ "$CUDA_MINOR" -ge 4 ]; then
        PYTORCH_CUDA_TAG="cu124"
    else
        PYTORCH_CUDA_TAG="cu121"
    fi
else
    PYTORCH_CUDA_TAG="cu118"
fi

PYTORCH_INDEX="https://download.pytorch.org/whl/${PYTORCH_CUDA_TAG}"
echo "[2/6] Installing PyTorch (CUDA ${CUDA_VERSION} -> ${PYTORCH_CUDA_TAG})..."
echo "  Index: ${PYTORCH_INDEX}"
echo "  This may take a while (downloading ~2GB)..."
${ENV_PIP} install -v torch torchvision torchaudio --index-url ${PYTORCH_INDEX} 2>&1 | tail -20

echo "Verifying PyTorch CUDA..."
${ENV_PYTHON} -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)')
"

# ---------- 3. Install vLLM ----------
echo "[3/6] Installing vLLM..."
${ENV_PIP} install -v vllm 2>&1 | tail -20

# ---------- 4. Install core dependencies ----------
echo "[4/6] Installing core dependencies..."
${ENV_PIP} install -v \
    "transformers>=4.45.0" \
    datasets \
    accelerate \
    pyserini \
    pytrec_eval \
    ftfy \
    dacite \
    toml \
    python-dotenv \
    wandb \
    pandas \
    numpy \
    tqdm \
    huggingface_hub 2>&1 | tail -20

# ---------- 5. Install JDK for pyserini ----------
echo "[5/6] Checking Java for pyserini..."
if ! command -v java &> /dev/null; then
    echo "Java not found. Installing OpenJDK 21 via conda..."
    conda install -n ${ENV_NAME} -c conda-forge openjdk=21 -y
else
    echo "Java found: $(java -version 2>&1 | head -1)"
fi

# ---------- 6. wandb login ----------
echo "[6/6] Setting up wandb..."
if [ -n "${WANDB_API_KEY}" ]; then
    ${CONDA_PREFIX}/bin/wandb login --relogin ${WANDB_API_KEY}
    echo "wandb logged in."
else
    echo "WANDB_API_KEY not set. Run 'wandb login' after activating the environment."
fi

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo "Next: conda activate ${ENV_NAME} && bash start.sh"
