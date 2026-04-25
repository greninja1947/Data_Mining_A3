#!/usr/bin/env bash
# One-time conda environment setup on IIT Delhi HPC.
# Run this ONCE on the login node before submitting any PBS jobs:
#   bash setup_hpc_env.sh
#
# Creates a conda env called "col761" in your home directory with
# PyTorch (CUDA 11.8) and all required packages.

set -euo pipefail

ENV_NAME="col761"
ENV_PATH="$HOME/$ENV_NAME"

module load apps/anaconda/3 2>/dev/null || true

echo "=== Creating conda env at $ENV_PATH ==="
if [[ -d "$ENV_PATH" ]]; then
    echo "env already exists — skipping creation (remove $ENV_PATH to recreate)"
else
    conda create --prefix "$ENV_PATH" python=3.10 -y
fi

source activate "$ENV_PATH"

echo "=== Installing PyTorch (cu118) ==="
pip install torch==2.1.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118 \
    --quiet

echo "=== Installing PyG ==="
pip install torch_geometric==2.4.0 --quiet
# pyg-lib and torch-sparse are needed for NeighborLoader on the server
pip install pyg-lib torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
    --quiet

echo "=== Installing other deps ==="
pip install scikit-learn numpy tqdm --quiet

echo ""
echo "Done. Verify with:"
echo "  source activate $ENV_PATH"
echo "  python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
