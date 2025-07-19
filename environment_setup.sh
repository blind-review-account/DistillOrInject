#!/usr/bin/env bash
set -euo pipefail

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ——— Ensure conda is installed ———
if ! command -v conda &> /dev/null; then
  echo "Conda not found. Running install_conda.sh…"
  bash "${SCRIPT_DIR}/install_conda.sh"
fi

eval "$(conda shell.bash hook)"
conda env create -f environment.yml
conda activate ast-gnn


# Get the "TORCH" part (strip off any +cu… suffix)
export TORCH=$(python - <<EOF
import torch
print(torch.__version__.split('+')[0])
EOF
)

# Get the "CUDA" part (or 'cpu' if none)
export CUDA=$(python - <<EOF
import torch
cuda_ver = torch.version.cuda
print('cpu' if cuda_ver is None else 'cu' + cuda_ver.replace('.',''))
EOF
)

echo "Using TORCH=$TORCH and CUDA=$CUDA"
