#!/usr/bin/env bash
set -euo pipefail



# ——— Base directories ———
# Resolve the folder this script lives in (POSIX-compatible)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CONFIG_BASE="${SCRIPT_DIR}/configs/devign/Inject"

# ——— Paths ———
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# ——— Load Conda if needed ———
if ! command -v conda &>/dev/null; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    echo "ERROR: conda not found under \$HOME/miniconda3."
    exit 1
  fi
fi

# ——— Create ast-gnn if missing ———
if ! conda env list | grep -qE '^[ *]*ast-gnn[[:space:]]'; then
  echo "ast-gnn env not found → creating from environment.yml…"
  conda env create -f "environment.yml"
fi

eval "$(conda shell.bash hook)"
# ——— Activate ———
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


# Experiments with CodeBERT
# ———> Feature extraction
python "${CONFIG_BASE}/preprocessing_parser.py" --config "${CONFIG_BASE}/microsoft_codebert_base_sweep/feature_extraction_bert_inject.yaml"
# ———> Training
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_codebert_base_sweep/gat_sweep_1.yaml"
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_codebert_base_sweep/gat_sweep_2.yaml"
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_codebert_base_sweep/gat_sweep_3.yaml"

python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_codebert_base_sweep/gat_sweep_4.yaml"
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_codebert_base_sweep/gat_sweep_5.yaml"
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_codebert_base_sweep/gat_sweep_6.yaml"

python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_codebert_base_sweep/gat_sweep_7.yaml"
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_codebert_base_sweep/gat_sweep_8.yaml"
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_codebert_base_sweep/gat_sweep_9.yaml"

python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_codebert_base_sweep/gat_sweep_10.yaml"
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_codebert_base_sweep/gat_sweep_11.yaml"
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_codebert_base_sweep/gat_sweep_12.yaml"

