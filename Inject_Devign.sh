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
python "${CONFIG_BASE}/preprocessing_parser.py" --config "${CONFIG_BASE}/microsoft_codebert_base/feature_extraction_bert_inject.yaml"
# ———> Training
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_codebert_base/gcn_experiment_bert_inject.yaml"
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_codebert_base/gat_experiment_bert_inject.yaml"
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_codebert_base/transformer_experiment_bert_inject.yaml"

# Experiments with UniXCoder
# ———> Feature extraction
python "${CONFIG_BASE}/preprocessing_parser.py" --config "${CONFIG_BASE}/microsoft_unixcoder_base_nine/feature_extraction_unixcoder_inject.yaml"
# ———> Training
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_unixcoder_base_nine/gcn_experiment_unixcoder_inject.yaml"
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_unixcoder_base_nine/gat_experiment_unixcoder_inject.yaml"
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/microsoft_unixcoder_base_nine/transformer_experiment_unixcoder_inject.yaml"

# Experiments with CodeT5p 220M
# ———> Feature extraction
python "${CONFIG_BASE}/preprocessing_parser.py" --config "${CONFIG_BASE}/salesforce_codet5p_220m/feature_extraction_codet5p220m_inject.yaml"
# ———> Training
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/salesforce_codet5p_220m/gcn_experiment_codet5p220m_inject.yaml"
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/salesforce_codet5p_220m/gat_experiment_codet5p220m_inject.yaml"
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/salesforce_codet5p_220m/transformer_experiment_codet5p220m_inject.yaml"

#Baseline GNN experiments
# ———> Feature extraction
python "${CONFIG_BASE}/preprocessing_parser.py" --config "${CONFIG_BASE}/gnn_baselines/feature_extraction_gnn_baseline.yaml"
# ———> Training
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/gnn_baselines/gcn_experiment_baseline.yaml"
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/gnn_baselines/gat_experiment_baseline.yaml"
python "${CONFIG_BASE}/experiment_parser.py" --config "${CONFIG_BASE}/gnn_baselines/transformer_experiment_baseline.yaml"
