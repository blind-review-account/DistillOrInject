
## Running the experiments

the following scripts are used to run the experiments :
- `Inject_Java250.sh`
- `Inject_Devign.sh`

They follow the same overall structure:

1. **Conda environment**  
   - Check for `conda` on your system (or source it from `~/miniconda3`).  
   - Create an `ast-gnn` environment from `environment.yml` if it doesn’t already exist.  
   - Activate `ast-gnn`.

2. **Data download (Java250 only)**  
   - For `Inject_Java250.sh`, if `./data/` is missing, fetch a zip from Google Drive via `gdown`, unzip it, and clean up.  
   - `Inject_Devign.sh` skips this step (assumes you’ve already placed Devign data under `./data/`).

3. **TORCH / CUDA detection**  
   - Probe your Python `torch` install for its version and CUDA suffix.  
   - Export `TORCH` and `CUDA` environment variables for downstream use.

4. **Config directory layout**  
   All configs live under:

Inside each `<feature-extractor-name>` folder you’ll find two types of YAMLs:
- **`feature_extraction_*.yaml`** → fed into `preprocessing_parser.py`  
- **`*_experiment_*.yaml`**   → fed into `experiment_parser.py`

5. **Per-Pretrained Transformer extractor workflow**  
For each pretrained transformer (e.g. `microsoft_codebert_base`, `microsoft_unixcoder_base_nine`, `salesforce_codet5p_220m`, and the GNN-only baseline), the script runs two phases in sequence:

```bash
# 1) Feature extraction
python "${CONFIG_BASE}/<extractor>/preprocessing_parser.py" \
  --config "${CONFIG_BASE}/<extractor>/feature_extraction_<...>.yaml"

# 2) Training experiments
python "${CONFIG_BASE}/<extractor>/experiment_parser.py" \
  --config "${CONFIG_BASE}/<extractor>/gcn_experiment_<...>.yaml"
python "${CONFIG_BASE}/<extractor>/experiment_parser.py" \
  --config "${CONFIG_BASE}/<extractor>/gat_experiment_<...>.yaml"
python "${CONFIG_BASE}/<extractor>/experiment_parser.py" \
  --config "${CONFIG_BASE}/<extractor>/transformer_experiment_<...>.yaml"
```

6. **Experiment parsers**  
   Each set of experiments contains two experiment specific parser files, driven by YAML configs:

   - `preprocessing_parser.py`  
     - Loads `feature_extraction_*.yaml`  
     - Runs your chosen parser (e.g. Python AST) + pretrained model to build and save `.pt` feature files  
     - Outputs into `data/java250/Inject/<extractor>/…` or `data/devign/Inject/<extractor>/…`  

   - `experiment_parser.py`  
     - Loads `*_experiment_*.yaml`  
     - Reads the pre‐saved `.pt` features, instantiates the GNN model (GCN/GAT/Transformer), and runs training  
     - Performs validation, early‐stopping, and (optionally) W&B logging 

7. **Output directories**  
   By default you’ll get two sibling folders:

   - **`./wandb/`** (if `log_wandb: true`)  
     - Contains W&B run metadata, metrics history, system logs, and links to the online dashboard  
     - Organized by project → entity → run name  

   - **`./experimental_results/.../<extractor>/`**  
     - Saved model checkpoints (e.g. `<gnn_type>_<extractor>.pt`)  
     - A `<gnn_type>_<extractor>_history.json` file with per‐epoch and per‐batch metrics (loss, accuracy, gradient norms, etc.)  
     - Mirrors the extractor name so you can compare results across feature‐extractors side‐by‐side  

    
    wandb usage can be turned off by not providing a `wandb_config.yaml` file at the project's root level.


### Real-time Tracking with Weights & Biases

If you’d like to monitor your runs live on W&B, create a file at the project root:

```yaml
# wandb_config.yaml
wandb_api_key: "YOUR API KEY"
entity:           "YOUR ENTITY NAME"
log_wandb:        true
