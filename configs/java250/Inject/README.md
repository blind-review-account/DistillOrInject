
# Java250 “Inject” Configs

This directory defines all of the YAML‐driven experiments for the Java250 classification task, under the “Inject” protocol (i.e. with injected pretrained Transformer features).

## Top-level scripts

- **`preprocessing_parser.py`**  parser for feature‐extraction.  
  ```bash
  python preprocessing_parser.py --config <path-to-feature-extraction-yaml>
  ```
    Reads a `feature_extraction_*.yaml`, builds AST+Transformer embeddings, and writes `.pt` files in a subfolder by default following this structure `data/java250/Inject/<extractor>/label/instance_id.pt`.
- **`experiment_parser.py`**  parser for training and evaluation.

    ```bash
    python experiment_parser.py --config <path-to-experiment-yaml>
    ```
    Reads a *_experiment_*.yaml, loads precomputed .pt features, runs GNN training (GCN/GAT/Transformer), and saves checkpoints + metrics.

# Extractor sub‐folders structure 
Each sub-folder follows the same four-YAML convention:
```
<extractor>/
├── feature_extraction_<extractor>_inject.yaml
├── gcn_experiment_<extractor>_inject.yaml
├── gat_experiment_<extractor>_inject.yaml
└── transformer_experiment_<extractor>_inject.yaml
```
- `gnn_baselines` – no pretrained features (AST only with positional encodings)
- `microsoft_codebert_base` 
- `microsoft_unixcoder_base_nine`
- `salesforce_codet5p_220m`

> **Note:**  
> `transformer_experiment_*` refers to our **Transformer-based GNN** variant, _not_ the feature-extractor itself.
