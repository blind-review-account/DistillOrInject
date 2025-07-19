# `utils/` Module README

This folder contains all low-level plumbing for:

- **AST construction**  
- **Dataset specific preprocessing utils & node-type mapping to include node type features**  
- **Feature extraction & PyG conversion**  

---

### `build_ast.py`

- **`build_networkx_ast(source, lang="python", use_posenc=False, pos_dim=16) → nx.DiGraph`**  
  Parse a source string into a Tree-sitter CST, convert it to a NetworkX `DiGraph`, and (optionally) attach Laplacian eigenvector positional encodings (`"pos"` node attribute).

- **`compute_laplacian_pe(G, k=16) → (pe_tensor, node_order)`**  
  Given a directed AST, build an undirected Laplacian, solve for the first `k` nontrivial eigenvectors, and return a `(N, k)` tensor plus the node ordering.

```python
from utils.build_ast import build_networkx_ast

G = build_networkx_ast(py_source, lang="python", use_posenc=True, pos_dim=32)
```

### `feature_extraction.py`

- **Graph + features → PyG**

    -  `nx_ast_and_features_to_pyg(G, feat_matrix, type2idx) → Data`
- **Transformer‐based feature extraction**

    -  `extract_graph_and_features(sources, langs, tokenizer, model, ...) → (feature_matrices, graphs)`
    -  `extract_transformer_features(sources, graphs, tokenizer, model, ...) → (feature_matrices, cleaned_graphs)`


```python
from utils.feature_extraction import extract_graph_and_features
feats, graphs = extract_graph_and_features(
    sources=["def foo(): pass"],
    langs=["python"],
    tokenizer=tok,
    model=mdl,
    use_posenc=True,
    pos_dim=16
)
```


## How It Fits Together

1. **AST construction** (`build_ast.py`)  
2. **Node-type mapping** (`dataset_processing.py`)  
3. **Transformer feature extraction** (`feature_extraction.py`)  
4. **PyG conversion** (`nx_ast_and_features_to_pyg`)

The high-level `preprocess_java250` and `preprocess_huggingfaceDataset` functions orchestrate these steps in batch, saving ready-to-train `.pt` files.
