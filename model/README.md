# Model Module README

This directory contains the heart of our Transformer‐enhanced GNN classifier. You can drop these files into any PyTorch + PyG project and get up and running in minutes.

---

## Files

- **`gnn_classifier.py`**  
  Defines the `GNN_Classifier` wrapper:  
  1. **NodeEmbedding** (from `node_type_layer.py`)  
  2. **GNN encoder** (one of GCN, GIN, SAGE, GAT, Transformer from `gnn_encoder.py`)  
  3. **Cls_Decoder** (from `gnn_decoder.py`)  
  4. Dispatches a `torch_geometric.data.Data` graph through embed → encode → classify.

- **`node_type_layer.py`**  
  `NodeEmbedding` module:  
  - Splits each node’s input vector into  
    - pretrained‐LLM slice (`llm_dim`)  
    - positional embedding slice (`pos_dim`)  
    - node‐type index  
  - Applies Embedding + LayerNorm + MLP to produce a unified `out_dim` feature per node.

- **`gnn_encoder.py`**  
  - **`BaseGNNEncoder`**: generic multi‐layer GNN with configurable  
    - convolution layer (GCNConv, GINConv, SAGEConv, GATConv, TransformerConv)  
    - depth, hidden/out dims, dropout, activation, normalization, self‐loops  
  - **`GCN_Encoder`**, **`GIN_Encoder`**, **`SAGE_Encoder`**, **`GAT_Encoder`**, **`Transformer_Encoder`**:  
    thin wrappers that pick the right `conv_layer` and hyperparameters.

- **`gnn_decoder.py`**  
  `Cls_Decoder` module:  
  - A gated attention (or mean/max/sum) pooling over nodes → graph embedding  
  - Final MLP projecting to `num_classes` logits.

---

## Quick Usage

```python
from model.gnn_classifier import GNN_Classifier
from torch_geometric.data import Data

# 1) Instantiate your model
model = GNN_Classifier(
    gnn_args={ 'depth': 3, 'heads': 4 },   # optional overrides
    n_emb=250,         # number of node‐type embeddings
    pos_dim=32,        # positional encoding size
    in_dim=801,        # total input dim per node (LLM + pos + type)
    emb_h_dim=1024,    # NodeEmbedding hidden dim
    emb_out_dim=1024,  # NodeEmbedding output dim
    gnn_h_dim=1024,    # GNN hidden/output dim
    num_classes=250,   # number of classes to predict
    gnn_type='GAT',    # one of: 'GCN','GIN','SAGE','GAT','Transformer'
    cls_pooling_method='attentional',  # one of: 'attentional','mean','max','sum'
    device='cuda'
)

# 2) Prepare a PyG Data object
#    data.x shape: [num_nodes, in_dim]
#    last column of x is the node‐type index (0 .. n_emb-1)
#    x[:, :llm_dim] are pretrained embeddings, x[:, llm_dim:llm_dim+pos_dim] are pos encodings
#    data.edge_index: [2, num_edges]
#    data.batch: [num_nodes] (for batched graphs)

data = Data(x=..., edge_index=..., batch=...)
logits = model(data)   # [num_graphs, num_classes]

