import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.nn.init as init

from model.gnn_encoder import *
from model.node_type_layer import NodeEmbedding
from model.gnn_decoder import Cls_Decoder

class GNN_Classifier(nn.Module):
    def __init__(self, gnn_args: dict = None,
                 n_emb=250, pos_dim=32, in_dim=801, emb_h_dim=1024, emb_out_dim=1024,
                 gnn_h_dim=1024, dropout=0.25, num_classes=250, gnn_type='GCN',
                 cls_pooling_method : str = "attentional", 
                 device=torch.device("cuda")):
        
        super().__init__()
        
        self.device = torch.device(device)
        self.n_emb = n_emb
        self.pos_dim = pos_dim
        self.in_dim = in_dim
        self.emb_h_dim = emb_h_dim
        self.emb_out_dim = emb_out_dim
        self.gnn_h_dim = gnn_h_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.gnn_type = gnn_type
        self.cls_pooling_method = cls_pooling_method
        
        # Initialize NodeEmbedding
        self.node_embedding = NodeEmbedding(n_emb = n_emb, pos_dim = pos_dim, 
                                            in_dim = in_dim, h_dim = emb_h_dim, 
                                            out_dim = emb_out_dim, dropout = dropout, 
                                            device = device)
        # Mapping of GNN types to respective classes
        gnn_types = {
            'GCN': GCN_Encoder,
            'GIN': GIN_Encoder,
            'SAGE': SAGE_Encoder,
            'GAT': GAT_Encoder,
            "Transformer" : Transformer_Encoder
        }

        # Ensure valid gnn_type
        if gnn_type not in gnn_types:
            raise ValueError(f"Unsupported GNN type: {gnn_type}. Available types: {', '.join(gnn_types.keys())}")

        #Get the encoder callable
        gnn_encoder = gnn_types[gnn_type]
        
        if gnn_args is None:
            gnn_args = {}
        gnn_args = dict(gnn_args)  # shallow copy
        gnn_args.update({
            'in_dim':      emb_out_dim,
            'hidden_dim':  gnn_h_dim,
            'out_dim':     gnn_h_dim,
            'dropout':     dropout,
            'device':      device
        })
        
        self.gnn_encoder = gnn_encoder(**gnn_args)
        
        # Decoder for classification
        self.cls_decoder = Cls_Decoder(in_dim=gnn_h_dim, 
                                       h_dim=gnn_h_dim, 
                                       out_dim=num_classes, 
                                       dropout=dropout, 
                                       pooling_method = cls_pooling_method,
                                       device=device)
        self.to(device)

    def forward(self, input_graph: Data):
        input_graph = input_graph.to(self.device)
        # Step 1: Get node embeddings
        input_graph = self.node_embedding(input_graph)
        # Step 2: Pass the node embeddings through the GNN layers
        x = self.gnn_encoder(input_graph)
        # Step 3: Get classification logits from the decoder
        input_graph.x = x  # Update node features with the output from the GNN layers
        logits = self.cls_decoder(input_graph)
        return logits