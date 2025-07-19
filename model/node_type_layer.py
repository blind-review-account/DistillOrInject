import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.nn.init as init


class NodeEmbedding(nn.Module): 
    def __init__(self, n_emb = 250, pos_dim = 32 ,in_dim = 801, h_dim = 1024,
                 out_dim = 1024, dropout = 0.25, device = torch.device("cuda")):
        super().__init__()
        self.device = torch.device(device)
        self.dropout = dropout 
        #Dimensions
        self.n_emb = n_emb
        self.pos_dim = pos_dim
        self.in_dim = in_dim 
        
        base_llm_dim = in_dim - pos_dim - 1
        self.use_llm = base_llm_dim > 0
        self.llm_dim = base_llm_dim if self.use_llm else 0
        
        self.h_dim = h_dim 
        self.out_dim = out_dim 
        
        emb_dim = self.llm_dim if self.use_llm else self.h_dim
        self.emb = nn.Embedding(n_emb, emb_dim).to(device)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)  # Stable init
        #Normalization Layers
        if self.use_llm:
            self.llm_norm = nn.LayerNorm(self.llm_dim).to(device)
            self.pos_norm = nn.LayerNorm(self.llm_dim).to(device)
            #positional features projection 
            self.pos_proj = nn.Sequential(
                nn.Linear(self.pos_dim, self.llm_dim),
                nn.LayerNorm(self.llm_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ).to(self.device)
        else:
            # Project combined pos + type embedding to h_dim
            self.pos_norm = nn.LayerNorm(self.h_dim).to(device)
            self.pos_proj = nn.Sequential(
                nn.Linear(self.pos_dim, self.h_dim),
                nn.LayerNorm(self.h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ).to(self.device)
            
        init.kaiming_uniform_(self.pos_proj[0].weight, mode='fan_in', nonlinearity='relu')
        if self.use_llm:
            mlp_in = self.llm_dim * 3
        else:
            mlp_in = self.h_dim * 2
            
        #MLP for final projection
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, self.h_dim),
            nn.LayerNorm(self.h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.h_dim, self.out_dim), 
            nn.LayerNorm(self.out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(self.device)
        
        # Apply He initialization to the Linear layers
        init.kaiming_uniform_(self.mlp[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.mlp[4].weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, input_graph : Data):
        input_graph = input_graph.to(self.device)
        x = input_graph.x
        
        #Extract features
        type_idx = x[:,-1].long()
        pos_raw  = x[:, self.llm_dim : self.llm_dim + self.pos_dim]
        if self.use_llm:
            llm_raw  = x[:, : self.llm_dim]

        #Normalized features 
        type_feat = self.emb(type_idx)
        pos_feat  = self.pos_norm(self.pos_proj(pos_raw))
        if self.use_llm:
            llm_feat  = self.llm_norm(llm_raw)

        #Concatenate features 
        if self.use_llm:
            feats = torch.cat([llm_feat, pos_feat, type_feat], dim = -1)
        else :
            feats = torch.cat([pos_feat, type_feat], dim = -1)

        #Project feats to dimension 
        out = self.mlp(feats)
        input_graph.x = out
        return input_graph