import os
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from datasets import load_dataset

from typing import Dict
from torch_geometric.loader import DataLoader as PyGDataLoader
from utils.dataset_processing import load_pyg_graph

class DevignDataset(Dataset):
    def __init__(
        self,
        huggingface_data_path: str = "DetectVul/devign",
        graphs_path: str = "data/devign/",
        split_key: str = "train",
        id_key: str = "id",
        label_key: str = "target",
        pt_suffix: str = ".pt",
        label_mode: str = "one_hot",
        label_to_idx: Dict[str, int] = None
    ):
        super().__init__()
        assert label_mode in ("str", "one_hot"), "label_mode must be 'str' or 'one_hot'."
        self.graphs_path = graphs_path
        self.pt_suffix   = pt_suffix
        self.id_key      = id_key
        self.label_key   = label_key
        self.label_mode  = label_mode

        # load the HuggingFace split
        self.data = load_dataset(huggingface_data_path)[split_key]

        # If no mapping was passed, build one from the unique labels in this split
        if label_to_idx is None:
            unique_labels = sorted({ex[self.label_key] for ex in self.data})
            self.label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx

        # prepare one-hot matrix once
        self.num_classes = len(self.label_to_idx)
        self.one_hot_matrix = torch.eye(self.num_classes, dtype=torch.float)
        
        # store all IDs so we can look up .pt files by index
        self.ids = [ex[self.id_key] for ex in self.data]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor]:
        # 1) which example?
        ex = self.data[idx]
        inst_id = ex[self.id_key]
        raw_label = ex[self.label_key]

        # 2) load the pre-saved PyG Data object
        path = os.path.join(self.graphs_path, f"{inst_id}{self.pt_suffix}")
        pyg_data: Data =  load_pyg_graph(path)

        # 3) build the label tensor
        label_idx = self.label_to_idx[raw_label]
        if self.label_mode == "one_hot":
            return pyg_data, self.one_hot_matrix[label_idx]
        else:
            # just a scalar index
            return pyg_data, torch.tensor(label_idx, dtype=torch.long)


def get_split_devign(huggingface_data_path: str = "DetectVul/devign",
                     graphs_path: str = "data/devign/",
                     model_name: str = None,
                     use_model_subfolder : bool = True,
                     train_split_key: str = "train",
                     valid_split_key: str = "validation",
                     test_split_key: str = "test",
                     id_key: str = "id",
                     label_key: str = "target",
                     pt_suffix: str = ".pt",
                     label_mode: str = "one_hot"):

    # Normalize and prepare features root
    graphs_path = os.path.abspath( os.path.normpath(graphs_path) )
    if use_model_subfolder:
        folder = model_name.replace('/', '_').replace('-', '_') if model_name else 'baseline'
        features_root = os.path.join(graphs_path, folder)
    else:
        features_root = graphs_path
        
    train_dataset = DevignDataset(
        huggingface_data_path = huggingface_data_path,
        graphs_path = features_root,
        split_key = train_split_key,
        id_key = id_key,
        label_key = label_key,
        pt_suffix = pt_suffix,
        label_mode = label_mode
    )

    valid_dataset = DevignDataset(
        huggingface_data_path = huggingface_data_path,
        graphs_path = features_root,
        split_key = valid_split_key,
        id_key = id_key,
        label_key = label_key,
        pt_suffix = pt_suffix,
        label_mode = label_mode,
        label_to_idx=train_dataset.label_to_idx)
    
    test_dataset= DevignDataset(
        huggingface_data_path = huggingface_data_path,
        graphs_path = features_root,
        split_key = test_split_key,
        id_key = id_key,
        label_key = label_key,
        pt_suffix = pt_suffix,
        label_mode = label_mode,
        label_to_idx=train_dataset.label_to_idx)

    return {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset
    }
