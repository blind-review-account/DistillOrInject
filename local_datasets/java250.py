import os
import pandas as pd
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from utils.dataset_processing import gather_java250_files, load_pyg_graph


from torch_geometric.loader import DataLoader as PyGDataLoader


class Java250Dataset(Dataset):
    """
    Dataset wrapping (instance_id, label_str, orig_path) triplets for Java250,
    where `orig_path` might end in .java (or anything). The dataset will auto‐replace
    that suffix (orig_suffix) with `pt_suffix` (e.g. ".pt") before loading.

    Args:
        triplets: List of (instance_id, label_str, orig_path) tuples.
                  For example, the output of gather_java250_files(path), where
                  `orig_path` typically ends in ".java".
        orig_suffix: The current suffix string in `orig_path` (default ".java").
        pt_suffix:   The new suffix you want to load (default ".pt").
        label_mode:  "str" or "one_hot". Determines what __getitem__ returns for the label.
                     - "str": returns raw label_str (e.g. "p00007").
                     - "one_hot": returns a FloatTensor one-hot vector of length num_classes.
        label_to_idx: Optional[dict]: if provided, should map each label_str → integer index.
                      If None, the class will build its own mapping by scanning all label_str
                      in `triplets`.
    """
    def __init__(
        self,
        triplets: List[Tuple[str, str, str]],
        features_path: str = None,
        orig_suffix: str = ".java",
        pt_suffix: str = ".pt",
        label_mode: str = "one_hot",
        label_to_idx: Dict[str, int] = None
    ):
        super().__init__()

        import warnings
        assert label_mode in ("str", "one_hot"), "label_mode must be 'str' or 'one_hot'."
        self.orig_suffix = orig_suffix
        self.pt_suffix = pt_suffix
        self.label_mode = label_mode
        self.features_path = features_path
        
        # 1) Filter triplets, replacing orig_suffix → pt_suffix
        self.triplets: List[Tuple[str, str, str]] = []
        dropped = 0
        for inst_id, label_str, orig_path in triplets:
            if not orig_path.endswith(orig_suffix):
                dropped += 1
                continue
            base_name = os.path.basename(orig_path)[:-len(orig_suffix)]
            if features_path:
                pt_path = os.path.join(features_path, label_str, base_name + pt_suffix)
            else:
                pt_path = os.path.splitext(orig_path)[0] + pt_suffix
            self.triplets.append((inst_id, label_str, pt_path))


        if dropped > 0:
            warnings.warn(f"Dropped {dropped} triplet(s) whose path did not end with '{orig_suffix}'.")

        if len(self.triplets) == 0:
            raise RuntimeError(f"No valid triplets left after filtering with suffix '{orig_suffix}'.")

        # 2) Build (or accept) label_to_idx / idx_to_label
        if label_to_idx is None:
            all_labels = sorted({label for (_id, label, _p) in self.triplets})
            self.label_to_idx = {lab: i for i, lab in enumerate(all_labels)}
        else:
            self.label_to_idx = dict(label_to_idx)  # copy
            missing = {label for (_id, label, _p) in self.triplets} - set(self.label_to_idx.keys())
            if missing:
                raise ValueError(f"The provided label_to_idx mapping is missing these labels: {missing}")

        self.idx_to_label = {i: lab for (lab, i) in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        self.one_hot_matrix = torch.eye(self.num_classes, dtype=torch.float)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor]:
        inst_id, label_str, pt_path = self.triplets[idx]

        if not os.path.isfile(pt_path):
            raise FileNotFoundError(f"Expected preprocessed file at {pt_path} not found.")
        data = load_pyg_graph(pt_path)
        if self.label_mode == "str":
            return data, label_str
        else:
            label_idx = self.label_to_idx[label_str]
            return data, self.one_hot_matrix[label_idx]

    def str_to_onehot(self, label_str: str) -> torch.Tensor:
        if label_str not in self.label_to_idx:
            raise KeyError(f"Label '{label_str}' not found in dataset.")
        idx = self.label_to_idx[label_str]
        return self.one_hot_matrix[idx]

    def onehot_to_str(self, onehot_vec: torch.Tensor) -> str:
        if onehot_vec.ndim != 1 or onehot_vec.shape[0] != self.num_classes:
            raise ValueError(
                f"Expected one-hot vector of length {self.num_classes}, got shape {tuple(onehot_vec.shape)}."
            )
        if (onehot_vec == 1.0).sum().item() != 1:
            raise ValueError("Provided tensor is not a valid one-hot encoding.")
        idx = int(onehot_vec.argmax().item())
        return self.idx_to_label[idx]

    def idx_to_onehot(self, idx: int) -> torch.Tensor:
        if not (0 <= idx < self.num_classes):
            raise IndexError(f"Index {idx} out of bounds [0, {self.num_classes}).")
        return self.one_hot_matrix[idx]

    def onehot_to_idx(self, onehot_vec: torch.Tensor) -> int:
        if onehot_vec.ndim != 1 or onehot_vec.shape[0] != self.num_classes:
            raise ValueError(
                f"Expected one-hot vector of length {self.num_classes}, got shape {tuple(onehot_vec.shape)}."
            )
        if (onehot_vec == 1.0).sum().item() != 1:
            raise ValueError("Provided tensor is not a valid one-hot encoding.")
        return int(onehot_vec.argmax().item())


def get_idx_split_java250(dataset_path: str):
    """
    Reads train, validation, and test indices from CSV split files in the dataset.

    Expects:
      data/Project_CodeNet/java250/
        └── split/
            ├── train.csv
            ├── valid.csv
            ├── test.csv
            ├──train_ids_split.csv
            ├──valid_ids_split.csv
            └──test_ids_split.csv

    Each CSV contains a single column of integer indices (zero-based), one per line.
    The ids_split files contain instance IDs instead

    Returns:
        { "train": [...], "valid": [...], "test": [...] }
    """
    split_folder = os.path.join(dataset_path, "split")
    if not os.path.isdir(split_folder):
        raise FileNotFoundError(f"Split folder not found at: {split_folder}")

    train_ids_csv = os.path.join(split_folder, "train_ids_split.csv")
    valid_ids_csv = os.path.join(split_folder, "valid_ids_split.csv")
    test_ids_csv = os.path.join(split_folder, "test_ids_split.csv")

    # If the split by IDs files exist, load them
    if os.path.isfile(train_ids_csv) and os.path.isfile(valid_ids_csv) and os.path.isfile(test_ids_csv):
        print("Loading splits by instance IDs from CSV files.")
        train_ids = pd.read_csv(train_ids_csv, header=None).iloc[:, 0].tolist()
        valid_ids = pd.read_csv(valid_ids_csv, header=None).iloc[:, 0].tolist()
        test_ids = pd.read_csv(test_ids_csv, header=None).iloc[:, 0].tolist()
    else: #Otherwise create them using the idx method (which tends to be udeterministic)
        print("split files by instance IDs not found. Generating splits using the indices instead.")
        
        #Only do this if you cant fine the ids .csv files (train_ids_split.csv, valid_ids_split.csv, test_ids_split.csv)
        train_idx = pd.read_csv(os.path.join(split_folder, "train.csv"), header=None).iloc[:, 0].tolist()
        valid_idx = pd.read_csv(os.path.join(split_folder, "valid.csv"), header=None).iloc[:, 0].tolist()
        test_idx = pd.read_csv(os.path.join(split_folder, "test.csv"), header=None).iloc[:, 0].tolist()
        
        triplets = gather_java250_files(dataset_path)
        train_ids = [triplets[idx][0] for idx in train_idx] 
        valid_ids = [triplets[idx][0] for idx in valid_idx]
        test_ids = [triplets[idx][0] for idx in test_idx]
        
        # Save the generated splits into CSV files for future use
        pd.DataFrame(train_ids).to_csv(train_ids_csv, header=None, index=False)
        pd.DataFrame(valid_ids).to_csv(valid_ids_csv, header=None, index=False)
        pd.DataFrame(test_ids).to_csv(test_ids_csv, header=None, index=False)
    #Otherwise load those ids splits here
    
    return {"train": train_ids, "valid": valid_ids, "test": test_ids}


def get_split_datasets_java250(
    model_name: str = None,
    use_model_subfolder: bool = True,
    dataset_path: str = "data/Project_CodeNet/java250/",
    orig_suffix: str = ".java",
    pt_suffix: str = ".pt",
    label_mode: str = "one_hot"
) -> Dict[str, Java250Dataset]:
    """
    1) Gather all (inst_id, label_str, orig_path) triplets via gather_java250_files.
    2) Load the CSV splits (train/valid/test indices).
    3) Build a label_to_idx mapping *solely from the training triplets*.
    4) Instantiate Java250Dataset(train), Java250Dataset(valid), Java250Dataset(test)
       all using the same label_to_idx (so validation/test maps to the exact same classes).

    Returns:
        {
          "train": Java250Dataset(train_triplets,  ..., label_to_idx=...),
          "valid": Java250Dataset(valid_triplets,  ..., label_to_idx=...),
          "test":  Java250Dataset(test_triplets,   ..., label_to_idx=...),
        }
    """

    # Normalize and prepare features root
    dataset_path = os.path.abspath(os.path.normpath(dataset_path))
    if use_model_subfolder:
        folder = model_name.replace('/', '_').replace('-', '_') if model_name else 'baseline'
        features_root = os.path.join(dataset_path, folder)
    else:
        features_root = None
    
    # 1) Gather all triplets
    triplets = gather_java250_files(dataset_path)
    if not isinstance(triplets, list) or len(triplets) == 0:
        raise RuntimeError(f"gather_java250_files returned no triplets for path: {dataset_path}")

    # 2) Load train/valid/test indices
    split_ids = get_idx_split_java250(dataset_path)
    train_ids = split_ids["train"]
    valid_ids = split_ids["valid"]
    test_ids  = split_ids["test"]

    #Build mapping for distributing the paths
    triplet_mapping = {instance_id : (instance_id, instance_label, instance_path) for (instance_id, instance_label, instance_path) in triplets}

    # 3) Partition the triplets
    train_triplets = [triplet_mapping[instance_id] for instance_id in train_ids]
    valid_triplets = [triplet_mapping[instance_id] for instance_id in valid_ids]
    test_triplets  = [triplet_mapping[instance_id] for instance_id in test_ids]

    # 4) Build label_to_idx from training set only
    all_train_labels = sorted({label_str for (_id, label_str, _path) in train_triplets})
    label_to_idx = {lab: idx for idx, lab in enumerate(all_train_labels)}

    # 5) Instantiate each split's dataset, reusing the same label_to_idx
    train_dataset = Java250Dataset(
        train_triplets,
        features_path= features_root,
        orig_suffix=orig_suffix,
        pt_suffix=pt_suffix,
        label_mode=label_mode,
        label_to_idx=label_to_idx
    )

    valid_dataset = Java250Dataset(
        valid_triplets,
        features_path= features_root,
        orig_suffix=orig_suffix,
        pt_suffix=pt_suffix,
        label_mode=label_mode,
        label_to_idx=label_to_idx
    )

    test_dataset = Java250Dataset(
        test_triplets,
        features_path = features_root,
        orig_suffix=orig_suffix,
        pt_suffix=pt_suffix,
        label_mode=label_mode,
        label_to_idx=label_to_idx
    )

    return {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset
    }
