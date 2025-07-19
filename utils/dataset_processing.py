import os
import math
from operator import itemgetter

import pickle
import torch
import networkx as nx
import warnings
import numpy as np
from tqdm import tqdm    
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

from utils.feature_extraction import extract_graph_and_features, nx_ast_and_features_to_pyg
from utils.feature_extraction import build_networkx_ast

from torch.serialization import add_safe_globals
import torch_geometric

def load_pyg_graph(path): 
    path = os.path.abspath(os.path.normpath(path))
    # 1) Make sure PyG’s Data class is in scope:
    from torch_geometric.data import Data, EdgeAttr 
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data
    
def crawl_directory(path, suffix=None):
    """
    Recursively crawl through the given folder and yield file paths, with optional suffix filtering.

    Args:
        path (str): Root directory to start crawling.
        suffix (str or None): File extension (without leading dot) to filter by. 
            If None, yield all files. Example: "java" will match files ending in ".java".

    Yields:
        str: Full path to each file that matches the suffix (or all files if suffix is None).
    """
    path = os.path.abspath(os.path.normpath(path))
    for root, dirs, files in os.walk(path):
        for filename in files:
            # If suffix is provided, check that filename ends with the extension (case-insensitive).
            if suffix is None or filename.lower().endswith(f".{suffix.lower()}"):
                yield os.path.join(root, filename)



def gather_dataset_files(path, suffix = None) : 
    path = os.path.abspath(os.path.normpath(path))
    data_files = [file for file in crawl_directory(path, suffix) ]
    return data_files 

def gather_java250_files(path): 
    path = os.path.abspath(os.path.normpath(path))
    files = gather_dataset_files(path = path, suffix ="java")
    #Relative paths
    rel_paths = [ os.path.relpath(instance_path, path )for instance_path in files ]
    #Extract the labels for each instance
    labels = [os.path.splitext(rel_p)[0].split(os.sep)[0] for rel_p in rel_paths ]
    ids = [os.path.splitext(rel_p)[0].split(os.sep)[1] for rel_p in rel_paths ]
    
    return [(instance_id, instance_label, instance_path)  
            for instance_id, instance_label, instance_path in 
            zip(ids, labels, files)]

## JAVA250 PROCESSING
def build_node_type_mapping_java250(folder="data/Project_CodeNet/java250/", batch_size=32):
    """
    Crawl all Java files in `folder`, build their NetworkX ASTs (without posenc),
    collect all distinct node‐type strings, sort them, and return a deterministic
    mapping { node_type_str: index }.

    Args:
        folder (str): Root directory containing Java250 subfolders.
        batch_size (int): Number of files to process in each batch.

    Returns:
        dict: { node_type: idx }, where idx runs from 0 .. (N_types-1) in sorted order.
    """
    # 1) Gather all (instance_id, label, path) triplets
    folder = os.path.abspath(os.path.normpath(folder))
    file_triplets = gather_java250_files(folder)
    total = len(file_triplets)
    print(f"Found {total} Java files under '{folder}'")
    num_batches = math.ceil(total / batch_size)
    
    # 2) Collect all distinct node‐type strings
    node_type_set = set()
    
    for batch_idx in tqdm(range(num_batches), desc="Collecting node types"):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = file_triplets[start:end]
        
        # 2a) Read source code snippets for this batch
        sources, langs = [], []
        for instance_id, instance_label, instance_path in batch:
            with open(instance_path, "r", encoding="utf-8", errors="ignore") as f:
                sources.append(f.read())
            langs.append("java")
        
        # 2b) Build NetworkX ASTs (no positional encodings)
        trees = [
            build_networkx_ast(source=snippet, lang=lg, use_posenc=False)
            for snippet, lg in zip(sources, langs)
        ]
        
        # 2c) For each tree, collect its node types
        for G in trees:
            for _, data in G.nodes(data=True):
                nt = data.get("type")
                if nt is not None:
                    node_type_set.add(nt)
    
    # 3) Sort node types for deterministic indexing
    sorted_types = sorted(node_type_set)
    
    # 4) Build mapping { type_str: index }
    type2idx = { nt: idx for idx, nt in enumerate(sorted_types) }
    
    return type2idx

def preprocess_java250(
    dataset_path,
    use_model_subfolder: bool = True,
    model_name = None,
    batch_size: int = 64,
    max_length: int = None,
    stride: int = 32,
    device: str = None,
    clean: bool = True,
    pos_dim: int = 32,
    use_posenc: bool = False,
    type2idx = None
):
    dataset_path = os.path.abspath(os.path.normpath(dataset_path))
    # Determine output root
    if use_model_subfolder:
        if model_name:
            folder_name = model_name.replace('/', '_').replace('-', '_')
        else:
            folder_name = 'baseline'
        out_root = os.path.join(dataset_path, folder_name)
    else:
        out_root = dataset_path
        
    if type2idx is None :
        type2idx = build_node_type_mapping_java250(folder=dataset_path, batch_size=batch_size)

    if model_name is not None:
        # 1. Load tokenizer & model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # 2. Move model to device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
    else: 
        model = None
        tokenizer = None
        device = None

    # 3. Gather all Java files
    file_triplets = gather_java250_files(dataset_path)
    
    # sort by the instance_path (3rd element)
    file_triplets = sorted(file_triplets, key=lambda triplet: triplet[2])

    total = len(file_triplets)
    print(f"Found {total} Java files under '{dataset_path}'")
    num_batches = math.ceil(total / batch_size)

    # 4. Suppress warnings so they don’t drown the tqdm bar
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # 5. Process in batches, with a notebook‐friendly progress bar
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch = file_triplets[start:end]

            # 5a. Read sources & languages
            sources = []
            langs = []
            for instance_id, instance_label, instance_path in batch:
                with open(instance_path, "r", encoding="utf-8", errors="ignore") as f:
                    src = f.read()
                sources.append(src)
                langs.append("java")
            # 5b. Extract features & graphs (inside, ensure tokenizer truncation)
            
            feature_matrices, cleaned_graphs = extract_graph_and_features(
                sources=sources,
                langs=langs,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,   # ensures truncation at 512
                stride=stride,
                device=device,
                clean=clean,
                pos_dim=pos_dim,
                use_posenc=use_posenc
            )
            
            # 5c) Convert each (G, feat_matrix) → PyG Data and save
            for (instance_id, instance_label, instance_path), feat_matrix, G in zip(
                batch,
                feature_matrices,
                cleaned_graphs
            ):
                pyg_data = nx_ast_and_features_to_pyg(G, feat_matrix, type2idx)

                # Compute relative path and output path
                rel_path = os.path.relpath(instance_path, dataset_path)
                rel_no_ext = os.path.splitext(rel_path)[0]
                out_path = os.path.join(out_root, rel_no_ext + '.pt')
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(pyg_data, out_path)
                
    return type2idx


## hugginFace datasets PREPROCESSING
def build_node_type_mapping_huggingfaceDataset(
    huggingface_data_path: str = "DetectVul/devign",
    key_of_interest: str = "func",
    split_keys: list[str] = ["train", "validation", "test"],
    id_key: str = "id",
    lang: str = "c"
) -> dict[str, int]:
    """
    Build a mapping from syntax tree node types to unique integer indices for a Hugging Face dataset.

    This function loads the specified dataset, parses each code snippet into a NetworkX AST graph using Tree-sitter,
    and collects all distinct CST node types. It returns a dictionary mapping each node type string to a sequential index.

    Args:
        huggingface_data_path (str):
            The path or identifier of the Hugging Face dataset (e.g., "DetectVul/devign").
        key_of_interest (str):
            The field name in each dataset example containing the source code snippet.
        split_keys (list[str]):
            The dataset splits to process (e.g., ["train", "validation", "test"]).
        id_key (str):
            The field name in each example containing a unique identifier for the snippet.
        lang (str):
            The programming language for parsing (passed to build_networkx_ast).

    Returns:
        dict[str, int]:
            A mapping from CST node type strings to integer indices (0..N-1) in sorted order.
    """
    # 1. Load the Hugging Face dataset
    dataset = load_dataset(huggingface_data_path)

    # 2. Gather all instances across specified splits
    instances = {
        example[id_key]: example[key_of_interest]
        for split in split_keys
        for example in dataset[split]
    }
    snippet_ids = list(instances.keys())
    print(f"Found {len(snippet_ids)} instances under '{huggingface_data_path}'")

    # 3. Collect unique node types from the ASTs
    node_type_set: set[str] = set()
    for snippet_id in tqdm(snippet_ids, desc="Collecting node types"):
        source_code = instances[snippet_id]
        # Parse source into a directed AST graph
        ast_graph: nx.DiGraph = build_networkx_ast(
            source=source_code,
            lang=lang,
            use_posenc=False
        )
        # Add each node's "type" attribute to the set
        for _, node_attrs in ast_graph.nodes(data=True):
            node_type = node_attrs.get("type")
            if node_type is not None:
                node_type_set.add(node_type)

    # 4. Sort node types for reproducible ordering
    sorted_types = sorted(node_type_set)

    # 5. Build and return the mapping { node_type_str: index }
    type_to_index = {nt: idx for idx, nt in enumerate(sorted_types)}
    return type_to_index


def preprocess_huggingfaceDataset(
    saving_path,
    model_name : str = None,
    huggingface_data_path: str = "DetectVul/devign",
    use_model_subfolder: bool = True,
    key_of_interest: str = "func",
    split_keys: list[str] = ["train", "validation", "test"],
    id_key: str = "id",
    lang: str = "c",
    
    batch_size: int = 64,
    max_length: int = None,
    stride: int = 32,
    device: str = None,
    clean: bool = True,
    pos_dim: int = 32,
    use_posenc: bool = False,
    type2idx = None
):
    if type2idx is None :
        type2idx = build_node_type_mapping_huggingfaceDataset(    huggingface_data_path = huggingface_data_path,
                                                              key_of_interest = key_of_interest,
                                                              split_keys = split_keys,
                                                              id_key = id_key,
                                                              lang = lang)
    # 0) make sure save dir exists
    saving_path = os.path.abspath(os.path.normpath(saving_path))
    if use_model_subfolder:
        folder_name = model_name.replace('/', '_').replace('-', '_') if model_name else 'baseline'
        out_root = os.path.join(saving_path, folder_name)
    else:
        out_root = saving_path
    os.makedirs(out_root, exist_ok=True)

    # 1. Load tokenizer & model
    model = None
    tokenizer = None
    if model_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        # 2. Move model to device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

    # 3. load the dataset from huggingface.
    dataset = load_dataset(huggingface_data_path)
    
    # build instance id to instance mapping
    instances = {
        example[id_key]: example[key_of_interest]
        for split in split_keys
        for example in dataset[split]
    }
    snippet_ids = list(instances.keys())
    total = len(snippet_ids)

    print(f"Found {len(snippet_ids)} instances under '{huggingface_data_path}'")
    num_batches = math.ceil(total / batch_size)

    # 4. Suppress warnings so they don’t drown the tqdm bar
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # 5. Process in batches, with a notebook‐friendly progress bar
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            # 5a. Read sources & languages
            sources = [ instances[sample_id] for sample_id in snippet_ids[start:end] ]
            langs = [lang for _ in snippet_ids[start : end]]
            # 5b. Extract features & graphs (inside, ensure tokenizer truncation)
            feature_matrices, cleaned_graphs = extract_graph_and_features(
                sources=sources,
                langs=langs,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,   # ensures truncation at 512
                stride=stride,
                device=device,
                clean=clean,
                pos_dim=pos_dim,
                use_posenc=use_posenc
            )
            
            # 5c) Convert each (G, feat_matrix) → PyG Data and save
            for instance_id, feature_matrix, G in zip(
                snippet_ids[start:end],
                feature_matrices,
                cleaned_graphs
            ):
                pyg_data = nx_ast_and_features_to_pyg(G, feature_matrix, type2idx)
                # save under saving_path/model_name (depending on flag)/<instance_id>.pt
                out_path = os.path.join(out_root, f"{instance_id}.pt")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(pyg_data, out_path)
                if not os.path.isfile(out_path):
                    raise RuntimeError(f"Failed to save {out_path}")
    return type2idx

