#Set path 
import sys
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).resolve().parent  # Current script directory
project_root = current_dir.parent.parent.parent  # Go up 3 levels to project root
sys.path.insert(0, str(project_root))

import yaml
import argparse
import pickle
import os
from utils.dataset_processing import preprocess_java250, build_node_type_mapping_java250


def get_config(config_path):
    config_path = os.path.normpath(config_path)
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def launch_preprocessing(config): 
    data_path = os.path.normpath(config["data_path"])
    data_path = os.path.abspath(data_path)
    
    type2idx = build_node_type_mapping_java250(folder=data_path, 
                                               batch_size = int(config["batch_size"])
    )
    preprocess_java250(dataset_path = data_path,
                       use_model_subfolder = config.get("use_model_subfolder", True),
                       model_name = config.get("model_name", None),
                       batch_size = int(config["batch_size"]),
                       stride = int(config["stride"]),
                       clean = bool(config["clean"]),
                       pos_dim = int(config["pos_dim"]),
                       use_posenc = bool(config["use_posenc"]),
                       type2idx = type2idx )
    
def parse_args():
    p = argparse.ArgumentParser(description="Preprocessing Java250 data.")
    p.add_argument("--config", "-c", required=True, help="Path to YAML config file for preprocessing.")
    return p.parse_args()

if __name__ == "__main__":
    args   = parse_args()
    config = get_config(args.config)
    launch_preprocessing(config)