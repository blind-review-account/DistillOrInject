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
from utils.dataset_processing import preprocess_huggingfaceDataset, build_node_type_mapping_huggingfaceDataset


def get_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def launch_preprocessing(config): 
    type2idx = build_node_type_mapping_huggingfaceDataset(huggingface_data_path=config["huggingface_data_path"], 
                                                          key_of_interest = config["key_of_interest"],
                                                          split_keys = config["split_keys"],
                                                          id_key = config["id_key"],
                                                          lang = config["lang"])

    preprocess_huggingfaceDataset(saving_path = config["saving_path"],
                                  model_name = config.get("model_name", None),
                                  huggingface_data_path =config["huggingface_data_path"],
                                  use_model_subfolder = config.get("use_model_subfolder", True),
                                  key_of_interest = config["key_of_interest"],
                                  split_keys = config["split_keys"],
                                  id_key = config["id_key"],
                                  lang = config["lang"],
                                  batch_size = int(config["batch_size"]),
                                  stride = int(config["stride"]),
                                  clean = bool(config["clean"]),
                                  pos_dim = int(config["pos_dim"]),
                                  use_posenc = bool(config["use_posenc"]),
                                  type2idx = type2idx )
    
def parse_args():
    p = argparse.ArgumentParser(description="Preprocessing Devign dataset.")
    p.add_argument("--config", "-c", required=True, help="Path to YAML config file for preprocessing.")
    return p.parse_args()

if __name__ == "__main__":
    args   = parse_args()
    config = get_config(args.config)
    launch_preprocessing(config)
