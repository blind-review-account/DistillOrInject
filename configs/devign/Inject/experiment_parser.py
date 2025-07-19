#Set path 
import sys
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).resolve().parent  # Current script directory
project_root = current_dir.parent.parent.parent  # Go up 3 levels to project root
sys.path.insert(0, str(project_root))

import argparse
import yaml
import os 
import random 
import numpy as np
import torch 

from model.gnn_classifier import GNN_Classifier

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from torch_geometric.loader import DataLoader as PyGDataLoader
from local_datasets.devign import get_split_devign
import multiprocessing

from utils.training.classification.training_utils import *

def get_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model(config):
    model_config = config['model']
    gnn_args = config['model'].get("gnn_args", {})
    
    model = GNN_Classifier(
        gnn_args = gnn_args,
        n_emb = model_config['n_emb'],
        pos_dim = model_config['pos_dim'],
        in_dim = model_config['in_dim'],
        emb_h_dim = model_config['emb_h_dim'],
        emb_out_dim = model_config['emb_out_dim'],
        gnn_h_dim = model_config['gnn_h_dim'],
        dropout = model_config['dropout'],
        num_classes = model_config['num_classes'],
        gnn_type = model_config['gnn_type'],
        cls_pooling_method = model_config['cls_pooling_method'],
        device = torch.device(model_config['device']) 
    ) 
    return model 

def create_optim(config, model):
    optim = AdamW(model.parameters(), 
                  lr=float(config['scheduler']['min_lr']),
                  weight_decay=float(config['optimizer']['weight_decay']) 
                 )
    return optim
    
def create_scheduler(config, optimizer,train_loader):
    scheduler_config = config["scheduler"]
    max_lr = float(scheduler_config["max_lr"])
    min_lr = float(scheduler_config["min_lr"])
    pct_start = float(scheduler_config["pct_start"])
    anneal_strategy = scheduler_config["anneal_strategy"]
    
    num_epochs = int(config["training_params"]["epochs"])
    total_steps = num_epochs*len(train_loader) 
    div_factor = max_lr / min_lr
    final_div_factor = 1    
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr= max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        div_factor=div_factor,           # start_lr = max_lr / div_factor = 5e-6
        final_div_factor=final_div_factor  # end_lr = max_lr / (div_factor * final_div_factor) = 5e-6
    )
    return scheduler
    

def create_data_loaders(config):
    data_config = config["data_loader"]

    dataset_path = os.path.normpath(data_config["dataset_path"])
    dataset_path = os.path.abspath(dataset_path)

    model_name = config.get("extractor_name", None)
    use_model_subfolder = bool(config.get("use_model_subfolder", True))
    
    datasets = get_split_devign(
        huggingface_data_path = data_config["huggingface_data_path"],
        graphs_path = dataset_path,
        model_name = model_name,
        use_model_subfolder = use_model_subfolder,
        train_split_key = data_config["train_split_key"],
        valid_split_key = data_config["valid_split_key"],
        test_split_key  = data_config["test_split_key"],
        id_key = data_config["id_key"],
        label_key = data_config["label_key"],
        pt_suffix = data_config["pt_suffix"],
        label_mode = data_config["label_mode"],
    )
    
    train_data, test_data, val_data = datasets["train"], datasets["test"], datasets["valid"]
    
    train_loader = PyGDataLoader(
        train_data,
        batch_size=data_config["batch_size"],
        shuffle=True,
        pin_memory=False,
        num_workers= multiprocessing.cpu_count()
         )

    val_loader = PyGDataLoader(
        val_data,
        batch_size=data_config["batch_size"],
        shuffle=False,
        pin_memory=False,
        num_workers= multiprocessing.cpu_count()
        )

    test_loader = PyGDataLoader(
        test_data,
        batch_size=data_config["batch_size"],
        shuffle=False,
        pin_memory=False,
        num_workers= multiprocessing.cpu_count()
        )

    return train_loader, val_loader, test_loader
    
def launch_training(config, wandb_config):
    
    training_config = config["training_params"]
    model = create_model(config)
    if wandb_config is None or wandb_config is False:
        training_config["log_wandb"] = False
        training_config["wandb_api_key"] = None
        training_config["entity"] = None
    else : 
        training_config["wandb_api_key"] = wandb_config["wandb_api_key"]
        training_config["log_wandb"] = wandb_config["log_wandb"]
        training_config["entity"] = wandb_config["entity"]


    extractor_name = config.get("extractor_name", None)
    if extractor_name :
        extractor_name = extractor_name.replace("/","_").replace("-","_")
    else: extractor_name = "baseline"
        
    gnn_type = config["model"]["gnn_type"]
    name_suffix = config.get("name_suffix", None)
    if name_suffix:
        model_name = f"{gnn_type}_{extractor_name}_{name_suffix}"
    else :
        model_name = f"{gnn_type}_{extractor_name}"
    #Assume the value of training_config["save_dir"] is "./experimental_results/devign/Inject"
    saving_dir = os.path.abspath(os.path.normpath(training_config["save_dir"].rstrip("/\\")))
    saving_dir = os.path.join(saving_dir, extractor_name) + os.sep
    os.makedirs(saving_dir, exist_ok=True)
    
    optimizer = create_optim(config, model)
    train_loader, val_loader, test_loader = create_data_loaders(config)
    scheduler = create_scheduler(config, optimizer,train_loader)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader, 
        test_loader=test_loader,
        device=model.device,
        optimizer=optimizer,
        scheduler=scheduler,
        
        epochs= int(training_config["epochs"]),
        patience= int(training_config["patience"]),
        delta = float(training_config["delta"]),
        max_grad_norm = float(training_config["max_grad_norm"]) if training_config["max_grad_norm"] is not None else None,
    
        model_name = model_name,
        project_name = training_config["project_name"],
        entity = training_config["entity"],           
        log_wandb = bool(training_config["log_wandb"]),
        wandb_api_key = training_config["wandb_api_key"],

        save_dir = saving_dir,
        save_model = bool(training_config["save_model"]),
        save_history = bool(training_config["save_history"])
        )

# Parse command line arguments for config file
def parse_args():
    parser = argparse.ArgumentParser(description="Run training with specified config file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file.")
    return parser.parse_args()
    
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CUDA / cuDNN determinism
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True)

if __name__ == "__main__":
    args = parse_args()
    config = get_config(args.config)
    
    #Get the wanbd config and API key
    try :
        wandb_config = get_config(project_root / "wandb_config.yaml")
    except FileNotFoundError:
        print(f"Could not find a wanbd_config.yaml file at {project_root}")
        wandb_config = None
        
    set_seed( seed = int(config["random_seed"]["random_seed"]) )
    launch_training(config, wandb_config)
