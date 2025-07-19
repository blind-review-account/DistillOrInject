import torch
import time

import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import time
import copy
import os
import json
import math
import wandb


def compute_loss(
    prediction: torch.Tensor,
    label: torch.Tensor,
    reduction: str = "mean",
    clamp_val: float = 1e2
) -> torch.Tensor:
    """
    Compute multi-class cross-entropy loss given:
      - `prediction`: raw logits, shape [B, C]
      - `label`: either
          • integer class indices, shape [B], with values in [0..C-1], or
          • one-hot vectors, shape [B, C].
    Uses F.cross_entropy under the hood (log_softmax + NLL) for numerical stability.

    Args:
      prediction: logits, float tensor of shape [batch_size, num_classes].
      label:      either LongTensor of class indices [batch_size]
                  or FloatTensor one-hot vectors [batch_size, num_classes].
      reduction:  "mean" or "sum" (same as PyTorch conventions).

    Returns:
      scalar loss Tensor.
    """
    # ensure logits are floats on correct device
    prediction = prediction.float()
    # prediction = torch.nan_to_num(prediction, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
    prediction = prediction.clamp(min=-clamp_val, max=clamp_val)
    # if label is one-hot, convert to class indices
    if label.dim() == prediction.dim():
        # assume one-hot
        label = label.float().to(prediction.device)
        label_idx = label.argmax(dim=1)
    else:
        # assume already indices
        label_idx = label.long().to(prediction.device)

    # use built-in cross-entropy
    return F.cross_entropy(prediction, label_idx, reduction=reduction)


def compute_accuracy(prediction: torch.Tensor, label: torch.Tensor) -> float:
    """
    Compute accuracy: percentage of correct predictions.
    Expects `prediction` as logits [B, C] and `label` as integer class [B].
    """
    preds = prediction.argmax(dim=1).to("cpu")
    labels = label.argmax(dim=1).to("cpu")
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

def get_eval_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate `model` on all batches in `loader`, returning average loss and accuracy.
    Uses compute_loss and compute_accuracy internally. Wraps inference in autocast for speed.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    loop = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for (graph, label) in loop:
            graph = graph.to(device)
            label = label.to(device)
            # with torch.autocast(device_type = model.device.type):
            #     logits = model(graph)  # [batch_size, num_classes]
            #     loss = compute_loss(logits, label)
            logits = model(graph)  # [batch_size, num_classes]
            loss = compute_loss(logits, label)

            bs = logits.size(0)  
            total_loss += loss.item() * bs
            acc = compute_accuracy(logits, label)
            total_correct += acc * bs
            total_samples += bs

            loop.set_postfix({
                "loss": f"{(total_loss / total_samples):.4f}",
                "acc":  f"{(total_correct / total_samples):.4f}"
            })
    model.train()
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples
    }


def wandb_safe_log(log_dict, step=None):
    try:
        if step is not None:
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)
    except Exception as e:
        tqdm.write(f"[W&B] Logging failed: {e}")
        
def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader :  torch.utils.data.DataLoader = None,
    device: str = "cuda",
    optimizer: torch.optim.Optimizer = None,
    scheduler = None,
    epochs: int = 10,
    patience: int = 3,
    delta: float = 0.0,
    max_grad_norm : float = 2.0,
    model_name: str = "best_model",
    project_name: str = "my_project",
    entity: str = None,
    log_wandb: bool = True,
    log_interval: int = 10, 
    wandb_api_key: str = None,

    save_dir: str = ".",
    save_model: bool = True,
    save_history: bool = True
) -> dict:
    """
    Train model with early stopping. Optionally save and log to W&B, including step-wise metrics and gradient norms.
    """
    if optimizer is None:
        raise ValueError("You must pass an instantiated optimizer.")
    if scheduler is None:
        raise ValueError("You must pass an instantiated scheduler.")

    # normalize & absolutize
    save_dir = os.path.abspath(os.path.normpath(save_dir))
    os.makedirs(save_dir, exist_ok=True)

    # build filenames & paths
    model_path = os.path.join(save_dir, model_name)

    #Make sure the model ends with a .pt
    base, _ = os.path.splitext(model_path)
    model_path = base + ".pt"

    hist_name  = os.path.splitext(model_name)[0] + "_history.json"
    hist_path  = os.path.join(save_dir, hist_name)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None
    
    # initialize history
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "test_loss":  [], "test_acc":   [],
        "batch_time": [],      # seconds per batch
        "batch_grad_norm": []  # total grad norm per batch
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # W&B setup
    if log_wandb:
        try:
            key = wandb_api_key or os.getenv("WANDB_API_KEY")
            wandb.login(key=key, relogin=False)
            run = wandb.init(
                project=project_name,
                entity=entity,
                name=os.path.splitext(model_name)[0],
                reinit='finish_previous',
                resume="never",
                config={
                    "epochs": epochs,
                    "batch_size": train_loader.batch_size,
                    "optimizer": optimizer.__class__.__name__,
                    "scheduler": scheduler.__class__.__name__,
                    "architecture": repr(model),
                }
            )
            wandb.watch(model, log="all", log_freq=1)
            # Bind everything to global_step
            for m in ["train_loss","train_acc","val_loss","val_acc","test_loss","test_acc","total_grad_norm",
                     "mem_alloc", "mem_reserved"]:
                wandb.define_metric(m, step_metric="global_step")
            global_step = 0
        except Exception as e:
            print(f"[W&B init failed] {e}")
            log_wandb = False
        
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = epoch_correct = epoch_samples = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)

        for batch_idx, (graph, label) in enumerate(loop):
            batch_start = time.time()
            graph, label = graph.to(device), label.to(device)
            optimizer.zero_grad()

            # with torch.autocast(device_type = model.device.type):
            #     logits = model(graph)
            #     loss = compute_loss(logits, label)
            logits = model(graph)
            loss = compute_loss(logits, label)

            # backward
            # if device.type == "cuda":
            #     scaler.scale(loss).backward()
            #     # unscale grads to undo AMP scaling before measuring
            #     scaler.unscale_(optimizer)
            # else:
            #     loss.backward()
            loss.backward()

            # gradient clipping
            total_grad_norm = torch.norm(
                torch.stack([
                    p.grad.data.norm(2)
                    for p in model.parameters() if p.grad is not None
                ])
                , 2).item()

            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= max_grad_norm)
            
            # gradient norms
            grad_norms = {}
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                grad_norms[name] = p.grad.data.norm(2).item()

            # optimizer step
            # if device.type == "cuda":
            #     scaler.step(optimizer)
            #     scaler.update()
            #     scheduler.step()
            # else:
            #     optimizer.step()
            #     scheduler.step()
            optimizer.step()
            scheduler.step()
            
            bs = graph.size(0) 
            epoch_loss += loss.item() * bs
            acc = compute_accuracy(logits, label)
            epoch_correct += acc * bs
            epoch_samples += bs

            # record batch time & grad norm
            batch_timing = time.time() - batch_start
            history["batch_time"].append(-1.0 if math.isnan(batch_timing) else batch_timing)
            history["batch_grad_norm"].append(-1.0 if math.isnan(total_grad_norm) else total_grad_norm)


            # increment step and prepare log
            # bump the step counter every batch
            if log_wandb:
                global_step += 1
    
                # log only every log_interval steps
                if global_step % log_interval == 0:
                    try:
                        log_dict = {
                            "global_step":     global_step,
                            "train_loss":      loss.item(),
                            "train_acc":       acc,
                            "train_lr":        optimizer.param_groups[0]["lr"],
                            "total_grad_norm": total_grad_norm,
                            "mem_alloc":       torch.cuda.memory_allocated(device)   if device.type=="cuda" else 0,
                            "mem_reserved":    torch.cuda.max_memory_reserved(device) if device.type=="cuda" else 0,
                            **{f"grad_norm/{n}": g for n, g in grad_norms.items()}
                        }
                        wandb.log(log_dict, step=global_step)
                    except Exception as e:
                        print(f"[W&B log failed] {e}")
            
            # Compute running averages
            running_loss = epoch_loss / epoch_samples
            running_acc = epoch_correct / epoch_samples
            
            # update tqdm
            loop.set_postfix({
                "loss": f"{running_loss:.4f}",
                "acc":  f"{running_acc:.4f}",
                "lr":   f"{optimizer.param_groups[0]['lr']:.2e}",
                "grad_norm": f"{total_grad_norm:.4f}"
            })

        # record epoch metrics
        train_loss = epoch_loss / epoch_samples
        train_acc  = epoch_correct / epoch_samples
        history["train_loss"].append(-1.0 if math.isnan(train_loss) else train_loss)
        history["train_acc"].append(-1.0 if math.isnan(train_acc) else train_acc)

        # validation
        val_metrics = get_eval_metrics(model, val_loader, device)
        val_loss = val_metrics["loss"]
        val_acc  = val_metrics["accuracy"]
        history["val_loss"].append(-1.0 if math.isnan(val_loss) else val_loss)
        history["val_acc"].append(-1.0 if math.isnan(val_acc) else val_acc)
        if log_wandb:
            wandb.log({
                "val_loss": val_loss,
                "val_acc":  val_acc,
            }, step=global_step)

        # early stopping & checkpoint
        improved = (val_loss + delta) < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if save_model:
                torch.save(best_model_wts, model_path)
                tqdm.write(f"Epoch {epoch}/{epochs} ▶ Saved best model.")
        else:
            epochs_no_improve += 1
            tqdm.write(f"Epoch {epoch}/{epochs} ▶ No improvement ({epochs_no_improve}/{patience})")
        if epochs_no_improve >= patience:
            tqdm.write(f"Early stopping at epoch {epoch}.")
            break

    # finalize
    model.load_state_dict(best_model_wts)
    if save_history:
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        tqdm.write(f"History saved to {hist_path}")
    # test evaluation & record
    if test_loader is not None:
        test_metrics = get_eval_metrics(model, test_loader, device)
        history["test_loss"].append(-1.0 if math.isnan(test_metrics["loss"]) else test_metrics["loss"]) 
        history["test_acc"].append(-1.0 if math.isnan(test_metrics["accuracy"]) else test_metrics["accuracy"]) 
        if log_wandb:
            try:
                wandb.log({
                    "test_loss": test_metrics["loss"],
                    "test_acc":  test_metrics["accuracy"],
                }, step=global_step)
            except:
                pass
            
    if log_wandb and run is not None:
        run.finish()
    return history
