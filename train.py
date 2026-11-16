"""
Training Pipeline for Proposed Deep Multi-Scale Residual Brain Tumor Classifier.
Features:
  - Linear Warmup followed by Cosine Annealing Learning Rate Scheduling
  - FP16 Automatic Mixed Precision (AMP)
  - Gradient Accumulation for Effective Batch Size Scaling
  - Multi-Class Focal Loss with Dynamic Class Imbalance Weighting & Label Smoothing
  - Early Stopping monitored on Validation Macro ROC-AUC
  - Checkpoint Serialization and Metric Export
"""

import os
import sys
import json
import math
import time
import pickle
import argparse
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from src.model import ProposedBrainTumorClassifier, get_model_summary
from src.dataset import (
    BrainTumorDataset,
    get_train_transforms,
    get_eval_transforms,
    scan_dataset_directory,
    stratified_split,
)
from src.loss import FocalLoss
from src.utils import set_seed, plot_training_curves, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Proposed Residual Classifier for Brain Tumor MRI Diagnosis"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/kaggle/input/datasets/mahamodulhasan05/mahamodulnd5/Training",
        help="Path to dataset root containing class subdirectories"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save checkpoints, logs, and evaluation figures"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.json",
        help="Path to JSON configuration file"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per forward step")
    parser.add_argument("--accum_steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=None, help="Peak learning rate")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Enable automatic mixed precision")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp", help="Disable automatic mixed precision")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    accumulation_steps: int = 2,
    use_amp: bool = True
) -> Tuple[float, float]:
    """Executes single training epoch with gradient accumulation and AMP."""
    model.train()
    total_loss, correct, total_samples = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, (xb, yb) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = xb.size(0)
        total_loss += loss.item() * accumulation_steps * bs
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total_samples += bs

    return total_loss / total_samples, correct / total_samples


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_amp: bool = True
) -> Tuple[float, float]:
    """Evaluates validation loss and macro ROC-AUC."""
    model.eval()
    all_probs, all_targets = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            probs = F.softmax(model(xb), dim=-1)
        all_probs.append(probs.cpu().float().numpy())
        all_targets.append(yb.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    val_ohe = np.zeros((len(all_targets), num_classes), dtype=np.float32)
    val_ohe[np.arange(len(all_targets)), all_targets] = 1.0

    val_auc = float(roc_auc_score(val_ohe, all_probs, average="macro", multi_class="ovr"))
    val_acc = float((all_probs.argmax(axis=1) == all_targets).mean())
    return val_auc, val_acc


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Resolve parameters from CLI or config
    seed = args.seed or cfg.get("seed", 42)
    set_seed(seed)

    epochs = args.epochs or cfg.get("epochs", 40)
    batch_size = args.batch_size or cfg.get("batch_size", 16)
    accum_steps = args.accum_steps or cfg.get("accumulation_steps", 2)
    lr_max = args.lr or cfg.get("lr_max", 8e-4)
    lr_min = cfg.get("lr_min", 1e-6)
    warmup_epochs = cfg.get("warmup_epochs", 5)
    weight_decay = cfg.get("weight_decay", 5e-4)
    patience = args.patience or cfg.get("patience", 12)
    image_size = cfg.get("image_size", 224)
    d_model = cfg.get("d_model", 64)
    classes = cfg.get("classes", ["glioma_tumor", "no_tumor", "meningioma_tumor", "pituitary_tumor"])
    num_classes = len(classes)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.use_amp and torch.cuda.is_available()

    print("=" * 75)
    print("  TRAINING PIPELINE: Proposed Deep Multi-Scale Residual Network")
    print("=" * 75)
    print(f"Device               : {device} (AMP: {use_amp})")
    print(f"Dataset Source       : {args.data_dir}")
    print(f"Target Output Folder : {args.output_dir}")
    print(f"Image Size / Classes : {image_size}x{image_size} | {num_classes} classes")
    print(f"Batch Size (Effective: {batch_size * accum_steps} ({batch_size} x {accum_steps})")
    print(f"Learning Rate (Max)  : {lr_max} (Warmup: {warmup_epochs} eps)")
    print(f"Seed                 : {seed}")
    print("=" * 75)

    # Ingest and partition dataset
    file_paths, labels = scan_dataset_directory(args.data_dir, classes)
    splits = stratified_split(
        file_paths,
        labels,
        train_split=cfg.get("train_split", 0.80),
        val_split=cfg.get("val_split", 0.10),
        test_split=cfg.get("test_split", 0.10),
        seed=seed
    )

    train_files, train_y = splits["train"]
    val_files, val_y = splits["val"]
    test_files, test_y = splits["test"]

    print(f"Partitions -> Train: {len(train_files):,} | Val: {len(val_files):,} | Test: {len(test_files):,}")

    train_ds = BrainTumorDataset(train_files, train_y, transform=get_train_transforms(image_size))
    val_ds = BrainTumorDataset(val_files, val_y, transform=get_eval_transforms(image_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Initialize Model
    model = ProposedBrainTumorClassifier(num_classes=num_classes, d_model=d_model).to(device)
    tot_params, train_params = get_model_summary(model)
    print(f"Model Initialized -> Total Parameters: {tot_params:,} ({tot_params/1e6:.3f}M)")

    # Loss, Optimizer, LR Scheduler
    cw_dict = cfg.get("class_weights", {
        "glioma_tumor": 1.6,
        "no_tumor": 1.0,
        "meningioma_tumor": 1.6,
        "pituitary_tumor": 1.1
    })
    cw_tensor = torch.tensor([cw_dict.get(c, 1.0) for c in classes], dtype=torch.float32, device=device)
    criterion = FocalLoss(gamma=cfg.get("focal_gamma", 2.0), label_smoothing=cfg.get("label_smoothing", 0.1), weight=cw_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay)

    def warmup_cosine_lr(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        t = epoch - warmup_epochs
        total_t = epochs - warmup_epochs
        return (lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t / max(total_t, 1)))) / lr_max

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lr)
    scaler = GradScaler(enabled=use_amp)

    # Training execution
    best_val_auc = 0.0
    best_epoch = 0
    patience_cnt = 0
    history = {"train_loss": [], "train_acc": [], "val_auc": [], "val_acc": []}

    best_ckpt_path = os.path.join(args.output_dir, "proposed_model_best.pt")
    full_ckpt_path = os.path.join(args.output_dir, "proposed_model_full.pt")

    print("-" * 75)
    print(f" {'Ep':>3} | {'TrLoss':>7} {'TrAcc':>7} | {'ValAUC':>7} {'ValAcc':>7} | {'LR':>8} | Status")
    print("-" * 75)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, accum_steps, use_amp
        )
        val_auc, val_acc = evaluate_validation(model, val_loader, device, num_classes, use_amp)

        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_auc"].append(val_auc)
        history["val_acc"].append(val_acc)

        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_cnt = 0
            save_checkpoint({
                "model_name": "ProposedBrainTumorClassifier",
                "epoch": epoch,
                "val_auc": best_val_auc,
                "val_acc": val_acc,
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "d_model": d_model,
                "num_classes": num_classes,
                "image_size": image_size,
            }, best_ckpt_path)
        else:
            patience_cnt += 1

        mark = " *BEST*" if is_best else ""
        print(f" {epoch:3d} | {tr_loss:7.4f} {tr_acc*100:6.2f}% | {val_auc:7.4f} {val_acc*100:6.2f}% | {curr_lr:8.2e} |{mark}")

        if patience_cnt >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best Val AUC: {best_val_auc:.4f} at epoch {best_epoch}")
            break

    total_time = (time.time() - start_time) / 60.0
    print("=" * 75)
    print(f"Training completed in {total_time:.2f} minutes.")
    print(f"Best Val ROC-AUC: {best_val_auc:.4f} (Epoch {best_epoch})")
    print(f"Best weights saved to: {best_ckpt_path}")

    # Save full state & metadata
    save_checkpoint({
        "model_name": "ProposedBrainTumorClassifier",
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "state_dict": model.state_dict(),
        "classes": classes,
        "d_model": d_model,
        "num_classes": num_classes,
        "image_size": image_size,
        "total_parameters": tot_params,
        "config": cfg,
    }, full_ckpt_path)

    # Save training curves plot and history log
    plot_training_curves(history, os.path.join(args.output_dir, "training_curves.png"))
    with open(os.path.join(args.output_dir, "training_history.pkl"), "wb") as f:
        pickle.dump(history, f)


if __name__ == "__main__":
    main()
