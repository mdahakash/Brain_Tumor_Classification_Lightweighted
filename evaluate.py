"""
Comprehensive Test Cohort Evaluation Engine with 3-Pass Test-Time Augmentation (TTA).
Generates publication-quality Confusion Matrix, Multi-Class ROC Curves, and JSON Metrics.
"""

import os
import json
import argparse
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from src.model import ProposedBrainTumorClassifier
from src.dataset import (
    BrainTumorDataset,
    get_eval_transforms,
    scan_dataset_directory,
    stratified_split,
)
from src.utils import (
    set_seed,
    load_checkpoint,
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Trained Brain Tumor Classifier")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/kaggle/input/datasets/mahamodulhasan05/mahamodulnd5/Training",
        help="Path to dataset root folder"
    )
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Directory to save figures and metrics")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split reproduction")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--no_tta", action="store_true", help="Disable 3-pass test-time augmentation")
    return parser.parse_args()


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs model evaluation. If use_tta is True, averages predictions across:
      1. Original image
      2. Horizontally flipped image
      3. Vertically flipped image
    """
    model.eval()
    all_probs = []
    all_targets = []
    use_amp = torch.cuda.is_available()

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            if use_tta:
                p1 = F.softmax(model(xb), dim=-1)
                p2 = F.softmax(model(torch.flip(xb, dims=[-1])), dim=-1)
                p3 = F.softmax(model(torch.flip(xb, dims=[-2])), dim=-1)
                probs = (p1 + p2 + p3) / 3.0
            else:
                probs = F.softmax(model(xb), dim=-1)

        all_probs.append(probs.cpu().float().numpy())
        all_targets.append(yb.numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  RIGOROUS EVALUATION ENGINE (3-PASS TTA)")
    print("=" * 70)
    print(f"Device        : {device}")
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"TTA Enabled   : {not args.no_tta}")
    print("=" * 70)

    # Load checkpoint
    ckpt = load_checkpoint(args.checkpoint, device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    classes = ckpt.get("classes", ["glioma_tumor", "no_tumor", "meningioma_tumor", "pituitary_tumor"])
    d_model = ckpt.get("d_model", 64)
    image_size = ckpt.get("image_size", 224)
    num_classes = len(classes)

    display_labels = ["Glioma", "No Tumor", "Meningioma", "Pituitary"]
    if len(classes) != len(display_labels):
        display_labels = classes

    # Initialize model
    model = ProposedBrainTumorClassifier(num_classes=num_classes, d_model=d_model).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Ingest and reproduce test partition
    file_paths, labels = scan_dataset_directory(args.data_dir, classes)
    splits = stratified_split(file_paths, labels, train_split=0.80, val_split=0.10, test_split=0.10, seed=args.seed)
    test_files, test_y = splits["test"]

    print(f"Evaluating held-out test cohort: {len(test_files):,} samples")

    test_ds = BrainTumorDataset(test_files, test_y, transform=get_eval_transforms(image_size))
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Run inference
    probs, targets = predict_with_tta(model, test_loader, device, use_tta=not args.no_tta)
    preds = probs.argmax(axis=1)

    # Compute metrics
    metrics = compute_classification_metrics(targets, probs, display_labels)

    print("\n" + "=" * 70)
    print("  HELD-OUT TEST SET DIAGNOSTIC PERFORMANCE")
    print("=" * 70)
    print(f"  Test Accuracy (TTA)  : {metrics['accuracy']:.2f}%")
    print(f"  Macro Precision      : {metrics['precision_macro']:.2f}%")
    print(f"  Macro Recall         : {metrics['recall_macro']:.2f}%")
    print(f"  Macro F1-Score       : {metrics['f1_macro']:.2f}%")
    print(f"  Macro ROC-AUC        : {metrics['roc_auc_macro']:.2f}%")
    print("-" * 70)
    print("  Per-Class F1-Scores:")
    for cls_name, f1_val in metrics["per_class_f1"].items():
        print(f"    - {cls_name:<16}: {f1_val:.2f}%")
    print("=" * 70)

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to -> {metrics_path}")

    # Plot Confusion Matrix
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(targets, preds, display_labels, cm_path)
    print(f"Confusion Matrix saved to -> {cm_path}")

    # Plot Multi-Class ROC Curves
    targets_ohe = np.zeros((len(targets), num_classes), dtype=np.float32)
    targets_ohe[np.arange(len(targets)), targets] = 1.0
    roc_path = os.path.join(args.output_dir, "roc_curves.png")
    plot_roc_curves(targets_ohe, probs, display_labels, roc_path)
    print(f"ROC Curves saved to -> {roc_path}")


if __name__ == "__main__":
    main()
