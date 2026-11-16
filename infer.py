"""
Single-Image Inference CLI for Brain Tumor MRI Diagnostic Prediction.
"""

import os
import argparse
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.model import ProposedBrainTumorClassifier
from src.dataset import get_eval_transforms
from src.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Tumor Subtype for a Single Brain MRI Scan")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input brain MRI scan (.jpg, .png)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--output_image", type=str, default=None, help="Optional path to save prediction visualization")
    parser.add_argument("--image_size", type=int, default=224, help="Input spatial dimension")
    return parser.parse_args()


def predict_mri(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    display_labels: list,
    image_size: int = 224
) -> Tuple[str, float, Dict[str, float], np.ndarray]:
    """Runs single MRI inference and returns top prediction, confidence, and full distribution."""
    model.eval()
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image file: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(cv2.resize(img_rgb, (image_size, image_size)))
    transform = get_eval_transforms(image_size)
    tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    top_idx = int(probs.argmax())
    top_label = display_labels[top_idx]
    top_conf = float(probs[top_idx])

    prob_dict = {display_labels[i]: float(probs[i]) for i in probs.argsort()[::-1]}
    return top_label, top_conf, prob_dict, img_rgb


def save_diagnostic_card(
    image_rgb: np.ndarray,
    top_label: str,
    top_conf: float,
    prob_dict: Dict[str, float],
    save_path: str
) -> None:
    """Renders and saves a diagnostic visualization figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    ax1.imshow(image_rgb)
    ax1.set_title(f"Diagnostic Scan\nDiagnosis: {top_label.upper()} ({top_conf*100:.1f}%)", fontsize=11, fontweight="bold")
    ax1.axis("off")

    classes = list(prob_dict.keys())
    values = [prob_dict[c] * 100 for c in classes]
    colors = ["#2ca02c" if c == top_label else "#1f77b4" for c in classes]

    bars = ax2.barh(classes[::-1], values[::-1], color=colors[::-1], edgecolor="black", alpha=0.85)
    ax2.set_xlim(0, 105)
    ax2.set_xlabel("Predicted Probability (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Differential Diagnosis Confidence", fontsize=11, fontweight="bold")

    for bar in bars:
        w = bar.get_width()
        ax2.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(args.checkpoint, device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    classes = ckpt.get("classes", ["glioma_tumor", "no_tumor", "meningioma_tumor", "pituitary_tumor"])
    d_model = ckpt.get("d_model", 64)
    num_classes = len(classes)
    display_labels = ["Glioma", "No Tumor", "Meningioma", "Pituitary"]
    if len(classes) != len(display_labels):
        display_labels = classes

    model = ProposedBrainTumorClassifier(num_classes=num_classes, d_model=d_model).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    top_label, top_conf, prob_dict, img_rgb = predict_mri(
        model, args.image_path, device, display_labels, args.image_size
    )

    print("=" * 60)
    print(f"  PREDICTED DIAGNOSIS : {top_label.upper()}")
    print(f"  Confidence Score    : {top_conf * 100:.2f}%")
    print("-" * 60)
    for cls_name, prob_val in prob_dict.items():
        bar = "█" * int(prob_val * 30)
        print(f"  {cls_name:<16} {bar:<30} {prob_val * 100:6.2f}%")
    print("=" * 60)

    if args.output_image:
        save_diagnostic_card(img_rgb, top_label, top_conf, prob_dict, args.output_image)
        print(f"Diagnostic report figure saved to: {args.output_image}")


if __name__ == "__main__":
    main()
