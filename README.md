# 🧠 Deep Multi-Scale Residual Network for Brain Tumor MRI Multi-Class Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-100%25%20Deterministic-brightgreen?style=flat-square)]()
[![Model Size](https://img.shields.io/badge/Parameters-0.818M-orange?style=flat-square)]()

**Research Laboratory:** ARCHE Research Lab  
**Domain:** Neuro-Oncology Computer Vision & Diagnostic Artificial Intelligence  
**Target Publication:** *Biomedical Signal Processing and Control* / Medical Imaging Venues  
**Primary Dataset:** ND5 Multi-Class Brain Tumor MRI Benchmark (17,784 slices across 4 subtypes)

---

## 📌 Executive Summary & Scientific Motivation

In computer-aided neuro-oncology, deep neural architectures must achieve near-perfect diagnostic sensitivity while remaining computationally lightweight for potential edge-device deployment in clinical MRI workstations. While heavyweight vision transformers and dense attention networks often exceed 20M–80M parameters, they are susceptible to optimization instability, high memory footprints, and slow inference latencies.

This repository presents our **Proposed Deep Multi-Scale Residual Network with Dual Global Pooling (Mean + Max)**:
- **Parameter Compactness:** Only **818,148 parameters (~0.818M)** — over $23\%$ more compact than standard hybrid multi-branch baselines.
- **Top-Tier Diagnostic Reliability:** Achieves **99.61% Test Accuracy**, **99.58% Macro F1-Score**, and **99.98% Macro ROC-AUC** on an independent held-out test cohort of 1,779 MRI slices.
- **Balanced Clinical Sensitivity:** Demonstrates uniform performance across all tumor pathologies without sacrificing specificity on non-neoplastic controls:
  - **Glioma:** Precision 99.65% | Recall 99.65% | F1 99.65%
  - **No Tumor (Healthy):** Precision 99.02% | Recall 99.67% | F1 99.35%
  - **Meningioma:** Precision 99.78% | Recall 99.56% | F1 99.67%
  - **Pituitary:** Precision 99.78% | Recall 99.56% | F1 99.67%

---

## 🏗️ Architectural Topology

```text
+----------------------------------------------------------------------------------------------------+
|                                    Input MRI Slice (224 x 224 x 3)                                 |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                      Shared Convolutional Stem                                     |
|                      Conv 7x7 (stride=2, out=32) -> BatchNorm -> ReLU -> MaxPool 3x3               |
|                                     Output: (B, 32, 56, 56)                                        |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                  Stage 1: Fine Residual Representation                             |
|          ConvBlock (32 -> 32, stride=2) + IdentityBlock(32) + IdentityBlock(32)                    |
|                                     Output: (B, 32, 28, 28)                                        |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                               Stage 2: Intermediate Residual Representation                        |
|          ConvBlock (32 -> 64, stride=2) + IdentityBlock(64) + IdentityBlock(64)                    |
|                                     Output: (B, 64, 14, 14)                                        |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                Stage 3: Deep Semantic Representation                               |
|          ConvBlock (64 -> 128, stride=2) + IdentityBlock(128)                                       |
|                                     Output: (B, 128, 7, 7)                                         |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                            Dual Global Pooling: Global Mean + Global Max                           |
|                         z = MeanPool(x, dims=[2,3]) + MaxPool(x, dims=[2,3])                       |
|                                          Output: (B, 128)                                          |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                Feature Dimension Projection: Linear(128 -> 64)                     |
|                                           Output: (B, 64)                                          |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                 Regularized MLP Classification Head                                |
|        FC(64 -> 128) -> ReLU -> BatchNorm1d(128) -> Dropout(0.4) -> FC(128 -> 64)                 |
|                             -> ReLU -> Dropout(0.3) -> FC(64 -> 4)                                 |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                    4-Class Softmax Probabilities                                   |
|                [ Glioma (0) | No Tumor (1) | Meningioma (2) | Pituitary (3) ]                       |
+----------------------------------------------------------------------------------------------------+
```

### Key Mathematical Formulations

1. **Dual Global Invariant Pooling:**  
   Standard global average pooling can dilute focal morphological activations of small micro-tumors, whereas global max pooling can over-emphasize localized artifacts. Our dual pooling operator captures both spatial density and peak activation:
   $$z = \frac{1}{H \times W} \sum_{h=1}^H \sum_{w=1}^W x_{c,h,w} + \max_{h,w} (x_{c,h,w})$$

2. **Multi-Class Focal Loss with Dynamic Class Imbalance Weighting:**  
   To prevent majority classes from dominating the gradient update while preventing overconfident probabilities:
   $$\mathcal{L}_{\text{Focal}} = -\frac{1}{B} \sum_{b=1}^B \sum_{c=1}^C w_c \cdot (1 - p_{b,c})^\gamma \cdot y_{b,c}^{\text{smooth}} \cdot \log(p_{b,c})$$
   where $\gamma = 2.0$, label smoothing $\alpha = 0.1$, and class weights $w = [1.6, 1.0, 1.6, 1.1]$.

---

## 📊 Comprehensive Experimental Findings

### 1. Overall Performance on Held-Out Test Cohort (1,779 slices)

| Evaluation Metric | Baseline CNN | Standard ResNet-18 | Hybrid ViT Baseline | **Proposed Model (Ours)** |
|:---|:---:|:---:|:---:|:---:|
| **Test Accuracy (TTA)** | 96.40% | 98.15% | 98.71% | **99.61%** |
| **Macro Precision** | 96.12% | 98.02% | 98.64% | **99.58%** |
| **Macro Recall** | 96.25% | 98.10% | 98.60% | **99.59%** |
| **Macro F1-Score** | 96.18% | 98.06% | 98.62% | **99.58%** |
| **Macro ROC-AUC** | 98.85% | 99.42% | 99.70% | **99.98%** |
| **Total Parameter Count** | 4.82M | 11.18M | 1.07M | **0.818M** |

### 2. Per-Class Diagnostic Breakdown

| Tumor Subtype | Test Samples | Precision | Recall (Sensitivity) | F1-Score | Specificity |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Glioma** | 569 | 99.65% | 99.65% | **99.65%** | 99.83% |
| **No Tumor (Healthy Control)** | 307 | 99.02% | 99.67% | **99.35%** | 99.80% |
| **Meningioma** | 452 | 99.78% | 99.56% | **99.67%** | 99.92% |
| **Pituitary** | 451 | 99.78% | 99.56% | **99.67%** | 99.92% |
| **Overall / Macro Average** | **1,779** | **99.58%** | **99.59%** | **99.58%** | **99.87%** |

---

## 📁 Repository Directory Structure

```text
brain_tumor_classification/
├── configs/
│   └── default_config.json        # Unified training & evaluation hyperparameters
├── src/
│   ├── __init__.py                # Package level exports
│   ├── model.py                   # PyTorch implementation of Proposed Residual Backbone
│   ├── dataset.py                 # In-memory and lazy loaders, augmentations, and splitters
│   ├── loss.py                    # Class-weighted multi-class Focal Loss with label smoothing
│   └── utils.py                   # Checkpoint I/O, seeding, plotting routines (ROC, CM, Curves)
├── notebooks/
│   └── proposed_model_demo.ipynb  # End-to-end interactive demonstration notebook
├── train.py                       # CLI training script with AMP, warmup-cosine LR, early stopping
├── evaluate.py                    # CLI evaluation tool with 3-Pass Test-Time Augmentation
├── infer.py                       # CLI single-image inference and diagnostic report generator
├── requirements.txt               # Pinned pip dependencies
├── environment.yml                # Conda environment definition
├── LICENSE                        # Open-source MIT License
└── README.md                      # Comprehensive academic documentation
```

---

## 🚀 Quickstart & Reproducibility Guide

### 1. Environment Installation

Clone the repository and set up your preferred environment:

```bash
# Option A: Using Conda (Recommended)
conda env create -f environment.yml
conda activate brain-tumor-cls

# Option B: Using Pip & Python venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset Setup

The benchmark uses the ND5 Brain Tumor MRI dataset (17,784 slices across 4 classes).
Organize your dataset folder as follows:

```text
ND5_Dataset/
├── Training/
│   ├── glioma_tumor/       # (5,692 images)
│   ├── no_tumor/           # (3,066 images)
│   ├── meningioma_tumor/   # (4,519 images)
│   └── pituitary_tumor/    # (4,507 images)
```

### 3. Model Training

Run the production training pipeline with deterministic seeding, automatic mixed precision (AMP), and gradient accumulation:

```bash
python train.py \
  --data_dir "/path/to/ND5_Dataset/Training" \
  --output_dir "./outputs" \
  --epochs 40 \
  --batch_size 16 \
  --accum_steps 2 \
  --lr 8e-4 \
  --seed 42
```

### 4. Rigorous Held-Out Evaluation (with 3-Pass TTA)

Evaluate model checkpoints using 3-Pass Test-Time Augmentation (Original, Horizontal Flip, Vertical Flip):

```bash
python evaluate.py \
  --checkpoint "./outputs/proposed_model_best.pt" \
  --data_dir "/path/to/ND5_Dataset/Training" \
  --output_dir "./eval_results" \
  --seed 42
```

Outputs generated:
- `eval_results/test_metrics.json`
- `eval_results/confusion_matrix.png`
- `eval_results/roc_curves.png`

### 5. Single Patient Diagnostic Inference

Run inference on an arbitrary MRI scan slice and generate an annotated diagnostic report:

```bash
python infer.py \
  --image_path "sample_mri.jpg" \
  --checkpoint "./outputs/proposed_model_best.pt" \
  --output_image "diagnostic_report.png"
```

Sample CLI output:
```text
============================================================
  PREDICTED DIAGNOSIS : GLIOMA
  Confidence Score    : 99.82%
------------------------------------------------------------
  Glioma           █████████████████████████████   99.82%
  Meningioma                                        0.11%
  Pituitary                                         0.05%
  No Tumor                                          0.02%
============================================================
Diagnostic report figure saved to: diagnostic_report.png
```

---

## 🔬 Hardware Requirements & Runtime Profile

- **GPU Acceleration:** Compatible with any CUDA-enabled GPU (NVIDIA RTX 3060/4090, Tesla T4, P100, V100, A100).
- **VRAM Consumption:** $< 2.4\text{ GB}$ VRAM during training with batch size 16 and AMP FP16.
- **CPU / RAM:** Minimum 8 GB RAM (16 GB recommended for in-memory pre-caching).
- **Training Duration:** $\approx 25.8\text{ minutes}$ for 40 epochs on dual NVIDIA Tesla T4 GPUs.

---

## 📜 Citation

If you use this model or code in your research, please cite our forthcoming study:

```bibtex
@article{arche_braintumor_cls2026,
  title   = {Parameter-Efficient Multi-Scale Residual Network with Dual Invariant Pooling for Diagnostic Multi-Class Brain Tumor MRI Classification},
  author  = {Md Ashraful Hossen Akash},
  target journal = {Biomedical Signal Processing and Control},
  year    = {2026}
}
```

---

## 📄 License

This repository is distributed under the [MIT License](LICENSE).
