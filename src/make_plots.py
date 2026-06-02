"""
Generates evaluation plots for the report:
  - Loss and accuracy curves (requires history JSON from train.py)
  - ROC curves with AUC
  - Precision-Recall curves
  - F1 / Precision / Recall comparison bar chart

Usage:
  python src/make_plots.py --task curves   # loss/accuracy curves only
  python src/make_plots.py --task roc      # ROC + PR curves only
  python src/make_plots.py --task bar      # F1 bar chart only
  python src/make_plots.py --task all      # everything
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
PLOTS_DIR    = OUTPUTS_DIR / "plots"
MODELS_DIR   = OUTPUTS_DIR / "models"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 224
BATCH_SIZE   = 64
NUM_WORKERS  = 0

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CKPT = {
    "main":   "efficientnetv2s_cbam_main_best.pth",
    "small":  "efficientnetv2s_cbam_small_best.pth",
    "medium": "efficientnetv2s_cbam_medium_best.pth",
    "large":  "efficientnetv2s_cbam_large_best.pth",
    "yawn":   "efficientnetv2s_cbam_yawn_best.pth",
}

HISTORY = {
    "main":   "efficientnetv2s_cbam_main_history.json",
    "small":  "efficientnetv2s_cbam_small_history.json",
    "medium": "efficientnetv2s_cbam_medium_history.json",
    "large":  "efficientnetv2s_cbam_large_history.json",
    "yawn":   "efficientnetv2s_cbam_yawn_history.json",
}


def _test_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def load_model(ckpt_name):
    path = MODELS_DIR / ckpt_name
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    m = build_model(num_classes=2, freeze_backbone=False).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    m.eval()
    return m


def get_probs(model, data_dir):
    ds     = datasets.ImageFolder(data_dir / "test", transform=_test_transform())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            probs = F.softmax(model(imgs.to(DEVICE)), dim=1).cpu()
            all_probs.append(probs)
            all_labels.extend(labels.tolist())
    return all_labels, torch.cat(all_probs)


# ── Loss / Accuracy Curves ────────────────────────────────────────────────────

def plot_curves(model_key: str, title: str):
    path = MODELS_DIR / HISTORY[model_key]
    if not path.exists():
        print(f"  [SKIP] history not found for '{model_key}' — retrain first")
        return

    with open(path) as f:
        h = json.load(f)

    epochs     = list(range(1, len(h["train_loss"]) + 1))
    boundary   = h.get("phase_boundary", 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, train_vals, val_vals, ylabel in [
        (ax1, h["train_loss"], h["val_loss"],  "Loss"),
        (ax2, h["train_acc"],  h["val_acc"],   "Accuracy"),
    ]:
        ax.plot(epochs, train_vals, label="Train", color="#1565C0", linewidth=2)
        ax.plot(epochs, val_vals,   label="Val",   color="#E65100", linewidth=2)
        if boundary:
            ax.axvline(boundary + 0.5, color="grey", linestyle="--", linewidth=1,
                       label="Phase 1 → 2")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save = PLOTS_DIR / f"curves_{model_key}.png"
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save.name}")


def plot_all_curves():
    print("\n=== Loss / Accuracy Curves ===")
    configs = [
        ("main",   "Main Eye Model — EfficientNetV2-S + CBAM"),
        ("small",  "Submodel: Small Sensor"),
        ("medium", "Submodel: Medium Sensor"),
        ("large",  "Submodel: Large Sensor"),
        ("yawn",   "Yawn Detection Model"),
    ]
    for key, title in configs:
        plot_curves(key, title)


# ── ROC + Precision-Recall Curves ────────────────────────────────────────────

def plot_roc_pr(model_key: str, data_dir: Path, label_name: str, title: str):
    model  = load_model(CKPT[model_key])
    labels, probs = get_probs(model, data_dir)

    # class index 0 = "closed" / "yawn" (the positive class for drowsiness)
    y_true  = np.array(labels)
    y_score = probs[:, 0].numpy()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=0)
    roc_auc     = auc(fpr, tpr)

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_true, y_score, pos_label=0)
    avg_prec     = average_precision_score(y_true, y_score, pos_label=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    ax1.plot(fpr, tpr, color="#1565C0", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(rec, prec, color="#6A1B9A", lw=2, label=f"AP = {avg_prec:.4f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision-Recall Curve ({label_name} = positive)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save = PLOTS_DIR / f"roc_pr_{model_key}.png"
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save.name}")


def plot_all_roc_pr():
    print("\n=== ROC + Precision-Recall Curves ===")
    plot_roc_pr("main",  PROJECT_ROOT / "data",      "closed", "Main Eye Model")
    plot_roc_pr("yawn",  PROJECT_ROOT / "yawn_data", "yawn",   "Yawn Detection Model")


# ── F1 / Precision / Recall Bar Chart ────────────────────────────────────────

def plot_bar_chart():
    print("\n=== F1 / Precision / Recall Bar Chart ===")

    configs = [
        ("main",  PROJECT_ROOT / "data",      "Main Eye\nModel"),
        ("small", PROJECT_ROOT / "data",      "Small\nSensor"),
        ("medium",PROJECT_ROOT / "data",      "Medium\nSensor"),
        ("large", PROJECT_ROOT / "data",      "Large\nSensor"),
        ("yawn",  PROJECT_ROOT / "yawn_data", "Yawn\nModel"),
    ]

    names, precisions, recalls, f1s, accs = [], [], [], [], []

    for key, data_dir, name in configs:
        model  = load_model(CKPT[key])
        labels, probs = get_probs(model, data_dir)
        preds  = probs.argmax(1).tolist()

        report = classification_report(labels, preds, output_dict=True)
        # weighted average across both classes
        names.append(name)
        precisions.append(report["weighted avg"]["precision"])
        recalls.append(report["weighted avg"]["recall"])
        f1s.append(report["weighted avg"]["f1-score"])
        accs.append(sum(l == p for l, p in zip(labels, preds)) / len(labels))

    x     = np.arange(len(names))
    width = 0.2
    colors = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A"]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 1.5*width, accs,       width, label="Accuracy",  color=colors[0])
    ax.bar(x - 0.5*width, precisions, width, label="Precision", color=colors[1])
    ax.bar(x + 0.5*width, recalls,    width, label="Recall",    color=colors[2])
    ax.bar(x + 1.5*width, f1s,        width, label="F1-Score",  color=colors[3])

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison (Weighted Average)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # add value labels on top of each bar
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

    plt.tight_layout()
    save = PLOTS_DIR / "bar_comparison.png"
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["curves", "roc", "bar", "all"], default="all")
    args = parser.parse_args()

    if args.task in ("curves", "all"):
        plot_all_curves()
    if args.task in ("roc", "all"):
        plot_all_roc_pr()
    if args.task in ("bar", "all"):
        plot_bar_chart()
