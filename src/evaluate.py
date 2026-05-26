"""
Usage:
  python src/evaluate.py --task main
  python src/evaluate.py --task majority_vote
  python src/evaluate.py --task yawn
  python src/evaluate.py --task all
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 224
BATCH_SIZE   = 64
NUM_WORKERS  = 4

# checkpoint filenames saved by train.py
CKPT = {
    "main":   "efficientnetv2s_cbam_main_best.pth",
    "small":  "efficientnetv2s_cbam_small_best.pth",
    "medium": "efficientnetv2s_cbam_medium_best.pth",
    "large":  "efficientnetv2s_cbam_large_best.pth",
    "yawn":   "efficientnetv2s_cbam_yawn_best.pth",
}

# this file stores the voting weights so demo.py can load them at runtime
MV_WEIGHTS_FILE = OUTPUTS_DIR / "models" / "majority_voting_weights.json"


def _test_transform():
    # same as validation transform in train.py: grayscale, resize, normalize
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_test_loader(data_dir: Path) -> Tuple[DataLoader, List[str]]:
    ds = datasets.ImageFolder(data_dir / "test", transform=_test_transform())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    return loader, ds.classes


def load_model(ckpt_name: str) -> torch.nn.Module:
    path = OUTPUTS_DIR / "models" / ckpt_name
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = build_model(num_classes=2, freeze_backbone=False).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def predict(model, loader) -> Tuple[List[int], List[int], torch.Tensor]:
    # runs the model on all test images and returns labels, predictions, and probabilities
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs  = imgs.to(DEVICE)
            probs = F.softmax(model(imgs), dim=1)
            all_probs.append(probs.cpu())
            all_preds.extend(probs.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())
    return all_labels, all_preds, torch.cat(all_probs)


def plot_cm(labels, preds, class_names, title, save_path: Path) -> None:
    # creates and saves a confusion matrix image
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path.name}")


def _acc(labels, preds) -> float:
    return sum(l == p for l, p in zip(labels, preds)) / len(labels)


def evaluate_main() -> float:
    print("\n=== Main Eye Model Evaluation ===")
    loader, classes = get_test_loader(PROJECT_ROOT / "data")
    model  = load_model(CKPT["main"])
    labels, preds, _ = predict(model, loader)
    acc = _acc(labels, preds)
    print(f"Test accuracy  : {acc:.4f}")
    print(classification_report(labels, preds, target_names=classes))
    plot_cm(labels, preds, classes,
            "EfficientNetV2-S + CBAM (Main)",
            OUTPUTS_DIR / "plots" / "cm_main.png")
    return acc


def evaluate_majority_vote() -> None:
    # runs all three sensor submodels on the same test set
    # then combines their predictions using weighted voting
    # the weights are based on each model's test accuracy
    print("\n=== Majority Voting Evaluation ===")
    sensors = ["small", "medium", "large"]

    # use the shared test set so all models run on the same images
    shared_loader, classes = get_test_loader(PROJECT_ROOT / "data")

    probs_per_sensor = {}
    weights = {}
    all_labels = None

    for sensor in sensors:
        model = load_model(CKPT[sensor])
        labels, preds, probs = predict(model, shared_loader)

        sensor_acc = _acc(labels, preds)
        print(f"  [{sensor}] test_acc={sensor_acc:.4f}")
        print(classification_report(labels, preds, target_names=classes))

        probs_per_sensor[sensor] = probs
        weights[sensor] = float(sensor_acc)
        if all_labels is None:
            all_labels = labels

    # normalize weights so they add up to 1
    total_w = sum(weights.values())
    weights = {s: w / total_w for s, w in weights.items()}
    print(f"\nNormalized voting weights: {weights}")

    # save weights to file so demo.py can load them without re-running evaluation
    MV_WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MV_WEIGHTS_FILE, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"  Weights saved: {MV_WEIGHTS_FILE.name}")

    # combine probabilities from all three models
    combined = sum(probs_per_sensor[s] * weights[s] for s in sensors)
    mv_preds = combined.argmax(1).tolist()

    mv_acc = _acc(all_labels, mv_preds)
    print(f"\nMajority Vote test_acc = {mv_acc:.4f}")
    print(classification_report(all_labels, mv_preds, target_names=classes))

    plot_cm(all_labels, mv_preds, classes,
            "Majority Voting (small + medium + large)",
            OUTPUTS_DIR / "plots" / "cm_majority_vote.png")


def evaluate_yawn() -> float:
    print("\n=== Yawn Model Evaluation ===")
    loader, classes = get_test_loader(PROJECT_ROOT / "yawn_data")
    model  = load_model(CKPT["yawn"])
    labels, preds, _ = predict(model, loader)
    acc = _acc(labels, preds)
    print(f"Test accuracy  : {acc:.4f}")
    print(classification_report(labels, preds, target_names=classes))
    plot_cm(labels, preds, classes,
            "Yawn Detection (EfficientNetV2-S + CBAM)",
            OUTPUTS_DIR / "plots" / "cm_yawn.png")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        choices=["main", "majority_vote", "yawn", "all"],
                        default="main")
    args = parser.parse_args()

    if args.task in ("main", "all"):
        evaluate_main()
    if args.task in ("majority_vote", "all"):
        evaluate_majority_vote()
    if args.task in ("yawn", "all"):
        evaluate_yawn()
