"""
Usage:
  python src/train.py                        # main eye model
  python src/train.py --sensor small         # submodel sensor 01
  python src/train.py --sensor medium        # submodel sensor 03
  python src/train.py --sensor large         # submodel sensor 02
  python src/train.py --task yawn            # yawn detection model
  python src/train.py --task eye --no-weighted
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset import make_weighted_sampler
from model import build_model, unfreeze_backbone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# training settings for each task
# yawn uses smaller batch and more epochs because the dataset is smaller
TASK_CFG = {
    "eye": dict(
        img_size      = 224,
        batch_size    = 64,
        phase1_epochs = 10,
        phase2_epochs = 20,
        phase1_lr     = 1e-3,
        phase2_lr     = 1e-4,
        weight_decay  = 1e-4,
        h_flip        = True,
        rotation      = 10,
        color_jitter  = (0.2, 0.2, 0.0, 0.0),
    ),
    "yawn": dict(
        img_size      = 224,
        batch_size    = 32,
        phase1_epochs = 15,
        phase2_epochs = 30,
        phase1_lr     = 1e-3,
        phase2_lr     = 5e-5,
        weight_decay  = 1e-3,
        h_flip        = True,
        rotation      = 15,
        color_jitter  = (0.3, 0.3, 0.1, 0.05),
    ),
}

# on Windows, using multiple workers with CUDA causes memory errors
# so we set it to 0 to run data loading on the main process
NUM_WORKERS = 0 if __import__("sys").platform == "win32" else 4


def get_transforms(cfg: dict):
    # we convert to grayscale first then repeat to 3 channels
    # this matches how the webcam frames are processed in demo.py
    jitter_args = cfg["color_jitter"]
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.RandomHorizontalFlip() if cfg["h_flip"] else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(cfg["rotation"]),
        transforms.ColorJitter(*jitter_args),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # validation doesn't use any augmentation, just resize and normalize
    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return train_tf, val_tf


def get_dataloaders(data_dir: Path, cfg: dict, use_weighted: bool):
    train_tf, val_tf = get_transforms(cfg)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(data_dir / "val",   transform=val_tf)

    print(f"Classes        : {train_ds.classes}")
    print(f"Train samples  : {len(train_ds)}")
    print(f"Val   samples  : {len(val_ds)}")

    if use_weighted:
        print("Weighted sampler (train):")
        sampler = make_weighted_sampler(train_ds)
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, train: bool):
    # runs one full pass over the data (either training or evaluation)
    model.train() if train else model.eval()
    total_loss = correct = total = 0

    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


def train(data_dir: Path, model_name: str, cfg: dict, use_weighted: bool):
    print(f"\nDevice         : {DEVICE}")
    print(f"Model name     : {model_name}")
    print(f"Data dir       : {data_dir}")
    print(f"Batch size     : {cfg['batch_size']}")

    train_loader, val_loader = get_dataloaders(data_dir, cfg, use_weighted)

    # start with backbone frozen (phase 1)
    model     = build_model(num_classes=2, freeze_backbone=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    # phase 1: only CBAM and the classification head are trained
    p1 = cfg["phase1_epochs"]
    print(f"\n--- Phase 1: CBAM + head only ({p1} epochs, backbone frozen) ---")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["phase1_lr"], weight_decay=cfg["weight_decay"],
    )
    # cosine schedule gradually lowers the learning rate over the epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p1)

    for epoch in range(1, p1 + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, None,      train=False)
        scheduler.step()
        print(
            f"  [{epoch:2d}/{p1}]"
            f"  train {tr_loss:.4f}/{tr_acc:.4f}"
            f"  val {vl_loss:.4f}/{vl_acc:.4f}"
            f"  ({time.time()-t0:.1f}s)"
        )
        # save the best checkpoint based on validation accuracy
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(),
                       OUTPUTS_DIR / "models" / f"{model_name}_phase1_best.pth")

    # phase 2: unfreeze everything and fine-tune with a smaller learning rate
    p2 = cfg["phase2_epochs"]
    print(f"\n--- Phase 2: full fine-tune ({p2} epochs) ---")
    unfreeze_backbone(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["phase2_lr"], weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p2)

    for epoch in range(1, p2 + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, None,      train=False)
        scheduler.step()
        print(
            f"  [{epoch:2d}/{p2}]"
            f"  train {tr_loss:.4f}/{tr_acc:.4f}"
            f"  val {vl_loss:.4f}/{vl_acc:.4f}"
            f"  ({time.time()-t0:.1f}s)"
        )
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(),
                       OUTPUTS_DIR / "models" / f"{model_name}_best.pth")

    print(f"\nBest val acc   : {best_val_acc:.4f}")
    print(f"Saved          : outputs/models/{model_name}_best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",    choices=["eye", "yawn"], default="eye")
    parser.add_argument("--sensor",  choices=["small", "medium", "large"], default=None,
                        help="Train per-sensor eye submodel (--task eye only).")
    parser.add_argument("--no-weighted", action="store_true",
                        help="Disable weighted sampling.")
    args = parser.parse_args()

    cfg = TASK_CFG[args.task]

    if args.task == "yawn":
        data_dir   = PROJECT_ROOT / "yawn_data"
        model_name = "efficientnetv2s_cbam_yawn"
    elif args.sensor:
        data_dir   = PROJECT_ROOT / "data_submodels" / args.sensor
        model_name = f"efficientnetv2s_cbam_{args.sensor}"
    else:
        data_dir   = PROJECT_ROOT / "data"
        model_name = "efficientnetv2s_cbam_main"

    train(data_dir, model_name, cfg, use_weighted=not args.no_weighted)
