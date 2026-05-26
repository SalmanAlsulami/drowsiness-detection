"""
Usage:
  python src/dataset.py --task eye
  python src/dataset.py --task yawn
  python src/dataset.py --task all
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List

import torch
from torch.utils.data import WeightedRandomSampler
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MRL_SOURCE   = PROJECT_ROOT / "mrl_data"
YAWN_SOURCE  = PROJECT_ROOT / "Yawn_Eye_Dataset"
DATA_DIR     = PROJECT_ROOT / "data"
SUBMODEL_DIR = PROJECT_ROOT / "data_submodels"
YAWN_DIR     = PROJECT_ROOT / "yawn_data"

# MRL uses "awake" and "sleepy" folder names, we rename them to "open" and "closed"
MRL_CLASS_MAP = {"awake": "open", "sleepy": "closed"}

# MRL filenames end with the sensor id (01, 02, 03)
# we group them into small, large, medium for the submodels
SENSOR_MAP    = {"01": "small", "02": "large", "03": "medium"}

SPLITS        = ("train", "val", "test")
YAWN_VAL_RATIO = 0.20   # 20% of train images go to validation
RANDOM_SEED    = 42


# reads the sensor id from the last part of the MRL filename
# example: s0001_01842_0_0_1_0_0_01.png  ->  "01"
def _sensor_id(filename: str) -> str:
    return Path(filename).stem.split("_")[-1]


# creates all the needed folders before copying images
def _make_eye_dirs() -> None:
    for split in SPLITS:
        for cls in ("open", "closed"):
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)
            for sensor in SENSOR_MAP.values():
                (SUBMODEL_DIR / sensor / split / cls).mkdir(parents=True, exist_ok=True)


def build_eye_splits() -> None:
    # copies images from mrl_data/ into data/ (main model)
    # and also into data_submodels/ sorted by sensor type
    if not MRL_SOURCE.exists():
        raise FileNotFoundError(f"MRL source not found: {MRL_SOURCE}")

    print("Building eye dataset from mrl_data/ ...")
    _make_eye_dirs()

    counts: dict     = defaultdict(lambda: defaultdict(int))
    sub_counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    skipped = 0

    for split in SPLITS:
        for src_cls, dst_cls in MRL_CLASS_MAP.items():
            src_folder = MRL_SOURCE / split / src_cls
            if not src_folder.exists():
                print(f"  WARNING: missing {src_folder}")
                continue
            for img in src_folder.glob("*.png"):
                # copy to main dataset
                shutil.copy2(img, DATA_DIR / split / dst_cls / img.name)
                counts[split][dst_cls] += 1

                # also copy to the matching sensor submodel folder
                sid = _sensor_id(img.name)
                if sid in SENSOR_MAP:
                    sensor = SENSOR_MAP[sid]
                    shutil.copy2(img, SUBMODEL_DIR / sensor / split / dst_cls / img.name)
                    sub_counts[sensor][split][dst_cls] += 1
                else:
                    skipped += 1

    print("\n=== Main Eye Dataset ===")
    for split in SPLITS:
        o, c = counts[split]["open"], counts[split]["closed"]
        print(f"  {split:5s}  open={o:6d}  closed={c:6d}  total={o+c:6d}")

    print("\n=== Submodel Eye Datasets ===")
    for sensor in ("small", "medium", "large"):
        print(f"  [{sensor}]")
        for split in SPLITS:
            o = sub_counts[sensor][split]["open"]
            c = sub_counts[sensor][split]["closed"]
            print(f"    {split:5s}  open={o:6d}  closed={c:6d}")

    if skipped:
        print(f"\n  Skipped (unknown sensor_id): {skipped}")
    print("Eye dataset build complete.\n")


def _make_yawn_dirs() -> None:
    for split in SPLITS:
        for cls in ("yawn", "no_yawn"):
            (YAWN_DIR / split / cls).mkdir(parents=True, exist_ok=True)


# splits a list of files into train and val randomly
def _stratified_split(files: List[Path], val_ratio: float, seed: int):
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def build_yawn_splits() -> None:
    # the yawn dataset has no validation folder, so we split train 80/20
    # we only use "yawn" and "no_yawn" classes and ignore "Open"/"Closed"
    if not YAWN_SOURCE.exists():
        raise FileNotFoundError(f"Yawn dataset not found: {YAWN_SOURCE}")

    print("Building yawn dataset from Yawn_Eye_Dataset/ ...")
    _make_yawn_dirs()

    counts: dict = defaultdict(lambda: defaultdict(int))

    for cls in ("yawn", "no_yawn"):
        src_train = YAWN_SOURCE / "train" / cls
        if not src_train.exists():
            print(f"  WARNING: missing {src_train}")
            continue

        all_files = sorted(src_train.glob("*.jpg")) + sorted(src_train.glob("*.png"))
        train_files, val_files = _stratified_split(all_files, YAWN_VAL_RATIO, RANDOM_SEED)

        for img in train_files:
            shutil.copy2(img, YAWN_DIR / "train" / cls / img.name)
            counts["train"][cls] += 1
        for img in val_files:
            shutil.copy2(img, YAWN_DIR / "val" / cls / img.name)
            counts["val"][cls] += 1

    # test set already exists so we just copy it as-is
    for cls in ("yawn", "no_yawn"):
        src_test = YAWN_SOURCE / "test" / cls
        if not src_test.exists():
            print(f"  WARNING: missing {src_test}")
            continue
        for img in list(src_test.glob("*.jpg")) + list(src_test.glob("*.png")):
            shutil.copy2(img, YAWN_DIR / "test" / cls / img.name)
            counts["test"][cls] += 1

    print("\n=== Yawn Dataset ===")
    for split in SPLITS:
        y  = counts[split]["yawn"]
        ny = counts[split]["no_yawn"]
        print(f"  {split:5s}  yawn={y:4d}  no_yawn={ny:4d}  total={y+ny:4d}")
    print("Yawn dataset build complete.\n")


def make_weighted_sampler(dataset: ImageFolder) -> WeightedRandomSampler:
    # some sensor classes have way more images than others (e.g. 18:1 ratio)
    # this sampler gives less common classes a higher chance of being picked
    # so the model sees a balanced mix during training
    class_counts: List[int] = [0] * len(dataset.classes)
    for _, label in dataset.samples:
        class_counts[label] += 1

    # weight = 1 / class_count  so rare classes get higher weight
    class_weights  = [1.0 / c if c > 0 else 0.0 for c in class_counts]
    sample_weights = torch.tensor(
        [class_weights[label] for _, label in dataset.samples],
        dtype=torch.float,
    )

    print(f"  Classes      : {dataset.classes}")
    print(f"  Class counts : {class_counts}")

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=["eye", "yawn", "all"], default="all",
        help="Which dataset to build (default: all)",
    )
    args = parser.parse_args()

    if args.task in ("eye", "all"):
        build_eye_splits()
    if args.task in ("yawn", "all"):
        build_yawn_splits()
