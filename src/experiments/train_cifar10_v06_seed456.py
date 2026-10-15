"""Train a CIFAR-10 ResNet-18 baseline with a fixed stratified 45k/5k split.

This script deliberately never constructs or evaluates the official CIFAR-10 test set.
Run from the Brahma-MetaCore project root:
    python -m src.experiments.train_cifar10_v02 --epochs 20
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "cifar10"
CHECKPOINT_DIR = ROOT / "checkpoints" / "milestone2_v06_seed456"
REPORT_DIR = ROOT / "reports" / "milestone2_v06_seed456"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class IndexedCIFAR10(Dataset):
    """View selected rows of one CIFAR-10 object with a split-specific transform."""

    def __init__(self, base: datasets.CIFAR10, indices: np.ndarray, transform: Any):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int):
        source_index = int(self.indices[index])
        image = Image.fromarray(self.base.data[source_index])
        label = int(self.base.targets[source_index])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Favor repeatability over maximum speed for this research baseline.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_stratified_split(targets: list[int], seed: int, val_per_class: int = 500):
    """Return deterministic train/validation indices with equal class representation."""
    targets_array = np.asarray(targets, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_id in np.unique(targets_array):
        class_indices = np.flatnonzero(targets_array == class_id)
        rng.shuffle(class_indices)
        if len(class_indices) <= val_per_class:
            raise ValueError(f"Class {class_id} has only {len(class_indices)} examples.")
        val_indices.extend(class_indices[:val_per_class].tolist())
        train_indices.extend(class_indices[val_per_class:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def build_model() -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def count_labels(targets: list[int], indices: np.ndarray) -> dict[str, int]:
    counts = Counter(int(targets[int(i)]) for i in indices)
    return {str(k): int(counts[k]) for k in sorted(counts)}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)
            loss_sum += float(loss.item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1), "examples": total}


def train_one_epoch(model, loader, optimizer, scaler, device, amp_enabled):
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(labels.numel())
        loss_sum += float(loss.detach().item()) * batch_size
        correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
        total += batch_size
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1), "examples": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")
    if not (DATA_ROOT / "cifar-10-batches-py").exists():
        raise FileNotFoundError(
            f"CIFAR-10 files were not found under {DATA_ROOT}. Expected the folder "
            "data/cifar10/cifar-10-batches-py. Extract the CIFAR-10 Python archive first."
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda") and not args.disable_amp

    print(f"Project root: {ROOT}")
    print(f"Device: {device}; AMP: {amp_enabled}")
    print("Loading CIFAR-10 training partition only (official test set is not loaded)...")
    base_train = datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=False, transform=None)
    source_split_path = ROOT / "checkpoints" / "milestone2_v02" / "split_indices_v02.npz"
    if not source_split_path.exists():
        raise FileNotFoundError(f"Source split not found: {source_split_path}")

    with np.load(source_split_path) as split:
        train_indices = split["train_indices"].copy()
        val_indices = split["val_indices"].copy()

    split_path = CHECKPOINT_DIR / "split_indices_v06_seed456.npz"
    np.savez_compressed(split_path, train_indices=train_indices, val_indices=val_indices)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = IndexedCIFAR10(base_train, train_indices, train_transform)
    val_ds = IndexedCIFAR10(base_train, val_indices, val_transform)
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, **loader_options)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, **loader_options)

    split_metadata = {
        "training_seed": args.seed,
        "split_source": "checkpoints/milestone2_v02/split_indices_v02.npz",
        "dataset": "CIFAR-10 official training partition only",
        "training_examples": int(len(train_indices)),
        "validation_examples": int(len(val_indices)),
        "training_class_counts": count_labels(base_train.targets, train_indices),
        "validation_class_counts": count_labels(base_train.targets, val_indices),
        "split_file": str(split_path.relative_to(ROOT)),
        "validation_transform": "ToTensor + Normalize; no random augmentation",
        "training_transform": "RandomCrop(32,padding=4) + RandomHorizontalFlip + Normalize",
        "official_test_set_used": False,
    }
    (REPORT_DIR / "split_metadata_v02.json").write_text(
        json.dumps(split_metadata, indent=2), encoding="utf-8"
    )
    print(f"Split: {len(train_indices):,} train / {len(val_indices):,} validation")
    print(f"Saved exact split indices: {split_path.relative_to(ROOT)}")

    model = build_model().to(device)
    initial_path = CHECKPOINT_DIR / "initial.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": 0,
        "seed": args.seed,
        "checkpoint_type": "random_initialization",
        "architecture": "torchvision_resnet18_cifar10_stem",
    }, initial_path)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_accuracy = -1.0
    best_loss = math.inf
    history: list[dict[str, Any]] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, amp_enabled)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": val_metrics["loss"],
            "validation_accuracy": val_metrics["accuracy"],
            "epoch_seconds": elapsed,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {row['train_loss']:.4f}, acc {row['train_accuracy']*100:.2f}% | "
            f"val loss {row['validation_loss']:.4f}, acc {row['validation_accuracy']*100:.2f}% | "
            f"{elapsed:.1f}s"
        )

        is_better = (val_metrics["accuracy"] > best_accuracy) or (
            math.isclose(val_metrics["accuracy"], best_accuracy) and val_metrics["loss"] < best_loss
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "seed": args.seed,
            "validation_accuracy": val_metrics["accuracy"],
            "validation_loss": val_metrics["loss"],
            "architecture": "torchvision_resnet18_cifar10_stem",
            "model_config": {
                "num_classes": 10,
                "conv1_kernel_size": 3,
                "conv1_stride": 1,
                "maxpool": "Identity",
            },
            "split_file": str(split_path.relative_to(ROOT)),
            "official_test_set_used": False,
        }
        torch.save(checkpoint, CHECKPOINT_DIR / "last.pt")
        if is_better:
            best_accuracy = val_metrics["accuracy"]
            best_loss = val_metrics["loss"]
            torch.save(checkpoint, CHECKPOINT_DIR / "best_validation.pt")
            print("  -> Saved new best_validation.pt")

        (REPORT_DIR / "training_history_v02.json").write_text(
            json.dumps({
                "experiment": "MetaAnalyzer v0.6 multi-seed target seed 456",
                "config": vars(args),
                "split": split_metadata,
                "best_checkpoint": "checkpoints/milestone2_v06_seed456/best_validation.pt",
                "best_validation_accuracy": best_accuracy,
                "best_validation_loss": best_loss,
                "history": history,
            }, indent=2), encoding="utf-8"
        )

    print(f"Training complete in {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")
    print(f"Best checkpoint: {CHECKPOINT_DIR / 'best_validation.pt'}")
    print(f"History report: {REPORT_DIR / 'training_history_v02.json'}")


if __name__ == "__main__":
    main()
