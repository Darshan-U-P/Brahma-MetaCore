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
SPLIT_PATH = ROOT / "checkpoints" / "milestone2_v02" / "split_indices_v02.npz"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class IndexedCIFAR10(Dataset):
    def __init__(self, base: datasets.CIFAR10, indices: np.ndarray, transform: Any):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        i = int(self.indices[index])
        image = Image.fromarray(self.base.data[i])
        label = int(self.base.targets[i])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model() -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def count_labels(targets, indices):
    counts = Counter(int(targets[int(i)]) for i in indices)
    return {str(k): int(counts[k]) for k in sorted(counts)}


def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            loss_sum += float(loss.item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "examples": total,
    }


def train_one_epoch(model, loader, optimizer, scaler, device, amp_enabled):
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=amp_enabled
        ):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n = int(labels.numel())
        loss_sum += float(loss.detach().item()) * n
        correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
        total += n
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "examples": total,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train an independent MetaTransfer v0.3 CIFAR-10 target model."
    )
    parser.add_argument("--seed", type=int, required=True)
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
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    if not (DATA_ROOT / "cifar-10-batches-py").exists():
        raise FileNotFoundError(f"CIFAR-10 data not found under {DATA_ROOT}")
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"Frozen split not found: {SPLIT_PATH}")

    tag = f"seed{args.seed}"
    checkpoint_dir = ROOT / "checkpoints" / f"metatransfer_v03_{tag}"
    report_dir = ROOT / "reports" / f"metatransfer_v03_{tag}"

    # Fail closed: never overwrite any previous experiment.
    if checkpoint_dir.exists() or report_dir.exists():
        raise FileExistsError(
            "Output already exists; refusing to overwrite:\n"
            f"  {checkpoint_dir}\n  {report_dir}"
        )

    # Validate split before creating outputs.
    with np.load(SPLIT_PATH) as split:
        train_indices = split["train_indices"].copy()
        val_indices = split["val_indices"].copy()

    if len(train_indices) != 45000 or len(val_indices) != 5000:
        raise ValueError("Frozen split has unexpected sizes.")
    if len(np.unique(train_indices)) != len(train_indices):
        raise ValueError("Duplicate training indices in frozen split.")
    if len(np.unique(val_indices)) != len(val_indices):
        raise ValueError("Duplicate validation indices in frozen split.")
    if np.intersect1d(train_indices, val_indices).size:
        raise ValueError("Training/validation overlap detected.")
    if len(np.unique(np.concatenate([train_indices, val_indices]))) != 50000:
        raise ValueError("Frozen split does not cover 50,000 unique images.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.disable_amp

    print(f"Project root: {ROOT}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}; AMP: {amp_enabled}")
    print(f"Train/validation: {len(train_indices)}/{len(val_indices)}")
    print("Official CIFAR-10 test set will not be loaded.")

    base_train = datasets.CIFAR10(
        root=str(DATA_ROOT), train=True, download=False, transform=None
    )
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
    # Explicit generator makes shuffle order reproducible for this seed.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=False, generator=loader_generator, **loader_options
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, **loader_options
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    split_copy = checkpoint_dir / "frozen_split_indices.npz"
    np.savez_compressed(
        split_copy, train_indices=train_indices, val_indices=val_indices
    )

    metadata = {
        "experiment": "MetaTransfer v0.3 independent target-seed replication",
        "seed": args.seed,
        "split_source": "checkpoints/milestone2_v02/split_indices_v02.npz",
        "split_copy": str(split_copy.relative_to(ROOT)),
        "training_examples": len(train_indices),
        "validation_examples": len(val_indices),
        "training_class_counts": count_labels(base_train.targets, train_indices),
        "validation_class_counts": count_labels(base_train.targets, val_indices),
        "training_transform": "RandomCrop(32,padding=4) + RandomHorizontalFlip + ToTensor + Normalize",
        "validation_transform": "ToTensor + Normalize",
        "official_test_set_used": False,
        "architecture": "torchvision_resnet18_cifar10_stem",
        "optimizer": "SGD",
        "scheduler": "CosineAnnealingLR",
        "config": vars(args),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "deterministic_cudnn": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    (report_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    model = build_model().to(device)
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": 0,
        "seed": args.seed,
        "checkpoint_type": "random_initialization",
        "architecture": "torchvision_resnet18_cifar10_stem",
        "split_file": str(split_copy.relative_to(ROOT)),
        "official_test_set_used": False,
    }, checkpoint_dir / "initial.pt")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate,
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_accuracy, best_loss = -1.0, math.inf
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, amp_enabled
        )
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate_after_scheduler_step": float(
                optimizer.param_groups[0]["lr"]
            ),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": val_metrics["loss"],
            "validation_accuracy": val_metrics["accuracy"],
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {row['train_loss']:.4f}, "
            f"acc {row['train_accuracy']*100:.2f}% | "
            f"val loss {row['validation_loss']:.4f}, "
            f"acc {row['validation_accuracy']*100:.2f}% | "
            f"{row['epoch_seconds']:.1f}s"
        )

        is_better = (
            val_metrics["accuracy"] > best_accuracy
            or (
                math.isclose(val_metrics["accuracy"], best_accuracy)
                and val_metrics["loss"] < best_loss
            )
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
            "split_file": str(split_copy.relative_to(ROOT)),
            "official_test_set_used": False,
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")
        if is_better:
            best_accuracy = val_metrics["accuracy"]
            best_loss = val_metrics["loss"]
            torch.save(checkpoint, checkpoint_dir / "best_validation.pt")
            print("  -> Saved new best_validation.pt")

        report = {
            **metadata,
            "best_checkpoint": str(
                (checkpoint_dir / "best_validation.pt").relative_to(ROOT)
            ),
            "best_validation_accuracy": best_accuracy,
            "best_validation_loss": best_loss,
            "elapsed_minutes": (time.time() - start_time) / 60,
            "history": history,
            "status": "training",
        }
        (report_dir / "training_history.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    report["status"] = "complete"
    report["elapsed_minutes"] = (time.time() - start_time) / 60
    (report_dir / "training_history.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Training complete in {report['elapsed_minutes']:.1f} minutes")
    print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")
    print(f"Checkpoint: {checkpoint_dir / 'best_validation.pt'}")
    print(f"Report: {report_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()
