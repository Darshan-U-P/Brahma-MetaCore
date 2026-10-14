
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.metaobserver.model_inspector import build_cifar10_resnet18


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loaders(data_dir: Path, batch_size: int, num_workers: int):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=test_transform,
    )

    generator = torch.Generator().manual_seed(42)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


def evaluate(model, loader, criterion, device, amp_enabled):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/cifar10_resnet18.json",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))

    seed = config["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    amp_enabled = bool(config["use_amp"] and device.type == "cuda")

    checkpoints_dir = project_root / "checkpoints" / "milestone2"
    reports_dir = project_root / "reports"
    data_dir = project_root / "data" / "cifar10"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = make_loaders(
        data_dir,
        config["batch_size"],
        config["num_workers"],
    )

    model = build_cifar10_resnet18().to(device)

    # Save the exact initial weights before training.
    initial_path = checkpoints_dir / "initial.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "config": config,
            "model_name": "ResNet-18-CIFAR10",
        },
        initial_path,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = []
    best_accuracy = 0.0

    print(f"Device: {device}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"Train examples: {len(train_loader.dataset):,}")
    print(f"Test examples: {len(test_loader.dataset):,}")
    print(f"Initial checkpoint: {initial_path}")

    for epoch in range(config["epochs"]):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * labels.size(0)
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_examples += labels.size(0)

        scheduler.step()

        train_metrics = {
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
        }

        # This is the held-out test set. For rigorous model selection,
        # use a separate validation split and reserve test evaluation
        # for the end.
        test_metrics = evaluate(
            model, test_loader, criterion, device, amp_enabled
        )

        epoch_record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "test": test_metrics,
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(epoch_record)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "metrics": epoch_record,
        }

        if config["save_every_epoch"]:
            torch.save(
                checkpoint,
                checkpoints_dir / f"epoch_{epoch + 1:03d}.pt",
            )

        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = test_metrics["accuracy"]
            torch.save(checkpoint, checkpoints_dir / "best.pt")

        print(
            f"Epoch {epoch + 1:02d}/{config['epochs']} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"train acc {train_metrics['accuracy']:.2%} | "
            f"test acc {test_metrics['accuracy']:.2%} | "
            f"{epoch_record['seconds']:.1f}s"
        )

        (reports_dir / "milestone2_training_history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    print(f"Best test accuracy during training: {best_accuracy:.2%}")
    print(f"Training history: {reports_dir / 'milestone2_training_history.json'}")


if __name__ == "__main__":
    main()
