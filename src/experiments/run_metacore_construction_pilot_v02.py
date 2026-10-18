from __future__ import annotations

import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18

ROOT = Path(r"D:\all project\Projects\AI projects\Brahma-MetaCore")
DATA_ROOT = ROOT / "data" / "cifar10"
SPLIT_PATH = ROOT / "checkpoints" / "milestone2_v02" / "split_indices_v02.npz"
SOURCE = ROOT / "checkpoints" / "milestone2_v02" / "best_validation.pt"
OUT = ROOT / "reports" / "metacore_construction_pilot_v02"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
SEED = 9001
RANDOM_CONTROL_SEED = 20261018
EPOCHS = 10
BATCH_SIZE = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 0

# Matched learned-stage and random-stage arms; full_source is a practical reference
# with a lower fine-tuning LR because LR=0.1 sharply degraded it in v0.1.
ARMS = [
    {"name": "random_init", "kind": "random", "selector": None, "lr": 0.1},
    {"name": "full_source", "kind": "learned", "selector": "FULL", "lr": 0.001},
]
for stage in range(1, 5):
    ARMS.append({"name": f"stage_layer{stage}_learned", "kind": "learned", "selector": f"layer{stage}.", "lr": 0.1})
    ARMS.append({"name": f"stage_layer{stage}_random_control", "kind": "random_control", "selector": f"layer{stage}.", "lr": 0.1})

class IndexedCIFAR10(Dataset):
    def __init__(self, base, indices, transform):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        i = int(self.indices[idx])
        image = Image.fromarray(self.base.data[i])
        label = int(self.base.targets[i])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_model():
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(images)
            loss_sum += float(criterion(logits.float(), labels).item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    return {"loss": loss_sum / total, "accuracy": correct / total, "examples": total}

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
        n = int(labels.numel())
        loss_sum += float(loss.detach().item()) * n
        correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
        total += n
    return {"loss": loss_sum / total, "accuracy": correct / total, "examples": total}

def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT}")
    for path in (SOURCE, SPLIT_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (DATA_ROOT / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(f"CIFAR-10 data missing under {DATA_ROOT}")

    source_hash_before = sha256_file(SOURCE)
    split_hash = sha256_file(SPLIT_PATH)
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=True)
    source_state = checkpoint["model_state_dict"]
    with np.load(SPLIT_PATH) as split:
        train_indices = split["train_indices"].copy()
        val_indices = split["val_indices"].copy()
    if len(train_indices) != 45000 or len(val_indices) != 5000:
        raise ValueError("Frozen split must contain 45,000 train and 5,000 validation indices.")
    if np.intersect1d(train_indices, val_indices).size:
        raise ValueError("Training/validation overlap detected.")
    if len(np.unique(np.concatenate([train_indices, val_indices]))) != 50000:
        raise ValueError("Frozen split does not cover 50,000 unique training-partition images.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    print(f"Device: {device}; AMP: {amp_enabled}")
    print(f"Dataset split: {len(train_indices)} train / {len(val_indices)} validation")
    print(f"Pilot v0.2: {len(ARMS)} arms x {EPOCHS} epochs; batch size {BATCH_SIZE}")
    print("Official CIFAR-10 test set will not be loaded.")
    print("Random-stage controls use randomly initialized stage tensors, not trained-source tensors.")

    base_train = datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=False, transform=None)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_ds = IndexedCIFAR10(base_train, train_indices, train_transform)
    val_ds = IndexedCIFAR10(base_train, val_indices, val_transform)
    loader_options = {"num_workers": NUM_WORKERS, "pin_memory": device.type == "cuda", "persistent_workers": False}

    set_seed(SEED)
    canonical = build_model()
    base_state = {k: v.detach().cpu().clone() for k, v in canonical.state_dict().items()}
    if set(base_state) != set(source_state):
        raise RuntimeError("Source checkpoint and target model keys differ.")
    shape_errors = [k for k in base_state if base_state[k].shape != source_state[k].shape]
    if shape_errors:
        raise RuntimeError(f"Shape mismatches: {shape_errors[:10]}")

    set_seed(RANDOM_CONTROL_SEED)
    random_donor = build_model()
    random_donor_state = {k: v.detach().cpu().clone() for k, v in random_donor.state_dict().items()}
    parameter_names = {k for k, _ in canonical.named_parameters()}
    total_params = sum(p.numel() for p in canonical.parameters())

    # Snapshot random donor before any training and verify it is genuinely independent.
    if all(torch.equal(random_donor_state[k], base_state[k]) for k in base_state):
        raise RuntimeError("Random donor unexpectedly equals canonical initialization.")

    arm_results = []
    experiment_start = time.perf_counter()
    for arm in ARMS:
        arm_name, kind, selector, lr = arm["name"], arm["kind"], arm["selector"], arm["lr"]
        print(f"\n=== ARM: {arm_name} | lr={lr} ===")
        set_seed(SEED)  # equal augmentation RNG stream at arm start
        state = {k: v.clone() for k, v in base_state.items()}
        copied_keys = []
        donor_state = random_donor_state if kind == "random_control" else source_state
        for key in state:
            copy_key = (kind == "learned" and selector == "FULL") or (selector is not None and key.startswith(selector))
            if copy_key:
                state[key].copy_(donor_state[key])
                copied_keys.append(key)

        model = build_model()
        model.load_state_dict(state, strict=True)
        loaded = model.state_dict()
        if any(not torch.equal(loaded[k], donor_state[k]) for k in copied_keys):
            raise RuntimeError(f"{arm_name}: donor tensor-copy integrity failure")
        if any(not torch.equal(loaded[k], base_state[k]) for k in state if k not in copied_keys):
            raise RuntimeError(f"{arm_name}: uncopied random tensor integrity failure")
        model = model.to(device)

        loader_generator = torch.Generator().manual_seed(SEED + 101)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                                  generator=loader_generator, **loader_options)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, **loader_options)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        arm_start = time.perf_counter()
        initial_metrics = evaluate(model, val_loader, device)
        print(f"Initial: val loss {initial_metrics['loss']:.4f}, accuracy {100*initial_metrics['accuracy']:.2f}%")

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        history = []
        best_accuracy, best_loss, best_epoch = initial_metrics["accuracy"], initial_metrics["loss"], 0
        for epoch in range(1, EPOCHS + 1):
            if device.type == "cuda": torch.cuda.synchronize(device)
            epoch_start = time.perf_counter()
            train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, amp_enabled)
            val_metrics = evaluate(model, val_loader, device)
            scheduler.step()
            if device.type == "cuda": torch.cuda.synchronize(device)
            row = {
                "epoch": epoch, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
                "validation_loss": val_metrics["loss"], "validation_accuracy": val_metrics["accuracy"],
                "epoch_seconds": time.perf_counter() - epoch_start,
                "learning_rate_after_scheduler_step": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            if (val_metrics["accuracy"] > best_accuracy or
                (math.isclose(val_metrics["accuracy"], best_accuracy) and val_metrics["loss"] < best_loss)):
                best_accuracy, best_loss, best_epoch = val_metrics["accuracy"], val_metrics["loss"], epoch
            print(f"Epoch {epoch:02d}/{EPOCHS}: train acc {100*row['train_accuracy']:.2f}% | "
                  f"val acc {100*row['validation_accuracy']:.2f}% | val loss {row['validation_loss']:.4f} | "
                  f"{row['epoch_seconds']:.1f}s")

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = int(torch.cuda.max_memory_allocated(device))
        else:
            peak_memory = None
        copied_elements = sum(donor_state[k].numel() for k in copied_keys)
        copied_parameters = sum(donor_state[k].numel() for k in copied_keys if k in parameter_names)
        result = {
            **arm,
            "copied_tensor_count": len(copied_keys),
            "copied_state_tensor_elements_including_buffers": copied_elements,
            "copied_parameter_elements": copied_parameters,
            "copied_parameter_fraction": copied_parameters / total_params,
            "initial_validation": initial_metrics,
            "history": history,
            "best_validation_accuracy_during_run": best_accuracy,
            "best_validation_loss_at_selected_epoch": best_loss,
            "best_epoch_including_initial_epoch_zero": best_epoch,
            "arm_elapsed_seconds_including_initial_eval": time.perf_counter() - arm_start,
            "peak_cuda_memory_allocated_bytes": peak_memory,
        }
        arm_results.append(result)
        del model, optimizer, scheduler, scaler, train_loader, val_loader
        if device.type == "cuda": torch.cuda.empty_cache()

    source_hash_after = sha256_file(SOURCE)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Source checkpoint hash changed during pilot.")

    report = {
        "title": "Brahma MetaCore Construction Pilot v0.2",
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda, "device": str(device), "amp_enabled": amp_enabled,
        "source_checkpoint": str(SOURCE), "source_checkpoint_sha256_before": source_hash_before,
        "source_checkpoint_sha256_after": source_hash_after, "source_checkpoint_unchanged": source_hash_before == source_hash_after,
        "split_file": str(SPLIT_PATH), "split_sha256": split_hash,
        "train_examples": len(train_indices), "validation_examples": len(val_indices),
        "official_cifar10_test_loaded": False, "source_seed": checkpoint.get("seed"),
        "source_epoch": checkpoint.get("epoch"), "source_validation_accuracy": checkpoint.get("validation_accuracy"),
        "source_validation_loss": checkpoint.get("validation_loss"), "seed_for_all_arms": SEED,
        "random_donor_seed": RANDOM_CONTROL_SEED, "epochs_per_arm": EPOCHS, "batch_size": BATCH_SIZE,
        "optimizer": "SGD", "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
        "learning_rate_policy": {"random_init_and_stage_arms": 0.1, "full_source": 0.001,
                                  "reason": "Full-source arm degraded sharply at lr=0.1 in v0.1; this is a practical tuned-reference arm, not a matched-LR causal comparison."},
        "scheduler": "CosineAnnealingLR, T_max=10", "fresh_optimizer_per_arm": True,
        "same_canonical_random_initialization_for_uncopied_tensors": True,
        "random_stage_controls_copy_random_donor_stage_tensors": True,
        "training_transform": "RandomCrop(32,padding=4) + RandomHorizontalFlip + ToTensor + Normalize",
        "validation_transform": "ToTensor + Normalize", "total_elapsed_seconds": time.perf_counter() - experiment_start,
        "arms": arm_results,
        "interpretation_limits": [
            "Exploratory experiment; not confirmatory and not a demonstration of model reconstruction.",
            "Source checkpoint was trained on the same CIFAR-10 task and training split; its validation score is not independent.",
            "Learned-stage and random-stage controls are matched by stage and parameter count, but stage sizes differ across stages.",
            "Full-source uses a different learning rate, so comparisons against other arms are not matched-hyperparameter causal estimates.",
            "All arms use the same training/validation split; official CIFAR-10 test set was not loaded.",
            "No claim about reduced pretraining, new-task transfer, or general model construction is justified by this experiment alone."
        ]
    }
    OUT.mkdir(parents=True, exist_ok=False)
    json_path, md_path = OUT / "pilot_report.json", OUT / "pilot_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Brahma MetaCore Construction Pilot v0.2", "", "**Status:** Completed", "",
        f"- Device: `{device}` (AMP: `{amp_enabled}`)",
        f"- Source checkpoint unchanged: `{source_hash_before == source_hash_after}`",
        f"- Frozen split SHA-256: `{split_hash}`", f"- Train/validation: {len(train_indices)}/{len(val_indices)}",
        f"- Budget: {EPOCHS} epochs per arm; batch size {BATCH_SIZE}",
        "- Optimizer: SGD; momentum=0.9; weight_decay=0.0005; cosine schedule over 10 epochs",
        "- Learning rates: scratch/partial arms 0.1; full-source reference 0.001 (not matched across all arms)",
        "- Random-stage controls copy the corresponding stage from an independent random initialization",
        "- Official CIFAR-10 test set loaded: No", "", "## Results", "",
        "| Arm | Copied parameters | Initial val acc | Best val acc | Best epoch | Final val loss | Elapsed (s) | Peak CUDA MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for a in arm_results:
        peak = a["peak_cuda_memory_allocated_bytes"]
        peak_mib = f"{peak / (1024**2):.1f}" if peak is not None else "n/a"
        final = a["history"][-1]
        lines.append(f"| `{a['name']}` | {a['copied_parameter_elements']:,} ({100*a['copied_parameter_fraction']:.2f}%) | "
                     f"{100*a['initial_validation']['accuracy']:.2f}% | {100*a['best_validation_accuracy_during_run']:.2f}% | "
                     f"{a['best_epoch_including_initial_epoch_zero']} | {final['validation_loss']:.4f} | "
                     f"{a['arm_elapsed_seconds_including_initial_eval']:.1f} | {peak_mib} |")
    lines += ["", "## Interpretation and limitations", ""]
    lines.extend(f"- {x}" for x in report["interpretation_limits"])
    lines += ["", f"Total elapsed seconds: {report['total_elapsed_seconds']:.1f}", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("\nPILOT v0.2 COMPLETED")
    print("JSON report:", json_path)
    print("Markdown report:", md_path)
    print("Source checkpoint unchanged:", source_hash_before == source_hash_after)

if __name__ == "__main__":
    main()
