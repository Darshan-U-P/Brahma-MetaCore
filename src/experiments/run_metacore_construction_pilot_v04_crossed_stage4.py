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
# Fully crossed design: every source checkpoint is tested with both target seeds.
# Unique source_id values keep each source-target pair separately auditable.
SOURCE_CHECKPOINTS = [
    {"source_id": "seed42_target9001", "source_family": "seed42", "path": ROOT / "checkpoints" / "milestone2_v02" / "best_validation.pt", "target_seed": 9001},
    {"source_id": "seed42_target9101", "source_family": "seed42", "path": ROOT / "checkpoints" / "milestone2_v02" / "best_validation.pt", "target_seed": 9101},
    {"source_id": "seed2027_target9001", "source_family": "seed2027", "path": ROOT / "checkpoints" / "metatransfer_v03_seed2027" / "best_validation.pt", "target_seed": 9001},
    {"source_id": "seed2027_target9101", "source_family": "seed2027", "path": ROOT / "checkpoints" / "metatransfer_v03_seed2027" / "best_validation.pt", "target_seed": 9101},
    {"source_id": "seed31415_target9001", "source_family": "seed31415", "path": ROOT / "checkpoints" / "metatransfer_v03_seed31415" / "best_validation.pt", "target_seed": 9001},
    {"source_id": "seed31415_target9101", "source_family": "seed31415", "path": ROOT / "checkpoints" / "metatransfer_v03_seed31415" / "best_validation.pt", "target_seed": 9101},
]
OUT = ROOT / "reports" / "metacore_construction_pilot_v04_crossed_stage4"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
EPOCHS = 10
BATCH_SIZE = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 0
STAGE_SELECTOR = "layer4."

# Within each source-checkpoint group, all arms share the same target random
# initialization and data-order seed. Only the designated donor tensors differ.
# The full-source arm uses a separately documented, lower fine-tuning LR.
ARM_TEMPLATES = [
    {"name": "stage4_learned", "kind": "learned_stage", "selector": STAGE_SELECTOR, "lr": 0.1},
    {"name": "stage4_random_control", "kind": "random_control", "selector": STAGE_SELECTOR, "lr": 0.1},
]


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


def validate_split(split_path: Path):
    with np.load(split_path) as split:
        train_indices = split["train_indices"].copy()
        val_indices = split["val_indices"].copy()
    if len(train_indices) != 45000 or len(val_indices) != 5000:
        raise ValueError("Frozen split must contain 45,000 train and 5,000 validation indices.")
    if len(np.unique(train_indices)) != len(train_indices) or len(np.unique(val_indices)) != len(val_indices):
        raise ValueError("Duplicate indices found in frozen split.")
    if np.intersect1d(train_indices, val_indices).size:
        raise ValueError("Training/validation overlap detected.")
    if len(np.unique(np.concatenate([train_indices, val_indices]))) != 50000:
        raise ValueError("Frozen split does not cover 50,000 unique training-partition images.")
    return train_indices, val_indices


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT}")
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(SPLIT_PATH)
    if not (DATA_ROOT / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(f"CIFAR-10 data missing under {DATA_ROOT}")
    for item in SOURCE_CHECKPOINTS:
        if not item["path"].is_file():
            raise FileNotFoundError(f"Missing source checkpoint for {item['source_id']}: {item['path']}")

    source_hashes_before = {item["source_id"]: sha256_file(item["path"]) for item in SOURCE_CHECKPOINTS}
    split_hash = sha256_file(SPLIT_PATH)
    loaded_sources = {}
    for item in SOURCE_CHECKPOINTS:
        ckpt = torch.load(item["path"], map_location="cpu", weights_only=True)
        if "model_state_dict" not in ckpt:
            raise KeyError(f"model_state_dict missing from {item['path']}")
        loaded_sources[item["source_id"]] = {"checkpoint": ckpt, "state": ckpt["model_state_dict"]}

    train_indices, val_indices = validate_split(SPLIT_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    print(f"Device: {device}; AMP: {amp_enabled}")
    print(f"Dataset split: {len(train_indices)} train / {len(val_indices)} validation")
    print(f"Pilot v0.4: {len(SOURCE_CHECKPOINTS)} source-target pairs x {len(ARM_TEMPLATES)} arms x {EPOCHS} epochs")
    print("Official CIFAR-10 test set will not be loaded.")
    print("Fully crossed: each of 3 source checkpoints uses both target seeds 9001 and 9101.")
    print("Primary comparison: learned Stage 4 versus random Stage 4 control within each source-target pair.")

    base_train = datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=False, transform=None)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_ds = IndexedCIFAR10(base_train, train_indices, train_transform)
    val_ds = IndexedCIFAR10(base_train, val_indices, val_transform)
    loader_options = {"num_workers": NUM_WORKERS, "pin_memory": device.type == "cuda", "persistent_workers": False}

    results = []
    experiment_start = time.perf_counter()
    for source_item in SOURCE_CHECKPOINTS:
        source_id = source_item["source_id"]
        source_state = loaded_sources[source_id]["state"]
        source_ckpt = loaded_sources[source_id]["checkpoint"]
        target_seed = source_item["target_seed"]
        print(f"\n######## SOURCE GROUP: {source_id} | target seed={target_seed} ########")

        set_seed(target_seed)
        canonical_model = build_model()
        base_state = {k: v.detach().cpu().clone() for k, v in canonical_model.state_dict().items()}
        if set(base_state) != set(source_state):
            raise RuntimeError(f"State keys mismatch for {source_id}")
        shape_errors = [k for k in base_state if base_state[k].shape != source_state[k].shape]
        if shape_errors:
            raise RuntimeError(f"State shape mismatch for {source_id}: {shape_errors[:10]}")
        parameter_names = {k for k, _ in canonical_model.named_parameters()}
        total_params = sum(p.numel() for p in canonical_model.parameters())
        del canonical_model

        # A distinct random donor per source group; held fixed across the two paired stage arms.
        set_seed(target_seed + 1000)
        random_donor_model = build_model()
        random_donor_state = {k: v.detach().cpu().clone() for k, v in random_donor_model.state_dict().items()}
        if all(torch.equal(random_donor_state[k], base_state[k]) for k in base_state):
            raise RuntimeError(f"Random donor unexpectedly equals canonical initialization for {source_id}")
        del random_donor_model

        for arm_template in ARM_TEMPLATES:
            arm = dict(arm_template)
            arm_name, kind, selector, lr = arm["name"], arm["kind"], arm["selector"], arm["lr"]
            print(f"\n=== SOURCE {source_id} | ARM: {arm_name} | lr={lr} ===")
            set_seed(target_seed)  # reset augmentation RNG stream identically for each arm in this group
            state = {k: v.clone() for k, v in base_state.items()}
            copied_keys = []
            if kind == "learned_full":
                donor_state = source_state
            elif kind == "learned_stage":
                donor_state = source_state
            elif kind == "random_control":
                donor_state = random_donor_state
            else:
                donor_state = base_state

            if selector is not None:
                for key in state:
                    if selector == "FULL" or key.startswith(selector):
                        state[key].copy_(donor_state[key])
                        copied_keys.append(key)

            model = build_model()
            model.load_state_dict(state, strict=True)
            loaded_state = model.state_dict()
            if any(not torch.equal(loaded_state[k], donor_state[k]) for k in copied_keys):
                raise RuntimeError(f"{source_id}/{arm_name}: copied tensor integrity failure")
            if any(not torch.equal(loaded_state[k], base_state[k]) for k in state if k not in copied_keys):
                raise RuntimeError(f"{source_id}/{arm_name}: uncopied tensor differs from canonical initialization")
            model = model.to(device)

            loader_generator = torch.Generator().manual_seed(target_seed + 101)
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
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                epoch_start = time.perf_counter()
                train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, amp_enabled)
                val_metrics = evaluate(model, val_loader, device)
                scheduler.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "validation_loss": val_metrics["loss"],
                    "validation_accuracy": val_metrics["accuracy"],
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
            final_row = history[-1]
            results.append({
                "source_id": source_id,
                "source_checkpoint": str(source_item["path"]),
                "source_checkpoint_sha256": source_hashes_before[source_id],
                "source_training_seed": source_ckpt.get("seed"),
                "source_checkpoint_epoch": source_ckpt.get("epoch"),
                "source_checkpoint_validation_accuracy": source_ckpt.get("validation_accuracy"),
                "target_seed": target_seed,
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
                "final_validation_accuracy": final_row["validation_accuracy"],
                "final_validation_loss": final_row["validation_loss"],
                "arm_elapsed_seconds_including_initial_eval": time.perf_counter() - arm_start,
                "peak_cuda_memory_allocated_bytes": peak_memory,
            })
            del model, optimizer, scheduler, scaler, train_loader, val_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()

    source_hashes_after = {item["source_id"]: sha256_file(item["path"]) for item in SOURCE_CHECKPOINTS}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("One or more source checkpoint hashes changed during the experiment.")

    paired = []
    for source_item in SOURCE_CHECKPOINTS:
        sid = source_item["source_id"]
        group = {r["name"]: r for r in results if r["source_id"] == sid}
        learned = group["stage4_learned"]
        control = group["stage4_random_control"]
        paired.append({
            "source_id": sid,
            "source_family": source_item.get("source_family", sid),
            "target_seed": source_item["target_seed"],
            "learned_stage4_best_accuracy": learned["best_validation_accuracy_during_run"],
            "random_stage4_control_best_accuracy": control["best_validation_accuracy_during_run"],
            "learned_minus_random_control_best_accuracy_pp": 100 * (learned["best_validation_accuracy_during_run"] - control["best_validation_accuracy_during_run"]),
            "learned_stage4_final_accuracy": learned["final_validation_accuracy"],
            "random_stage4_control_final_accuracy": control["final_validation_accuracy"],
            "learned_minus_random_control_final_accuracy_pp": 100 * (learned["final_validation_accuracy"] - control["final_validation_accuracy"]),
        })
    deltas = [x["learned_minus_random_control_best_accuracy_pp"] for x in paired]
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0

    report = {
        "title": "Brahma MetaCore Construction Pilot v0.4 — Crossed Stage 4 Replication",
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "split_file": str(SPLIT_PATH),
        "split_sha256": split_hash,
        "train_examples": len(train_indices),
        "validation_examples": len(val_indices),
        "official_cifar10_test_loaded": False,
        "epochs_per_arm": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "SGD",
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR, T_max=10",
        "learning_rate_policy": {"random_init_and_stage4_arms": 0.1, "full_source": 0.001,
                                  "reason": "Full-source uses a practical lower fine-tuning LR; comparisons against it are not matched-LR causal estimates."},
        "design": "Fully crossed 3 source checkpoints x 2 target seeds (9001, 9101) x 2 matched Stage 4 arms.",
        "primary_comparison": "Within each source-target pair: learned Stage 4 versus random Stage 4 control, using final-epoch validation accuracy as primary endpoint. Best epoch is secondary.",
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "all_source_checkpoints_unchanged": source_hashes_before == source_hashes_after,
        "arms": results,
        "paired_stage4_comparisons": paired,
        "stage4_best_accuracy_delta_mean_pp": mean_delta,
        "stage4_best_accuracy_delta_sample_sd_pp": std_delta,
        "stage4_final_accuracy_delta_mean_pp": float(np.mean([x["learned_minus_random_control_final_accuracy_pp"] for x in paired])),
        "stage4_final_accuracy_delta_sample_sd_pp": float(np.std([x["learned_minus_random_control_final_accuracy_pp"] for x in paired], ddof=1)),
        "stage4_positive_best_delta_count": sum(d > 0 for d in deltas),
        "stage4_replication_count": len(deltas),
        "total_elapsed_seconds": time.perf_counter() - experiment_start,
        "interpretation_limits": [
            "Fully crosses three source checkpoints with two target seeds, but only two target seeds are sampled; broader replication remains necessary.",
            "The source checkpoints were trained on the same CIFAR-10 task and frozen training split; this is not cross-task transfer.",
            "Validation checkpoints were selected using the same validation partition, so source validation metrics are not independent evaluation.",
            "Final-epoch validation accuracy is the primary endpoint; best accuracy over epochs 0-10 is secondary and optimistic due to validation-based selection. Confirm on a genuinely untouched test set before strong claims.",
            "This pilot isolates the learned-versus-random Stage 4 comparison; it does not compare against full-source fine-tuning or random initialization.",
            "The three source checkpoints and their channel/parameter tensors are not independent samples from all architectures or tasks.",
            "No claim about reduced pretraining, architecture discovery, or general model construction is established by this experiment alone."
        ]
    }

    OUT.mkdir(parents=True, exist_ok=False)
    json_path = OUT / "pilot_report.json"
    md_path = OUT / "pilot_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Brahma MetaCore Construction Pilot v0.4 — Crossed Stage 4 Replication", "",
        "**Status:** Completed", "",
        f"- Device: `{device}` (AMP: `{amp_enabled}`)",
        f"- Source checkpoints unchanged: `{report['all_source_checkpoints_unchanged']}`",
        f"- Frozen split SHA-256: `{split_hash}`",
        f"- Budget: {EPOCHS} epochs per arm; batch size {BATCH_SIZE}",
        "- Design: 3 source checkpoints x 2 target seeds x 2 arms = 12 training runs",
        "- Arms: learned Stage 4 versus random Stage 4 control, both at learning rate 0.1",
        "- Target seeds: 9001 and 9101 are each crossed with every source checkpoint",
        "- Primary endpoint: final-epoch validation accuracy difference; best-epoch difference is secondary",
        "- Official CIFAR-10 test set loaded: No", "", "## Primary paired comparisons", "",
        "| Source checkpoint | Target seed | Learned Stage 4 best | Random Stage 4 best | Best difference (pp) | Learned final | Random final | Final difference (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for item in paired:
        lines.append(
            f"| `{item['source_id']}` | {item['target_seed']} | {100*item['learned_stage4_best_accuracy']:.2f}% | "
            f"{100*item['random_stage4_control_best_accuracy']:.2f}% | {item['learned_minus_random_control_best_accuracy_pp']:+.2f} | "
            f"{100*item['learned_stage4_final_accuracy']:.2f}% | {100*item['random_stage4_control_final_accuracy']:.2f}% | "
            f"{item['learned_minus_random_control_final_accuracy_pp']:+.2f} |"
        )
    final_deltas = [x["learned_minus_random_control_final_accuracy_pp"] for x in paired]
    lines += ["", f"- Mean learned-minus-random Stage 4 **best** accuracy difference (secondary): {mean_delta:+.2f} pp",
              f"- Mean learned-minus-random Stage 4 **final-epoch** accuracy difference (primary): {float(np.mean(final_deltas)):+.2f} pp",
              f"- Sample standard deviation of best-accuracy differences across 6 paired runs: {std_delta:.2f} pp",
              f"- Positive best-accuracy differences: {sum(d > 0 for d in deltas)}/{len(deltas)}", "",
              "## All arms", "", "| Source-target pair | Arm | Initial acc | Best acc | Best epoch | Final acc | Final loss | Elapsed (s) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(
            f"| `{r['source_id']}` | `{r['name']}` | {100*r['initial_validation']['accuracy']:.2f}% | "
            f"{100*r['best_validation_accuracy_during_run']:.2f}% | {r['best_epoch_including_initial_epoch_zero']} | "
            f"{100*r['final_validation_accuracy']:.2f}% | {r['final_validation_loss']:.4f} | {r['arm_elapsed_seconds_including_initial_eval']:.1f} |"
        )
    lines += ["", "## Interpretation limits", ""]
    lines.extend(f"- {x}" for x in report["interpretation_limits"])
    lines += ["", f"Total elapsed seconds: {report['total_elapsed_seconds']:.1f}", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("\nPILOT v0.4 COMPLETED")
    print("JSON report:", json_path)
    print("Markdown report:", md_path)
    print("All source checkpoints unchanged:", report["all_source_checkpoints_unchanged"])
    print("Stage 4 best-accuracy differences (pp):", [round(x, 3) for x in deltas])
    print("Mean Stage 4 best-accuracy difference (pp):", round(mean_delta, 3))


if __name__ == "__main__":
    main()
