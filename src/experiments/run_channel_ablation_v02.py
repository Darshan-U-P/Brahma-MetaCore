"""MetaAnalyzer v0.2: validation-set channel ablation with random controls.

The official CIFAR-10 test set is never loaded. Each channel intervention is
applied on every validation batch for one unique BN1 output in a residual block.
Run from the Brahma-MetaCore project root:
    python -m src.experiments.run_channel_ablation_v02
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "cifar10"
CHECKPOINT_DIR = ROOT / "checkpoints" / "milestone2_v02"
REPORT_DIR = ROOT / "reports" / "milestone2_v02"
DEFAULT_LAYERS = ("layer2.1.bn1", "layer3.1.bn1", "layer4.1.bn1")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class IndexedCIFAR10(Dataset):
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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model() -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    current: nn.Module = model
    for part in name.split("."):
        if part.isdigit():
            current = current[int(part)]  # type: ignore[index]
        else:
            if not hasattr(current, part):
                raise KeyError(f"Module path does not exist: {name} (failed at {part})")
            current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise TypeError(f"Path is not a module: {name}")
    return current


def mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "std": None}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def evaluate_baseline(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_predictions: list[int] = []
    all_labels: list[int] = []
    per_batch: list[dict[str, Any]] = []
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            per_sample_losses = F.cross_entropy(logits, labels, reduction="none")
            predictions = logits.argmax(dim=1)
            n = int(labels.numel())
            batch_loss_sum = float(per_sample_losses.sum().item())
            batch_correct = int((predictions == labels).sum().item())
            batch_accuracy = batch_correct / n
            loss_sum += batch_loss_sum
            correct += batch_correct
            total += n
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            per_batch.append({
                "batch_index": batch_idx,
                "n_examples": n,
                "loss": batch_loss_sum / n,
                "accuracy": batch_accuracy,
            })
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "examples": total,
        "predictions": np.asarray(all_predictions, dtype=np.int64),
        "labels": np.asarray(all_labels, dtype=np.int64),
        "per_batch": per_batch,
    }


def evaluate_with_channel_ablation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    module: nn.Module,
    channel_index: int,
    baseline_predictions: np.ndarray,
    baseline_batches: list[dict[str, Any]],
):
    """Zero the chosen feature channel for every validation batch, not just once."""
    state = {"hook_calls": 0}

    def ablation_hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise RuntimeError("Selected module did not return a [batch, channels, height, width] tensor.")
        if channel_index < 0 or channel_index >= output.shape[1]:
            raise IndexError(f"Channel {channel_index} outside activation width {output.shape[1]}.")
        modified = output.clone()
        modified[:, channel_index, :, :] = 0
        state["hook_calls"] += 1
        return modified

    handle = module.register_forward_hook(ablation_hook)
    model.eval()
    predictions_all: list[int] = []
    loss_sum = 0.0
    correct = 0
    total = 0
    batch_results: list[dict[str, Any]] = []
    offset = 0
    try:
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                per_sample_losses = F.cross_entropy(logits, labels, reduction="none")
                predictions = logits.argmax(dim=1)
                pred_np = predictions.cpu().numpy().astype(np.int64, copy=False)
                n = int(labels.numel())
                batch_loss_sum = float(per_sample_losses.sum().item())
                batch_correct = int((predictions == labels).sum().item())
                batch_accuracy = batch_correct / n
                base_slice = baseline_predictions[offset:offset + n]
                flip_count = int(np.count_nonzero(pred_np != base_slice))
                base_batch = baseline_batches[batch_idx]
                base_loss = float(base_batch["loss"])
                base_accuracy = float(base_batch["accuracy"])
                batch_results.append({
                    "batch_index": batch_idx,
                    "n_examples": n,
                    "baseline_loss": base_loss,
                    "ablated_loss": batch_loss_sum / n,
                    "delta_loss": (batch_loss_sum / n) - base_loss,
                    "baseline_accuracy": base_accuracy,
                    "ablated_accuracy": batch_accuracy,
                    "delta_accuracy_pp": (batch_accuracy - base_accuracy) * 100.0,
                    "prediction_flip_rate": flip_count / n,
                })
                predictions_all.extend(pred_np.tolist())
                loss_sum += batch_loss_sum
                correct += batch_correct
                total += n
                offset += n
    finally:
        handle.remove()

    expected_batches = len(loader)
    if state["hook_calls"] != expected_batches:
        raise RuntimeError(
            f"Intervention hook ran {state['hook_calls']} times, expected once per batch "
            f"({expected_batches}). Refusing to save invalid ablation results."
        )

    predictions_np = np.asarray(predictions_all, dtype=np.int64)
    batch_delta_losses = [float(row["delta_loss"]) for row in batch_results]
    batch_delta_acc = [float(row["delta_accuracy_pp"]) for row in batch_results]
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "examples": total,
        "predictions": predictions_np,
        "prediction_flip_rate": float(np.mean(predictions_np != baseline_predictions)),
        "hook_calls": state["hook_calls"],
        "intervened_batches": state["hook_calls"],
        "batch_metrics": batch_results,
        "fraction_batches_loss_increased": float(np.mean(np.asarray(batch_delta_losses) > 0)),
        "fraction_batches_accuracy_decreased": float(np.mean(np.asarray(batch_delta_acc) < 0)),
        "batch_delta_loss_mean": float(np.mean(batch_delta_losses)),
        "batch_delta_loss_std": float(np.std(batch_delta_losses)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "best_validation.pt"))
    parser.add_argument("--split-file", default=str(CHECKPOINT_DIR / "split_indices_v02.npz"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--candidate-channels", type=int, default=32,
                        help="Randomly sampled discovery channels per layer; default 32.")
    parser.add_argument("--all-channels", action="store_true",
                        help="Measure every channel in each selected layer; random controls remain a separate, preselected comparison group.")
    parser.add_argument("--random-controls", type=int, default=8,
                        help="Additional disjoint, randomly selected control channels per layer.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of highest-loss-impact discovery channels to compare with controls.")
    parser.add_argument("--max-validation-examples", type=int, default=0,
                        help="Optional quick pilot limit; 0 evaluates the entire 5,000-image validation split.")
    parser.add_argument("--layers", default=",".join(DEFAULT_LAYERS),
                        help="Comma-separated module paths; defaults to layer2.1.bn1,layer3.1.bn1,layer4.1.bn1.")
    args = parser.parse_args()
    if args.candidate_channels < 1 or args.random_controls < 1 or args.top_k < 1:
        parser.error("candidate-channels, random-controls, and top-k must be positive.")
    if args.max_validation_examples < 0:
        parser.error("max-validation-examples cannot be negative.")

    checkpoint_path = Path(args.checkpoint)
    split_path = Path(args.split_file)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train the v0.2 baseline first.")
    if not split_path.exists():
        raise FileNotFoundError(f"Saved split not found: {split_path}. Train the v0.2 baseline first.")
    if not (DATA_ROOT / "cifar-10-batches-py").exists():
        raise FileNotFoundError(f"CIFAR-10 data folder not found under {DATA_ROOT}.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Loading only the saved validation indices; official test set is not loaded...")

    with np.load(split_path) as saved_split:
        val_indices = saved_split["val_indices"].copy()
    base_train = datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=False, transform=None)
    validation_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    if args.max_validation_examples:
        val_indices = val_indices[:args.max_validation_examples]
    val_ds = IndexedCIFAR10(base_train, val_indices, validation_transform)
    loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    start_time = time.time()
    baseline = evaluate_baseline(model, loader, device)
    print(
        f"Baseline on {baseline['examples']} validation images: "
        f"accuracy={baseline['accuracy'] * 100:.2f}%, loss={baseline['loss']:.4f}"
    )

    layer_results: list[dict[str, Any]] = []
    interventions_completed = 0
    for layer_name in [part.strip() for part in args.layers.split(",") if part.strip()]:
        module = get_module_by_name(model, layer_name)
        if not isinstance(module, nn.BatchNorm2d):
            raise TypeError(
                f"{layer_name} is {type(module).__name__}, expected BatchNorm2d. "
                "Use a BN1 module path so each chosen activation tensor is intercepted exactly once per forward pass."
            )
        channel_count = int(module.num_features)
        if args.random_controls >= channel_count:
            raise ValueError(f"Layer {layer_name} has {channel_count} channels; random_controls must be smaller.")
        if (not args.all_channels) and (args.candidate_channels + args.random_controls > channel_count):
            raise ValueError(
                f"Layer {layer_name} has {channel_count} channels, but requested "
                f"{args.candidate_channels} candidates + {args.random_controls} controls."
            )
        all_channels = np.arange(channel_count, dtype=np.int64)
        control_channels = rng.choice(all_channels, size=args.random_controls, replace=False)
        remaining_channels = np.setdiff1d(all_channels, control_channels, assume_unique=False)
        rng.shuffle(remaining_channels)
        if args.all_channels:
            candidate_channels = remaining_channels
        else:
            candidate_channels = rng.choice(remaining_channels, size=args.candidate_channels, replace=False)

        print(f"\nLayer {layer_name}: {channel_count} channels; "
              f"testing {len(candidate_channels)} discovery + {len(control_channels)} random controls")
        measured: list[dict[str, Any]] = []
        for role, channel_list in (("candidate", candidate_channels), ("random_control", control_channels)):
            for local_index, channel in enumerate(channel_list.tolist(), start=1):
                result = evaluate_with_channel_ablation(
                    model=model,
                    loader=loader,
                    device=device,
                    module=module,
                    channel_index=int(channel),
                    baseline_predictions=baseline["predictions"],
                    baseline_batches=baseline["per_batch"],
                )
                interventions_completed += 1
                row = {
                    "layer": layer_name,
                    "channel_index": int(channel),
                    "role": role,
                    "loss": float(result["loss"]),
                    "accuracy": float(result["accuracy"]),
                    "delta_loss": float(result["loss"] - baseline["loss"]),
                    "delta_accuracy_pp": float((result["accuracy"] - baseline["accuracy"]) * 100.0),
                    "prediction_flip_rate": float(result["prediction_flip_rate"]),
                    "hook_calls": int(result["hook_calls"]),
                    "intervened_batches": int(result["intervened_batches"]),
                    "fraction_batches_loss_increased": float(result["fraction_batches_loss_increased"]),
                    "fraction_batches_accuracy_decreased": float(result["fraction_batches_accuracy_decreased"]),
                    "batch_delta_loss_mean": float(result["batch_delta_loss_mean"]),
                    "batch_delta_loss_std": float(result["batch_delta_loss_std"]),
                    "batch_metrics": result["batch_metrics"],
                }
                measured.append(row)
                print(
                    f"  {role:13s} {local_index:02d}/{len(channel_list):02d} channel={channel:03d} | "
                    f"Δloss={row['delta_loss']:+.4f}, Δacc={row['delta_accuracy_pp']:+.2f} pp, "
                    f"flips={row['prediction_flip_rate']*100:.2f}%"
                )

        candidates = [r for r in measured if r["role"] == "candidate"]
        controls = [r for r in measured if r["role"] == "random_control"]
        ranked_by_loss = sorted(candidates, key=lambda r: r["delta_loss"], reverse=True)
        ranked_by_accuracy_damage = sorted(candidates, key=lambda r: r["delta_accuracy_pp"])
        for rank, row in enumerate(ranked_by_loss, start=1):
            row["rank_by_loss_impact"] = rank
        for rank, row in enumerate(ranked_by_accuracy_damage, start=1):
            row["rank_by_accuracy_damage"] = rank

        top_count = min(args.top_k, len(ranked_by_loss))
        top_loss = ranked_by_loss[:top_count]
        def group_stats(group: list[dict[str, Any]]) -> dict[str, Any]:
            output: dict[str, Any] = {"n_channels": len(group)}
            for metric in ("delta_loss", "delta_accuracy_pp", "prediction_flip_rate",
                           "fraction_batches_loss_increased", "fraction_batches_accuracy_decreased"):
                values = [float(row[metric]) for row in group]
                output[metric] = mean_std(values)
            return output

        top_stats = group_stats(top_loss)
        control_stats = group_stats(controls)
        layer_summary = {
            "layer": layer_name,
            "activation_module": "BatchNorm2d output (before ReLU); zeroing the selected channel makes its subsequent ReLU activation zero",
            "available_channels": channel_count,
            "candidate_channels_tested": len(candidates),
            "random_controls_tested": len(controls),
            "candidate_channel_indices": [int(x) for x in candidate_channels.tolist()],
            "random_control_channel_indices": [int(x) for x in control_channels.tolist()],
            "top_k_by_loss_impact": [
                {"channel_index": r["channel_index"], "delta_loss": r["delta_loss"],
                 "delta_accuracy_pp": r["delta_accuracy_pp"],
                 "prediction_flip_rate": r["prediction_flip_rate"],
                 "fraction_batches_loss_increased": r["fraction_batches_loss_increased"]}
                for r in top_loss
            ],
            "top_k_by_accuracy_damage": [
                {"channel_index": r["channel_index"], "delta_loss": r["delta_loss"],
                 "delta_accuracy_pp": r["delta_accuracy_pp"],
                 "prediction_flip_rate": r["prediction_flip_rate"]}
                for r in ranked_by_accuracy_damage[:top_count]
            ],
            "top_loss_impact_group_statistics": top_stats,
            "random_control_group_statistics": control_stats,
            "top_vs_random_control_mean_difference": {
                "delta_loss": top_stats["delta_loss"]["mean"] - control_stats["delta_loss"]["mean"],
                "delta_accuracy_pp": top_stats["delta_accuracy_pp"]["mean"] - control_stats["delta_accuracy_pp"]["mean"],
                "prediction_flip_rate": top_stats["prediction_flip_rate"]["mean"] - control_stats["prediction_flip_rate"]["mean"],
            },
            "measured_channels": sorted(measured, key=lambda r: (r["role"] != "candidate", -r.get("rank_by_loss_impact", 0))),
        }
        layer_results.append(layer_summary)
        print("  Top-by-loss channels:", ", ".join(
            f"{r['channel_index']} (Δloss {r['delta_loss']:+.3f})" for r in top_loss
        ))

    output = {
        "experiment": "Brahma MetaCore MetaAnalyzer v0.2 channel-level ablation",
        "method": "Zero one channel in selected BatchNorm2d outputs for every batch of a fixed validation subset.",
        "seed": args.seed,
        "checkpoint": str(checkpoint_path.relative_to(ROOT) if checkpoint_path.is_relative_to(ROOT) else checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_selection": "best validation accuracy (tie-break: lower validation loss)",
        "split_file": str(split_path.relative_to(ROOT) if split_path.is_relative_to(ROOT) else split_path),
        "validation_examples": int(baseline["examples"]),
        "validation_subset_note": (
            "Full saved 5,000-example validation split." if not args.max_validation_examples
            else f"Pilot run on the first {args.max_validation_examples} examples of the fixed validation split; not a full validation evaluation."
        ),
        "official_test_set_used": False,
        "baseline": {
            "loss": float(baseline["loss"]),
            "accuracy": float(baseline["accuracy"]),
            "examples": int(baseline["examples"]),
            "per_batch": baseline["per_batch"],
        },
        "configuration": {
            "candidate_channels_per_layer": "all non-control channels" if args.all_channels else args.candidate_channels,
            "all_channels_mode": args.all_channels,
            "random_controls_per_layer": args.random_controls,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "layers": [r["layer"] for r in layer_results],
            "validation_transform": "ToTensor + Normalize; no random augmentation",
            "intervention": "Set output[:, channel, :, :] to zero in the selected BN1 module, for every validation batch.",
            "control_sampling": "Random controls sampled uniformly per layer without replacement; candidate channels are sampled separately from the remaining channels.",
        },
        "interventions_completed": interventions_completed,
        "elapsed_seconds": time.time() - start_time,
        "interpretation_caution": (
            "Ablation measures sensitivity to this particular intervention, not intelligence or semantic meaning. "
            "Correlated and redundant channels can complicate interpretation. Channel selection and all comparisons are exploratory."
        ),
        "layers": layer_results,
    }
    output_path = REPORT_DIR / "meta_analyzer_v0_2_channel_ablation.json"
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nCompleted {interventions_completed} channel interventions in {output['elapsed_seconds']/60:.1f} minutes.")
    print(f"Saved structured report: {output_path}")


if __name__ == "__main__":
    main()
