from pathlib import Path
import json
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from src.experiments.run_channel_ablation_v02 import build_model


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "cifar10"
CHECKPOINT = ROOT / "checkpoints" / "milestone2_v02" / "best_validation.pt"
OUTPUT = ROOT / "reports" / "milestone2_v03" / "frozen_channel_confirmation.json"

BATCH_SIZE = 256

FROZEN_CHANNELS = {
    "layer2.1.bn1": {
        "candidate": [6, 81, 57, 5, 98],
        "random_control": [103, 21, 47, 10, 79, 45, 3, 58],
    },
    "layer3.1.bn1": {
        "candidate": [115, 130, 145, 192, 2],
        "random_control": [165, 159, 253, 138, 233, 17, 199, 19],
    },
    "layer4.1.bn1": {
        "candidate": [40, 245, 128, 67, 189],
        "random_control": [169, 474, 70, 298, 477, 55, 5, 311],
    },
}


def evaluate(model, loader, device, baseline_predictions=None):
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_flips = 0
    offset = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            losses = F.cross_entropy(logits, labels, reduction="sum")
            predictions = logits.argmax(dim=1)

            total_loss += losses.item()
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.numel()

            if baseline_predictions is not None:
                reference = baseline_predictions[offset:offset + labels.numel()]
                total_flips += (predictions.cpu() != reference).sum().item()

            offset += labels.numel()

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "correct": total_correct,
        "examples": total_examples,
        "prediction_flip_rate": (
            total_flips / total_examples
            if baseline_predictions is not None else None
        ),
    }


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not (DATA_ROOT / "cifar-10-batches-py" / "test_batch").exists():
        raise FileNotFoundError("CIFAR-10 test_batch was not found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Standard CIFAR-10 evaluation normalization; no random augmentation.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    test_dataset = datasets.CIFAR10(
        root=str(DATA_ROOT),
        train=False,
        download=False,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    started = time.perf_counter()
    baseline = evaluate(model, loader, device)

    # Cache baseline predictions in the same deterministic loader order.
    baseline_predictions = []
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            baseline_predictions.append(model(images).argmax(dim=1).cpu())
    baseline_predictions = torch.cat(baseline_predictions)

    print(
        f"Baseline: accuracy={baseline['accuracy'] * 100:.2f}% | "
        f"loss={baseline['loss']:.6f} | "
        f"examples={baseline['examples']}"
    )

    results = []
    expected_interventions = sum(
        len(role_channels)
        for layer in FROZEN_CHANNELS.values()
        for role_channels in layer.values()
    )
    completed = 0

    for layer_name, roles in FROZEN_CHANNELS.items():
        module = model.get_submodule(layer_name)
        channel_count = module.num_features

        for role, channels in roles.items():
            for channel in channels:
                if not 0 <= channel < channel_count:
                    raise ValueError(
                        f"Invalid channel {channel} for {layer_name} "
                        f"with {channel_count} channels"
                    )

                def ablation_hook(_module, _inputs, output, ch=channel):
                    modified = output.clone()
                    modified[:, ch, :, :] = 0
                    return modified

                handle = module.register_forward_hook(ablation_hook)
                try:
                    measured = evaluate(
                        model, loader, device, baseline_predictions
                    )
                finally:
                    handle.remove()

                row = {
                    "layer": layer_name,
                    "role": role,
                    "channel": channel,
                    "loss": measured["loss"],
                    "delta_loss": measured["loss"] - baseline["loss"],
                    "accuracy": measured["accuracy"],
                    "delta_accuracy_pp": (
                        measured["accuracy"] - baseline["accuracy"]
                    ) * 100,
                    "prediction_flip_rate": measured["prediction_flip_rate"],
                    "examples": measured["examples"],
                }
                results.append(row)
                completed += 1

                print(
                    f"[{completed:02d}/{expected_interventions}] "
                    f"{layer_name} | {role:13s} | ch={channel:3d} | "
                    f"dLoss={row['delta_loss']:+.6f} | "
                    f"dAcc={row['delta_accuracy_pp']:+.3f} pp | "
                    f"flips={row['prediction_flip_rate'] * 100:.2f}%"
                )

    summaries = {}
    for layer_name in FROZEN_CHANNELS:
        summaries[layer_name] = {}
        for role in ("candidate", "random_control"):
            group = [
                r for r in results
                if r["layer"] == layer_name and r["role"] == role
            ]
            summaries[layer_name][role] = {
                "n": len(group),
                "mean_delta_loss": sum(
                    r["delta_loss"] for r in group
                ) / len(group),
                "mean_delta_accuracy_pp": sum(
                    r["delta_accuracy_pp"] for r in group
                ) / len(group),
                "mean_flip_rate": sum(
                    r["prediction_flip_rate"] for r in group
                ) / len(group),
            }

    report = {
        "experiment": "Brahma MetaCore MetaAnalyzer v0.3 frozen-channel confirmation",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset": "CIFAR-10 official test split",
        "test_examples": len(test_dataset),
        "official_test_set_used": True,
        "test_set_history_caution": (
            "The official test set was inspected during an earlier training "
            "experiment; this is not a completely untouched benchmark."
        ),
        "batch_size": BATCH_SIZE,
        "device": str(device),
        "baseline": baseline,
        "frozen_channels": FROZEN_CHANNELS,
        "interventions_completed": completed,
        "results": results,
        "group_summaries": summaries,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_caution": (
            "Ablation measures sensitivity to this intervention, not semantic "
            "meaning. Candidate channels were selected on validation results; "
            "compare against controls cautiously."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"\nCompleted {completed} interventions.")
    print(f"Report saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
