
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.metaobserver.model_inspector import build_cifar10_resnet18


TARGET_LAYERS = [
    "layer2.1.relu",
    "layer3.1.relu",
    "layer4.1.relu",
]

NUM_IMAGES = 1000
BATCH_SIZE = 128


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    ablate_layer: str | None = None,
) -> dict:
    """Evaluate the same images, optionally zeroing one module's outputs."""
    model.eval()
    all_logits = []
    all_labels = []
    hook_calls = 0
    handle = None

    if ablate_layer is not None:
        modules = dict(model.named_modules())
        if ablate_layer not in modules:
            raise ValueError(f"Layer not found: {ablate_layer}")

        target_module = modules[ablate_layer]

        def zero_output_hook(module, inputs, output):
            nonlocal hook_calls
            hook_calls += 1

            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"{ablate_layer} returned a non-tensor output"
                )

            return torch.zeros_like(output)

        handle = target_module.register_forward_hook(zero_output_hook)

    try:
        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(images)
                all_logits.append(logits.float().cpu())
                all_labels.append(labels.cpu())
    finally:
        if handle is not None:
            handle.remove()

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    losses = F.cross_entropy(logits, labels, reduction="none")
    predictions = logits.argmax(dim=1)
    correct_class_logits = logits.gather(
        1, labels.unsqueeze(1)
    ).squeeze(1)

    return {
        "logits": logits,
        "labels": labels,
        "predictions": predictions,
        "accuracy": (predictions == labels).float().mean().item(),
        "mean_cross_entropy_loss": losses.mean().item(),
        "mean_correct_class_logit": correct_class_logits.mean().item(),
        "hook_calls": hook_calls,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    checkpoint_path = (
        root / "checkpoints" / "milestone2" / "epoch_020.pt"
    )
    report_path = (
        root / "reports" / "metaanalyzer_v0_1_ablation.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Match the evaluation preprocessing used during training.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    test_set = datasets.CIFAR10(
        root=str(root / "data" / "cifar10"),
        train=False,
        download=False,
        transform=transform,
    )

    # Deterministic, identical subset for every condition.
    num_images = min(NUM_IMAGES, len(test_set))
    subset = Subset(test_set, range(num_images))

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_cifar10_resnet18()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Fixed test images: {num_images}")
    print("Running baseline...")

    baseline = evaluate(model, loader, device)
    baseline_logits = baseline["logits"]
    baseline_predictions = baseline["predictions"]
    labels = baseline["labels"]

    results = {}

    print(
        f"Baseline accuracy: {baseline['accuracy']:.2%} | "
        f"loss: {baseline['mean_cross_entropy_loss']:.4f}"
    )

    for layer_name in TARGET_LAYERS:
        print(f"Testing zero ablation: {layer_name}")

        ablated = evaluate(
            model,
            loader,
            device,
            ablate_layer=layer_name,
        )

        # Positive means the intervention increased the true-class logit.
        baseline_true_logits = baseline_logits.gather(
            1, labels.unsqueeze(1)
        ).squeeze(1)
        ablated_true_logits = ablated["logits"].gather(
            1, labels.unsqueeze(1)
        ).squeeze(1)

        results[layer_name] = {
            "intervention": "replace every output of this ReLU module with zeros",
            "hook_calls": ablated["hook_calls"],
            "accuracy": ablated["accuracy"],
            "accuracy_change_percentage_points": (
                ablated["accuracy"] - baseline["accuracy"]
            ) * 100,
            "mean_cross_entropy_loss": ablated[
                "mean_cross_entropy_loss"
            ],
            "loss_change": (
                ablated["mean_cross_entropy_loss"]
                - baseline["mean_cross_entropy_loss"]
            ),
            "mean_correct_class_logit": ablated[
                "mean_correct_class_logit"
            ],
            "mean_correct_class_logit_change": (
                ablated_true_logits - baseline_true_logits
            ).mean().item(),
            "prediction_flip_rate": (
                ablated["predictions"] != baseline_predictions
            ).float().mean().item(),
        }

    report = {
        "experiment": "MetaAnalyzer-v0.1",
        "purpose": "Exploratory activation ablation",
        "checkpoint": "epoch_020.pt",
        "checkpoint_selection_note": (
            "This checkpoint comes from a run that evaluated the test "
            "set each epoch; results are exploratory, not an unbiased "
            "final benchmark."
        ),
        "device": str(device),
        "num_images": num_images,
        "baseline": {
            "accuracy": baseline["accuracy"],
            "mean_cross_entropy_loss": baseline[
                "mean_cross_entropy_loss"
            ],
            "mean_correct_class_logit": baseline[
                "mean_correct_class_logit"
            ],
        },
        "ablations": results,
        "interpretation_note": (
            "Zero ablation measures sensitivity to this intervention. "
            "It does not establish semantic meaning or causal importance "
            "across all inputs. Each target ReLU module is reused twice "
            "in its BasicBlock, so both invocations are ablated."
        ),
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nABLATION RESULTS")
    print("-" * 88)

    for layer_name, result in results.items():
        print(
            f"{layer_name:20s} "
            f"accuracy={result['accuracy']:.2%} "
            f"delta={result['accuracy_change_percentage_points']:+.2f} pp "
            f"loss={result['mean_cross_entropy_loss']:.4f} "
            f"flip_rate={result['prediction_flip_rate']:.2%}"
        )

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
