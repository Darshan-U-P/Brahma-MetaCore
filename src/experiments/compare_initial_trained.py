
import json
from pathlib import Path

import torch
from torchvision import datasets, transforms

from src.metaobserver.model_inspector import (
    build_cifar10_resnet18,
    capture_activations,
)


def main():
    root = Path(__file__).resolve().parents[2]
    checkpoint_dir = root / "checkpoints" / "milestone2"
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    initial_checkpoint = torch.load(
        checkpoint_dir / "initial.pt",
        map_location="cpu",
        weights_only=True,
    )
    trained_checkpoint = torch.load(
        checkpoint_dir / "epoch_020.pt",
        map_location="cpu",
        weights_only=True,
    )

    initial_model = build_cifar10_resnet18()
    trained_model = build_cifar10_resnet18()

    initial_model.load_state_dict(
        initial_checkpoint["model_state_dict"]
    )
    trained_model.load_state_dict(
        trained_checkpoint["model_state_dict"]
    )

    initial_model = initial_model.to(device).eval()
    trained_model = trained_model.to(device).eval()

    # Use a real CIFAR-10 image and identical preprocessing for both models.
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

    image, label = test_set[0]
    inputs = image.unsqueeze(0).to(device)

    # Compare parameter changes, tensor by tensor.
    parameter_changes = {}
    changed_tensors = 0
    total_delta_sq = 0.0
    total_initial_sq = 0.0
    total_parameters = 0

    initial_state = initial_model.state_dict()
    trained_state = trained_model.state_dict()

    for name, initial_tensor in initial_state.items():
        trained_tensor = trained_state[name]
        if not initial_tensor.is_floating_point():
            continue

        delta = trained_tensor.float() - initial_tensor.float()
        delta_norm = delta.norm().item()
        initial_norm = initial_tensor.float().norm().item()

        if delta_norm > 0:
            changed_tensors += 1

        total_delta_sq += delta_norm ** 2
        total_initial_sq += initial_norm ** 2
        total_parameters += initial_tensor.numel()

        parameter_changes[name] = {
            "numel": initial_tensor.numel(),
            "initial_l2_norm": initial_norm,
            "delta_l2_norm": delta_norm,
            "mean_absolute_change": delta.abs().mean().item(),
        }

    relative_change = (
        total_delta_sq ** 0.5 / max(total_initial_sq ** 0.5, 1e-12)
    )

    initial_logits, initial_activations = capture_activations(
        initial_model, inputs
    )
    trained_logits, trained_activations = capture_activations(
        trained_model, inputs
    )

    activation_comparison = {}
    for name in initial_activations.keys() & trained_activations.keys():
        before = initial_activations[name]
        after = trained_activations[name]

        activation_comparison[name] = {
            "shape": before["shape"],
            "initial_mean": before["mean"],
            "trained_mean": after["mean"],
            "initial_std": before["std"],
            "trained_std": after["std"],
            "mean_change": after["mean"] - before["mean"],
            "std_change": after["std"] - before["std"],
        }

    report = {
        "experiment": "MetaCore-Milestone2-Initial-vs-Trained",
        "device": str(device),
        "trained_checkpoint": "epoch_020.pt",
        "sample": {
            "test_dataset_index": 0,
            "ground_truth_class_index": int(label),
            "initial_predicted_class": int(initial_logits.argmax(1).item()),
            "trained_predicted_class": int(trained_logits.argmax(1).item()),
            "initial_logits": initial_logits[0].cpu().tolist(),
            "trained_logits": trained_logits[0].cpu().tolist(),
        },
        "weight_comparison": {
            "floating_point_parameters_compared": total_parameters,
            "tensors_with_nonzero_change": changed_tensors,
            "relative_global_l2_change": relative_change,
            "per_tensor": parameter_changes,
        },
        "activation_comparison": activation_comparison,
        "interpretation_note": (
            "Activation summaries describe this input only. "
            "They do not establish causal importance."
        ),
    }

    output = report_dir / "milestone2_initial_vs_trained.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Sample ground-truth class: {label}")
    print(
        "Initial prediction:",
        report["sample"]["initial_predicted_class"],
    )
    print(
        "Trained prediction:",
        report["sample"]["trained_predicted_class"],
    )
    print(f"Changed tensors: {changed_tensors}")
    print(f"Relative global L2 weight change: {relative_change:.6f}")
    print(f"Compared activation modules: {len(activation_comparison)}")
    print(f"Report saved to: {output}")


if __name__ == "__main__":
    main()
