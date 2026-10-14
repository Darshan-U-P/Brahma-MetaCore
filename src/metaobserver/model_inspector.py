
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision.models import resnet18


def build_cifar10_resnet18() -> nn.Module:
    """Build a ResNet-18 adapted for 32x32 CIFAR-10 images."""
    model = resnet18(weights=None)

    # CIFAR-10 images are 32x32, so use a smaller initial convolution.
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    # Remove ImageNet's initial downsampling operation.
    model.maxpool = nn.Identity()

    # CIFAR-10 has 10 classes instead of ImageNet's 1000.
    model.fc = nn.Linear(model.fc.in_features, 10)

    return model


def inspect_model(model: nn.Module) -> dict[str, Any]:
    """Collect architecture and parameter metadata."""
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    layers = []
    for name, module in model.named_modules():
        if not name:
            continue

        direct_parameters = sum(
            p.numel() for p in module.parameters(recurse=False)
        )

        layers.append(
            {
                "name": name,
                "module_type": type(module).__name__,
                "direct_parameter_count": direct_parameters,
            }
        )

    tensors = []
    for name, tensor in model.state_dict().items():
        tensors.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
            }
        )

    return {
        "model_name": "ResNet-18-CIFAR10",
        "initialization": "random",
        "num_parameters": parameter_count,
        "num_trainable_parameters": trainable_count,
        "num_layers": len(layers),
        "layers": layers,
        "tensors": tensors,
    }


def capture_activations(
    model: nn.Module,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, dict[str, Any]]]:
    """Capture output shapes and summary statistics for leaf modules."""
    activations: dict[str, dict[str, Any]] = {}
    handles = []

    def make_hook(name: str):
        def hook(
            module: nn.Module,
            module_inputs: tuple[torch.Tensor, ...],
            output: Any,
        ) -> None:
            if not isinstance(output, torch.Tensor):
                return

            values = output.detach().float()

            activations[name] = {
                "module_type": type(module).__name__,
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "mean": values.mean().item(),
                "std": values.std(unbiased=False).item(),
                "min": values.min().item(),
                "max": values.max().item(),
            }

        return hook

    for name, module in model.named_modules():
        if name and not any(module.children()):
            handles.append(
                module.register_forward_hook(make_hook(name))
            )

    try:
        model.eval()
        with torch.inference_mode():
            logits = model(inputs)
    finally:
        for handle in handles:
            handle.remove()

    return logits, activations
