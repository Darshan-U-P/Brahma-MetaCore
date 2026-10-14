
import json

import torch
import torch.nn as nn

from src.metaobserver.model_inspector import (
    build_cifar10_resnet18,
    capture_activations,
    inspect_model,
)


def test_model_has_expected_parameter_count():
    model = build_cifar10_resnet18()

    expected_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    report = inspect_model(model)

    assert expected_count == 11_173_962
    assert report["num_parameters"] == expected_count
    assert report["num_trainable_parameters"] == expected_count


def test_model_outputs_ten_logits():
    model = build_cifar10_resnet18()
    model.eval()

    sample = torch.randn(1, 3, 32, 32)

    with torch.inference_mode():
        logits = model(sample)

    assert logits.shape == (1, 10)
    assert torch.isfinite(logits).all()


def test_architecture_report_contains_tensor_metadata():
    model = build_cifar10_resnet18()
    report = inspect_model(model)

    assert report["model_name"] == "ResNet-18-CIFAR10"
    assert report["initialization"] == "random"
    assert report["layers"]
    assert report["tensors"]

    tensor_names = {tensor["name"] for tensor in report["tensors"]}
    assert "conv1.weight" in tensor_names
    assert "fc.weight" in tensor_names

    for tensor in report["tensors"]:
        assert isinstance(tensor["shape"], list)
        assert tensor["numel"] > 0
        assert "dtype" in tensor


def test_activation_capture_records_outputs_and_removes_hooks():
    model = build_cifar10_resnet18()
    sample = torch.randn(1, 3, 32, 32)

    original_hook_counts = {
        name: len(module._forward_hooks)
        for name, module in model.named_modules()
    }

    logits, activations = capture_activations(model, sample)

    assert logits.shape == (1, 10)
    assert activations

    assert "conv1" in activations
    assert activations["conv1"]["shape"] == [1, 64, 32, 32]

    for name, activation in activations.items():
        assert activation["shape"]
        assert all(dimension > 0 for dimension in activation["shape"])
        assert torch.isfinite(torch.tensor(activation["mean"]))
        assert torch.isfinite(torch.tensor(activation["std"]))

    # Hooks registered by capture_activations must be removed.
    for name, module in model.named_modules():
        assert len(module._forward_hooks) == original_hook_counts[name]


def test_report_can_be_serialized_to_json(tmp_path):
    model = build_cifar10_resnet18()
    sample = torch.randn(1, 3, 32, 32)

    architecture = inspect_model(model)
    logits, activations = capture_activations(model, sample)

    report = {
        "architecture": architecture,
        "inference": {
            "logits_shape": list(logits.shape),
            "activation_count": len(activations),
            "activations": activations,
        },
    }

    report_path = tmp_path / "test_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    loaded = json.loads(report_path.read_text(encoding="utf-8"))

    assert loaded["architecture"]["num_parameters"] == 11_173_962
    assert loaded["inference"]["logits_shape"] == [1, 10]
    assert loaded["inference"]["activation_count"] > 0


def test_activation_capture_does_not_change_model_parameters():
    model = build_cifar10_resnet18()
    sample = torch.randn(1, 3, 32, 32)

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    capture_activations(model, sample)

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_model_contains_expected_output_layer():
    model = build_cifar10_resnet18()

    assert isinstance(model.fc, nn.Linear)
    assert model.fc.in_features == 512
    assert model.fc.out_features == 10
