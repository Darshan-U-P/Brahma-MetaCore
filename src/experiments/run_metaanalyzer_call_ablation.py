
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


def evaluate(model, loader, device, target=None, call_to_zero=None):
    model.eval()
    logits_all, labels_all = [], []
    calls = 0
    handle = None

    if target is not None:
        module = dict(model.named_modules())[target]

        def hook(module, inputs, output):
            nonlocal calls
            calls += 1

            if calls == call_to_zero:
                return torch.zeros_like(output)

            return output

        handle = module.register_forward_hook(hook)

    try:
        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device)
                logits = model(images)
                logits_all.append(logits.float().cpu())
                labels_all.append(labels.cpu())
    finally:
        if handle is not None:
            handle.remove()

    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    predictions = logits.argmax(dim=1)

    return {
        "accuracy": (predictions == labels).float().mean().item(),
        "loss": F.cross_entropy(logits, labels).item(),
        "logits": logits,
        "predictions": predictions,
        "labels": labels,
        "hook_calls": calls,
    }


def main():
    root = Path(__file__).resolve().parents[2]
    checkpoint_path = (
        root / "checkpoints" / "milestone2" / "epoch_020.pt"
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    dataset = datasets.CIFAR10(
        root=str(root / "data" / "cifar10"),
        train=False,
        download=False,
        transform=transform,
    )
    loader = DataLoader(
        Subset(dataset, range(min(NUM_IMAGES, len(dataset)))),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = build_cifar10_resnet18()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    baseline = evaluate(model, loader, device)
    print(
        f"Baseline: accuracy={baseline['accuracy']:.2%}, "
        f"loss={baseline['loss']:.4f}"
    )

    for layer in TARGET_LAYERS:
        for call_index in (1, 2):
            result = evaluate(
                model, loader, device,
                target=layer,
                call_to_zero=call_index,
            )

            flip_rate = (
                result["predictions"] != baseline["predictions"]
            ).float().mean().item()

            print(
                f"{layer} | zero call {call_index} | "
                f"hook calls={result['hook_calls']} | "
                f"accuracy={result['accuracy']:.2%} | "
                f"delta={(result['accuracy'] - baseline['accuracy']) * 100:+.2f} pp | "
                f"loss={result['loss']:.4f} | "
                f"flip rate={flip_rate:.2%}"
            )

    repeated = evaluate(model, loader, device)
    print(
        f"Repeated baseline: accuracy={repeated['accuracy']:.2%}, "
        f"loss={repeated['loss']:.4f}"
    )


if __name__ == "__main__":
    main()
