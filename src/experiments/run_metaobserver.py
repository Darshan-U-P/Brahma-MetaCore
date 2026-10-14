
import json
from pathlib import Path

import torch

from src.metaobserver.model_inspector import (
    build_cifar10_resnet18,
    capture_activations,
    inspect_model,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_cifar10_resnet18().to(device)

    # Synthetic input: no dataset download is needed for this test.
    sample = torch.randn(1, 3, 32, 32, device=device)

    architecture = inspect_model(model)
    logits, activations = capture_activations(model, sample)

    report = {
        "experiment": "MetaObserver-v0.1",
        "device": str(device),
        "architecture": architecture,
        "inference": {
            "input_shape": list(sample.shape),
            "logits_shape": list(logits.shape),
            "logits": logits[0].detach().cpu().tolist(),
            "predicted_class_index": int(logits.argmax(dim=1).item()),
            "activation_count": len(activations),
            "activations": activations,
        },
    }

    report_path = reports_dir / "metaobserver_v0_1.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Device: {device}")
    print(f"Parameters: {architecture['num_parameters']:,}")
    print(f"Layers: {architecture['num_layers']}")
    print(f"Captured activations: {len(activations)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
