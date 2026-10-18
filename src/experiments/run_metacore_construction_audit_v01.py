from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import torch
import torchvision
from torchvision.models import resnet18

ROOT = Path(r"D:\all project\Projects\AI projects\Brahma-MetaCore")
SOURCE = ROOT / "checkpoints/milestone2_v02/best_validation.pt"
OUT = ROOT / "reports/metacore_construction_audit_v01"
SEED = 9001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARMS = {
    "random_init": None,
    "full_source": "FULL",
    "stage_layer1": "layer1.",
    "stage_layer2": "layer2.",
    "stage_layer3": "layer3.",
    "stage_layer4": "layer4.",
}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def build_model():
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = torch.nn.Identity()
    return model

def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")

    source_hash_before = sha256(SOURCE)
    ckpt = torch.load(SOURCE, map_location="cpu", weights_only=True)
    source = ckpt["model_state_dict"]

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    base_model = build_model()
    base = {k: v.detach().cpu().clone()
            for k, v in base_model.state_dict().items()}

    if set(base) != set(source):
        raise RuntimeError(
            f"Key mismatch: model-only={sorted(set(base)-set(source))}; "
            f"source-only={sorted(set(source)-set(base))}"
        )

    shape_errors = [
        (k, tuple(base[k].shape), tuple(source[k].shape))
        for k in base if base[k].shape != source[k].shape
    ]
    if shape_errors:
        raise RuntimeError(f"Shape mismatch: {shape_errors[:10]}")

    parameter_names = {k for k, _ in base_model.named_parameters()}
    total_params = sum(v.numel() for _, v in base_model.named_parameters())
    total_state_elements = sum(v.numel() for v in source.values())
    results = []

    for arm, selector in ARMS.items():
        state = {k: v.clone() for k, v in base.items()}
        copied = []

        for key in state:
            should_copy = (
                selector == "FULL" or
                (selector is not None and key.startswith(selector))
            )
            if should_copy:
                state[key].copy_(source[key])
                copied.append(key)

        model = build_model()
        model.load_state_dict(state, strict=True)

        loaded = model.state_dict()
        copied_errors = [
            k for k in copied
            if not torch.equal(loaded[k].cpu(), source[k])
        ]
        untouched_errors = [
            k for k in state if k not in copied
            and not torch.equal(loaded[k].cpu(), base[k])
        ]
        if copied_errors or untouched_errors:
            raise RuntimeError(
                f"{arm}: copied mismatches={copied_errors[:5]}, "
                f"untouched mismatches={untouched_errors[:5]}"
            )

        model = model.to(DEVICE).eval()
        generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
        x = torch.randn(2, 3, 32, 32, generator=generator).to(DEVICE)

        with torch.inference_mode():
            logits = model(x)

        if logits.shape != (2, 10) or not torch.isfinite(logits).all():
            raise RuntimeError(f"{arm}: invalid output {tuple(logits.shape)}")

        copied_elements = sum(source[k].numel() for k in copied)
        copied_params = sum(
            source[k].numel() for k in copied if k in parameter_names
        )

        results.append({
            "arm": arm,
            "status": "passed",
            "copied_tensor_count": len(copied),
            "copied_tensor_elements_including_buffers": copied_elements,
            "copied_trainable_parameter_elements": copied_params,
            "fraction_trainable_parameters_copied": copied_params / total_params,
            "fraction_state_tensor_elements_copied":
                copied_elements / total_state_elements,
            "copied_keys_match_source": True,
            "uncopied_keys_match_shared_random_initialization": True,
            "forward_shape": list(logits.shape),
            "forward_finite": True,
        })

        print(
            f"PASS {arm}: {len(copied)} tensors; "
            f"{copied_params:,} parameter elements copied "
            f"({100 * copied_params / total_params:.2f}%)"
        )

        del model, x, logits
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    source_hash_after = sha256(SOURCE)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Source checkpoint hash changed!")

    report = {
        "title": "MetaCore Construction Initialization Audit v0.1",
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(DEVICE),
        "source_checkpoint": str(SOURCE),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_unchanged": True,
        "source_seed": ckpt.get("seed"),
        "source_epoch": ckpt.get("epoch"),
        "architecture": ckpt.get("architecture"),
        "model_config": ckpt.get("model_config"),
        "target_random_seed": SEED,
        "state_dict_tensor_count": len(source),
        "total_state_tensor_elements_including_buffers": total_state_elements,
        "total_trainable_parameter_elements": total_params,
        "training_performed": False,
        "dataset_loaded": False,
        "arms": results,
        "limitations": [
            "This validates initialization mechanics, not training efficacy.",
            "Stage arms transfer different parameter counts.",
            "All uncopied tensors share the same random initialization.",
            "Only model weights and buffers are copied; optimizer state is not loaded.",
            "Source model and target experiment use the same task and training split."
        ],
    }

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "initialization_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        "# MetaCore Construction Initialization Audit v0.1",
        "",
        "**Status:** PASSED",
        "",
        f"- Device: `{DEVICE}`",
        f"- PyTorch: `{torch.__version__}`",
        f"- torchvision: `{torchvision.__version__}`",
        f"- Source SHA-256: `{source_hash_before}`",
        f"- Source unchanged: `{source_hash_before == source_hash_after}`",
        f"- State-dict tensors: {len(source)}",
        f"- Trainable parameter elements: {total_params:,}",
        "- Training performed: No",
        "- Dataset loaded: No",
        "",
        "| Arm | Tensors copied | Tensor elements copied | Parameter elements copied | % parameters copied |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| `{r['arm']}` | {r['copied_tensor_count']} | "
            f"{r['copied_tensor_elements_including_buffers']:,} | "
            f"{r['copied_trainable_parameter_elements']:,} | "
            f"{100*r['fraction_trainable_parameters_copied']:.2f}% |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "All arms passed exact tensor-copy checks and finite forward inference.",
        "This is an initialization-integrity audit, not evidence of transfer efficacy.",
        "Stage arms differ in size and are not equal-budget controls.",
        "The source checkpoint was trained on the same CIFAR-10 task and training split.",
        "",
    ]
    (OUT / "initialization_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("\nAUDIT PASSED")
    print("Report directory:", OUT)
    print("Source checkpoint unchanged:", source_hash_before == source_hash_after)

if __name__ == "__main__":
    main()
