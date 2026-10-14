
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "cifar10"
REPORT_V02 = (
    ROOT / "reports" / "milestone2_v02"
    / "meta_analyzer_v0_2_channel_ablation_full.json"
)
CHECKPOINT = ROOT / "checkpoints" / "milestone2_v02" / "best_validation.pt"
SPLIT_FILE = ROOT / "checkpoints" / "milestone2_v02" / "split_indices_v02.npz"
OUTPUT = ROOT / "reports" / "milestone2_v04" / "predictive_channel_importance.json"

LAYERS = ("layer2.1.bn1", "layer3.1.bn1", "layer4.1.bn1")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
SEED = 2026


class IndexedCIFAR10(Dataset):
    def __init__(self, base, indices, transform):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = int(self.indices[index])
        image = Image.fromarray(self.base.data[source_index])
        label = int(self.base.targets[source_index])
        return self.transform(image), label


def load_model(device):
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        state = None
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
        if state is None and all(torch.is_tensor(v) for v in checkpoint.values()):
            state = checkpoint
        if state is None:
            raise ValueError(
                f"Cannot locate model state in checkpoint. Keys: {list(checkpoint)}"
            )
    else:
        raise TypeError("Unexpected checkpoint format.")

    state = {
        (key.removeprefix("module.")): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def layer_modules(model, layer_name):
    parts = layer_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)

    bn = getattr(parent, parts[-1])
    conv_in = parent.conv1
    conv_out = parent.conv2

    if not isinstance(bn, nn.BatchNorm2d):
        raise TypeError(f"{layer_name} is not BatchNorm2d")

    return parent, bn, conv_in, conv_out


@torch.inference_mode()
def extract_features(model, loader, device):
    modules = {}
    stats = {}

    for layer in LAYERS:
        parent, bn, conv_in, conv_out = layer_modules(model, layer)
        modules[layer] = (bn, conv_in, conv_out)

        channels = bn.num_features
        stats[layer] = {
            "count": 0,
            "sum": torch.zeros(channels, dtype=torch.float64),
            "sum_sq": torch.zeros(channels, dtype=torch.float64),
            "sum_abs": torch.zeros(channels, dtype=torch.float64),
            "near_zero": torch.zeros(channels, dtype=torch.float64),
        }

    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output) or output.ndim != 4:
                raise RuntimeError(f"Unexpected activation shape in {layer}")

            x = output.detach().to(torch.float32)
            item = stats[layer]
            item["count"] += x.shape[0] * x.shape[2] * x.shape[3]
            item["sum"] += x.sum((0, 2, 3), dtype=torch.float64).cpu()
            item["sum_sq"] += (
                x.square().sum((0, 2, 3), dtype=torch.float64).cpu()
            )
            item["sum_abs"] += x.abs().sum((0, 2, 3), dtype=torch.float64).cpu()
            item["near_zero"] += (
                (x.abs() < 1e-6).sum((0, 2, 3)).to(torch.float64).cpu()
            )
        return hook

    for layer, (bn, _, _) in modules.items():
        handles.append(bn.register_forward_hook(make_hook(layer)))

    try:
        for images, _labels in loader:
            model(images.to(device, non_blocking=True))
    finally:
        for handle in handles:
            handle.remove()

    feature_rows = {}

    for layer, (bn, conv_in, conv_out) in modules.items():
        item = stats[layer]
        count = item["count"]
        if count == 0:
            raise RuntimeError(f"No activations captured for {layer}")

        mean = item["sum"] / count
        variance = (item["sum_sq"] / count - mean.square()).clamp_min(0)
        activation_std = variance.sqrt()
        mean_abs = item["sum_abs"] / count
        zero_fraction = item["near_zero"] / count

        # Conv1 output channels align with BN channels.
        incoming_norm = conv_in.weight.detach().float().flatten(1).norm(dim=1)

        # Conv2 input channels align with BN channels.
        outgoing_norm = conv_out.weight.detach().float().permute(
            1, 0, 2, 3
        ).flatten(1).norm(dim=1)

        gamma = bn.weight.detach().float()
        beta = bn.bias.detach().float()

        feature_rows[layer] = []
        for ch in range(bn.num_features):
            feature_rows[layer].append({
                "layer": layer,
                "channel": ch,
                "activation_mean": float(mean[ch]),
                "activation_std": float(activation_std[ch]),
                "activation_mean_abs": float(mean_abs[ch]),
                "activation_near_zero_fraction": float(zero_fraction[ch]),
                "bn_gamma_abs": float(gamma[ch].abs()),
                "bn_beta_abs": float(beta[ch].abs()),
                "incoming_weight_norm": float(incoming_norm[ch]),
                "outgoing_weight_norm": float(outgoing_norm[ch]),
            })

    return feature_rows


def safe_spearman(y, score):
    result = spearmanr(y, score)
    value = float(result.statistic)
    return value if np.isfinite(value) else None


def ranking_metrics(y, score, k):
    order = np.argsort(-score)
    actual = np.argsort(-y)
    top_pred = set(order[:k].tolist())
    top_actual = set(actual[:k].tolist())

    return {
        "spearman": safe_spearman(y, score),
        "top_k": int(k),
        "top_k_overlap": len(top_pred & top_actual),
        "top_k_precision": len(top_pred & top_actual) / max(k, 1),
    }


def main():
    start = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    discovery = json.loads(REPORT_V02.read_text(encoding="utf-8"))
    if discovery["interventions_completed"] != 896:
        raise RuntimeError("Expected the completed 896-intervention report.")
    if discovery["validation_examples"] != 5000:
        raise RuntimeError("Unexpected discovery validation size.")

    with np.load(SPLIT_FILE) as split:
        val_indices = split["val_indices"].copy()

    base = datasets.CIFAR10(
        root=str(DATA_ROOT), train=True, download=False, transform=None
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = IndexedCIFAR10(base, val_indices, transform)
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = load_model(device)
    features = extract_features(model, loader, device)

    measurements = {}
    for layer_data in discovery["layers"]:
        layer = layer_data["layer"]
        for row in layer_data["measured_channels"]:
            key = (layer, int(row["channel_index"]))
            measurements[key] = row

    rows = []
    for layer in LAYERS:
        for feature in features[layer]:
            key = (layer, feature["channel"])
            if key not in measurements:
                raise KeyError(f"Missing ablation label for {key}")

            measured = measurements[key]
            # Absolute loss change represents intervention sensitivity.
            target = abs(float(measured["delta_loss"]))

            rows.append({
                **feature,
                "target_abs_delta_loss": target,
                "signed_delta_loss": float(measured["delta_loss"]),
            })

    feature_names = [
        "activation_mean",
        "activation_std",
        "activation_mean_abs",
        "activation_near_zero_fraction",
        "bn_gamma_abs",
        "bn_beta_abs",
        "incoming_weight_norm",
        "outgoing_weight_norm",
    ]

    # Hold out channels separately inside each layer, so every layer
    # contributes to both fitting and evaluation.
    rng = np.random.default_rng(SEED)
    train_indices, test_indices = [], []
    for layer in LAYERS:
        indices = np.array([
            i for i, row in enumerate(rows) if row["layer"] == layer
        ])
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * 0.30)))
        test_indices.extend(indices[:n_test].tolist())
        train_indices.extend(indices[n_test:].tolist())

    def matrix(indices):
        return np.asarray([
            [rows[i][name] for name in feature_names]
            for i in indices
        ], dtype=np.float64)

    y_all = np.asarray([r["target_abs_delta_loss"] for r in rows])
    X_train = matrix(train_indices)
    X_test = matrix(test_indices)
    y_train = y_all[train_indices]
    y_test = y_all[test_indices]

    predictor = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=SEED,
        n_jobs=-1,
    )
    predictor.fit(X_train, y_train)
    predictions = predictor.predict(X_test)

    # Baseline scores: use each feature as a simple importance heuristic.
    test_rows = [rows[i] for i in test_indices]
    baseline_features = {
        "incoming_weight_norm": np.asarray([
            r["incoming_weight_norm"] for r in test_rows
        ]),
        "outgoing_weight_norm": np.asarray([
            r["outgoing_weight_norm"] for r in test_rows
        ]),
        "activation_magnitude": np.asarray([
            r["activation_mean_abs"] for r in test_rows
        ]),
        "activation_std": np.asarray([
            r["activation_std"] for r in test_rows
        ]),
    }

    # Evaluate rankings within each layer; cross-layer target distributions
    # may differ because layer widths and representations differ.
    per_layer = {}
    for layer in LAYERS:
        local = [
            j for j, row in enumerate(test_rows) if row["layer"] == layer
        ]
        y = y_test[local]
        pred = predictions[local]
        k = max(1, int(np.ceil(len(local) * 0.10)))

        result = {
            "held_out_channels": len(local),
            "predictor": ranking_metrics(y, pred, k),
            "baselines": {},
        }

        for name, scores in baseline_features.items():
            result["baselines"][name] = ranking_metrics(
                y, scores[local], k
            )

        result["predictor_regression"] = {
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        }
        per_layer[layer] = result

    # Random-ranking reference distribution, computed separately per layer.
    random_baseline = {}
    for layer in LAYERS:
        local = [
            j for j, row in enumerate(test_rows) if row["layer"] == layer
        ]
        y = y_test[local]
        k = max(1, int(np.ceil(len(local) * 0.10)))
        trials = []
        for _ in range(500):
            scores = rng.random(len(local))
            trials.append(ranking_metrics(y, scores, k)["top_k_precision"])
        random_baseline[layer] = {
            "trials": len(trials),
            "mean_top_k_precision": float(np.mean(trials)),
            "std_top_k_precision": float(np.std(trials)),
        }

    report = {
        "experiment": "Brahma MetaCore MetaAnalyzer v0.4 predictive channel importance",
        "seed": SEED,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_epoch": discovery["checkpoint_epoch"],
        "feature_dataset": "Saved 5,000-example validation split",
        "label_source": str(REPORT_V02.relative_to(ROOT)),
        "channels_total": len(rows),
        "train_channels": len(train_indices),
        "held_out_channels": len(test_indices),
        "split_method": (
            "Random channel holdout within each layer; 70% train, 30% test. "
            "This evaluates generalization to held-out channels, not to a new model."
        ),
        "target": "absolute delta_loss after zeroing one BN2d output channel",
        "features": feature_names,
        "predictor": "RandomForestRegressor",
        "per_layer_results": per_layer,
        "random_ranking_reference": random_baseline,
        "feature_importance": {
            name: float(value)
            for name, value in zip(
                feature_names, predictor.feature_importances_
            )
        },
        "elapsed_seconds": time.time() - start,
        "limitations": [
            "The labels are ablation measurements on the same validation split used for activation feature extraction.",
            "Channels from one trained model are not independent samples.",
            "The held-out channel split tests channel-level generalization only.",
            "This is a pilot with one checkpoint and one dataset.",
            "Ablation sensitivity is not evidence of semantic understanding or general intelligence.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\nMetaAnalyzer v0.4 completed.")
    print("Channels:", len(rows))
    print("Training channels:", len(train_indices))
    print("Held-out channels:", len(test_indices))
    print("Report:", OUTPUT)
    print("\nPer-layer held-out results:")
    for layer, result in per_layer.items():
        print(
            f"{layer}: "
            f"predictor Spearman={result['predictor']['spearman']}, "
            f"top-k precision={result['predictor']['top_k_precision']:.3f}"
        )
        for name, baseline in result["baselines"].items():
            print(
                f"  {name}: Spearman={baseline['spearman']}, "
                f"top-k precision={baseline['top_k_precision']:.3f}"
            )


if __name__ == "__main__":
    main()
