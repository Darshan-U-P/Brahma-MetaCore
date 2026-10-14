from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.experiments import run_metaanalyzer_v04 as v04
from src.experiments import run_channel_ablation_v02 as ablation


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CHECKPOINT = ROOT / "checkpoints/milestone2_v02/best_validation.pt"
TARGET_CHECKPOINT = ROOT / "checkpoints/milestone2_v05_target/best_validation.pt"
SPLIT_FILE = ROOT / "checkpoints/milestone2_v02/split_indices_v02.npz"
OUTPUT = ROOT / "reports/milestone2_v05/cross_model_generalization.json"

LAYERS = ("layer2.1.bn1", "layer3.1.bn1", "layer4.1.bn1")
FEATURE_NAMES = [
    "activation_mean",
    "activation_std",
    "activation_mean_abs",
    "activation_near_zero_fraction",
    "bn_gamma_abs",
    "bn_beta_abs",
    "incoming_weight_norm",
    "outgoing_weight_norm",
]
SEED = 20261015
BATCH_SIZE = 128


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unexpected checkpoint structure: {checkpoint_path}")

    model = ablation.build_model()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def make_loader(base, indices, device):
    dataset = v04.IndexedCIFAR10(
        base,
        indices,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(v04.MEAN, v04.STD),
        ]),
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def measure_channel_effects(model, loader, device, model_label):
    """Measure absolute loss change for every channel on the label subset."""
    baseline = ablation.evaluate_baseline(model, loader, device)
    print(
        f"{model_label} intervention-subset baseline: "
        f"accuracy={baseline['accuracy'] * 100:.2f}%, "
        f"loss={baseline['loss']:.5f}"
    )

    labels = {}
    total = sum(
        ablation.get_module_by_name(model, layer).num_features
        for layer in LAYERS
    )
    completed = 0

    for layer in LAYERS:
        module = ablation.get_module_by_name(model, layer)
        for channel in range(module.num_features):
            result = ablation.evaluate_with_channel_ablation(
                model=model,
                loader=loader,
                device=device,
                module=module,
                channel_index=channel,
                baseline_predictions=baseline["predictions"],
                baseline_batches=baseline["per_batch"],
            )
            delta = float(result["loss"] - baseline["loss"])
            labels[(layer, channel)] = {
                "signed_delta_loss": delta,
                "abs_delta_loss": abs(delta),
            }
            completed += 1

            if completed % 64 == 0 or completed == total:
                print(
                    f"  {model_label}: {completed}/{total} channels measured"
                )

    return labels, {
        "loss": float(baseline["loss"]),
        "accuracy": float(baseline["accuracy"]),
        "examples": int(baseline["examples"]),
    }


def make_feature_map(model, loader, device):
    extracted = v04.extract_features(model, loader, device)
    feature_map = {}
    for layer in LAYERS:
        for row in extracted[layer]:
            feature_map[(layer, int(row["channel"]))] = row
    return feature_map


def safe_spearman(actual, scores):
    if len(actual) < 2 or np.std(actual) == 0 or np.std(scores) == 0:
        return None
    value = float(spearmanr(actual, scores).statistic)
    return value if np.isfinite(value) else None


def ranking_metrics(actual, scores, k=20):
    actual = np.asarray(actual, dtype=float)
    scores = np.asarray(scores, dtype=float)
    k = min(k, len(actual))
    predicted_top = set(np.argsort(scores)[-k:].tolist())
    actual_top = set(np.argsort(actual)[-k:].tolist())
    return {
        "spearman": safe_spearman(actual, scores),
        "mae": float(mean_absolute_error(actual, scores)),
        "rmse": float(np.sqrt(mean_squared_error(actual, scores))),
        "top_k": k,
        "top_k_precision": len(predicted_top & actual_top) / k,
        "random_top_k_precision_reference": k / len(actual),
    }


def main():
    start = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    for path in (SOURCE_CHECKPOINT, TARGET_CHECKPOINT, SPLIT_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("MetaAnalyzer v0.5 — cross-model generalization")
    print("Device:", device)
    print("Loading CIFAR-10 training partition only; official test set is not loaded.")

    with np.load(SPLIT_FILE) as saved:
        val_indices = saved["val_indices"].copy()

    if len(val_indices) != 5000:
        raise RuntimeError(f"Expected 5,000 validation indices, got {len(val_indices)}")

    # Randomize once, then create disjoint feature and intervention subsets.
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(val_indices)
    feature_indices = shuffled[:2500]
    intervention_indices = shuffled[2500:]

    if np.intersect1d(feature_indices, intervention_indices).size:
        raise RuntimeError("Feature and intervention subsets overlap.")

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    base = datasets.CIFAR10(
        root=str(ROOT / "data" / "cifar10"),
        train=True,
        download=False,
        transform=None,
    )
    feature_loader = make_loader(base, feature_indices, device)
    intervention_loader = make_loader(base, intervention_indices, device)

    source_model, source_ckpt = load_model(SOURCE_CHECKPOINT, device)
    target_model, target_ckpt = load_model(TARGET_CHECKPOINT, device)

    print("\nExtracting source-model features...")
    source_features = make_feature_map(source_model, feature_loader, device)

    print("Extracting target-model features...")
    target_features = make_feature_map(target_model, feature_loader, device)

    print("\nMeasuring source-model channel interventions...")
    source_labels, source_baseline = measure_channel_effects(
        source_model, intervention_loader, device, "SOURCE"
    )

    print("\nMeasuring target-model channel interventions...")
    target_labels, target_baseline = measure_channel_effects(
        target_model, intervention_loader, device, "TARGET"
    )

    expected_keys = {
        (layer, channel)
        for layer in LAYERS
        for channel in range(
            ablation.get_module_by_name(source_model, layer).num_features
        )
    }
    for name, mapping in (
        ("source features", source_features),
        ("target features", target_features),
        ("source labels", source_labels),
        ("target labels", target_labels),
    ):
        if set(mapping) != expected_keys:
            raise RuntimeError(
                f"{name} has {len(mapping)} channels; expected {len(expected_keys)}."
            )

    keys = sorted(expected_keys)
    X_source = np.asarray(
        [[source_features[key][name] for name in FEATURE_NAMES] for key in keys],
        dtype=np.float64,
    )
    X_target = np.asarray(
        [[target_features[key][name] for name in FEATURE_NAMES] for key in keys],
        dtype=np.float64,
    )
    y_source = np.asarray(
        [source_labels[key]["abs_delta_loss"] for key in keys],
        dtype=np.float64,
    )
    y_target = np.asarray(
        [target_labels[key]["abs_delta_loss"] for key in keys],
        dtype=np.float64,
    )

    # Fit only on source-model measurements. Target labels are evaluation-only.
    predictor = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=SEED,
        n_jobs=-1,
    )
    predictor.fit(X_source, y_source)
    target_predictions = predictor.predict(X_target)

    overall = {
        "random_forest_cross_model": ranking_metrics(y_target, target_predictions),
        "feature_heuristics": {},
    }

    heuristic_columns = {
        "abs_activation_mean": 0,
        "activation_std": 1,
        "activation_mean_abs": 2,
        "bn_gamma_abs": 4,
        "bn_beta_abs": 5,
        "incoming_weight_norm": 6,
        "outgoing_weight_norm": 7,
    }
    for name, column in heuristic_columns.items():
        overall["feature_heuristics"][name] = ranking_metrics(
            y_target, np.abs(X_target[:, column])
        )

    per_layer = {}
    for layer in LAYERS:
        indices = [i for i, key in enumerate(keys) if key[0] == layer]
        layer_actual = y_target[indices]
        layer_predictions = target_predictions[indices]
        per_layer[layer] = {
            "channels": len(indices),
            "random_forest_cross_model": ranking_metrics(
                layer_actual, layer_predictions, k=min(10, len(indices))
            ),
            "heuristics": {},
        }
        for name, column in heuristic_columns.items():
            per_layer[layer]["heuristics"][name] = ranking_metrics(
                layer_actual,
                np.abs(X_target[indices, column]),
                k=min(10, len(indices)),
            )

    report = {
        "experiment": "Brahma MetaCore MetaAnalyzer v0.5 cross-model generalization",
        "seed": SEED,
        "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
        "source_checkpoint_epoch": int(source_ckpt.get("epoch", -1)),
        "target_checkpoint": str(TARGET_CHECKPOINT.relative_to(ROOT)),
        "target_checkpoint_epoch": int(target_ckpt.get("epoch", -1)),
        "split_file": str(SPLIT_FILE.relative_to(ROOT)),
        "feature_examples": len(feature_indices),
        "intervention_examples": len(intervention_indices),
        "subsets_disjoint": True,
        "official_test_set_used": False,
        "channels_evaluated": len(keys),
        "layers": list(LAYERS),
        "feature_names": FEATURE_NAMES,
        "target": "absolute change in mean cross-entropy loss after zeroing one BN2d output channel",
        "training_protocol": (
            "Random Forest fitted on source-model channel features and source-model "
            "intervention labels; evaluated on target-model features against target "
            "intervention labels. Target labels are not used for fitting."
        ),
        "source_intervention_baseline": source_baseline,
        "target_intervention_baseline": target_baseline,
        "overall_results": overall,
        "per_layer_results": per_layer,
        "random_forest_feature_importance": {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, predictor.feature_importances_)
        },
        "elapsed_seconds": time.time() - start,
        "limitations": [
            "This is a two-checkpoint pilot on one dataset and architecture.",
            "Channels with the same index in independently trained models may not be functionally aligned.",
            "Channel measurements within one model are correlated and are not independent model-level samples.",
            "Intervention effects depend on the chosen zeroing intervention and validation subset.",
            "The validation partition is used for research measurements, not as a final untouched benchmark.",
            "Predictive channel importance does not demonstrate semantic understanding, model construction, or general intelligence.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nCompleted in {(time.time() - start) / 60:.1f} minutes.")
    print("Report:", OUTPUT)
    print("\nOverall Random Forest results:")
    print(json.dumps(overall["random_forest_cross_model"], indent=2))


if __name__ == "__main__":
    main()
