"""MetaCore Parameter Intervention v0.2 — independent-checkpoint replication.

Two-stage protocol:
  1) --prepare: extract model-map features, measure training-channel interventions,
     train predictors, and save held-out-channel predictions BEFORE their outcomes
     are measured.
  2) --evaluate: load the frozen predictions, measure held-out-channel interventions
     on a disjoint image subset, and score predictions against outcomes.

Run from the Brahma-MetaCore repository root in the existing venv_cuda environment.
This replication uses checkpoints\\metatransfer_v03_seed2027\\best_validation.pt and writes only to reports\\parameter_intervention_v02_seed2027. No training, package installation, official CIFAR-10 test-set loading, or checkpoint modification is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Subset

from src.experiments import run_metaanalyzer_v04 as v04
from src.experiments import run_metaanalyzer_v06 as v06
from src.experiments import run_metatransfer_v02 as v02

ROOT = Path(__file__).resolve().parents[2]
SOURCE_CHECKPOINT = ROOT / "checkpoints" / "metatransfer_v03_seed2027" / "best_validation.pt"
OUT_DIR = ROOT / "reports" / "parameter_intervention_v02_seed2027"
PREPARED_PATH = OUT_DIR / "prepared_predictions.json"
REPORT_PATH = OUT_DIR / "evaluation_report.json"
MARKDOWN_PATH = OUT_DIR / "evaluation_report.md"

LAYERS = ("layer2.1.bn1", "layer3.1.bn1", "layer4.1.bn1")
WIDTHS = {"layer2.1.bn1": 128, "layer3.1.bn1": 256, "layer4.1.bn1": 512}
FEATURE_NAMES = (
    "activation_mean", "activation_std", "activation_mean_abs",
    "activation_near_zero_fraction", "bn_gamma_abs", "bn_beta_abs",
    "incoming_weight_norm", "outgoing_weight_norm",
)
SEED = 20261016
BATCH_SIZE = 128
TRAIN_CHANNEL_FRACTION = 0.70
RF_SEED = 20261016


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_indices(indices: np.ndarray) -> str:
    return sha256_bytes(np.asarray(indices, dtype=np.int64).tobytes())


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return obj


def load_inputs():
    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {SOURCE_CHECKPOINT}")
    checkpoint_hash = sha256_file(SOURCE_CHECKPOINT)
    checkpoint = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Source checkpoint must contain model_state_dict.")

    images, labels, data_hashes = v02.validate_dataset_files()
    if len(images) != 2000 or len(labels) != 2000:
        raise ValueError("Expected CIFAR-10.1 v6 with exactly 2,000 images and labels.")
    return checkpoint, checkpoint_hash, images, labels, data_hashes


def split_indices(n: int = 2000):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n)
    feature_ids = order[:1000]
    train_effect_ids = order[1000:1500]
    heldout_effect_ids = order[1500:2000]
    assert len(set(feature_ids) & set(train_effect_ids)) == 0
    assert len(set(feature_ids) & set(heldout_effect_ids)) == 0
    assert len(set(train_effect_ids) & set(heldout_effect_ids)) == 0
    return feature_ids, train_effect_ids, heldout_effect_ids


def make_loader(dataset, indices: np.ndarray, device: torch.device):
    subset = Subset(dataset, np.asarray(indices, dtype=np.int64).tolist())
    return DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )


def get_channel_features(model, loader, device):
    # Existing feature extractor uses activation statistics and weight/BN statistics.
    extracted = v04.extract_features(model, loader, device)
    rows = {}
    for layer, channel_rows in extracted.items():
        for row in channel_rows:
            channel = int(row["channel"])
            rows[f"{layer}|{channel}"] = {
                name: float(row[name]) for name in FEATURE_NAMES
            }
    for layer, width in WIDTHS.items():
        if sum(1 for key in rows if key.startswith(layer + "|")) != width:
            raise RuntimeError(f"Feature extraction did not return all {width} channels for {layer}.")
    return rows


def eval_metrics(model, loader, device):
    # Uses existing baseline evaluator; returns per-example predictions and aggregate metrics.
    from src.experiments import run_channel_ablation_v02 as ablation
    return ablation.evaluate_baseline(model, loader, device)


def measure_filter_zeroing(model, layer: str, channel: int, loader, device, baseline):
    """Zero one learned conv1 output filter, measure outcomes, and restore exact weights."""
    from src.experiments import run_metaanalyzer_v04 as feature_module
    parent, bn, conv_in, conv_out = feature_module.layer_modules(model, layer)
    if channel < 0 or channel >= conv_in.weight.shape[0] or channel >= bn.num_features:
        raise IndexError(f"Channel {channel} is invalid for {layer}.")
    original = conv_in.weight.detach()[channel].clone()
    try:
        with torch.no_grad():
            conv_in.weight[channel].zero_()
        changed = eval_metrics(model, loader, device)
    finally:
        with torch.no_grad():
            conv_in.weight[channel].copy_(original)

    base_predictions = np.asarray(baseline["predictions"], dtype=np.int64)
    changed_predictions = np.asarray(changed["predictions"], dtype=np.int64)
    if len(base_predictions) != len(changed_predictions):
        raise RuntimeError("Baseline/intervention prediction lengths differ.")
    delta_loss = float(changed["loss"] - baseline["loss"])
    delta_accuracy_pp = float((changed["accuracy"] - baseline["accuracy"]) * 100.0)
    flip_rate = float(np.mean(changed_predictions != base_predictions))
    return {
        "layer": layer,
        "channel": int(channel),
        "delta_loss": delta_loss,
        "abs_delta_loss": abs(delta_loss),
        "delta_accuracy_pp": delta_accuracy_pp,
        "prediction_flip_rate": flip_rate,
        "examples": int(changed["examples"]),
    }


def channel_split():
    rng = np.random.default_rng(SEED)
    train_ids, heldout_ids = {}, {}
    for layer, width in WIDTHS.items():
        ids = np.arange(width, dtype=np.int64)
        rng.shuffle(ids)
        n_train = int(round(width * TRAIN_CHANNEL_FRACTION))
        train_ids[layer] = sorted(ids[:n_train].tolist())
        heldout_ids[layer] = sorted(ids[n_train:].tolist())
    return train_ids, heldout_ids


def metrics_regression(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if len(actual) == 0:
        return {"n": 0, "mae": None, "rmse": None, "spearman": None}
    rho = spearmanr(actual, predicted).statistic if np.std(actual) > 0 and np.std(predicted) > 0 else np.nan
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "spearman": float(rho) if np.isfinite(rho) else None,
    }


def bootstrap_mae_improvement(actual, prediction, baseline_prediction, *, n_boot=2000, seed=20261017):
    """Paired within-layer channel bootstrap for MAE(baseline) - MAE(model).

    Positive values favor the learned predictor. This interval describes
    channel-sampling uncertainty within this one checkpoint, not model-level
    uncertainty or a population-level significance test.
    """
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    baseline_prediction = np.asarray(baseline_prediction, dtype=np.float64)
    if not (len(actual) == len(prediction) == len(baseline_prediction)) or len(actual) < 2:
        return {"n": int(len(actual)), "mean_improvement": None, "ci95_low": None, "ci95_high": None}
    per_channel_gain = np.abs(actual - baseline_prediction) - np.abs(actual - prediction)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(per_channel_gain), size=(n_boot, len(per_channel_gain)))
    boot_means = per_channel_gain[indices].mean(axis=1)
    return {
        "n": int(len(actual)),
        "mean_improvement": float(per_channel_gain.mean()),
        "ci95_low": float(np.quantile(boot_means, 0.025)),
        "ci95_high": float(np.quantile(boot_means, 0.975)),
        "bootstrap_replicates": int(n_boot),
        "interpretation": "Positive favors the learned predictor; within-checkpoint channel bootstrap only.",
    }


def make_predictor_comparisons(train_rows, heldout_rows, target_key):
    """Fit on training-channel outcomes and predict held-out channels without labels.

    Held-out outcomes must not be accessed in the preparation stage. Evaluation
    metrics are calculated only later, after frozen predictions are saved and
    the held-out interventions are measured by --evaluate-only.
    """
    Xtr = np.asarray([[r["features"][name] for name in FEATURE_NAMES] for r in train_rows], dtype=np.float64)
    Xte = np.asarray([[r["features"][name] for name in FEATURE_NAMES] for r in heldout_rows], dtype=np.float64)
    ytr = np.asarray([r[target_key] for r in train_rows], dtype=np.float64)
    models = {
        "no_change_zero": None,
        "training_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=4, max_features=0.8,
            random_state=RF_SEED, n_jobs=-1,
        ),
    }
    predictions = {}
    for name, model in models.items():
        if model is None:
            pred = np.zeros(len(heldout_rows), dtype=np.float64)
        else:
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
        predictions[name] = pred
    return predictions


def audit():
    checkpoint, checkpoint_hash, images, labels, data_hashes = load_inputs()
    feature_ids, train_ids, heldout_ids = split_indices()
    train_channels, heldout_channels = channel_split()
    if PREPARED_PATH.exists() or REPORT_PATH.exists() or MARKDOWN_PATH.exists():
        raise FileExistsError(
            f"Output collision under {OUT_DIR}. Existing artifacts are preserved; "
            "choose a new output directory in a future script revision rather than overwrite."
        )
    print("Source checkpoint: PASS", SOURCE_CHECKPOINT.relative_to(ROOT))
    print("Checkpoint SHA-256:", checkpoint_hash)
    print("CIFAR-10.1 v6 hashes: PASS", data_hashes)
    print("Image split sizes: features=1000, train effects=500, held-out effects=500")
    print("Feature split SHA-256:", sha256_indices(feature_ids))
    print("Training-effect split SHA-256:", sha256_indices(train_ids))
    print("Held-out-effect split SHA-256:", sha256_indices(heldout_ids))
    print("Channel split: train", sum(map(len, train_channels.values())),
          "held-out", sum(map(len, heldout_channels.values())))
    print("Audit PASS: no model inference, interventions, cache writes, or report writes.")
    return checkpoint, checkpoint_hash, images, labels, data_hashes, feature_ids, train_ids, heldout_ids, train_channels, heldout_channels


def prepare():
    (checkpoint, checkpoint_hash, images, labels, data_hashes,
     feature_ids, train_image_ids, heldout_image_ids, train_channels, heldout_channels) = audit()
    # audit() guarantees output files do not already exist.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print("Device:", device)

    dataset = v02.ArrayCIFAR10Dataset(images, labels)
    feature_loader = make_loader(dataset, feature_ids, device)
    train_loader = make_loader(dataset, train_image_ids, device)
    heldout_loader = make_loader(dataset, heldout_image_ids, device)

    model, loaded_checkpoint = v06.load_model(SOURCE_CHECKPOINT, device)
    baseline_train = eval_metrics(model, train_loader, device)
    baseline_heldout = eval_metrics(model, heldout_loader, device)
    print(f"Source baseline train-effect subset: acc={baseline_train['accuracy']*100:.2f}%, loss={baseline_train['loss']:.6f}")
    print(f"Source baseline held-out subset: acc={baseline_heldout['accuracy']*100:.2f}%, loss={baseline_heldout['loss']:.6f}")
    print("Extracting model-map features from the separate 1,000-image feature subset...")
    features = get_channel_features(model, feature_loader, device)

    train_effect_rows = []
    total = sum(len(v) for v in train_channels.values())
    done = 0
    for layer in LAYERS:
        for channel in train_channels[layer]:
            outcome = measure_filter_zeroing(model, layer, channel, train_loader, device, baseline_train)
            key = f"{layer}|{channel}"
            train_effect_rows.append({
                "layer": layer, "channel": channel, "features": features[key],
                "delta_loss": outcome["delta_loss"],
                "abs_delta_loss": outcome["abs_delta_loss"],
                "delta_accuracy_pp": outcome["delta_accuracy_pp"],
                "prediction_flip_rate": outcome["prediction_flip_rate"],
                "examples": outcome["examples"],
            })
            done += 1
            if done % 32 == 0 or done == total:
                print(f"Training intervention labels: {done}/{total}")

    # Create held-out predictions and serialize them before measuring any held-out outcomes.
    prepared_predictions = {}
    for layer in LAYERS:
        train_rows = [r for r in train_effect_rows if r["layer"] == layer]
        held_rows = [
            {"layer": layer, "channel": ch, "features": features[f"{layer}|{ch}"]}
            for ch in heldout_channels[layer]
        ]
        for target_key in ("delta_loss", "abs_delta_loss"):
            pred_by_model = make_predictor_comparisons(train_rows, held_rows, target_key)
            for row_index, row in enumerate(held_rows):
                key = f"{layer}|{row['channel']}"
                entry = prepared_predictions.setdefault(key, {
                    "layer": layer, "channel": int(row["channel"]),
                    "features": row["features"],
                })
                entry[f"{target_key}_predictions"] = {
                    method: float(values[row_index])
                    for method, values in pred_by_model.items()
                }
    payload = {
        "experiment": "Brahma MetaCore Parameter Intervention v0.2 independent-checkpoint replication",
        "status": "predictions_frozen_before_heldout_interventions",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
        "source_checkpoint_sha256": checkpoint_hash,
        "source_checkpoint_epoch": int(loaded_checkpoint.get("epoch", -1)),
        "dataset": "CIFAR-10.1 v6",
        "dataset_hashes": data_hashes,
        "official_cifar10_test_set_used": False,
        "seed": SEED,
        "device": str(device),
        "feature_image_count": len(feature_ids),
        "train_effect_image_count": len(train_image_ids),
        "heldout_effect_image_count": len(heldout_image_ids),
        "image_split_hashes": {
            "feature": sha256_indices(feature_ids),
            "train_effect": sha256_indices(train_image_ids),
            "heldout_effect": sha256_indices(heldout_image_ids),
        },
        "channel_split": {
            layer: {"train": train_channels[layer], "heldout": heldout_channels[layer]}
            for layer in LAYERS
        },
        "feature_names": list(FEATURE_NAMES),
        "intervention": "zero one trained conv1 output filter in the selected BasicBlock; restore exact filter after evaluation",
        "baseline_train": {
            "loss": float(baseline_train["loss"]), "accuracy": float(baseline_train["accuracy"]),
            "examples": int(baseline_train["examples"]),
        },
        "baseline_heldout": {
            "loss": float(baseline_heldout["loss"]), "accuracy": float(baseline_heldout["accuracy"]),
            "examples": int(baseline_heldout["examples"]),
        },
        "training_intervention_outcomes": train_effect_rows,
        "heldout_predictions": list(prepared_predictions.values()),
        "prediction_protocol_note": "Held-out channel intervention outcomes and evaluation metrics have not been measured at the time this file is written. Predictions are frozen first; metrics are calculated only by --evaluate-only.",
    }
    write_json_new(PREPARED_PATH, payload)
    print("Frozen predictions saved BEFORE held-out interventions:", PREPARED_PATH.relative_to(ROOT))
    print("Next, run --evaluate-only. Do not edit prepared_predictions.json.")


def evaluate():
    if not PREPARED_PATH.is_file():
        raise FileNotFoundError(f"Prepared predictions missing: {PREPARED_PATH}. Run --prepare first.")
    if REPORT_PATH.exists() or MARKDOWN_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation outputs under {OUT_DIR}.")
    prepared = load_json(PREPARED_PATH)
    if prepared.get("status") != "predictions_frozen_before_heldout_interventions":
        raise ValueError("Prepared artifact status is invalid.")
    checkpoint, checkpoint_hash, images, labels, data_hashes = load_inputs()
    if checkpoint_hash != prepared.get("source_checkpoint_sha256"):
        raise ValueError("Source checkpoint changed since predictions were frozen.")
    if data_hashes != prepared.get("dataset_hashes"):
        raise ValueError("Dataset hashes changed since predictions were frozen.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset = v02.ArrayCIFAR10Dataset(images, labels)
    _, _, heldout_image_ids = split_indices()
    heldout_loader = make_loader(dataset, heldout_image_ids, device)
    model, loaded_checkpoint = v06.load_model(SOURCE_CHECKPOINT, device)
    baseline = eval_metrics(model, heldout_loader, device)

    outcome_rows = []
    rows = prepared.get("heldout_predictions", [])
    for index, row in enumerate(rows, start=1):
        outcome = measure_filter_zeroing(
            model, row["layer"], int(row["channel"]), heldout_loader, device, baseline
        )
        outcome_rows.append({
            **outcome,
            "features": row["features"],
            "predictions": {
                "delta_loss": row["delta_loss_predictions"],
                "abs_delta_loss": row["abs_delta_loss_predictions"],
            },
        })
        if index % 32 == 0 or index == len(rows):
            print(f"Held-out intervention outcomes: {index}/{len(rows)}")

    results = {}
    for layer in LAYERS:
        local = [r for r in outcome_rows if r["layer"] == layer]
        results[layer] = {}
        for target_key, actual_key in (("delta_loss", "delta_loss"), ("abs_delta_loss", "abs_delta_loss")):
            actual = np.asarray([r[actual_key] for r in local], dtype=np.float64)
            methods = ("no_change_zero", "training_mean", "ridge", "random_forest")
            metrics = {}
            for method in methods:
                predicted = np.asarray([r["predictions"][target_key][method] for r in local], dtype=np.float64)
                metrics[method] = metrics_regression(actual, predicted)
            results[layer][target_key] = {"methods": metrics}
        # Prespecified primary endpoint: absolute loss-change magnitude.
        # Compare each learned predictor against both simple baselines with paired
        # bootstrap CIs over held-out channels within this layer.
        actual_abs = np.asarray([r["abs_delta_loss"] for r in local], dtype=np.float64)
        bootstrap_comparisons = {}
        for learned_method in ("ridge", "random_forest"):
            learned_pred = np.asarray([r["predictions"]["abs_delta_loss"][learned_method] for r in local], dtype=np.float64)
            for baseline_method in ("no_change_zero", "training_mean"):
                baseline_pred = np.asarray([r["predictions"]["abs_delta_loss"][baseline_method] for r in local], dtype=np.float64)
                bootstrap_comparisons[f"{learned_method}_vs_{baseline_method}"] = bootstrap_mae_improvement(
                    actual_abs, learned_pred, baseline_pred,
                    seed=20261017 + LAYERS.index(layer) * 100 + (0 if learned_method == "ridge" else 10) + (0 if baseline_method == "no_change_zero" else 1),
                )
        results[layer]["abs_effect_mae_bootstrap_comparisons"] = bootstrap_comparisons

        actual_signed = np.asarray([r["delta_loss"] for r in local], dtype=np.float64)
        pred_signed = np.asarray([r["predictions"]["delta_loss"]["random_forest"] for r in local], dtype=np.float64)
        results[layer]["direction_accuracy"] = {
            "random_forest": float(np.mean((actual_signed > 0) == (pred_signed > 0))),
            "note": "Sign accuracy is descriptive; exact/near-zero effects make sign unstable."
        }

    report = {
        "experiment": "Brahma MetaCore Parameter Intervention v0.2 independent-checkpoint replication",
        "status": "completed",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_predictions_sha256": sha256_file(PREPARED_PATH),
        "source_checkpoint": prepared["source_checkpoint"],
        "source_checkpoint_sha256": checkpoint_hash,
        "dataset": "CIFAR-10.1 v6",
        "dataset_hashes": data_hashes,
        "official_cifar10_test_set_used": False,
        "intervention": prepared["intervention"],
        "heldout_channel_count": len(outcome_rows),
        "heldout_outcomes": outcome_rows,
        "metrics_by_layer": results,
        "limitations": [
            "This run uses one independent source checkpoint, one architecture family, and one external dataset; replication across further checkpoints is still required.",
            "Channel interventions within one model are dependent observations, not independent model replications.",
            "CIFAR-10.1 v6 has only 2,000 examples, split into disjoint feature/train-effect/held-out-effect subsets.",
            "This tests prediction of direct filter-zeroing effects; it does not demonstrate model reconstruction or reduced pretraining.",
            "A held-out channel split is not a held-out model split.",
        ],
    }
    write_json_new(REPORT_PATH, report)
    markdown = [
        "# MetaCore Parameter Intervention v0.2 — Independent-checkpoint replication",
        "",
        "**Status:** Completed",
        "",
        "This experiment predicts the effects of zeroing selected learned convolution filters, then measures held-out intervention outcomes on disjoint images. Predictions were saved before held-out outcomes were measured.",
        "",
        "## Results by layer",
        "",
        "| Layer | Target | Method | N | MAE | RMSE | Spearman |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for layer, layer_data in results.items():
        for target_key in ("delta_loss", "abs_delta_loss"):
            for method, m in layer_data[target_key]["methods"].items():
                rho_text = "n/a" if m["spearman"] is None else f"{m['spearman']:.4f}"
                markdown.append(
                    f"| `{layer}` | `{target_key}` | `{method}` | {m['n']} | "
                    f"{m['mae']:.6g} | {m['rmse']:.6g} | {rho_text} |"
                )
    markdown += [
        "",
        "## Absolute-effect MAE improvement: paired channel bootstrap",
        "",
        "Positive improvement means the learned predictor has lower MAE than the named baseline. 95% percentile intervals resample held-out channels within each layer; they are not model-level confidence intervals.",
        "",
        "| Layer | Comparison | Mean MAE improvement | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for layer in LAYERS:
        for comparison, stats in results[layer]["abs_effect_mae_bootstrap_comparisons"].items():
            markdown.append(
                f"| `{layer}` | `{comparison}` | {stats['mean_improvement']:.6g} | "
                f"[{stats['ci95_low']:.6g}, {stats['ci95_high']:.6g}] |"
            )
    markdown += [
        "",
        "## Prespecified replication criterion",
        "",
        "Primary endpoint: absolute loss-change magnitude. A layer counts as positive evidence only when a learned predictor's MAE improvement over a simple baseline has a 95% within-layer channel-bootstrap interval entirely above zero. This is a descriptive within-checkpoint criterion; it does not substitute for multiple-checkpoint replication or account for all model-level uncertainty.",
        "",
        "## Interpretation boundary",
        "",
        "This is an independent-checkpoint replication of the Phase A predictive-intervention pilot. Any observed advantage applies only to this checkpoint, intervention definition, dataset, and split. It does not establish model reconstruction, capability preservation, or reduced pretraining.",
        "",
        "## Limitations",
        "",
    ] + [f"- {item}" for item in report["limitations"]]
    if MARKDOWN_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite {MARKDOWN_PATH}")
    MARKDOWN_PATH.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("Evaluation report saved:", REPORT_PATH.relative_to(ROOT))
    print("Markdown report saved:", MARKDOWN_PATH.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audit-only", action="store_true")
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        audit()
    elif args.prepare:
        prepare()
    else:
        evaluate()


if __name__ == "__main__":
    main()
