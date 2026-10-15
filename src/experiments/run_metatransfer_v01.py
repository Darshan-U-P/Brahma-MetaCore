"""MetaTransfer v0.1: disjoint cross-model channel-importance transfer.

Run from the Brahma-MetaCore project root:
    python -m src.experiments.run_metatransfer_v01 --audit-only
    python -m src.experiments.run_metatransfer_v01

Design:
- Reuse the frozen v0.6 per-image activation signatures (2,500 feature images).
- Divide the original 2,500-image v0.6 intervention subset into two disjoint
  1,250-image sets: source importance selection and target evaluation.
- Measure seed42 source channel effects only on the source-scoring set.
- Measure target channel effects for seeds 123/456/789 only on the target set.
- Compare same-index transfer, activation-aligned transfer, and randomized
  one-to-one channel mappings using all cached signatures but new effect labels.

No model training is performed. The full run does perform fresh channel
ablations; this is necessary because v0.6 effects are aggregated over all 2,500
intervention images and cannot be split post hoc into disjoint source/target
labels. The official CIFAR-10 test set is never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from torchvision import datasets

from src.experiments import run_metaanalyzer_v06 as v06

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "metatransfer_v01"
CACHE_DIR = OUT_DIR / "cache"
DEFAULT_OUTPUT = OUT_DIR / "transfer_report.json"

SOURCE_MODEL = "seed42"
TARGET_MODELS = ("seed123", "seed456", "seed789")
FEATURE_EXAMPLES = 2500
DEFAULT_SOURCE_EXAMPLES = 1250
DEFAULT_TARGET_EXAMPLES = 1250
PRIMARY_FRACTION = 0.10
FRACTIONS = (0.05, 0.10, 0.20)
DEFAULT_RANDOM_MAPPINGS = 1000
RANDOM_SEED = 20261016


def sha256_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def read_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required cache/report is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "payload_json" not in archive:
            raise ValueError(f"Cache has no payload_json entry: {path}")
        payload = json.loads(str(archive["payload_json"].item()))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_split_and_signatures():
    """Recreate the exact v0.6 index order and load only its feature signatures."""
    report_path = v06.DEFAULT_OUTPUT
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing frozen v0.6 report: {report_path}")
    old_report = json.loads(report_path.read_text(encoding="utf-8"))
    if old_report.get("pilot") is not False:
        raise ValueError("The v0.6 report is not marked as a full, non-pilot run.")
    if old_report.get("feature_examples") != FEATURE_EXAMPLES:
        raise ValueError("The v0.6 report does not use exactly 2,500 feature images.")
    if old_report.get("intervention_examples") != 2500:
        raise ValueError("The v0.6 report does not use exactly 2,500 intervention images.")

    with np.load(v06.SPLIT_FILE, allow_pickle=False) as split:
        if "val_indices" not in split:
            raise KeyError(f"Missing val_indices in {v06.SPLIT_FILE}")
        validation_indices = split["val_indices"].copy()

    if len(validation_indices) != 5000:
        raise ValueError(
            f"Expected the saved 5,000-image validation split; got {len(validation_indices)}. "
            "Do not silently change the split for this experiment."
        )

    shuffled = np.random.default_rng(v06.SEED).permutation(validation_indices)
    feature_indices = shuffled[:FEATURE_EXAMPLES]
    old_intervention_indices = shuffled[FEATURE_EXAMPLES:FEATURE_EXAMPLES + 2500]

    expected_feature_hash = old_report.get("feature_indices_sha256")
    expected_intervention_hash = old_report.get("intervention_indices_sha256")
    if expected_feature_hash and sha256_indices(feature_indices) != expected_feature_hash:
        raise ValueError("Reconstructed feature indices do not match the v0.6 report.")
    if expected_intervention_hash and sha256_indices(old_intervention_indices) != expected_intervention_hash:
        raise ValueError("Reconstructed intervention indices do not match the v0.6 report.")

    signatures: dict[str, dict[str, np.ndarray]] = {}
    signature_paths: dict[str, str] = {}
    signature_hashes: dict[str, str] = {}
    for model_key in (SOURCE_MODEL, *TARGET_MODELS):
        path = v06.CACHE_DIR / f"full_f2500_i2500_{model_key}_signatures.npz"
        payload = read_payload(path)
        model_signatures: dict[str, np.ndarray] = {}
        for layer in v06.LAYERS:
            if layer not in payload:
                raise KeyError(f"{path} has no signature data for {layer}")
            arr = np.asarray(payload[layer], dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] != FEATURE_EXAMPLES:
                raise ValueError(
                    f"Unexpected signatures for {model_key}/{layer}: {arr.shape}; "
                    f"expected {FEATURE_EXAMPLES} x channels."
                )
            if not np.isfinite(arr).all():
                raise ValueError(f"Non-finite activation signature in {path}, {layer}")
            model_signatures[layer] = arr
        signatures[model_key] = model_signatures
        signature_paths[model_key] = str(path.relative_to(ROOT))
        signature_hashes[model_key] = sha256_file(path)

    for layer in v06.LAYERS:
        widths = {key: value[layer].shape[1] for key, value in signatures.items()}
        if len(set(widths.values())) != 1:
            raise ValueError(f"Channel widths do not match for {layer}: {widths}")

    return old_report, feature_indices, old_intervention_indices, signatures, signature_paths, signature_hashes


def cache_effects_for_subset(
    model_key: str,
    role: str,
    indices: np.ndarray,
    device: torch.device,
    base_dataset,
) -> dict[str, Any]:
    """Load a validated fresh effects cache, or measure and cache channel effects."""
    index_hash = sha256_indices(indices)
    checkpoint_path = v06.CHECKPOINTS[model_key]
    checkpoint_label = str(checkpoint_path.relative_to(ROOT))
    cache_path = CACHE_DIR / f"{role}_effects_{model_key}_n{len(indices)}_{index_hash[:12]}.npz"
    expected_meta = {
        "model_key": model_key,
        "role": role,
        "indices_sha256": index_hash,
        "examples": int(len(indices)),
        "checkpoint": checkpoint_label,
    }

    if cache_path.exists():
        print(f"Reusing effects cache: {cache_path.relative_to(ROOT)}")
        payload = read_payload(cache_path)
        if payload.get("metadata") != expected_meta:
            raise ValueError(
                f"Cached metadata mismatch in {cache_path}. Preserve the file and "
                "investigate rather than overwriting it."
            )
    else:
        loader = v06.make_loader(base_dataset, indices, device)
        model, checkpoint = v06.load_model(checkpoint_path, device)
        print(
            f"\nMeasuring {role}: {model_key}; examples={len(indices)}; "
            f"checkpoint epoch={int(checkpoint.get('epoch', -1))}"
        )
        effects, baseline = v06.measure_channel_effects(
            model=model,
            loader=loader,
            device=device,
            model_key=f"{model_key}/{role}",
            channels_per_layer=None,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        payload = {
            "metadata": expected_meta,
            "effects": effects,
            "baseline": baseline,
        }
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Cache names contain the exact subset hash and never overwrite existing files.
        if cache_path.exists():
            raise FileExistsError(f"Refusing to overwrite cache created during run: {cache_path}")
        v06.save_cache(cache_path, payload)
        print(f"Saved effects cache: {cache_path.relative_to(ROOT)}")

    effects = payload.get("effects")
    baseline = payload.get("baseline")
    if not isinstance(effects, dict) or not isinstance(baseline, dict):
        raise ValueError(f"Invalid effects/baseline payload in {cache_path}")
    if int(baseline.get("examples", -1)) != len(indices):
        raise ValueError(f"Baseline example count mismatch in {cache_path}")
    if len(effects) != sum(signatures_width(model_key, layer) for layer in v06.LAYERS):
        raise ValueError(f"Expected effects for every channel; got {len(effects)} in {cache_path}")

    for layer in v06.LAYERS:
        width = signatures_width(model_key, layer)
        for channel in range(width):
            key = f"{layer}|{channel}"
            if key not in effects:
                raise KeyError(f"Missing channel effect {key} in {cache_path}")
            item = effects[key]
            if "signed_delta_loss" not in item or "abs_delta_loss" not in item:
                raise KeyError(f"Missing signed/absolute loss effect in {cache_path}: {key}")
            if int(item.get("examples", -1)) != len(indices):
                raise ValueError(f"Channel effect example count mismatch for {key} in {cache_path}")
    payload["cache_path"] = str(cache_path.relative_to(ROOT))
    payload["cache_sha256"] = sha256_file(cache_path)
    return payload


# Widths are set once the frozen signature caches have loaded.
_CHANNEL_WIDTHS: dict[tuple[str, str], int] = {}


def signatures_width(model_key: str, layer: str) -> int:
    try:
        return _CHANNEL_WIDTHS[(model_key, layer)]
    except KeyError as exc:
        raise RuntimeError("Signature widths were not initialized before effects validation.") from exc


def topk_metrics(actual_abs: np.ndarray, predicted_abs: np.ndarray, fractions=FRACTIONS) -> dict[str, Any]:
    actual_abs = np.asarray(actual_abs, dtype=np.float64)
    predicted_abs = np.asarray(predicted_abs, dtype=np.float64)
    n = len(actual_abs)
    if n != len(predicted_abs) or n == 0:
        raise ValueError("Actual and predicted importance arrays must have equal nonzero length.")
    actual_top_order = np.argsort(actual_abs)
    out: dict[str, Any] = {}
    for fraction in fractions:
        k = max(1, int(np.ceil(n * fraction)))
        predicted_top = set(np.argsort(predicted_abs)[-k:].tolist())
        actual_top = set(actual_top_order[-k:].tolist())
        selected_mean = float(actual_abs[list(predicted_top)].mean())
        global_mean = float(actual_abs.mean())
        out[f"top_{int(round(fraction * 100))}pct"] = {
            "fraction": float(fraction),
            "k": int(k),
            "precision": float(len(predicted_top & actual_top) / k),
            "recall": float(len(predicted_top & actual_top) / k),
            "overlap": int(len(predicted_top & actual_top)),
            "mean_target_abs_delta_loss_selected": selected_mean,
            "mean_target_abs_delta_loss_all_channels": global_mean,
            "importance_enrichment": float(selected_mean / global_mean) if global_mean > 0 else None,
            "random_selection_expected_precision": float(k / n),
        }
    return out


def score_transfer(actual_abs: np.ndarray, predicted_abs: np.ndarray) -> dict[str, Any]:
    actual_abs = np.asarray(actual_abs, dtype=np.float64)
    predicted_abs = np.asarray(predicted_abs, dtype=np.float64)
    rho = None
    if np.std(actual_abs) > 0 and np.std(predicted_abs) > 0:
        value = float(spearmanr(actual_abs, predicted_abs).statistic)
        rho = value if np.isfinite(value) else None
    return {
        "spearman_abs_importance": rho,
        "mae_abs_importance": float(np.mean(np.abs(actual_abs - predicted_abs))),
        "rmse_abs_importance": float(np.sqrt(np.mean((actual_abs - predicted_abs) ** 2))),
        "topk": topk_metrics(actual_abs, predicted_abs),
    }


def signed_effect_metrics(actual_signed: np.ndarray, predicted_signed: np.ndarray) -> dict[str, Any]:
    actual_signed = np.asarray(actual_signed, dtype=np.float64)
    predicted_signed = np.asarray(predicted_signed, dtype=np.float64)
    rho = None
    if np.std(actual_signed) > 0 and np.std(predicted_signed) > 0:
        value = float(spearmanr(actual_signed, predicted_signed).statistic)
        rho = value if np.isfinite(value) else None
    return {
        "spearman_signed_delta_loss": rho,
        "sign_agreement_all_channels": float(np.mean(np.sign(actual_signed) == np.sign(predicted_signed))),
        "mean_actual_signed_delta_loss": float(actual_signed.mean()),
        "mean_predicted_signed_delta_loss": float(predicted_signed.mean()),
        "note": "Secondary exploratory metric; signs of very small effects may be unstable.",
    }


def target_effect_arrays(payload: dict[str, Any], layer: str, width: int):
    effects = payload["effects"]
    signed = np.empty(width, dtype=np.float64)
    absolute = np.empty(width, dtype=np.float64)
    for channel in range(width):
        item = effects[f"{layer}|{channel}"]
        signed[channel] = float(item["signed_delta_loss"])
        absolute[channel] = float(item["abs_delta_loss"])
    return signed, absolute


def transfer_score_arrays(source_effects: dict[str, Any], layer: str, width: int):
    signed = np.empty(width, dtype=np.float64)
    absolute = np.empty(width, dtype=np.float64)
    for channel in range(width):
        item = source_effects[f"{layer}|{channel}"]
        signed[channel] = float(item["signed_delta_loss"])
        absolute[channel] = float(item["abs_delta_loss"])
    return signed, absolute


def summarize_random_mappings(
    actual_target_abs: np.ndarray,
    source_score_abs: np.ndarray,
    aligned_metrics: dict[str, Any],
    same_metrics: dict[str, Any],
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n = len(actual_target_abs)
    primary_key = f"top_{int(round(PRIMARY_FRACTION * 100))}pct"
    aligned_p = aligned_metrics["topk"][primary_key]["precision"]
    same_p = same_metrics["topk"][primary_key]["precision"]
    random_precisions = np.empty(repetitions, dtype=np.float64)
    random_rhos = np.full(repetitions, np.nan, dtype=np.float64)

    for i in range(repetitions):
        # A random permutation preserves the source score distribution while
        # removing the learned source->target channel correspondence.
        permutation = rng.permutation(n)  # source index -> target index
        predicted = np.empty(n, dtype=np.float64)
        predicted[permutation] = source_score_abs
        random_metric = topk_metrics(actual_target_abs, predicted, fractions=(PRIMARY_FRACTION,))
        random_precisions[i] = random_metric[primary_key]["precision"]
        if np.std(actual_target_abs) > 0 and np.std(predicted) > 0:
            random_rhos[i] = float(spearmanr(actual_target_abs, predicted).statistic)

    random_finite_rhos = random_rhos[np.isfinite(random_rhos)]
    return {
        "n_random_mappings": int(repetitions),
        "primary_fraction": PRIMARY_FRACTION,
        "precision_mean": float(random_precisions.mean()),
        "precision_std": float(random_precisions.std(ddof=1)) if repetitions > 1 else 0.0,
        "precision_2_5_percentile": float(np.quantile(random_precisions, 0.025)),
        "precision_97_5_percentile": float(np.quantile(random_precisions, 0.975)),
        "empirical_p_random_precision_ge_aligned": float((1 + np.sum(random_precisions >= aligned_p)) / (repetitions + 1)),
        "empirical_p_random_precision_ge_same_index": float((1 + np.sum(random_precisions >= same_p)) / (repetitions + 1)),
        "spearman_mean": float(random_finite_rhos.mean()) if len(random_finite_rhos) else None,
        "spearman_2_5_percentile": float(np.quantile(random_finite_rhos, 0.025)) if len(random_finite_rhos) else None,
        "spearman_97_5_percentile": float(np.quantile(random_finite_rhos, 0.975)) if len(random_finite_rhos) else None,
        "interpretation_note": "Permutation reference only; empirical p-values do not account for dependence across layers or target seeds and are not stand-alone population significance tests.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-only", action="store_true",
        help="Validate v0.6 report, split hashes, and all signature caches without running ablations.",
    )
    parser.add_argument(
        "--random-mappings", type=int, default=DEFAULT_RANDOM_MAPPINGS,
        help=f"Random channel permutations per target/layer (default: {DEFAULT_RANDOM_MAPPINGS}).",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed for permutation controls (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Report path; existing files are never overwritten.",
    )
    args = parser.parse_args()
    if args.random_mappings < 100:
        parser.error("--random-mappings must be at least 100.")

    output_path = v06.resolve_from_root(args.output) if args.output else DEFAULT_OUTPUT
    markdown_output_path = output_path.with_suffix(".md")
    if not args.audit_only and (output_path.exists() or markdown_output_path.exists()):
        raise FileExistsError(
            "Refusing to overwrite an existing report. Choose a new JSON filename "
            f"with --output. JSON exists={output_path.exists()}, Markdown exists={markdown_output_path.exists()}."
        )

    started = time.time()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    old_report, feature_indices, old_intervention_indices, signatures, signature_paths, signature_hashes = load_split_and_signatures()

    for model_key, layers in signatures.items():
        for layer, matrix in layers.items():
            _CHANNEL_WIDTHS[(model_key, layer)] = int(matrix.shape[1])

    for model_key, checkpoint in v06.CHECKPOINTS.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {model_key}: {checkpoint}")
    if not (v06.DATA_ROOT / "cifar-10-batches-py").exists():
        raise FileNotFoundError(f"CIFAR-10 extracted data not found under {v06.DATA_ROOT}")

    if args.audit_only:
        print("MetaTransfer v0.1 cache/split audit: PASS")
        print(f"v0.6 report: {v06.DEFAULT_OUTPUT.relative_to(ROOT)}")
        print(f"Feature subset: {len(feature_indices)} images; SHA-256={sha256_indices(feature_indices)}")
        print(f"Original intervention subset: {len(old_intervention_indices)} images; SHA-256={sha256_indices(old_intervention_indices)}")
        for model_key in (SOURCE_MODEL, *TARGET_MODELS):
            print(f"{model_key}: signatures validated; shapes=" + ", ".join(f"{layer}={signatures[model_key][layer].shape}" for layer in v06.LAYERS))
            print(f"  cache: {signature_paths[model_key]}")
            print(f"  sha256: {signature_hashes[model_key]}")
        print("No model inference, channel ablations, or writes were performed.")
        return

    if args.random_mappings < 100:
        raise ValueError("Random mapping count is too small for the planned controls.")
    source_examples = DEFAULT_SOURCE_EXAMPLES
    target_examples = DEFAULT_TARGET_EXAMPLES
    if source_examples + target_examples != len(old_intervention_indices):
        raise RuntimeError("Source and target subsets must exactly partition the cached v0.6 intervention subset.")
    source_score_indices = old_intervention_indices[:source_examples]
    target_eval_indices = old_intervention_indices[source_examples:source_examples + target_examples]
    if np.intersect1d(feature_indices, source_score_indices).size or np.intersect1d(feature_indices, target_eval_indices).size:
        raise RuntimeError("Feature and intervention subsets overlap.")
    if np.intersect1d(source_score_indices, target_eval_indices).size:
        raise RuntimeError("Source score and target evaluation subsets overlap.")
    if len(np.unique(source_score_indices)) != len(source_score_indices) or len(np.unique(target_eval_indices)) != len(target_eval_indices):
        raise RuntimeError("Source/target effect subsets contain duplicate image indices.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    base_dataset = datasets.CIFAR10(root=str(v06.DATA_ROOT), train=True, download=False, transform=None)

    source_payload = cache_effects_for_subset(
        SOURCE_MODEL, "source_score", source_score_indices, device, base_dataset
    )
    target_payloads: dict[str, dict[str, Any]] = {}
    for target_model in TARGET_MODELS:
        target_payloads[target_model] = cache_effects_for_subset(
            target_model, "target_eval", target_eval_indices, device, base_dataset
        )

    rng = np.random.default_rng(args.seed)
    target_results: dict[str, Any] = {}
    target_seed_primary_deltas = []

    for target_model in TARGET_MODELS:
        layer_results: dict[str, Any] = {}
        per_target_primary_aligned = []
        per_target_primary_same = []
        per_target_primary_random = []

        for layer in v06.LAYERS:
            source_sig = signatures[SOURCE_MODEL][layer]
            target_sig = signatures[target_model][layer]
            mapping, alignment_stats = v06.align_channels(source_sig, target_sig)
            width = source_sig.shape[1]
            if len(mapping) != width or not np.array_equal(np.sort(mapping), np.arange(width)):
                raise ValueError(f"Non-permutation alignment produced for {target_model}/{layer}")

            source_signed, source_abs = transfer_score_arrays(source_payload["effects"], layer, width)
            target_signed, target_abs = target_effect_arrays(target_payloads[target_model], layer, width)

            # Identity mapping: source channel i predicts target channel i.
            same_abs = source_abs.copy()
            same_signed = source_signed.copy()

            # Hungarian mapping is source-index -> target-index. Put each source
            # effect score into its mapped target-channel slot.
            aligned_abs = np.empty(width, dtype=np.float64)
            aligned_signed = np.empty(width, dtype=np.float64)
            aligned_abs[mapping] = source_abs
            aligned_signed[mapping] = source_signed

            same_metrics = score_transfer(target_abs, same_abs)
            aligned_metrics = score_transfer(target_abs, aligned_abs)
            random_metrics = summarize_random_mappings(
                actual_target_abs=target_abs,
                source_score_abs=source_abs,
                aligned_metrics=aligned_metrics,
                same_metrics=same_metrics,
                repetitions=args.random_mappings,
                rng=rng,
            )

            layer_result = {
                "channels": int(width),
                "alignment": {
                    **json_safe(alignment_stats),
                    "fraction_mapped_to_same_index": float(np.mean(mapping == np.arange(width))),
                    "mapping_source_index_to_target_index": mapping.astype(int).tolist(),
                },
                "source_selection_baseline": {
                    "accuracy": float(source_payload["baseline"]["accuracy"]),
                    "loss": float(source_payload["baseline"]["loss"]),
                    "examples": int(source_payload["baseline"]["examples"]),
                },
                "target_evaluation_baseline": {
                    "accuracy": float(target_payloads[target_model]["baseline"]["accuracy"]),
                    "loss": float(target_payloads[target_model]["baseline"]["loss"]),
                    "examples": int(target_payloads[target_model]["baseline"]["examples"]),
                },
                "same_index_transfer": same_metrics,
                "activation_aligned_transfer": aligned_metrics,
                "random_mapping_control": random_metrics,
                "signed_effect_transfer_secondary": {
                    "same_index": signed_effect_metrics(target_signed, same_signed),
                    "activation_aligned": signed_effect_metrics(target_signed, aligned_signed),
                },
            }
            layer_results[layer] = layer_result
            primary_key = f"top_{int(round(PRIMARY_FRACTION * 100))}pct"
            per_target_primary_same.append(same_metrics["topk"][primary_key]["precision"])
            per_target_primary_aligned.append(aligned_metrics["topk"][primary_key]["precision"])
            per_target_primary_random.append(random_metrics["precision_mean"])

            print(
                f"{target_model} / {layer}: P@{int(PRIMARY_FRACTION*100)}% "
                f"same={per_target_primary_same[-1]:.3f}, "
                f"aligned={per_target_primary_aligned[-1]:.3f}, "
                f"random-map mean={per_target_primary_random[-1]:.3f}; "
                f"rho same={same_metrics['spearman_abs_importance']}, "
                f"aligned={aligned_metrics['spearman_abs_importance']}"
            )

        target_delta = float(np.mean(np.asarray(per_target_primary_aligned) - np.asarray(per_target_primary_same)))
        target_seed_primary_deltas.append(target_delta)
        target_results[target_model] = {
            "source_model": SOURCE_MODEL,
            "layers": layer_results,
            "mean_primary_precision_across_layers": {
                "same_index": float(np.mean(per_target_primary_same)),
                "activation_aligned": float(np.mean(per_target_primary_aligned)),
                "random_mapping_mean": float(np.mean(per_target_primary_random)),
                "aligned_minus_same_index": target_delta,
            },
        }

    all_rows = [
        (target, layer, result)
        for target, item in target_results.items()
        for layer, result in item["layers"].items()
    ]
    primary_key = f"top_{int(round(PRIMARY_FRACTION * 100))}pct"
    aligned_precision = [r["activation_aligned_transfer"]["topk"][primary_key]["precision"] for _, _, r in all_rows]
    same_precision = [r["same_index_transfer"]["topk"][primary_key]["precision"] for _, _, r in all_rows]
    random_precision = [r["random_mapping_control"]["precision_mean"] for _, _, r in all_rows]

    report = {
        "experiment": "Brahma MetaCore MetaTransfer v0.1",
        "status": "completed_by_script_when_run",
        "purpose": "Test whether activation-based source-to-target channel alignment improves importance transfer on disjoint source-scoring and target-evaluation image subsets.",
        "source_model": SOURCE_MODEL,
        "target_models": list(TARGET_MODELS),
        "layers": list(v06.LAYERS),
        "seed": int(args.seed),
        "random_mappings_per_target_layer": int(args.random_mappings),
        "primary_metric": f"Top-{int(PRIMARY_FRACTION*100)}% precision for identifying the target model's truly top-{int(PRIMARY_FRACTION*100)}% channels by absolute loss change.",
        "secondary_metrics": ["Spearman rank correlation", "MAE/RMSE of absolute effects", "target-effect enrichment", "signed-effect correlation and sign agreement"],
        "protocol": {
            "feature_examples": int(len(feature_indices)),
            "source_effect_scoring_examples": int(len(source_score_indices)),
            "target_effect_evaluation_examples": int(len(target_eval_indices)),
            "feature_source_target_subsets_pairwise_disjoint": True,
            "source_score_indices_sha256": sha256_indices(source_score_indices),
            "target_eval_indices_sha256": sha256_indices(target_eval_indices),
            "feature_indices_sha256": sha256_indices(feature_indices),
            "source_and_target_effect_sets_disjoint": True,
            "official_cifar10_test_set_loaded": False,
            "feature_signatures_reused_from_v06": True,
            "source_effects_and_target_effects_recomputed_on_disjoint_subsets": True,
            "subset_selection": "Reconstruct v0.6 seeded validation permutation; preserve its first 2,500 feature images; split the remaining 2,500 intervention images into the first 1,250 source-score images and last 1,250 target-evaluation images.",
        },
        "source_score_cache": {
            "path": source_payload["cache_path"],
            "sha256": source_payload["cache_sha256"],
            "baseline": source_payload["baseline"],
        },
        "target_eval_caches": {
            key: {
                "path": payload["cache_path"],
                "sha256": payload["cache_sha256"],
                "baseline": payload["baseline"],
            }
            for key, payload in target_payloads.items()
        },
        "signature_cache_paths": signature_paths,
        "signature_cache_sha256": signature_hashes,
        "results": target_results,
        "aggregate_primary_metric": {
            "number_of_target_layer_comparisons": len(all_rows),
            "mean_aligned_top_fraction_precision": float(np.mean(aligned_precision)),
            "mean_same_index_top_fraction_precision": float(np.mean(same_precision)),
            "mean_random_mapping_top_fraction_precision": float(np.mean(random_precision)),
            "aligned_beats_same_index_comparisons": int(np.sum(np.asarray(aligned_precision) > np.asarray(same_precision))),
            "ties_vs_same_index_comparisons": int(np.sum(np.asarray(aligned_precision) == np.asarray(same_precision))),
            "aligned_beats_random_mapping_mean_comparisons": int(np.sum(np.asarray(aligned_precision) > np.asarray(random_precision))),
            "target_seed_mean_aligned_minus_same_index_precision": {
                key: target_results[key]["mean_primary_precision_across_layers"]["aligned_minus_same_index"]
                for key in TARGET_MODELS
            },
            "number_of_target_seeds_with_positive_mean_delta": int(np.sum(np.asarray(target_seed_primary_deltas) > 0)),
        },
        "preregistered_interpretation_rule": {
            "support_for_transfer": "Directional support requires mean aligned Top-10% precision across the 9 target-layer comparisons to exceed both same-index transfer and the mean randomized-mapping control, and the layer-averaged aligned-minus-same-index difference to be positive for at least 2 of the 3 target seeds.",
            "not_sufficient": "A single winning layer, a high activation correlation alone, or a favourable result against only one baseline is not sufficient.",
        },
        "limitations": [
            "The 5,000-image validation partition has previously been used for checkpoint selection; this is not a pristine external benchmark.",
            "There are three target checkpoints; layer rows within a checkpoint are dependent and must not be treated as nine independent model replications.",
            "The experiment tests channel-importance transfer within one dataset and one architecture, not semantic understanding, general intelligence, architecture generation, or reduced pretraining requirements.",
            "Single-channel ablation importance is intervention-dependent and does not capture channel interactions or global causal structure.",
            "Signed-effect correlation is secondary and may be unstable for channels with near-zero effects.",
            "Random-mapping empirical probabilities are permutation controls, not stand-alone population-level significance tests.",
        ],
        "elapsed_seconds": float(time.time() - started),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or markdown_output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite report created during this run: {output_path}"
        )
    lines = [
        "# MetaTransfer v0.1 — Disjoint Channel-Importance Transfer",
        "",
        f"- Primary metric: **Top-{int(PRIMARY_FRACTION*100)}% precision**.",
        f"- Source score examples: **{len(source_score_indices)}**; target evaluation examples: **{len(target_eval_indices)}**.",
        f"- Random channel permutations per target/layer: **{args.random_mappings}**.",
        "",
        "## Primary results",
        "",
        "| Target | Layer | Same-index P@10% | Aligned P@10% | Random-map mean P@10% | Aligned Δ vs same |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for target, layer, result in all_rows:
        same_p = result["same_index_transfer"]["topk"][primary_key]["precision"]
        align_p = result["activation_aligned_transfer"]["topk"][primary_key]["precision"]
        rand_p = result["random_mapping_control"]["precision_mean"]
        lines.append(f"| {target} | `{layer}` | {same_p:.3f} | {align_p:.3f} | {rand_p:.3f} | {align_p-same_p:+.3f} |")
    lines.extend([
        "",
        "## Aggregate",
        "",
        f"- Mean aligned P@10%: **{report['aggregate_primary_metric']['mean_aligned_top_fraction_precision']:.4f}**",
        f"- Mean same-index P@10%: **{report['aggregate_primary_metric']['mean_same_index_top_fraction_precision']:.4f}**",
        f"- Mean random-map P@10%: **{report['aggregate_primary_metric']['mean_random_mapping_top_fraction_precision']:.4f}**",
        f"- Aligned beats same-index: **{report['aggregate_primary_metric']['aligned_beats_same_index_comparisons']}/{len(all_rows)}** comparisons",
        f"- Target seeds with positive layer-averaged delta: **{report['aggregate_primary_metric']['number_of_target_seeds_with_positive_mean_delta']}/3**",
        "",
        "## Limitations",
        "",
    ])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    markdown_output_path.write_text("\n".join(lines), encoding="utf-8")
    output_path.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8")

    print(f"\nJSON report saved: {output_path.relative_to(ROOT)}")
    print(f"Markdown report saved: {markdown_output_path.relative_to(ROOT)}")
    print(f"Elapsed: {report['elapsed_seconds']:.1f} seconds")
    print("No training or official CIFAR-10 test-set evaluation was performed.")


if __name__ == "__main__":
    main()
