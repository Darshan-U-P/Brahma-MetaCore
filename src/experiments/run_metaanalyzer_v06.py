"""MetaAnalyzer v0.6: activation-based channel alignment across trained models.

Run from the Brahma-MetaCore project root:

    python -m src.experiments.run_metaanalyzer_v06 --pilot
    python -m src.experiments.run_metaanalyzer_v06
    python -m src.experiments.run_metaanalyzer_v06 --output reports/milestone2_v06/channel_alignment_repeat1.json

The pilot uses 64 feature images, 64 intervention images, and 8 evenly spaced
channels per layer. It validates the pipeline; it is not a scientific result.
The full run uses disjoint feature/intervention subsets and measures every
channel in the three selected BatchNorm layers for four checkpoints.

No official CIFAR-10 test examples are loaded.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.experiments import run_metaanalyzer_v04 as v04
from src.experiments import run_channel_ablation_v02 as ablation


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "cifar10"
SPLIT_FILE = ROOT / "checkpoints" / "milestone2_v02" / "split_indices_v02.npz"
DEFAULT_OUTPUT = ROOT / "reports" / "milestone2_v06" / "channel_alignment.json"
CACHE_DIR = ROOT / "reports" / "milestone2_v06" / "cache"

LAYERS = ("layer2.1.bn1", "layer3.1.bn1", "layer4.1.bn1")
FEATURE_NAMES = (
    "activation_mean",
    "activation_std",
    "activation_mean_abs",
    "activation_near_zero_fraction",
    "bn_gamma_abs",
    "bn_beta_abs",
    "incoming_weight_norm",
    "outgoing_weight_norm",
)
SEED = 20261015
BATCH_SIZE = 128

CHECKPOINTS = {
    "seed42": ROOT / "checkpoints" / "milestone2_v02" / "best_validation.pt",
    "seed123": ROOT / "checkpoints" / "milestone2_v05_target" / "best_validation.pt",
    "seed456": ROOT / "checkpoints" / "milestone2_v06_seed456" / "best_validation.pt",
    "seed789": ROOT / "checkpoints" / "milestone2_v06_seed789" / "best_validation.pt",
}


def resolve_from_root(path_string: str) -> Path:
    path = Path(path_string)
    return path if path.is_absolute() else ROOT / path


def json_safe(value: Any) -> Any:
    """Convert nested NumPy/Python values to JSON-compatible values."""
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


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Unexpected checkpoint structure in {checkpoint_path}; "
            "expected a 'model_state_dict' entry."
        )
    model = ablation.build_model()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def make_loader(base_dataset, indices: np.ndarray, device: torch.device):
    dataset = v04.IndexedCIFAR10(
        base_dataset,
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
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


@torch.inference_mode()
def extract_signatures(
    model, loader: DataLoader, device: torch.device
) -> dict[str, np.ndarray]:
    """Return N x C per-image signatures from spatially pooled BN outputs."""
    modules = {
        layer: ablation.get_module_by_name(model, layer)
        for layer in LAYERS
    }
    chunks: dict[str, list[np.ndarray]] = {layer: [] for layer in LAYERS}
    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output) or output.ndim != 4:
                raise RuntimeError(
                    f"Expected a 4-D activation at {layer}, "
                    f"received {type(output)} / "
                    f"{getattr(output, 'shape', None)}"
                )
            # Preserve image identity; average only spatial dimensions.
            pooled = output.detach().float().mean(dim=(2, 3)).cpu().numpy()
            chunks[layer].append(pooled)
        return hook

    for layer, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(layer)))

    try:
        for images, _labels in loader:
            model(images.to(device, non_blocking=True))
    finally:
        for handle in handles:
            handle.remove()

    signatures = {}
    for layer, layer_chunks in chunks.items():
        if not layer_chunks:
            raise RuntimeError(f"No activation signatures captured for {layer}.")
        signatures[layer] = np.concatenate(layer_chunks, axis=0)
    return signatures


def extract_feature_map(model, loader, device) -> dict[str, dict[str, float]]:
    """Return aggregate channel features keyed by 'layer|channel'."""
    extracted = v04.extract_features(model, loader, device)
    result: dict[str, dict[str, float]] = {}
    for layer, rows in extracted.items():
        for row in rows:
            key = f"{layer}|{int(row['channel'])}"
            result[key] = {
                name: float(row[name]) for name in FEATURE_NAMES
            }
    return result


def measure_channel_effects(
    model,
    loader: DataLoader,
    device: torch.device,
    model_key: str,
    channels_per_layer: int | None,
):
    """Measure absolute/signed loss changes from zeroing BN output channels."""
    baseline = ablation.evaluate_baseline(model, loader, device)
    print(
        f"{model_key} intervention baseline: "
        f"accuracy={baseline['accuracy'] * 100:.2f}%, "
        f"loss={baseline['loss']:.6f}, examples={baseline['examples']}"
    )

    effects: dict[str, dict[str, float | int]] = {}
    channel_plan = {}
    total_to_measure = 0

    for layer in LAYERS:
        width = int(ablation.get_module_by_name(model, layer).num_features)
        if channels_per_layer is None:
            ids = list(range(width))
        else:
            count = min(channels_per_layer, width)
            ids = np.linspace(0, width - 1, count, dtype=int).tolist()
        channel_plan[layer] = ids
        total_to_measure += len(ids)

    completed = 0
    for layer, channel_ids in channel_plan.items():
        module = ablation.get_module_by_name(model, layer)
        for channel in channel_ids:
            result = ablation.evaluate_with_channel_ablation(
                model=model,
                loader=loader,
                device=device,
                module=module,
                channel_index=int(channel),
                baseline_predictions=baseline["predictions"],
                baseline_batches=baseline["per_batch"],
            )
            if int(result["examples"]) != int(baseline["examples"]):
                raise RuntimeError(
                    f"{model_key} {layer} channel {channel}: "
                    "intervention example count differs from baseline."
                )
            if int(result["hook_calls"]) != len(loader):
                raise RuntimeError(
                    f"{model_key} {layer} channel {channel}: expected "
                    f"{len(loader)} hook calls, got {result['hook_calls']}."
                )

            delta = float(result["loss"] - baseline["loss"])
            effects[f"{layer}|{channel}"] = {
                "signed_delta_loss": delta,
                "abs_delta_loss": abs(delta),
                "hook_calls": int(result["hook_calls"]),
                "examples": int(result["examples"]),
            }
            completed += 1
            if completed % 32 == 0 or completed == total_to_measure:
                print(
                    f"  {model_key}: measured {completed}/{total_to_measure} "
                    "channel interventions"
                )

    baseline_summary = {
        "accuracy": float(baseline["accuracy"]),
        "loss": float(baseline["loss"]),
        "examples": int(baseline["examples"]),
    }
    return effects, baseline_summary


def cache_path(model_key: str, kind: str, run_tag: str) -> Path:
    return CACHE_DIR / f"{run_tag}_{model_key}_{kind}.npz"


def save_cache(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(json_safe(payload), separators=(",", ":"))
    np.savez_compressed(path, payload_json=np.asarray(encoded))


def read_cache(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        if "payload_json" not in archive:
            raise ValueError(f"Cache is missing payload_json: {path}")
        return json.loads(str(archive["payload_json"].item()))


def get_or_compute(
    model_key: str,
    kind: str,
    run_tag: str,
    compute_fn,
):
    path = cache_path(model_key, kind, run_tag)
    if path.exists():
        print(f"Using cache: {path.relative_to(ROOT)}")
        return read_cache(path)
    payload = compute_fn()
    save_cache(path, payload)
    print(f"Cached: {path.relative_to(ROOT)}")
    return payload


def standardize_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, keepdims=True)
    return np.divide(
        centered, std, out=np.zeros_like(centered), where=std > 1e-12
    )


def align_channels(
    source_signatures: np.ndarray,
    target_signatures: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return mapping[source_channel] = matched_target_channel.

    Hungarian assignment maximizes absolute Pearson correlation across the
    shared feature images. The mapping is constructed without intervention
    labels. Both models must have signatures for the same images in the same
    order and the same number of channels in this layer.
    """
    if source_signatures.ndim != 2 or target_signatures.ndim != 2:
        raise ValueError("Signatures must be two-dimensional N x C matrices.")
    if source_signatures.shape[0] != target_signatures.shape[0]:
        raise ValueError("Source/target signatures must use the same images.")
    if source_signatures.shape[1] != target_signatures.shape[1]:
        raise ValueError("This experiment expects equal channel widths per layer.")

    source = standardize_columns(source_signatures)
    target = standardize_columns(target_signatures)
    correlation = (source.T @ target) / max(source.shape[0], 1)
    correlation = np.nan_to_num(
        correlation, nan=0.0, posinf=0.0, neginf=0.0
    )

    source_ids, target_ids = linear_sum_assignment(-np.abs(correlation))
    mapping = np.empty(source_signatures.shape[1], dtype=np.int64)
    mapping[source_ids] = target_ids
    matched = correlation[source_ids, target_ids]

    return mapping, {
        "mean_abs_matched_correlation": float(np.mean(np.abs(matched))),
        "median_abs_matched_correlation": float(np.median(np.abs(matched))),
        "mean_signed_matched_correlation": float(np.mean(matched)),
        "mean_abs_same_index_correlation": float(
            np.mean(np.abs(np.diag(correlation)))
        ),
    }


def spearman_safe(actual: np.ndarray, scores: np.ndarray):
    if len(actual) < 2 or np.std(actual) == 0 or np.std(scores) == 0:
        return None
    value = float(spearmanr(actual, scores).statistic)
    return value if np.isfinite(value) else None


def ranking_metrics(
    actual: np.ndarray, scores: np.ndarray, k: int = 20
) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(actual) != len(scores):
        raise ValueError("Actual values and scores must have equal length.")
    n = len(actual)
    if n == 0:
        return {"n": 0, "spearman": None, "top_k": 0}

    k = min(max(1, int(k)), n)
    predicted_top = set(np.argsort(scores)[-k:].tolist())
    actual_top = set(np.argsort(actual)[-k:].tolist())
    return {
        "n": n,
        "spearman": spearman_safe(actual, scores),
        "mae": float(mean_absolute_error(actual, scores)),
        "rmse": float(np.sqrt(mean_squared_error(actual, scores))),
        "top_k": k,
        "top_k_overlap": len(predicted_top & actual_top),
        "top_k_precision": len(predicted_top & actual_top) / k,
        "random_expected_top_k_precision": k / n,
    }


def random_ranking_reference(
    actual: np.ndarray, k: int, repetitions: int, seed: int
) -> dict[str, Any]:
    """Monte Carlo reference; each random ranking is independent of actual labels."""
    actual = np.asarray(actual, dtype=np.float64)
    n = len(actual)
    if n == 0:
        return {
            "n_random_rankings": repetitions,
            "expected_top_k_precision": None,
            "monte_carlo_mean_top_k_precision": None,
            "monte_carlo_std_top_k_precision": None,
        }

    k = min(max(1, int(k)), n)
    actual_top = set(np.argsort(actual)[-k:].tolist())
    rng = np.random.default_rng(seed)
    precisions = np.empty(repetitions, dtype=np.float64)
    for i in range(repetitions):
        random_scores = rng.random(n)
        random_top = set(np.argsort(random_scores)[-k:].tolist())
        precisions[i] = len(random_top & actual_top) / k
    return {
        "n_random_rankings": int(repetitions),
        "expected_top_k_precision": float(k / n),
        "monte_carlo_mean_top_k_precision": float(precisions.mean()),
        "monte_carlo_std_top_k_precision": float(precisions.std(ddof=1)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Quick wiring test: 64 feature images, 64 intervention images, "
             "and 8 channels per layer.",
    )
    parser.add_argument(
        "--feature-examples", type=int, default=2500,
        help="Number of validation images used for activation/features (default: 2500).",
    )
    parser.add_argument(
        "--intervention-examples", type=int, default=2500,
        help="Number of disjoint validation images used for ablations (default: 2500).",
    )
    parser.add_argument(
        "--random-rankings", type=int, default=1000,
        help="Number of Monte Carlo random rankings per target/layer (minimum 100).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path. Existing files are never overwritten. "
             "Defaults to channel_alignment_pilot.json in pilot mode and "
             "channel_alignment.json in full mode.",
    )
    parser.add_argument(
        "--recompute-cache", action="store_true",
        help="Recompute cache entries for this run tag instead of reusing them.",
    )
    args = parser.parse_args()

    if args.feature_examples < 2 or args.intervention_examples < 2:
        parser.error("Feature and intervention subsets must each contain >= 2 images.")
    if args.random_rankings < 100:
        parser.error("--random-rankings must be at least 100.")

    if args.output is None:
        output_path = (
            DEFAULT_OUTPUT.with_name("channel_alignment_pilot.json")
            if args.pilot else DEFAULT_OUTPUT
        )
    else:
        output_path = resolve_from_root(args.output)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report: {output_path}. "
            "Choose a new path with --output."
        )

    start_time = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("MetaAnalyzer v0.6 — activation-based channel alignment")
    print(f"Device: {device}")
    print(f"Pilot mode: {args.pilot}")

    for model_key, checkpoint_path in CHECKPOINTS.items():
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for {model_key}: {checkpoint_path}"
            )
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Missing validation split: {SPLIT_FILE}")
    if not (DATA_ROOT / "cifar-10-batches-py").exists():
        raise FileNotFoundError(f"Missing CIFAR-10 data under {DATA_ROOT}")

    with np.load(SPLIT_FILE) as split:
        if "val_indices" not in split:
            raise KeyError(f"{SPLIT_FILE} does not contain val_indices.")
        validation_indices = split["val_indices"].copy()

    if len(validation_indices) < 4:
        raise RuntimeError("Validation split is too small for disjoint subsets.")

    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(validation_indices)

    if args.pilot:
        n_feature = min(64, args.feature_examples)
        n_intervention = min(64, args.intervention_examples)
        channels_per_layer = 8
        run_tag = "pilot_f64_i64_c8"
    else:
        n_feature = args.feature_examples
        n_intervention = args.intervention_examples
        channels_per_layer = None
        run_tag = f"full_f{n_feature}_i{n_intervention}"

    if n_feature + n_intervention > len(shuffled):
        raise ValueError(
            f"Requested {n_feature + n_intervention} images, but only "
            f"{len(shuffled)} validation indices are available."
        )

    feature_indices = shuffled[:n_feature]
    intervention_indices = shuffled[n_feature:n_feature + n_intervention]
    if np.intersect1d(feature_indices, intervention_indices).size:
        raise RuntimeError("Feature and intervention subsets overlap.")
    if len(np.unique(feature_indices)) != len(feature_indices):
        raise RuntimeError("Feature subset contains duplicate indices.")
    if len(np.unique(intervention_indices)) != len(intervention_indices):
        raise RuntimeError("Intervention subset contains duplicate indices.")

    print(
        f"Feature examples: {len(feature_indices)}; "
        f"intervention examples: {len(intervention_indices)}"
    )
    print("Official CIFAR-10 test set: NOT loaded.")

    base_dataset = datasets.CIFAR10(
        root=str(DATA_ROOT), train=True, download=False, transform=None
    )
    feature_loader = make_loader(base_dataset, feature_indices, device)
    intervention_loader = make_loader(base_dataset, intervention_indices, device)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Process one model at a time to limit GPU memory use.
    model_payloads: dict[str, dict[str, Any]] = {}
    for model_key, checkpoint_path in CHECKPOINTS.items():
        model, checkpoint = load_model(checkpoint_path, device)
        epoch = int(checkpoint.get("epoch", -1))
        print(f"\n[{model_key}] checkpoint epoch={epoch}")

        def get_or_compute_local(kind, compute_fn):
            path = cache_path(model_key, kind, run_tag)
            if path.exists() and not args.recompute_cache:
                print(f"Using cache: {path.relative_to(ROOT)}")
                return read_cache(path)
            payload = compute_fn()
            save_cache(path, payload)
            print(f"Cached: {path.relative_to(ROOT)}")
            return payload

        signature_payload = get_or_compute_local(
            "signatures",
            lambda: {
                layer: values.tolist()
                for layer, values in extract_signatures(
                    model, feature_loader, device
                ).items()
            },
        )
        feature_payload = get_or_compute_local(
            "features",
            lambda: extract_feature_map(model, feature_loader, device),
        )
        def compute_effect_payload():
            effects, baseline = measure_channel_effects(
                model,
                intervention_loader,
                device,
                model_key,
                channels_per_layer,
            )
            return {"effects": effects, "baseline": baseline}

        effect_payload = get_or_compute_local("effects", compute_effect_payload)

        # The effects cache stores the baseline together with the channel labels.
        # Validate it before use; the helper above is replaced below if legacy or
        # incomplete cache data is encountered.
        if (
            not isinstance(effect_payload, dict)
            or "effects" not in effect_payload
            or "baseline" not in effect_payload
            or not isinstance(effect_payload["effects"], dict)
            or not isinstance(effect_payload["baseline"], dict)
            or "loss" not in effect_payload["baseline"]
        ):
            raise ValueError(
                f"Invalid intervention cache for {model_key}. "
                "Use --recompute-cache to rebuild it."
            )

        signatures = {
            layer: np.asarray(values, dtype=np.float64)
            for layer, values in signature_payload.items()
        }
        for layer in LAYERS:
            expected_width = int(
                ablation.get_module_by_name(model, layer).num_features
            )
            if signatures[layer].shape != (n_feature, expected_width):
                raise ValueError(
                    f"{model_key} {layer} signature shape "
                    f"{signatures[layer].shape}; expected "
                    f"{(n_feature, expected_width)}. "
                    "Use --recompute-cache if the cache is stale."
                )

        model_payloads[model_key] = {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "epoch": epoch,
            "signatures": signatures,
            "features": feature_payload,
            "effects": effect_payload["effects"],
            "baseline": effect_payload["baseline"],
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # All four models must have identical feature-channel keys per layer.
    for layer in LAYERS:
        widths = {
            key: model_payloads[key]["signatures"][layer].shape[1]
            for key in model_payloads
        }
        if len(set(widths.values())) != 1:
            raise RuntimeError(f"Channel widths differ in {layer}: {widths}")

    report: dict[str, Any] = {
        "experiment": "Brahma MetaCore MetaAnalyzer v0.6",
        "description": (
            "Cross-model channel importance transfer using activation-derived "
            "Hungarian channel matching."
        ),
        "seed": SEED,
        "pilot": bool(args.pilot),
        "run_tag": run_tag,
        "feature_examples": int(n_feature),
        "intervention_examples": int(n_intervention),
        "feature_and_intervention_subsets_disjoint": True,
        "feature_indices_sha256": __import__("hashlib").sha256(
            np.asarray(feature_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "intervention_indices_sha256": __import__("hashlib").sha256(
            np.asarray(intervention_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "official_test_set_used": False,
        "layers": list(LAYERS),
        "feature_names": list(FEATURE_NAMES),
        "alignment_method": (
            "Hungarian assignment maximizing absolute Pearson correlation "
            "between per-image spatially pooled activation signatures."
        ),
        "importance_target": (
            "Absolute change in mean cross-entropy loss after zeroing one "
            "BatchNorm2d output channel on the intervention subset."
        ),
        "transfer_score_definition": (
            "Source model's measured absolute channel-ablation loss change "
            "is transferred either to the same target channel index or to "
            "the activation-matched target channel. Target intervention "
            "labels are used only for evaluation."
        ),
        "models": {},
        "target_results": {},
        "limitations": [
            "Pilot mode measures only eight evenly spaced channels per layer; its results are wiring checks, not scientific evidence.",
            "Activation matching is learned from the feature subset only; intervention labels are not used to create the mapping.",
            "Absolute correlation treats sign-inverted activation patterns as similar.",
            "Spatially pooled activations may hide spatially localized feature behavior.",
            "Models share architecture, dataset, and validation partition; results do not establish semantic understanding or general intelligence.",
            "Channel importance can depend on interactions with other channels; single-channel ablation is not a complete causal explanation.",
            "Channels are correlated observations, so channel-level metrics are descriptive and should not be treated as independent-sample statistical proof.",
        ],
    }

    for model_key, payload in model_payloads.items():
        report["models"][model_key] = {
            "checkpoint": payload["checkpoint"],
            "epoch": payload["epoch"],
            "intervention_baseline": payload["baseline"],
            "channels_with_effect_labels": len(payload["effects"]),
        }

    source_payload = model_payloads["seed42"]
    for target_key in ("seed123", "seed456", "seed789"):
        layer_results = {}
        for layer in LAYERS:
            source_sig = source_payload["signatures"][layer]
            target_sig = model_payloads[target_key]["signatures"][layer]
            mapping, alignment_stats = align_channels(source_sig, target_sig)

            source_effects = source_payload["effects"]
            target_effects = model_payloads[target_key]["effects"]
            width = source_sig.shape[1]

            # Build a paired comparison. Each source channel contributes only if
            # its same-index target channel AND its aligned target channel both
            # have measured intervention labels. This matters in pilot mode,
            # where only eight channels per layer are intervened upon.
            same_actual, same_scores = [], []
            aligned_actual, aligned_scores = [], []
            paired_source_channels = []

            for source_channel in range(width):
                source_key = f"{layer}|{source_channel}"
                same_target_key = f"{layer}|{source_channel}"
                aligned_target_channel = int(mapping[source_channel])
                aligned_target_key = f"{layer}|{aligned_target_channel}"

                if (
                    source_key not in source_effects
                    or same_target_key not in target_effects
                    or aligned_target_key not in target_effects
                ):
                    continue

                source_score = float(
                    source_effects[source_key]["abs_delta_loss"]
                )
                paired_source_channels.append(source_channel)
                same_actual.append(
                    float(target_effects[same_target_key]["abs_delta_loss"])
                )
                same_scores.append(source_score)
                aligned_actual.append(
                    float(target_effects[aligned_target_key]["abs_delta_loss"])
                )
                aligned_scores.append(source_score)

            same_actual_array = np.asarray(same_actual, dtype=np.float64)
            same_scores_array = np.asarray(same_scores, dtype=np.float64)
            aligned_actual_array = np.asarray(aligned_actual, dtype=np.float64)
            aligned_scores_array = np.asarray(aligned_scores, dtype=np.float64)

            k = min(20, len(same_actual_array))
            same_metrics = ranking_metrics(
                same_actual_array, same_scores_array, k=max(1, k)
            )
            aligned_metrics = ranking_metrics(
                aligned_actual_array, aligned_scores_array, k=max(1, k)
            )
            random_reference = random_ranking_reference(
                same_actual_array,
                k=max(1, k),
                repetitions=args.random_rankings,
                seed=SEED + len(target_key) + len(layer),
            )

            # Measure how much the alignment changes channel identity and compare
            # activation correlation before/after matching.
            identity_fraction = float(np.mean(mapping == np.arange(width)))
            layer_results[layer] = {
                "channels_in_layer": int(width),
                "paired_channels_evaluated": int(len(same_actual_array)),
                "alignment": {
                    **alignment_stats,
                    "fraction_matched_to_same_index": identity_fraction,
                },
                "same_index_importance_transfer": same_metrics,
                "activation_aligned_importance_transfer": aligned_metrics,
                "random_ranking_reference": random_reference,
                "paired_comparison_note": (
                    "Same-index and aligned rankings use the same source-channel "
                    "importance scores. In pilot mode, only source channels whose "
                    "same-index and aligned target channels both have intervention "
                    "labels are included."
                ),
            }

        report["target_results"][target_key] = {
            "source_model": "seed42",
            "layers": layer_results,
        }

    report["elapsed_seconds"] = float(time.time() - start_time)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Check again immediately before writing; never overwrite a prior result.
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite report created during this run: {output_path}"
        )
    output_path.write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nReport saved: {output_path}")
    print(f"Elapsed: {report['elapsed_seconds']:.1f} seconds")
    if args.pilot:
        print("Pilot complete. Inspect results, then run the full experiment.")
    else:
        print("Full experiment complete. Review the report and its limitations.")



if __name__ == "__main__":
    main()
