r"""
Brahma MetaCore — MetaAnalyzer v0.7 robust validation
====================================================

Reads the completed v0.6 JSON report and its cached activation signatures /
channel-ablation effects. It does NOT retrain models or rerun channel ablations.

Outputs (new files only):
  reports/milestone2_v07/validation_report.json
  reports/milestone2_v07/validation_report.md

Run from the repository root:
  & "D:\\all project\\venv_cuda\\Scripts\python.exe" -m src.experiments.run_metaanalyzer_v07

Optional:
  --stability-repeats 50
  --subset-fractions 0.5 0.75
  --bootstrap-repeats 2000
  --random-mappings 500
  --output reports/milestone2_v07/validation_report.json

Important interpretation:
- Nine target-layer comparisons are summarized, but are not nine independent
  model-level replications: layers and channels are dependent.
- Channel bootstrap intervals are descriptive sensitivity intervals, not
  population-level confidence intervals. There are only three target seeds.
- Stability subsets are drawn from the existing 2,500-image feature subset.
  This tests sensitivity to the selected examples, not independent dataset
  replication.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "reports" / "milestone2_v06" / "cache"
V06_REPORT = ROOT / "reports" / "milestone2_v06" / "channel_alignment.json"
DEFAULT_OUTPUT = ROOT / "reports" / "milestone2_v07" / "validation_report.json"
LAYERS = ("layer2.1.bn1", "layer3.1.bn1", "layer4.1.bn1")
TARGETS = ("seed123", "seed456", "seed789")
SOURCE = "seed42"
RUN_TAG = "full_f2500_i2500"


def read_payload(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required v0.6 cache not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "payload_json" not in archive:
            raise ValueError(f"Cache has no payload_json field: {path}")
        return json.loads(str(archive["payload_json"].item()))


def cache_path(model: str, kind: str) -> Path:
    return CACHE_DIR / f"{RUN_TAG}_{model}_{kind}.npz"


def signatures_for(model: str) -> dict[str, np.ndarray]:
    raw = read_payload(cache_path(model, "signatures"))
    return {layer: np.asarray(raw[layer], dtype=np.float64) for layer in LAYERS}


def effects_for(model: str) -> dict[str, Any]:
    raw = read_payload(cache_path(model, "effects"))
    if not isinstance(raw, dict) or "effects" not in raw:
        raise ValueError(f"Unexpected effects cache structure for {model}")
    return raw["effects"]


def standardize_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, keepdims=True)
    return np.divide(centered, std, out=np.zeros_like(centered), where=std > 1e-12)


def correlation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError(f"Expected same-shaped N x C arrays, got {a.shape}, {b.shape}")
    return np.nan_to_num(
        (standardize_columns(a).T @ standardize_columns(b)) / max(1, a.shape[0]),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


def hungarian_map(corr: np.ndarray) -> np.ndarray:
    rows, cols = linear_sum_assignment(-np.abs(corr))
    mapping = np.empty(corr.shape[0], dtype=np.int64)
    mapping[rows] = cols
    return mapping


def spearman_safe(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return None
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else None


def rank_metrics(actual: np.ndarray, scores: np.ndarray, k: int = 20) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(actual)
    if n == 0:
        return {"n": 0, "spearman": None, "top_k": 0, "top_k_precision": None}
    k = min(max(1, int(k)), n)
    true_top = set(np.argsort(actual)[-k:].tolist())
    pred_top = set(np.argsort(scores)[-k:].tolist())
    return {
        "n": int(n),
        "spearman": spearman_safe(actual, scores),
        "top_k": int(k),
        "top_k_overlap": int(len(true_top & pred_top)),
        "top_k_precision": float(len(true_top & pred_top) / k),
        "random_expected_top_k_precision": float(k / n),
    }


def percentile_interval(values: list[float] | np.ndarray, alpha: float = 0.05):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"lower": None, "median": None, "upper": None, "n": 0}
    return {
        "lower": float(np.quantile(values, alpha / 2)),
        "median": float(np.quantile(values, 0.5)),
        "upper": float(np.quantile(values, 1 - alpha / 2)),
        "n": int(len(values)),
    }


def bootstrap_metric_differences(
    actual_same: np.ndarray,
    actual_aligned: np.ndarray,
    source_scores: np.ndarray,
    k: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """
    Resample channel rows with replacement. This is intentionally labelled
    channel-level/descriptive because channel observations are dependent.
    """
    n = len(source_scores)
    if n < 3:
        return {"method": "paired channel bootstrap", "n": n, "warning": "too few channels"}
    rng = np.random.default_rng(seed)
    rho_delta, precision_delta = [], []
    for _ in range(repeats):
        ids = rng.integers(0, n, size=n)
        sm = rank_metrics(actual_same[ids], source_scores[ids], k)
        am = rank_metrics(actual_aligned[ids], source_scores[ids], k)
        if sm["spearman"] is not None and am["spearman"] is not None:
            rho_delta.append(am["spearman"] - sm["spearman"])
        if sm["top_k_precision"] is not None and am["top_k_precision"] is not None:
            precision_delta.append(am["top_k_precision"] - sm["top_k_precision"])
    return {
        "method": "paired channel-row bootstrap; descriptive only because channels are dependent",
        "repeats": int(repeats),
        "spearman_delta_aligned_minus_same": percentile_interval(rho_delta),
        "top_k_precision_delta_aligned_minus_same": percentile_interval(precision_delta),
        "interpretation_note": (
            "Intervals quantify sensitivity to channel resampling within this fixed model/layer; "
            "they are not model-seed-level confidence intervals."
        ),
    }


def random_mapping_reference(
    source_scores: np.ndarray,
    target_scores: np.ndarray,
    k: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """
    Compare source importance scores with target importance reindexed by a
    random one-to-one mapping. This is the correct random-mapping comparator
    for alignment; it differs from a random ranking of target labels.
    """
    n = len(source_scores)
    rng = np.random.default_rng(seed)
    rhos, precisions = [], []
    for _ in range(repeats):
        perm = rng.permutation(n)
        metrics = rank_metrics(target_scores[perm], source_scores, k)
        if metrics["spearman"] is not None:
            rhos.append(metrics["spearman"])
        precisions.append(metrics["top_k_precision"])
    return {
        "repeats": int(repeats),
        "spearman": percentile_interval(rhos),
        "top_k_precision": percentile_interval(precisions),
        "expected_top_k_precision_under_random_mapping": float(k / n) if n else None,
    }


def get_effect_vector(effects: dict[str, Any], layer: str, width: int) -> np.ndarray:
    values = []
    for channel in range(width):
        key = f"{layer}|{channel}"
        if key not in effects:
            raise KeyError(f"Missing channel effect: {key}")
        values.append(float(effects[key]["abs_delta_loss"]))
    return np.asarray(values, dtype=np.float64)


def summarize_existing_nine(v06: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for target in TARGETS:
        for layer in LAYERS:
            item = v06["target_results"][target]["layers"][layer]
            same = item["same_index_importance_transfer"]
            aligned = item["activation_aligned_importance_transfer"]
            rows.append({
                "target": target,
                "layer": layer,
                "channels": item.get("channels_in_layer"),
                "paired_channels": item.get("paired_channels_evaluated"),
                "matched_abs_corr": item["alignment"].get("mean_abs_matched_correlation"),
                "same_index_abs_corr": item["alignment"].get("mean_abs_same_index_correlation"),
                "same_spearman": same.get("spearman"),
                "aligned_spearman": aligned.get("spearman"),
                "spearman_delta": (
                    aligned.get("spearman") - same.get("spearman")
                    if aligned.get("spearman") is not None and same.get("spearman") is not None else None
                ),
                "same_top_k_precision": same.get("top_k_precision"),
                "aligned_top_k_precision": aligned.get("top_k_precision"),
                "top_k_precision_delta": (
                    aligned.get("top_k_precision") - same.get("top_k_precision")
                    if aligned.get("top_k_precision") is not None and same.get("top_k_precision") is not None else None
                ),
                "same_mae": same.get("mae"),
                "aligned_mae": aligned.get("mae"),
                "same_rmse": same.get("rmse"),
                "aligned_rmse": aligned.get("rmse"),
            })
    return {"n_comparisons": len(rows), "rows": rows}


def run_stability(
    signatures: dict[str, dict[str, np.ndarray]],
    repeats: int,
    fractions: list[float],
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    results = []
    for target in TARGETS:
        for layer in LAYERS:
            src = signatures[SOURCE][layer]
            tgt = signatures[target][layer]
            full_map = hungarian_map(correlation_matrix(src, tgt))
            full_corr = correlation_matrix(src, tgt)
            width = src.shape[1]
            for fraction in fractions:
                subset_n = max(20, int(round(len(src) * fraction)))
                if subset_n >= len(src):
                    continue
                agreements, matched_corrs, matched_corr_deltas = [], [], []
                for rep in range(repeats):
                    ids = np.sort(rng.choice(len(src), size=subset_n, replace=False))
                    sub_corr = correlation_matrix(src[ids], tgt[ids])
                    sub_map = hungarian_map(sub_corr)
                    agreements.append(float(np.mean(sub_map == full_map)))
                    matched_corrs.append(float(np.mean(np.abs(sub_corr[np.arange(width), sub_map]))))
                    matched_corr_deltas.append(
                        float(np.mean(np.abs(sub_corr[np.arange(width), sub_map])) -
                              np.mean(np.abs(np.diag(sub_corr))))
                    )
                results.append({
                    "target": target,
                    "layer": layer,
                    "subset_fraction": float(fraction),
                    "subset_examples": int(subset_n),
                    "full_feature_examples": int(len(src)),
                    "repeats": int(repeats),
                    "mapping_agreement_with_full_subset": percentile_interval(agreements),
                    "subset_mean_abs_matched_correlation": percentile_interval(matched_corrs),
                    "subset_matched_minus_same_index_correlation": percentile_interval(matched_corr_deltas),
                    "random_mapping_identity_agreement_reference": float(1 / width),
                })
    return {
        "definition": (
            "For each target/layer, compare Hungarian mappings learned on repeated random subsets "
            "of the existing feature examples against the mapping learned on all 2,500 examples. "
            "Mapping agreement is the fraction of source channels assigned to the same target channel."
        ),
        "repeats_per_target_layer_fraction": int(repeats),
        "fractions": fractions,
        "results": results,
        "limitations": [
            "Subsets are drawn from the same 2,500 feature examples; this is not an independent-data replication.",
            "Subset assignments are dependent and should be read as stability diagnostics, not independent experiments.",
            "Exact one-to-one assignments can be unstable among near-tied channels even when matched correlation remains high.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT.relative_to(ROOT)))
    parser.add_argument("--stability-repeats", type=int, default=50)
    parser.add_argument("--subset-fractions", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--random-mappings", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()

    if args.stability_repeats < 10:
        parser.error("--stability-repeats must be at least 10")
    if args.bootstrap_repeats < 100:
        parser.error("--bootstrap-repeats must be at least 100")
    if args.random_mappings < 100:
        parser.error("--random-mappings must be at least 100")
    if any(not 0 < x < 1 for x in args.subset_fractions):
        parser.error("Each --subset-fractions value must be strictly between 0 and 1")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output_path}")
    if not V06_REPORT.exists():
        raise FileNotFoundError(f"Missing completed v0.6 report: {V06_REPORT}")

    started = time.time()
    v06 = json.loads(V06_REPORT.read_text(encoding="utf-8"))
    if v06.get("pilot", False):
        raise ValueError("v0.6 report is marked pilot; v0.7 requires the full v0.6 run.")
    if len(TARGETS) * len(LAYERS) != 9:
        raise RuntimeError("Expected exactly nine target-layer comparisons.")

    signatures = {key: signatures_for(key) for key in (SOURCE, *TARGETS)}
    effects = {key: effects_for(key) for key in (SOURCE, *TARGETS)}

    # Ensure cache compatibility and ensure signatures are the full run's cached matrices.
    for model in signatures:
        for layer in LAYERS:
            if signatures[model][layer].ndim != 2 or signatures[model][layer].shape[0] != 2500:
                raise ValueError(
                    f"{model}/{layer} signature shape {signatures[model][layer].shape}; "
                    "expected (2500, channel_count). Ensure full_f2500_i2500 caches are present."
                )

    nine = summarize_existing_nine(v06)
    rows = nine["rows"]
    layer_summary = {}
    for layer in LAYERS:
        subset = [r for r in rows if r["layer"] == layer]
        rho_deltas = [r["spearman_delta"] for r in subset if r["spearman_delta"] is not None]
        p_deltas = [r["top_k_precision_delta"] for r in subset if r["top_k_precision_delta"] is not None]
        corr_deltas = [
            r["matched_abs_corr"] - r["same_index_abs_corr"]
            for r in subset
            if r["matched_abs_corr"] is not None and r["same_index_abs_corr"] is not None
        ]
        layer_summary[layer] = {
            "target_seed_comparisons": len(subset),
            "matched_minus_same_index_abs_correlation": {
                "values_by_target": corr_deltas,
                "mean": float(np.mean(corr_deltas)) if corr_deltas else None,
                "all_positive": bool(all(x > 0 for x in corr_deltas)) if corr_deltas else None,
            },
            "aligned_minus_same_spearman": {
                "values_by_target": rho_deltas,
                "mean": float(np.mean(rho_deltas)) if rho_deltas else None,
                "positive_comparisons": int(sum(x > 0 for x in rho_deltas)),
                "n": len(rho_deltas),
            },
            "aligned_minus_same_top_k_precision": {
                "values_by_target": p_deltas,
                "mean": float(np.mean(p_deltas)) if p_deltas else None,
                "positive_comparisons": int(sum(x > 0 for x in p_deltas)),
                "n": len(p_deltas),
            },
        }

    # Recompute channel-level metrics from the v0.6 cached effects for consistent
    # random-mapping baselines and channel-bootstrap sensitivity intervals.
    transfer_validation = []
    for target_i, target in enumerate(TARGETS):
        for layer_i, layer in enumerate(LAYERS):
            src_sig = signatures[SOURCE][layer]
            tgt_sig = signatures[target][layer]
            corr = correlation_matrix(src_sig, tgt_sig)
            mapping = hungarian_map(corr)
            width = src_sig.shape[1]
            source_scores = get_effect_vector(effects[SOURCE], layer, width)
            target_scores = get_effect_vector(effects[target], layer, width)
            k = min(20, width)
            same_metrics = rank_metrics(target_scores, source_scores, k)
            aligned_target_scores = target_scores[mapping]
            aligned_metrics = rank_metrics(aligned_target_scores, source_scores, k)

            # Bootstrap same/aligned as paired channel rows.
            boot = bootstrap_metric_differences(
                target_scores, aligned_target_scores, source_scores,
                k=k, repeats=args.bootstrap_repeats,
                seed=args.seed + 100 * target_i + layer_i,
            )
            random_ref = random_mapping_reference(
                source_scores, target_scores, k=k,
                repeats=args.random_mappings,
                seed=args.seed + 1000 + 100 * target_i + layer_i,
            )
            transfer_validation.append({
                "target": target,
                "layer": layer,
                "n_channels": int(width),
                "same_index": same_metrics,
                "activation_aligned": aligned_metrics,
                "aligned_minus_same_spearman": (
                    aligned_metrics["spearman"] - same_metrics["spearman"]
                    if aligned_metrics["spearman"] is not None and same_metrics["spearman"] is not None else None
                ),
                "aligned_minus_same_top_k_precision": (
                    aligned_metrics["top_k_precision"] - same_metrics["top_k_precision"]
                ),
                "random_one_to_one_mapping_reference": random_ref,
                "channel_bootstrap_sensitivity": boot,
                "mapping_same_index_fraction": float(np.mean(mapping == np.arange(width))),
                "mean_abs_matched_activation_correlation": float(np.mean(np.abs(corr[np.arange(width), mapping]))),
                "mean_abs_same_index_activation_correlation": float(np.mean(np.abs(np.diag(corr)))),
            })

    stability = run_stability(signatures, args.stability_repeats, args.subset_fractions, args.seed + 777)
    output = {
        "experiment": "Brahma MetaCore MetaAnalyzer v0.7",
        "purpose": "Robust validation of v0.6 activation-based channel alignment and importance correspondence",
        "status": "completed_by_script_when_run",
        "source_report": str(V06_REPORT.relative_to(ROOT)),
        "source_run_tag": RUN_TAG,
        "source_v06_report_sha256": hashlib.sha256(V06_REPORT.read_bytes()).hexdigest(),
        "cache_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for model in (SOURCE, *TARGETS)
            for kind in ("signatures", "effects")
            for path in [cache_path(model, kind)]
        },
        "seed": args.seed,
        "protocol": {
            "models": [SOURCE, *TARGETS],
            "layers": list(LAYERS),
            "v06_comparisons_aggregated": nine["n_comparisons"],
            "stability_feature_examples": 2500,
            "stability_subsets": "repeated random subsets of the same existing 2,500-image feature set",
            "intervention_labels": "existing v0.6 effects cache; no interventions rerun",
            "official_test_set_used": False,
            "v06_artifacts_modified": False,
        },
        "aggregate_nine_comparisons": nine,
        "per_layer_summary": layer_summary,
        "importance_transfer_reanalysis": transfer_validation,
        "activation_matching_stability": stability,
        "overall_interpretation": [
            "Activation matching should be judged separately from importance transfer: high matched activation correlation does not guarantee useful importance transfer.",
            "The nine target-layer comparisons are clustered within three target models and three layers; they are not nine independent model-level replications.",
            "Random one-to-one mapping is the primary random control for correspondence. The expected top-k overlap is k divided by the layer width.",
            "Channel-bootstrap intervals are descriptive because channels within a layer are dependent.",
            "Subset stability is conditional on the original 2,500 feature examples and does not establish stability on a new dataset sample.",
            "No evidence here establishes semantic understanding, general intelligence, model construction, or reduced pretraining requirements.",
        ],
        "elapsed_seconds": float(time.time() - started),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output_path}")
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")

    md_path = output_path.with_suffix(".md")
    if md_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing Markdown report: {md_path}")

    lines = [
        "# MetaAnalyzer v0.7 — Robust Validation",
        "",
        f"- Source report: `{output['source_report']}`",
        f"- v0.6 comparisons aggregated: **{nine['n_comparisons']}**",
        f"- Feature signatures: **2,500 images/model**",
        f"- Subset repeats per target/layer/fraction: **{args.stability_repeats}**",
        f"- Random mappings per target/layer: **{args.random_mappings}**",
        f"- Channel-bootstrap repeats: **{args.bootstrap_repeats}**",
        "",
        "## Layer summary",
        "",
        "| Layer | Mean Δ matched-vs-same activation correlation | Importance Spearman Δ positive | Top-k precision Δ positive |",
        "|---|---:|---:|---:|",
    ]
    for layer in LAYERS:
        s = layer_summary[layer]
        corr_mean = s["matched_minus_same_index_abs_correlation"]["mean"]
        rho = s["aligned_minus_same_spearman"]
        prec = s["aligned_minus_same_top_k_precision"]
        lines.append(
            f"| `{layer}` | {corr_mean:.4f} | {rho['positive_comparisons']}/{rho['n']} (mean {rho['mean']:.4f}) | "
            f"{prec['positive_comparisons']}/{prec['n']} (mean {prec['mean']:.4f}) |"
        )
    lines += [
        "",
        "## Nine-comparison table",
        "",
        "| Target | Layer | Matched abs corr | Same-index abs corr | Δ Spearman | Same top-k P | Aligned top-k P | Δ top-k P |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        def fmt(v, digits=3):
            return "NA" if v is None else f"{v:.{digits}f}"
        lines.append(
            f"| {r['target']} | `{r['layer']}` | {fmt(r['matched_abs_corr'])} | {fmt(r['same_index_abs_corr'])} | "
            f"{fmt(r['spearman_delta'])} | {fmt(r['same_top_k_precision'])} | {fmt(r['aligned_top_k_precision'])} | "
            f"{fmt(r['top_k_precision_delta'])} |"
        )
    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "- Activation correspondence and importance transfer are different outcomes.",
        "- Treat channel bootstrap intervals as within-layer sensitivity diagnostics, not population-level confidence intervals.",
        "- The three target seeds, not the nine layer rows, are the key independent replication units available here.",
        "- Matching stability is evaluated on subsets of the existing feature examples; no new images are introduced.",
        "- The official CIFAR-10 test set is not used by this script.",
        "- This analysis does not demonstrate semantic understanding, model construction, or reduced pretraining requirements.",
        "",
        "See the JSON report for per-target bootstrap intervals, random one-to-one mapping baselines, subset stability distributions, and artifact hashes.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON report: {output_path}")
    print(f"Markdown report: {md_path}")
    print(f"Aggregated comparisons: {nine['n_comparisons']}")
    print(f"Elapsed: {output['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
