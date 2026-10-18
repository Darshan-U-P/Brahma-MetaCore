import argparse, json, math
from datetime import datetime, timezone
from pathlib import Path

REPORTS = [
    ("v0.1_seed42", Path("reports/parameter_intervention_v01/evaluation_report.json")),
    ("v0.2_seed2027", Path("reports/parameter_intervention_v02_seed2027/evaluation_report.json")),
    ("v0.3_seed31415", Path("reports/parameter_intervention_v03_seed31415/evaluation_report.json")),
]
DEFAULT_OUTPUT = Path("reports/parameter_intervention_synthesis_v01")

def fmt(x):
    if x is None: return "n/a"
    try:
        x = float(x)
        return f"{x:.6g}" if math.isfinite(x) else "n/a"
    except (TypeError, ValueError): return str(x)

def main():
    ap = argparse.ArgumentParser(description="Read-only synthesis of MetaCore Parameter Intervention v0.1-v0.3 reports.")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists(): raise SystemExit(f"STOP: output directory already exists: {out}. Choose a new --output path; no files overwritten.")
    loaded, missing = [], []
    for label, path in REPORTS:
        if not path.is_file(): missing.append(str(path)); continue
        with path.open("r", encoding="utf-8-sig") as f: report = json.load(f)
        layers = report.get("metrics_by_layer")
        if not isinstance(layers, dict) or not layers: raise SystemExit(f"STOP: metrics_by_layer missing or invalid in {path}")
        loaded.append({"label": label, "path": str(path), "report": report, "layers": layers})
    if missing: raise SystemExit("STOP: required reports missing:\n" + "\n".join(missing))
    if len(loaded) != 3: raise SystemExit(f"STOP: expected 3 reports, loaded {len(loaded)}")
    layer_names = sorted(set().union(*(set(x["layers"]) for x in loaded)))
    result_rows, boot_rows = [], []
    for run in loaded:
        for layer in layer_names:
            obj = run["layers"].get(layer)
            if obj is None: continue
            methods = obj.get("abs_delta_loss", {}).get("methods", {})
            row = {"run": run["label"], "layer": layer,
                "checkpoint": run["report"].get("source_checkpoint"),
                "checkpoint_sha256": run["report"].get("source_checkpoint_sha256"),
                "n": methods.get("ridge", {}).get("n"),
                "mae": {m: methods.get(m, {}).get("mae") for m in ("no_change_zero", "training_mean", "ridge", "random_forest")},
                "rmse": {m: methods.get(m, {}).get("rmse") for m in ("no_change_zero", "training_mean", "ridge", "random_forest")},
                "spearman": {m: methods.get(m, {}).get("spearman") for m in ("no_change_zero", "training_mean", "ridge", "random_forest")}, "bootstrap": {}}
            boots = obj.get("abs_effect_mae_bootstrap_comparisons") or obj.get("abs_delta_loss_bootstrap_comparisons") or {}
            for method in ("ridge", "random_forest"):
                for baseline in ("no_change_zero", "training_mean"):
                    key = f"{method}_vs_{baseline}"; b = boots.get(key)
                    if b:
                        lo, hi = b.get("ci95_low"), b.get("ci95_high")
                        passed = lo is not None and hi is not None and float(lo) > 0 and float(hi) > 0
                        entry = {"mean_improvement": b.get("mean_improvement"), "ci95_low": lo, "ci95_high": hi, "n": b.get("n"), "bootstrap_replicates": b.get("bootstrap_replicates"), "criterion_ci_entirely_above_zero": passed}
                    else: entry = {"available": False, "criterion_ci_entirely_above_zero": None}
                    row["bootstrap"][key] = entry
                    boot_rows.append({"run": run["label"], "layer": layer, "comparison": key, **entry})
            result_rows.append(row)
    consistency = []
    for layer in layer_names:
        for comp in ("ridge_vs_no_change_zero", "ridge_vs_training_mean", "random_forest_vs_no_change_zero", "random_forest_vs_training_mean"):
            rows = [r for r in boot_rows if r["layer"] == layer and r["comparison"] == comp and r.get("criterion_ci_entirely_above_zero") is not None]
            consistency.append({"layer": layer, "comparison": comp, "available_runs": len(rows), "positive_runs": sum(r["criterion_ci_entirely_above_zero"] for r in rows), "positive_run_labels": [r["run"] for r in rows if r["criterion_ci_entirely_above_zero"]], "interpretation": "Descriptive count only; not a pooled significance test."})
    limitations = [
        "This synthesis reads existing reports only; no interventions were rerun.",
        "Channel bootstrap intervals quantify within-checkpoint uncertainty, not model-level uncertainty.",
        "Channels within a model are dependent; counts across runs are descriptive and are not pooled significance tests.",
        "The checkpoints share one architecture family and CIFAR-10.1 v6 dataset; cross-architecture generalization is untested.",
        "The endpoint is prediction of direct convolution-filter zeroing effects, not model reconstruction or training-efficiency improvement.",
        "No multiplicity correction is applied across predictors, layers, and baselines."]
    summary = {"experiment": "Brahma MetaCore Parameter Intervention cross-checkpoint synthesis v0.1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "completed",
        "source_reports": [{"label": x["label"], "path": x["path"], "source_checkpoint": x["report"].get("source_checkpoint"), "source_checkpoint_sha256": x["report"].get("source_checkpoint_sha256"), "status": x["report"].get("status"), "official_cifar10_test_set_used": x["report"].get("official_cifar10_test_set_used"), "dataset_hashes": x["report"].get("dataset_hashes")} for x in loaded],
        "layers": layer_names, "per_checkpoint_layer_results": result_rows, "bootstrap_comparisons": boot_rows, "descriptive_consistency_counts": consistency, "limitations": limitations}
    out.mkdir(parents=True, exist_ok=False)
    json_path, md_path = out / "cross_checkpoint_synthesis.json", out / "cross_checkpoint_synthesis.md"
    with json_path.open("x", encoding="utf-8") as f: json.dump(summary, f, indent=2, ensure_ascii=False); f.write("\n")
    lines = ["# Brahma MetaCore Parameter Intervention — Cross-checkpoint Synthesis v0.1", "", "**Status:** Completed  ", f"**Generated (UTC):** {summary['created_at_utc']}  ", f"**Source runs:** {', '.join(x['label'] for x in loaded)}", "", "## Scope", "", "This report synthesizes existing JSON results only. It does not rerun interventions or modify original artifacts.", "", "## Checkpoint provenance", "", "| Run | Checkpoint | SHA-256 | Official CIFAR-10 test used? |", "|---|---|---|---|"]
    for x in loaded:
        r=x["report"]; lines.append(f"| {x['label']} | `{r.get('source_checkpoint')}` | `{r.get('source_checkpoint_sha256')}` | {r.get('official_cifar10_test_set_used')} |")
    lines += ["", "## Absolute-effect MAE by checkpoint and layer", "", "Lower MAE is better. Values are not pooled across checkpoints.", "", "| Run | Layer | N | Zero MAE | Mean MAE | Ridge MAE | Random Forest MAE |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in result_rows:
        m=r["mae"]; lines.append(f"| {r['run']} | `{r['layer']}` | {r['n']} | {fmt(m['no_change_zero'])} | {fmt(m['training_mean'])} | {fmt(m['ridge'])} | {fmt(m['random_forest'])} |")
    lines += ["", "## Bootstrap comparisons", "", "Positive improvement favors the learned predictor. Criterion is met when the full 95% within-checkpoint channel-bootstrap interval is above zero.", "", "| Run | Layer | Comparison | Improvement | 95% CI | Criterion |", "|---|---|---|---:|---|---|"]
    for r in boot_rows:
        passed=r.get("criterion_ci_entirely_above_zero")
        if passed is None: continue
        lines.append(f"| {r['run']} | `{r['layer']}` | `{r['comparison']}` | {fmt(r.get('mean_improvement'))} | [{fmt(r.get('ci95_low'))}, {fmt(r.get('ci95_high'))}] | {'Met' if passed else 'Not met'} |")
    lines += ["", "## Descriptive consistency counts", "", "These counts are not pooled significance tests and do not make channels independent across models.", "", "| Layer | Comparison | Available runs | Positive runs | Positive runs |", "|---|---|---:|---:|---|"]
    for r in consistency: lines.append(f"| `{r['layer']}` | `{r['comparison']}` | {r['available_runs']} | {r['positive_runs']} | {', '.join(r['positive_run_labels']) or 'None'} |")
    lines += ["", "## Interpretation", "", "Repeated within-checkpoint advantages would support a narrow predictive signal in model-map features for absolute filter-zeroing effects. They do not demonstrate reconstruction, capability preservation, or reduced training data/compute.", "", "## Limitations", ""] + [f"- {x}" for x in limitations] + ["", "## Source integrity", "", "Original reports, checkpoints, predictions, datasets, and experiment scripts were read-only inputs. Only the two new files in this output directory were created.", ""]
    with md_path.open("x", encoding="utf-8") as f: f.write("\n".join(lines))
    print(f"Synthesis JSON saved: {json_path}"); print(f"Synthesis Markdown saved: {md_path}"); print(f"Runs loaded: {len(loaded)}; layers: {', '.join(layer_names)}; bootstrap comparisons: {len(boot_rows)}"); print("Original reports and checkpoints were not modified.")
    return 0

if __name__ == "__main__": raise SystemExit(main())
