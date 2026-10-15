"""MetaTransfer v0.2: external-image confirmation on CIFAR-10.1 v6.

Run from the Brahma-MetaCore repository root:
    python -m src.experiments.run_metatransfer_v02 --download-data
    python -m src.experiments.run_metatransfer_v02 --audit-only
    python -m src.experiments.run_metatransfer_v02

Design:
- Freeze the v0.1 source importance scores and source->target mappings.
- Reuse the three existing target checkpoints (seeds 123, 456, 789).
- Measure target channel ablation effects on CIFAR-10.1 v6 (2,000 new images).
- Compare aligned transfer, same-index transfer, and randomized mappings.
- Do not retrain models, modify the frozen v0.1 report/cache, or load the
  official CIFAR-10 test set.

This is an external-image replication using the same three target checkpoints,
not a new-model-seed replication. If it succeeds, a subsequent milestone should
repeat on new target checkpoints before claiming robust generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.experiments import run_metaanalyzer_v06 as v06
from src.experiments import run_metatransfer_v01 as v01
from src.experiments import run_metaanalyzer_v04 as v04

ROOT = Path(__file__).resolve().parents[2]
V01_REPORT = ROOT / "reports" / "metatransfer_v01" / "transfer_report.json"
DATA_DIR = ROOT / "data" / "cifar10_1"
IMAGES_PATH = DATA_DIR / "cifar10.1_v6_data.npy"
LABELS_PATH = DATA_DIR / "cifar10.1_v6_labels.npy"
OUT_DIR = ROOT / "reports" / "metatransfer_v02_external"
CACHE_DIR = OUT_DIR / "cache"
DEFAULT_OUTPUT = OUT_DIR / "cifar10_1_v6_transfer_report.json"

TARGET_MODELS = ("seed123", "seed456", "seed789")
PRIMARY_FRACTION = 0.10
FRACTIONS = (0.05, 0.10, 0.20)
DEFAULT_RANDOM_MAPPINGS = 1000
DEFAULT_SEED = 20261017
BATCH_SIZE = 128
DEFAULT_CHANNELS_PER_FORWARD = 4
DATA_URLS = {
    "images": "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_data.npy",
    "labels": "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_labels.npy",
}
DATASET_SOURCE = "https://github.com/modestyachts/CIFAR-10.1"


class ArrayCIFAR10Dataset(Dataset):
    """CIFAR-10.1 NPY dataset using the same normalization as CIFAR-10 runs."""
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        if images.shape != (2000, 32, 32, 3):
            raise ValueError(f"Expected CIFAR-10.1 v6 images (2000,32,32,3), got {images.shape}")
        if labels.shape != (2000,):
            raise ValueError(f"Expected CIFAR-10.1 v6 labels (2000,), got {labels.shape}")
        if images.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image pixels, got {images.dtype}")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"Expected integer labels, got {labels.dtype}")
        if not np.isfinite(images).all() or labels.min() < 0 or labels.max() > 9:
            raise ValueError("CIFAR-10.1 data contains invalid pixels or labels")
        self.images = images
        self.labels = labels.astype(np.int64, copy=False)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(v04.MEAN, v04.STD),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.transform(np.array(self.images[index], copy=True))
        return image, int(self.labels[index])


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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def load_cache(path: Path) -> dict[str, Any]:
    return v01.read_payload(path)


def validate_dataset_files() -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    if not IMAGES_PATH.is_file() or not LABELS_PATH.is_file():
        raise FileNotFoundError(
            "CIFAR-10.1 v6 data files are missing. First run: "
            "python -m src.experiments.run_metatransfer_v02 --download-data"
        )
    images_hash = sha256_file(IMAGES_PATH)
    labels_hash = sha256_file(LABELS_PATH)
    images = np.load(IMAGES_PATH, allow_pickle=False, mmap_mode="r")
    labels = np.load(LABELS_PATH, allow_pickle=False, mmap_mode="r")
    # Validate exact known release dimensions, pixel representation, and labels.
    if images.shape != (2000, 32, 32, 3) or images.dtype != np.uint8:
        raise ValueError(f"Unexpected CIFAR-10.1 image array: shape={images.shape}, dtype={images.dtype}")
    if labels.shape != (2000,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Unexpected CIFAR-10.1 labels array: shape={labels.shape}, dtype={labels.dtype}")
    if labels.min() < 0 or labels.max() > 9:
        raise ValueError("CIFAR-10.1 labels must be integers from 0 through 9")
    hashes = {"images_sha256": images_hash, "labels_sha256": labels_hash}
    return images, labels, hashes


def download_file(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".download")
    if destination.exists():
        return
    if temporary.exists():
        temporary.unlink()
    print(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "Brahma-MetaCore research script"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    temporary.replace(destination)


def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing_images, existing_labels = IMAGES_PATH.exists(), LABELS_PATH.exists()
    if existing_images != existing_labels:
        raise FileExistsError(
            f"Only one CIFAR-10.1 file exists in {DATA_DIR}. To avoid overwriting user data, "
            "inspect that directory and either restore the missing matching file or move the "
            "existing file aside before downloading."
        )
    if existing_images and existing_labels:
        images, labels, hashes = validate_dataset_files()
        print("CIFAR-10.1 v6 already exists; files were validated and left unchanged.")
        print(f"Images: {images.shape}, {images.dtype}; SHA-256={hashes['images_sha256']}")
        print(f"Labels: {labels.shape}, {labels.dtype}; SHA-256={hashes['labels_sha256']}")
        return

    staged_images = DATA_DIR / "cifar10.1_v6_data.staged.npy"
    staged_labels = DATA_DIR / "cifar10.1_v6_labels.staged.npy"
    try:
        download_file(DATA_URLS["images"], staged_images)
        download_file(DATA_URLS["labels"], staged_labels)
        staged_image_array = np.load(staged_images, allow_pickle=False, mmap_mode="r")
        staged_label_array = np.load(staged_labels, allow_pickle=False, mmap_mode="r")
        if staged_image_array.shape != (2000, 32, 32, 3) or staged_image_array.dtype != np.uint8:
            raise ValueError(f"Unexpected staged image data: {staged_image_array.shape}, {staged_image_array.dtype}")
        if staged_label_array.shape != (2000,) or not np.issubdtype(staged_label_array.dtype, np.integer):
            raise ValueError(f"Unexpected staged labels: {staged_label_array.shape}, {staged_label_array.dtype}")
        if staged_label_array.min() < 0 or staged_label_array.max() > 9:
            raise ValueError("Staged CIFAR-10.1 labels are outside the range 0..9")
        # Close memory-mapped staged files before renaming them. On Windows,
        # an open np.load(..., mmap_mode="r") mapping prevents os.replace/Path.replace.
        for staged_array in (staged_image_array, staged_label_array):
            mapping = getattr(staged_array, "_mmap", None)
            if mapping is not None:
                mapping.close()
        del staged_image_array, staged_label_array

        # Only publish the final filenames after both files validate and are closed.
        staged_images.replace(IMAGES_PATH)
        staged_labels.replace(LABELS_PATH)
        images, labels, hashes = validate_dataset_files()
    except Exception:
        print("Download or validation failed. Final dataset files were not intentionally overwritten.")
        raise
    finally:
        for temp_path in (staged_images, staged_labels, staged_images.with_suffix(staged_images.suffix + ".download"), staged_labels.with_suffix(staged_labels.suffix + ".download")):
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
    print("CIFAR-10.1 v6 downloaded and validated.")
    print(f"Images: {images.shape}, {images.dtype}; SHA-256={hashes['images_sha256']}")
    print(f"Labels: {labels.shape}, {labels.dtype}; SHA-256={hashes['labels_sha256']}")
    print(f"Source: {DATASET_SOURCE}")


def load_frozen_inputs():
    v01_report = load_json(V01_REPORT)
    if v01_report.get("experiment") != "Brahma MetaCore MetaTransfer v0.1":
        raise ValueError("The supplied report is not a MetaTransfer v0.1 report.")
    if v01_report.get("source_model") != "seed42":
        raise ValueError("Unexpected source model in frozen v0.1 report.")
    if tuple(v01_report.get("target_models", [])) != TARGET_MODELS:
        raise ValueError(f"Expected v0.1 target models {TARGET_MODELS}, got {v01_report.get('target_models')}")

    source_info = v01_report.get("source_score_cache", {})
    source_cache_path = v06.resolve_from_root(source_info.get("path", ""))
    if not source_cache_path.is_file():
        raise FileNotFoundError(f"Frozen v0.1 source-score cache is missing: {source_cache_path}")
    source_cache_hash = sha256_file(source_cache_path)
    if source_info.get("sha256") and source_cache_hash != source_info["sha256"]:
        raise ValueError("Frozen v0.1 source-score cache SHA-256 does not match transfer_report.json.")
    source_payload = load_cache(source_cache_path)
    effects = source_payload.get("effects")
    if not isinstance(effects, dict):
        raise ValueError("Frozen v0.1 source-score cache has no effects dictionary.")

    for target in TARGET_MODELS:
        if target not in v01_report.get("results", {}):
            raise KeyError(f"Frozen v0.1 report has no results for {target}")
        for layer in v06.LAYERS:
            layer_data = v01_report["results"][target]["layers"].get(layer)
            if not layer_data:
                raise KeyError(f"Frozen v0.1 report is missing {target}/{layer}")
            mapping = np.asarray(
                layer_data["alignment"]["mapping_source_index_to_target_index"],
                dtype=np.int64,
            )
            width = int(layer_data["channels"])
            if mapping.shape != (width,) or not np.array_equal(np.sort(mapping), np.arange(width)):
                raise ValueError(f"Invalid frozen mapping for {target}/{layer}")
            for channel in range(width):
                key = f"{layer}|{channel}"
                if key not in effects:
                    raise KeyError(f"Frozen source cache missing channel {key}")
                if "signed_delta_loss" not in effects[key] or "abs_delta_loss" not in effects[key]:
                    raise KeyError(f"Frozen source effect is missing signed/absolute loss for {key}")
    return v01_report, source_payload, source_cache_path, source_cache_hash


def prepare_external_tensors(images: np.ndarray, labels: np.ndarray, device: torch.device):
    """Decode/normalize CIFAR-10.1 once instead of transforming it for every ablation.

    The resulting image tensor stays in host memory; batches are transferred to the
    device as needed. This is numerically equivalent to ToTensor + Normalize used
    by the original single-channel implementation.
    """
    pixels = torch.from_numpy(np.array(images, copy=True)).permute(0, 3, 1, 2).contiguous()
    x = pixels.to(dtype=torch.float32).div_(255.0)
    mean = torch.tensor(v04.MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(v04.STD, dtype=torch.float32).view(1, 3, 1, 1)
    x.sub_(mean).div_(std)
    y = torch.from_numpy(np.array(labels, dtype=np.int64, copy=True))
    if device.type == "cuda":
        # CIFAR-10.1 is small (~25 MB normalized); pin once, not once per channel.
        x = x.pin_memory()
        y = y.pin_memory()
    return x, y


@torch.inference_mode()
def measure_channel_effects_batched(
    model,
    images_tensor: torch.Tensor,
    labels_tensor: torch.Tensor,
    device: torch.device,
    model_key: str,
    channels_per_forward: int = DEFAULT_CHANNELS_PER_FORWARD,
    image_batch_size: int = BATCH_SIZE,
):
    """Measure channel ablations in grouped forwards to reduce Python/GPU launch overhead.

    Each grouped forward repeats a batch K times, with a different single channel
    zeroed in each K-sized group. Thus every channel still sees every evaluation
    image, and effects remain single-channel ablation effects. Only the execution
    batching changes; the model, examples, mappings, loss definition, and metrics do not.
    """
    if channels_per_forward < 1:
        raise ValueError("channels_per_forward must be at least 1")
    model.eval()
    n_examples = int(labels_tensor.numel())
    if int(images_tensor.shape[0]) != n_examples:
        raise ValueError("Image and label counts do not match")

    def batch_slices():
        for start in range(0, n_examples, image_batch_size):
            stop = min(start + image_batch_size, n_examples)
            yield start, stop

    # Baseline is computed once, not repeated separately for every channel.
    baseline_loss_sum = 0.0
    baseline_correct = 0
    baseline_total = 0
    for start, stop in batch_slices():
        xb = images_tensor[start:stop].to(device, non_blocking=True)
        yb = labels_tensor[start:stop].to(device, non_blocking=True)
        logits = model(xb)
        per_example_loss = F.cross_entropy(logits, yb, reduction="none")
        baseline_loss_sum += float(per_example_loss.sum().item())
        baseline_correct += int((logits.argmax(dim=1) == yb).sum().item())
        baseline_total += int(yb.numel())
    if baseline_total != n_examples:
        raise RuntimeError(f"{model_key}: baseline processed {baseline_total}/{n_examples} examples")
    baseline_loss = baseline_loss_sum / n_examples
    baseline_summary = {
        "accuracy": baseline_correct / n_examples,
        "loss": baseline_loss,
        "examples": n_examples,
    }
    print(
        f"{model_key} intervention baseline: accuracy={baseline_summary['accuracy'] * 100:.2f}%, "
        f"loss={baseline_loss:.6f}, examples={n_examples}"
    )

    effects: dict[str, dict[str, float | int]] = {}
    total_channels = sum(int(v06.ablation.get_module_by_name(model, layer).num_features) for layer in v06.LAYERS)
    completed = 0
    for layer in v06.LAYERS:
        module = v06.ablation.get_module_by_name(model, layer)
        width = int(module.num_features)
        for chunk_start in range(0, width, channels_per_forward):
            channel_ids = list(range(chunk_start, min(chunk_start + channels_per_forward, width)))
            k = len(channel_ids)
            state = {"batch_size": None}

            def ablation_group_hook(_module, _inputs, output):
                if not isinstance(output, torch.Tensor) or output.ndim != 4:
                    raise RuntimeError(f"Unexpected activation output at {layer}: {getattr(output, 'shape', None)}")
                base_n = state["batch_size"]
                if base_n is None or output.shape[0] != k * base_n:
                    raise RuntimeError(
                        f"Grouped ablation batch mismatch at {layer}: activation={output.shape[0]}, "
                        f"expected={k}*{base_n}"
                    )
                if output.shape[1] != width:
                    raise RuntimeError(f"Activation width changed unexpectedly at {layer}")
                modified = output.clone(memory_format=torch.contiguous_format)
                grouped = modified.view(k, base_n, output.shape[1], output.shape[2], output.shape[3])
                for group_index, channel_index in enumerate(channel_ids):
                    grouped[group_index, :, channel_index, :, :] = 0
                return grouped.view_as(output)

            loss_sums = np.zeros(k, dtype=np.float64)
            correct_counts = np.zeros(k, dtype=np.int64)
            hook_calls = 0
            handle = module.register_forward_hook(ablation_group_hook)
            try:
                for start, stop in batch_slices():
                    xb = images_tensor[start:stop].to(device, non_blocking=True)
                    yb = labels_tensor[start:stop].to(device, non_blocking=True)
                    base_n = int(yb.numel())
                    state["batch_size"] = base_n
                    # Repeat whole image batches K times; hook applies one mask per group.
                    grouped_x = xb.repeat((k, 1, 1, 1))
                    logits = model(grouped_x).reshape(k, base_n, -1)
                    grouped_y = yb.unsqueeze(0).expand(k, -1)
                    losses = F.cross_entropy(
                        logits.reshape(k * base_n, -1),
                        grouped_y.reshape(k * base_n),
                        reduction="none",
                    ).view(k, base_n)
                    loss_sums += losses.sum(dim=1).double().cpu().numpy()
                    correct_counts += (logits.argmax(dim=2) == grouped_y).sum(dim=1).cpu().numpy()
                    hook_calls += 1
            finally:
                handle.remove()

            expected_batches = (n_examples + image_batch_size - 1) // image_batch_size
            if hook_calls != expected_batches:
                raise RuntimeError(
                    f"{model_key} {layer} channels {channel_ids}: hook called {hook_calls} times, "
                    f"expected {expected_batches}"
                )
            for group_index, channel_index in enumerate(channel_ids):
                ablated_loss = float(loss_sums[group_index] / n_examples)
                delta = ablated_loss - baseline_loss
                effects[f"{layer}|{channel_index}"] = {
                    "signed_delta_loss": float(delta),
                    "abs_delta_loss": float(abs(delta)),
                    "hook_calls": int(hook_calls),
                    "examples": n_examples,
                    "execution_group_size": k,
                }
            completed += k
            if completed % 32 == 0 or completed == total_channels:
                print(f"  {model_key}: measured {completed}/{total_channels} channel interventions")

    if len(effects) != total_channels:
        raise RuntimeError(f"{model_key}: expected {total_channels} channel effects, got {len(effects)}")
    return effects, baseline_summary


def measure_target_effects(target_model: str, images, labels, device: torch.device, dataset_hashes: dict[str, str], channels_per_forward: int):
    checkpoint = v06.CHECKPOINTS[target_model]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing target checkpoint: {checkpoint}")
    checkpoint_rel = str(checkpoint.relative_to(ROOT))
    cache_path = CACHE_DIR / (
        f"target_eval_cifar10_1_v6_effects_{target_model}_n2000_"
        f"{dataset_hashes['images_sha256'][:10]}_{dataset_hashes['labels_sha256'][:10]}.npz"
    )
    expected_metadata = {
        "experiment": "MetaTransfer v0.2 external image evaluation",
        "dataset": "CIFAR-10.1 v6",
        "model_key": target_model,
        "checkpoint": checkpoint_rel,
        "checkpoint_sha256": sha256_file(checkpoint),
        "examples": 2000,
        "images_sha256": dataset_hashes["images_sha256"],
        "labels_sha256": dataset_hashes["labels_sha256"],
        "normalization_mean": list(v04.MEAN),
        "normalization_std": list(v04.STD),
    }

    if cache_path.exists():
        payload = load_cache(cache_path)
        if payload.get("metadata") != expected_metadata:
            raise ValueError(
                f"Existing cache metadata does not match current inputs: {cache_path}. "
                "The file is preserved; investigate instead of overwriting it."
            )
        print(f"Reusing validated external-effect cache: {cache_path.relative_to(ROOT)}")
    else:
        images_tensor, labels_tensor = prepare_external_tensors(images, labels, device)
        model, checkpoint_payload = v06.load_model(checkpoint, device)
        print(
            f"\nMeasuring external target effects: {target_model}; examples=2000; "
            f"checkpoint epoch={int(checkpoint_payload.get('epoch', -1))}; "
            f"channels per forward={channels_per_forward}; image batch={BATCH_SIZE}"
        )
        measured, baseline = measure_channel_effects_batched(
            model=model,
            images_tensor=images_tensor,
            labels_tensor=labels_tensor,
            device=device,
            model_key=f"{target_model}/CIFAR-10.1-v6",
            channels_per_forward=channels_per_forward,
            image_batch_size=BATCH_SIZE,
        )
        del images_tensor, labels_tensor
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        payload = {
            "metadata": expected_metadata,
            "effects": measured,
            "baseline": baseline,
            "execution_metadata": {
                "method": "grouped_single_channel_ablation_forwards",
                "channels_per_forward_requested": int(channels_per_forward),
                "image_batch_size": int(BATCH_SIZE),
            },
        }
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            raise FileExistsError(f"Refusing to overwrite cache created during run: {cache_path}")
        v06.save_cache(cache_path, payload)
        print(f"Saved external-effect cache: {cache_path.relative_to(ROOT)}")

    baseline = payload.get("baseline")
    effects = payload.get("effects")
    if not isinstance(baseline, dict) or not isinstance(effects, dict):
        raise ValueError(f"Invalid cached effects/baseline data: {cache_path}")
    if int(baseline.get("examples", -1)) != 2000:
        raise ValueError(f"Expected 2,000 external examples in baseline: {cache_path}")
    for layer in v06.LAYERS:
        width = {"layer2.1.bn1": 128, "layer3.1.bn1": 256, "layer4.1.bn1": 512}[layer]
        for channel in range(width):
            item = effects.get(f"{layer}|{channel}")
            if not isinstance(item, dict) or int(item.get("examples", -1)) != 2000:
                raise ValueError(f"Missing or invalid external effect for {target_model}/{layer}/{channel}")
    if "execution_metadata" not in payload:
        # Existing caches made by the original v0.2 implementation remain valid;
        # label their execution method accurately rather than pretending they used batching.
        any_grouped_entry = any(
            isinstance(item, dict) and "execution_group_size" in item
            for item in effects.values()
        )
        payload["execution_metadata"] = {
            "method": "grouped_single_channel_ablation_forwards" if any_grouped_entry else "legacy_single_channel_forwards",
            "channels_per_forward_requested": int(channels_per_forward) if any_grouped_entry else 1,
            "image_batch_size": int(BATCH_SIZE),
            "note": "Existing effect cache reused without modification." if not any_grouped_entry else "Grouped channel interventions were cached.",
        }
    payload["cache_path"] = str(cache_path.relative_to(ROOT))
    payload["cache_sha256"] = sha256_file(cache_path)
    return payload


def effect_arrays(effects: dict[str, Any], layer: str, width: int):
    signed = np.empty(width, dtype=np.float64)
    absolute = np.empty(width, dtype=np.float64)
    for channel in range(width):
        item = effects[f"{layer}|{channel}"]
        signed[channel] = float(item["signed_delta_loss"])
        absolute[channel] = float(item["abs_delta_loss"])
    return signed, absolute


def random_control(actual_abs, source_abs, aligned_metrics, same_metrics, repetitions, rng):
    n = len(actual_abs)
    actual_top_n = max(1, int(np.ceil(n * PRIMARY_FRACTION)))
    actual_top = set(np.argsort(actual_abs)[-actual_top_n:].tolist())
    primary_key = f"top_{int(round(PRIMARY_FRACTION * 100))}pct"
    aligned_precision = aligned_metrics["topk"][primary_key]["precision"]
    same_precision = same_metrics["topk"][primary_key]["precision"]
    precisions = np.empty(repetitions, dtype=np.float64)
    rhos = np.full(repetitions, np.nan, dtype=np.float64)
    for i in range(repetitions):
        mapping = rng.permutation(n)
        predicted = np.empty(n, dtype=np.float64)
        predicted[mapping] = source_abs
        predicted_top_n = max(1, int(np.ceil(n * PRIMARY_FRACTION)))
        predicted_top = set(np.argsort(predicted)[-predicted_top_n:].tolist())
        precisions[i] = len(actual_top & predicted_top) / predicted_top_n
        if np.std(actual_abs) > 0 and np.std(predicted) > 0:
            rhos[i] = float(spearmanr(actual_abs, predicted).statistic)
    finite = rhos[np.isfinite(rhos)]
    return {
        "n_random_mappings": int(repetitions),
        "primary_fraction": PRIMARY_FRACTION,
        "precision_mean": float(precisions.mean()),
        "precision_std": float(precisions.std(ddof=1)) if repetitions > 1 else 0.0,
        "precision_2_5_percentile": float(np.quantile(precisions, 0.025)),
        "precision_97_5_percentile": float(np.quantile(precisions, 0.975)),
        "empirical_p_random_precision_ge_aligned": float((1 + np.sum(precisions >= aligned_precision)) / (repetitions + 1)),
        "empirical_p_random_precision_ge_same_index": float((1 + np.sum(precisions >= same_precision)) / (repetitions + 1)),
        "spearman_mean": float(finite.mean()) if len(finite) else None,
        "spearman_2_5_percentile": float(np.quantile(finite, 0.025)) if len(finite) else None,
        "spearman_97_5_percentile": float(np.quantile(finite, 0.975)) if len(finite) else None,
        "note": "Permutation reference only; not a population-level significance test.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-data", action="store_true", help="Download and validate CIFAR-10.1 v6; exit without inference.")
    parser.add_argument("--audit-only", action="store_true", help="Validate frozen v0.1 inputs, external data files, and checkpoints; no inference or report writes.")
    parser.add_argument("--random-mappings", type=int, default=DEFAULT_RANDOM_MAPPINGS, help="Random source-to-target permutations per target/layer (minimum 100).")
    parser.add_argument("--channels-per-forward", type=int, default=DEFAULT_CHANNELS_PER_FORWARD,
                        help="Channel interventions evaluated in one grouped forward (default 4; use 2 if CUDA runs out of memory).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for random mapping controls.")
    parser.add_argument("--output", default=None, help="JSON output path; refuses to overwrite existing JSON or Markdown reports.")
    args = parser.parse_args()
    if sum([args.download_data, args.audit_only]) > 1:
        parser.error("Use only one of --download-data and --audit-only at a time.")
    if args.random_mappings < 100:
        parser.error("--random-mappings must be at least 100.")
    if args.channels_per_forward < 1 or args.channels_per_forward > 16:
        parser.error("--channels-per-forward must be between 1 and 16.")

    if args.download_data:
        download_dataset()
        return

    output_path = v06.resolve_from_root(args.output) if args.output else DEFAULT_OUTPUT
    md_path = output_path.with_suffix(".md")
    if not args.audit_only and (output_path.exists() or md_path.exists()):
        raise FileExistsError(
            f"Refusing to overwrite reports. JSON exists={output_path.exists()}, Markdown exists={md_path.exists()}. "
            "Pass --output to use a new report path."
        )

    v01_report, source_payload, source_cache_path, source_cache_hash = load_frozen_inputs()
    images, labels, data_hashes = validate_dataset_files()
    for model_key in ("seed42", *TARGET_MODELS):
        if not v06.CHECKPOINTS[model_key].is_file():
            raise FileNotFoundError(f"Missing checkpoint for {model_key}: {v06.CHECKPOINTS[model_key]}")

    if args.audit_only:
        print("MetaTransfer v0.2 external evaluation audit: PASS")
        print(f"Frozen v0.1 report: {V01_REPORT.relative_to(ROOT)}")
        print(f"Frozen source effects: {source_cache_path.relative_to(ROOT)}")
        print(f"Source cache SHA-256: {source_cache_hash}")
        print(f"CIFAR-10.1 v6 images: {images.shape}, {images.dtype}; SHA-256={data_hashes['images_sha256']}")
        print(f"CIFAR-10.1 v6 labels: {labels.shape}, {labels.dtype}; SHA-256={data_hashes['labels_sha256']}")
        print("Target checkpoints available: " + ", ".join(TARGET_MODELS))
        print("No model inference, ablations, or report writes were performed.")
        return

    started = time.time()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Protocol: frozen source scores and mappings; 2,000 external target-evaluation images.")
    print("The official CIFAR-10 test set is not loaded.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target_payloads = {}
    for target in TARGET_MODELS:
        target_payloads[target] = measure_target_effects(
            target, images, labels, device, data_hashes,
            channels_per_forward=args.channels_per_forward,
        )

    rng = np.random.default_rng(args.seed)
    results = {}
    target_mean_deltas = []
    for target in TARGET_MODELS:
        layer_results = {}
        source_layer_same, source_layer_aligned, source_layer_random = [], [], []
        for layer in v06.LAYERS:
            width = {"layer2.1.bn1": 128, "layer3.1.bn1": 256, "layer4.1.bn1": 512}[layer]
            source_signed, source_abs = effect_arrays(source_payload["effects"], layer, width)
            target_signed, target_abs = effect_arrays(target_payloads[target]["effects"], layer, width)
            layer_report = v01_report["results"][target]["layers"][layer]
            mapping = np.asarray(layer_report["alignment"]["mapping_source_index_to_target_index"], dtype=np.int64)
            same_abs = source_abs.copy()
            same_signed = source_signed.copy()
            aligned_abs = np.empty(width, dtype=np.float64)
            aligned_signed = np.empty(width, dtype=np.float64)
            aligned_abs[mapping] = source_abs
            aligned_signed[mapping] = source_signed

            same_metrics = v01.score_transfer(target_abs, same_abs)
            aligned_metrics = v01.score_transfer(target_abs, aligned_abs)
            random_metrics = random_control(target_abs, source_abs, aligned_metrics, same_metrics, args.random_mappings, rng)
            layer_results[layer] = {
                "channels": width,
                "frozen_alignment_from_v01": {
                    "mean_abs_matched_correlation": layer_report["alignment"]["mean_abs_matched_correlation"],
                    "mean_abs_same_index_correlation": layer_report["alignment"]["mean_abs_same_index_correlation"],
                    "mapping_sha256": hashlib.sha256(mapping.astype(np.int64).tobytes()).hexdigest(),
                },
                "target_evaluation_baseline": target_payloads[target]["baseline"],
                "same_index_transfer": same_metrics,
                "activation_aligned_transfer": aligned_metrics,
                "random_mapping_control": random_metrics,
                "signed_effect_transfer_secondary": {
                    "same_index": v01.signed_effect_metrics(target_signed, same_signed),
                    "activation_aligned": v01.signed_effect_metrics(target_signed, aligned_signed),
                },
            }
            key = "top_10pct"
            same_p = same_metrics["topk"][key]["precision"]
            aligned_p = aligned_metrics["topk"][key]["precision"]
            random_p = random_metrics["precision_mean"]
            source_layer_same.append(same_p)
            source_layer_aligned.append(aligned_p)
            source_layer_random.append(random_p)
            print(f"{target} / {layer}: P@10% same={same_p:.3f}, aligned={aligned_p:.3f}, random mean={random_p:.3f}; delta={aligned_p-same_p:+.3f}")
        mean_delta = float(np.mean(np.asarray(source_layer_aligned) - np.asarray(source_layer_same)))
        target_mean_deltas.append(mean_delta)
        results[target] = {
            "source_model": "seed42",
            "layers": layer_results,
            "mean_primary_precision_across_layers": {
                "same_index": float(np.mean(source_layer_same)),
                "activation_aligned": float(np.mean(source_layer_aligned)),
                "random_mapping_mean": float(np.mean(source_layer_random)),
                "aligned_minus_same_index": mean_delta,
            },
        }

    rows = [(target, layer, lr) for target, tr in results.items() for layer, lr in tr["layers"].items()]
    aligned_ps = [r["activation_aligned_transfer"]["topk"]["top_10pct"]["precision"] for _, _, r in rows]
    same_ps = [r["same_index_transfer"]["topk"]["top_10pct"]["precision"] for _, _, r in rows]
    random_ps = [r["random_mapping_control"]["precision_mean"] for _, _, r in rows]
    rule_pass = bool(np.mean(aligned_ps) > np.mean(same_ps) and np.mean(aligned_ps) > np.mean(random_ps) and sum(delta > 0 for delta in target_mean_deltas) >= 2)

    report = {
        "experiment": "Brahma MetaCore MetaTransfer v0.2 External Evaluation",
        "status": "completed_by_script_when_run",
        "purpose": "Test the frozen v0.1 transfer mappings and source importance rankings on an independent CIFAR-10.1 v6 image set.",
        "dataset": {
            "name": "CIFAR-10.1 v6",
            "source_url": DATASET_SOURCE,
            "examples": 2000,
            "images_path": str(IMAGES_PATH.relative_to(ROOT)),
            "labels_path": str(LABELS_PATH.relative_to(ROOT)),
            **data_hashes,
            "official_cifar10_test_set_loaded": False,
        },
        "source_model": "seed42",
        "target_models": list(TARGET_MODELS),
        "layers": list(v06.LAYERS),
        "seed": int(args.seed),
        "random_mappings_per_target_layer": int(args.random_mappings),
        "execution_optimization": {
            "method": "grouped_single_channel_ablation_forwards",
            "channels_per_forward_requested": int(args.channels_per_forward),
            "image_batch_size": int(BATCH_SIZE),
            "normalization_precomputed_once_per_target": True,
            "scientific_intervention_unchanged": "Each group duplicates the same images and zeros a different single channel in each image group; every channel still evaluates the full CIFAR-10.1 v6 image set.",
        },
        "primary_metric": "Top-10% precision for identifying the target model's top-10% channels by absolute loss change measured on CIFAR-10.1 v6.",
        "frozen_inputs": {
            "v01_report_path": str(V01_REPORT.relative_to(ROOT)),
            "v01_report_sha256": sha256_file(V01_REPORT),
            "source_effect_cache_path": str(source_cache_path.relative_to(ROOT)),
            "source_effect_cache_sha256": source_cache_hash,
            "source_scores_reused_without_refitting": True,
            "v01_alignments_reused_without_refitting": True,
        },
        "protocol": {
            "target_evaluation_examples": 2000,
            "external_target_evaluation_set": True,
            "target_importance_measured_fresh_on_external_images": True,
            "source_scores_and_alignment_frozen_before_external_evaluation": True,
            "existing_target_checkpoints_reused": True,
            "new_target_seed_replication": False,
            "official_cifar10_test_set_loaded": False,
        },
        "target_eval_caches": {
            target: {
                "path": payload["cache_path"],
                "sha256": payload["cache_sha256"],
                "baseline": payload["baseline"],
                "execution_metadata": payload.get("execution_metadata", {}),
            }
            for target, payload in target_payloads.items()
        },
        "results": results,
        "aggregate_primary_metric": {
            "number_of_target_layer_comparisons": len(rows),
            "mean_aligned_top10_precision": float(np.mean(aligned_ps)),
            "mean_same_index_top10_precision": float(np.mean(same_ps)),
            "mean_random_mapping_top10_precision": float(np.mean(random_ps)),
            "aligned_beats_same_index_comparisons": int(np.sum(np.asarray(aligned_ps) > np.asarray(same_ps))),
            "ties_vs_same_index_comparisons": int(np.sum(np.asarray(aligned_ps) == np.asarray(same_ps))),
            "aligned_beats_random_mapping_mean_comparisons": int(np.sum(np.asarray(aligned_ps) > np.asarray(random_ps))),
            "target_seed_mean_aligned_minus_same_index_precision": {target: results[target]["mean_primary_precision_across_layers"]["aligned_minus_same_index"] for target in TARGET_MODELS},
            "number_of_target_seeds_with_positive_mean_delta": int(sum(delta > 0 for delta in target_mean_deltas)),
        },
        "predefined_directional_rule": {
            "rule": "Directional replication support requires mean aligned Top-10% precision to exceed both same-index and mean randomized-mapping controls, and positive mean aligned-minus-same-index precision for at least 2 of 3 target checkpoints.",
            "passed": rule_pass,
            "interpretation": "This is a directional screening rule, not a statistical significance test. A pass does not establish new-model-seed generalization.",
        },
        "limitations": [
            "The target models are the same three checkpoints examined in v0.1; this is external-image replication, not new-model-seed replication.",
            "CIFAR-10.1 v6 has 2,000 images and may not represent all shifts or datasets.",
            "The alignment and source importance ranking are frozen from the prior validation-based v0.1 pipeline; the current test images were not used to choose them.",
            "Channel rows and layer comparisons are dependent; nine rows are not nine independent model replications.",
            "Random-map controls are a permutation reference, not a population-level significance test.",
            "Single-channel ablation does not measure channel interactions, semantic understanding, model construction, or reduced pretraining requirements.",
        ],
        "elapsed_seconds": float(time.time() - started),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or md_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output_path} / {md_path}")
    lines = [
        "# MetaTransfer v0.2 — External Evaluation on CIFAR-10.1 v6",
        "",
        "- Frozen v0.1 source importance and channel alignments; no tuning on external labels.",
        "- Target evaluation images: **2,000 CIFAR-10.1 v6 examples**.",
        f"- Random mappings per target/layer: **{args.random_mappings}**.",
        "- New target seeds: **not included in this stage**.",
        f"- Execution optimization: grouped single-channel ablations, up to **{args.channels_per_forward} channels per forward** with image batch **{BATCH_SIZE}**; the ablation definition is unchanged.",
        "",
        "## Primary results",
        "",
        "| Target | Layer | Same-index P@10% | Aligned P@10% | Random-map mean | Aligned Δ vs same |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for target, layer, layer_result in rows:
        same_p = layer_result["same_index_transfer"]["topk"]["top_10pct"]["precision"]
        aligned_p = layer_result["activation_aligned_transfer"]["topk"]["top_10pct"]["precision"]
        random_p = layer_result["random_mapping_control"]["precision_mean"]
        lines.append(f"| {target} | `{layer}` | {same_p:.3f} | {aligned_p:.3f} | {random_p:.3f} | {aligned_p-same_p:+.3f} |")
    aggregate = report["aggregate_primary_metric"]
    lines.extend([
        "",
        "## Aggregate",
        "",
        f"- Mean aligned P@10%: **{aggregate['mean_aligned_top10_precision']:.4f}**",
        f"- Mean same-index P@10%: **{aggregate['mean_same_index_top10_precision']:.4f}**",
        f"- Mean random-map P@10%: **{aggregate['mean_random_mapping_top10_precision']:.4f}**",
        f"- Aligned beats same-index: **{aggregate['aligned_beats_same_index_comparisons']}/{len(rows)}** comparisons",
        f"- Directional rule passed: **{rule_pass}**",
        "",
        "## Limitations",
        "",
    ])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    output_path.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nJSON report saved: {output_path.relative_to(ROOT)}")
    print(f"Markdown report saved: {md_path.relative_to(ROOT)}")
    print(f"Elapsed: {report['elapsed_seconds']:.1f} seconds")
    print("No model training or official CIFAR-10 test-set evaluation was performed.")


if __name__ == "__main__":
    main()
