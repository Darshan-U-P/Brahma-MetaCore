
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    path = root / "reports" / "milestone2_initial_vs_trained.json"
    report = json.loads(path.read_text(encoding="utf-8"))

    weights = report["weight_comparison"]["per_tensor"]
    largest_changes = sorted(
        weights.items(),
        key=lambda item: item[1]["delta_l2_norm"],
        reverse=True,
    )[:15]

    print("\nTOP 15 TENSOR CHANGES")
    print("-" * 90)

    for name, metrics in largest_changes:
        print(
            f"{name:45s} "
            f"delta_l2={metrics['delta_l2_norm']:.5f} "
            f"mean_abs={metrics['mean_absolute_change']:.6f}"
        )

    activations = report["activation_comparison"]
    largest_mean_changes = sorted(
        activations.items(),
        key=lambda item: abs(item[1]["mean_change"]),
        reverse=True,
    )[:15]

    print("\nTOP 15 ABSOLUTE ACTIVATION-MEAN CHANGES")
    print("-" * 90)

    for name, metrics in largest_mean_changes:
        print(
            f"{name:45s} "
            f"mean_delta={metrics['mean_change']:+.5f} "
            f"std_delta={metrics['std_change']:+.5f}"
        )


if __name__ == "__main__":
    main()
