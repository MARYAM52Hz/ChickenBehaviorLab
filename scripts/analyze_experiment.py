from __future__ import annotations

import argparse
import json
from pathlib import Path

from chicken_behavior_lab.analysis import (
    ConfusionMatrixAnalyzer,
    TrainingHistoryAnalyzer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a ChickenBehaviorLab "
            "experiment."
        )
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help=(
            "Path to the experiment directory."
        ),
    )

    return parser.parse_args()


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:

    args = parse_args()

    experiment_dir = Path(
        args.experiment
    )

    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory not found: "
            f"{experiment_dir}"
        )

    history_path = (
        experiment_dir
        / "training_history.json"
    )

    metrics_path = (
        experiment_dir
        / "test_metrics.json"
    )

    config_path = (
        experiment_dir
        / "experiment_config.json"
    )

    # --------------------------------------------------
    # Training history
    # --------------------------------------------------

    history_analyzer = (
        TrainingHistoryAnalyzer()
    )

    history = history_analyzer.load(
        history_path
    )

    training_summary = (
        history_analyzer.summarize(
            history
        )
    )

    overfitting = (
        history_analyzer.detect_overfitting(
            history
        )
    )

    print()
    print("=" * 70)
    print("TRAINING ANALYSIS")
    print("=" * 70)

    print(
        f"Epochs: "
        f"{training_summary['num_epochs']}"
    )

    print(
        f"Best epoch: "
        f"{training_summary['best_epoch']}"
    )

    print(
        f"Best validation Macro-F1: "
        f"{training_summary['best_validation_macro_f1']:.4f}"
    )

    print(
        f"Final validation accuracy: "
        f"{training_summary['final_validation_accuracy']:.4f}"
    )

    print(
        f"Potential overfitting: "
        f"{overfitting}"
    )

    # --------------------------------------------------
    # Test metrics
    # --------------------------------------------------

    metrics = load_json(
        metrics_path
    )

    confusion_matrix = (
        metrics["confusion_matrix"]
    )

    config = load_json(
        config_path
    )

    label_to_index = config.get(
        "label_to_index",
        None,
    )

    if label_to_index is None:
        print()
        print(
            "No label mapping found in "
            "experiment configuration."
        )

        return

    class_names = [
        label
        for label, _ in sorted(
            label_to_index.items(),
            key=lambda item: item[1],
        )
    ]

    analyzer = (
        ConfusionMatrixAnalyzer(
            matrix=confusion_matrix,
            class_names=class_names,
        )
    )

    print()
    print("=" * 70)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 70)

    print()
    print("Per-class summary:")

    for item in analyzer.per_class_summary():

        print(
            f"  {item['class']}: "
            f"TP={item['true_positive']} "
            f"FP={item['false_positive']} "
            f"FN={item['false_negative']} "
            f"support={item['support']}"
        )

    print()
    print("Most confused behavior pairs:")

    pairs = analyzer.most_confused_pairs()

    if not pairs:
        print(
            "  No off-diagonal errors found."
        )

    for pair in pairs:

        print(
            f"  {pair['true_class']} "
            f"→ "
            f"{pair['predicted_class']}: "
            f"{pair['count']}"
        )


if __name__ == "__main__":
    main()
