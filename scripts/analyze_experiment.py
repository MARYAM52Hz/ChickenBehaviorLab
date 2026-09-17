from __future__ import annotations

import argparse
import json
from pathlib import Path

from chicken_behavior_lab.analysis import (
    ConfusionMatrixAnalyzer,
    PredictionErrorAnalyzer,
    TrainingHistoryAnalyzer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a ChickenBehaviorLab experiment."
        )
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to the experiment directory.",
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
            "Experiment directory not found: "
            f"{experiment_dir}"
        )

    # --------------------------------------------------
    # Training analysis
    # --------------------------------------------------

    history_path = (
        experiment_dir
        / "training_history.json"
    )

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
        "Best validation Macro-F1: "
        f"{training_summary['best_validation_macro_f1']:.4f}"
    )

    print(
        "Final validation accuracy: "
        f"{training_summary['final_validation_accuracy']:.4f}"
    )

    print(
        f"Potential overfitting: "
        f"{overfitting}"
    )

    # --------------------------------------------------
    # Test metrics
    # --------------------------------------------------

    metrics_path = (
        experiment_dir
        / "test_metrics.json"
    )

    config_path = (
        experiment_dir
        / "experiment_config.json"
    )

    metrics = load_json(
        metrics_path
    )

    config = load_json(
        config_path
    )

    label_to_index = config.get(
        "label_to_index"
    )

    if label_to_index is None:
        print()
        print(
            "No label mapping found in "
            "experiment configuration."
        )
        return

    index_to_label = {
        int(index): label
        for label, index
        in label_to_index.items()
    }

    class_names = [
        index_to_label[index]
        for index in sorted(
            index_to_label
        )
    ]

    confusion_matrix = (
        metrics["confusion_matrix"]
    )

    analyzer = ConfusionMatrixAnalyzer(
        matrix=confusion_matrix,
        class_names=class_names,
    )

    print()
    print("=" * 70)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 70)

    print()
    print("Per-class summary:")

    for item in (
        analyzer.per_class_summary()
    ):
        print(
            f"  {item['class']}: "
            f"TP={item['true_positive']} "
            f"FP={item['false_positive']} "
            f"FN={item['false_negative']} "
            f"support={item['support']}"
        )

    print()
    print(
        "Most confused behavior pairs:"
    )

    pairs = (
        analyzer.most_confused_pairs()
    )

    if not pairs:
        print(
            "  No off-diagonal errors found."
        )

    for pair in pairs:
        print(
            f"  {pair['true_class']} → "
            f"{pair['predicted_class']}: "
            f"{pair['count']}"
        )

    # --------------------------------------------------
    # Per-sample error analysis
    # --------------------------------------------------

    prediction_path = (
        experiment_dir
        / "prediction_errors.json"
    )

    if not prediction_path.exists():
        print()
        print(
            "No prediction_errors.json found."
        )
        return

    error_analyzer = (
        PredictionErrorAnalyzer.load_json(
            prediction_path
        )
    )

    print()
    print("=" * 70)
    print("PER-SAMPLE ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"Predictions: "
        f"{len(error_analyzer)}"
    )

    print(
        f"Correct: "
        f"{len(error_analyzer.correct_predictions)}"
    )

    print(
        f"Errors: "
        f"{len(error_analyzer.errors)}"
    )

    print(
        f"Accuracy: "
        f"{error_analyzer.accuracy():.4f}"
    )

    print(
        f"Error rate: "
        f"{error_analyzer.error_rate():.4f}"
    )

    print()
    print(
        "High-confidence errors:"
    )

    high_confidence_errors = (
        error_analyzer.high_confidence_errors(
            threshold=0.80
        )
    )

    if not high_confidence_errors:
        print(
            "  No high-confidence errors."
        )

    for error in (
        high_confidence_errors
    ):
        print(
            f"  {error.sample_id}: "
            f"{error.true_behavior} → "
            f"{error.predicted_behavior} "
            f"(confidence="
            f"{error.confidence:.3f}, "
            f"frames="
            f"{error.start_frame}-"
            f"{error.end_frame})"
        )

    print()
    print(
        "Error rate by true behavior:"
    )

    for item in (
        error_analyzer
        .error_rate_by_true_behavior()
    ):
        print(
            f"  {item['behavior']}: "
            f"{item['errors']}/"
            f"{item['total']} "
            f"("
            f"{item['error_rate']:.2%}"
            f")"
        )


if __name__ == "__main__":
    main()
