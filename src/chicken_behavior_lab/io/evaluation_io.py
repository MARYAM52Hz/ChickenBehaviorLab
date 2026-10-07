from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _to_serializable(value: Any) -> Any:
    """Convert common scientific Python objects to JSON-compatible values."""

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): _to_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]

    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except (ValueError, TypeError):
            pass

    return value


def save_temporal_evaluation(
    experiment_dir: str | Path,
    result: Any,
) -> None:
    """
    Save temporal evaluation metrics and prediction-level errors.

    The result is expected to be a TemporalEvaluationResult or an object
    exposing `metrics` and `prediction_records`.
    """

    experiment_path = Path(experiment_dir)
    experiment_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = getattr(result, "metrics", None)

    if metrics is None:
        raise ValueError(
            "Evaluation result does not contain `metrics`."
        )

    prediction_records = getattr(
        result,
        "prediction_records",
        [],
    )

    serializable_metrics = _to_serializable(metrics)

    metrics_path = experiment_path / "test_metrics.json"

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    serializable_predictions: list[dict[str, Any]] = []

    for record in prediction_records:
        record_dict = dict(record)

        true_index = record_dict.get("true_index")
        predicted_index = record_dict.get("predicted_index")

        if (
            true_index is not None
            and predicted_index is not None
        ):
            record_dict["correct"] = (
                int(true_index) == int(predicted_index)
            )

        serializable_predictions.append(
            _to_serializable(record_dict)
        )

    prediction_path = (
        experiment_path / "prediction_errors.json"
    )

    with prediction_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_predictions,
            file,
            indent=2,
            ensure_ascii=False,
        )
