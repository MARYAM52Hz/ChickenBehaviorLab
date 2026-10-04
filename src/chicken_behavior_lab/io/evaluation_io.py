from __future__ import annotations

from pathlib import Path
from typing import Any

from chicken_behavior_lab.io.experiment_io import (
    ExperimentOutputWriter,
)
from chicken_behavior_lab.training.temporal_evaluator import (
    TemporalEvaluationResult,
)


def save_temporal_evaluation(
    experiment_dir: str | Path,
    result: TemporalEvaluationResult,
) -> None:

    writer = ExperimentOutputWriter()

    prediction_errors = []

    for record in result.prediction_records:

        prediction_errors.append(
            {
                "sample_id": record.sample_id,
                "video_id": record.video_id,
                "track_id": record.track_id,
                "start_frame": record.start_frame,
                "end_frame": record.end_frame,
                "true_index": record.true_index,
                "predicted_index": record.predicted_index,
                "confidence": record.confidence,
                "correct": (
                    record.true_index
                    == record.predicted_index
                ),
            }
        )

    writer.save_prediction_errors(
        experiment_dir,
        prediction_errors,
    )

    writer.save_test_metrics(
        experiment_dir,
        result.metrics,
    )
