from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chicken_behavior_lab.training.temporal_evaluator import (
    TemporalEvaluationResult,
)


@dataclass(slots=True)
class TemporalPredictionError:
    sample_id: str
    video_id: str
    track_id: int
    start_frame: int
    end_frame: int
    true_behavior: str
    predicted_behavior: str
    confidence: float


class TemporalPredictionErrorAnalyzer:
    """
    Analyze temporal behavior classification errors.

    Unlike frame-level analysis, each prediction corresponds to a temporal
    window rather than a single frame.
    """

    def __init__(
        self,
        errors: list[TemporalPredictionError],
        correct_predictions: list[TemporalPredictionError],
    ) -> None:
        self.errors = errors
        self.correct_predictions = correct_predictions

    @classmethod
    def from_evaluation_result(
        cls,
        evaluation_result: TemporalEvaluationResult,
        index_to_label: dict[int, str],
    ) -> "TemporalPredictionErrorAnalyzer":
        errors: list[TemporalPredictionError] = []
        correct: list[TemporalPredictionError] = []

        for record in evaluation_result.prediction_records:
            true_behavior = index_to_label.get(
                record.true_index,
                str(record.true_index),
            )

            predicted_behavior = index_to_label.get(
                record.predicted_index,
                str(record.predicted_index),
            )

            item = TemporalPredictionError(
                sample_id=record.sample_id,
                video_id=record.video_id,
                track_id=record.track_id,
                start_frame=record.start_frame,
                end_frame=record.end_frame,
                true_behavior=true_behavior,
                predicted_behavior=predicted_behavior,
                confidence=record.confidence,
            )

            if record.true_index == record.predicted_index:
                correct.append(item)
            else:
                errors.append(item)

        return cls(
            errors=errors,
            correct_predictions=correct,
        )

    def accuracy(self) -> float:
        total = len(self.errors) + len(
            self.correct_predictions
        )

        if total == 0:
            return 0.0

        return len(self.correct_predictions) / total

    def error_rate(self) -> float:
        total = len(self.errors) + len(
            self.correct_predictions
        )

        if total == 0:
            return 0.0

        return len(self.errors) / total

    def high_confidence_errors(
        self,
        threshold: float = 0.80,
    ) -> list[TemporalPredictionError]:
        return [
            error
            for error in self.errors
            if error.confidence >= threshold
        ]

    def low_confidence_errors(
        self,
        threshold: float = 0.50,
    ) -> list[TemporalPredictionError]:
        return [
            error
            for error in self.errors
            if error.confidence < threshold
        ]

    def confusion_pairs(self) -> dict[tuple[str, str], int]:
        pairs: dict[tuple[str, str], int] = {}

        for error in self.errors:
            key = (
                error.true_behavior,
                error.predicted_behavior,
            )
            pairs[key] = pairs.get(key, 0) + 1

        return dict(
            sorted(
                pairs.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    def error_rate_by_true_behavior(
        self,
    ) -> dict[str, float]:
        totals: dict[str, int] = {}
        errors: dict[str, int] = {}

        for item in self.correct_predictions:
            totals[item.true_behavior] = (
                totals.get(item.true_behavior, 0) + 1
            )

        for item in self.errors:
            totals[item.true_behavior] = (
                totals.get(item.true_behavior, 0) + 1
            )
            errors[item.true_behavior] = (
                errors.get(item.true_behavior, 0) + 1
            )

        result: dict[str, float] = {}

        for behavior, total in totals.items():
            result[behavior] = (
                errors.get(behavior, 0) / total
            )

        return dict(
            sorted(
                result.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    def summary(self) -> dict[str, Any]:
        return {
            "num_errors": len(self.errors),
            "num_correct": len(
                self.correct_predictions
            ),
            "accuracy": self.accuracy(),
            "error_rate": self.error_rate(),
            "num_high_confidence_errors": len(
                self.high_confidence_errors()
            ),
            "num_low_confidence_errors": len(
                self.low_confidence_errors()
            ),
            "confusion_pairs": {
                f"{true}->{predicted}": count
                for (true, predicted), count
                in self.confusion_pairs().items()
            },
            "error_rate_by_true_behavior":
                self.error_rate_by_true_behavior(),
        }
