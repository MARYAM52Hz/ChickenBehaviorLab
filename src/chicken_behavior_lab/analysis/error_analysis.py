from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """
    Store one model prediction for one graph sample.
    """

    sample_id: str
    video_id: str
    track_id: int
    start_frame: int
    end_frame: int
    true_behavior: str
    predicted_behavior: str
    confidence: float

    @property
    def is_correct(self) -> bool:
        return (
            self.true_behavior
            == self.predicted_behavior
        )

    @property
    def is_error(self) -> bool:
        return not self.is_correct

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError(
                "sample_id cannot be empty."
            )

        if not self.video_id:
            raise ValueError(
                "video_id cannot be empty."
            )

        if self.track_id < 0:
            raise ValueError(
                "track_id cannot be negative."
            )

        if self.start_frame < 0:
            raise ValueError(
                "start_frame cannot be negative."
            )

        if self.end_frame < self.start_frame:
            raise ValueError(
                "end_frame must be greater than "
                "or equal to start_frame."
            )

        if not self.true_behavior:
            raise ValueError(
                "true_behavior cannot be empty."
            )

        if not self.predicted_behavior:
            raise ValueError(
                "predicted_behavior cannot be empty."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0.0 and 1.0."
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()

        data = asdict(self)

        data["is_correct"] = self.is_correct
        data["is_error"] = self.is_error

        return data


class PredictionErrorAnalyzer:
    """
    Analyze per-sample model predictions and errors.
    """

    def __init__(
        self,
        predictions: list[PredictionRecord],
    ) -> None:
        self.predictions = list(predictions)
        self._validate()

    def _validate(self) -> None:
        sample_ids: set[str] = set()

        for prediction in self.predictions:
            if not isinstance(
                prediction,
                PredictionRecord,
            ):
                raise TypeError(
                    "Every prediction must be a "
                    "PredictionRecord."
                )

            prediction.validate()

            if prediction.sample_id in sample_ids:
                raise ValueError(
                    f"Duplicate sample_id: "
                    f"{prediction.sample_id}"
                )

            sample_ids.add(
                prediction.sample_id
            )

    @classmethod
    def from_evaluation_result(
        cls,
        evaluation_result,
        index_to_label: dict[int, str],
    ) -> "PredictionErrorAnalyzer":
        """
        Convert integer class predictions into canonical
        behavior IDs.
        """

        if not hasattr(
            evaluation_result,
            "prediction_records",
        ):
            raise ValueError(
                "EvaluationResult does not contain "
                "prediction_records."
            )

        records: list[PredictionRecord] = []

        if len(
            evaluation_result.prediction_records
        ) != len(evaluation_result.y_true):
            raise ValueError(
                "Prediction record count does not match "
                "evaluation target count."
            )

        for index, raw_record in enumerate(
            evaluation_result.prediction_records
        ):
            true_index = int(
                evaluation_result.y_true[index]
            )

            predicted_index = int(
                evaluation_result.y_pred[index]
            )

            if true_index not in index_to_label:
                raise KeyError(
                    f"Unknown true class index: "
                    f"{true_index}"
                )

            if predicted_index not in index_to_label:
                raise KeyError(
                    "Unknown predicted class index: "
                    f"{predicted_index}"
                )

            records.append(
                PredictionRecord(
                    sample_id=raw_record.sample_id,
                    video_id=raw_record.video_id,
                    track_id=raw_record.track_id,
                    start_frame=raw_record.start_frame,
                    end_frame=raw_record.end_frame,
                    true_behavior=index_to_label[
                        true_index
                    ],
                    predicted_behavior=index_to_label[
                        predicted_index
                    ],
                    confidence=raw_record.confidence,
                )
            )

        return cls(records)

    def _validate(self) -> None:
        sample_ids: set[str] = set()

        for prediction in self.predictions:
            prediction.validate()

            if prediction.sample_id in sample_ids:
                raise ValueError(
                    f"Duplicate sample_id: "
                    f"{prediction.sample_id}"
                )

            sample_ids.add(
                prediction.sample_id
            )

    def __len__(self) -> int:
        return len(self.predictions)

    @property
    def errors(
        self,
    ) -> list[PredictionRecord]:
        return [
            prediction
            for prediction in self.predictions
            if prediction.is_error
        ]

    @property
    def correct_predictions(
        self,
    ) -> list[PredictionRecord]:
        return [
            prediction
            for prediction in self.predictions
            if prediction.is_correct
        ]

    def accuracy(self) -> float:
        if not self.predictions:
            return 0.0

        return (
            len(self.correct_predictions)
            / len(self.predictions)
        )

    def error_rate(self) -> float:
        if not self.predictions:
            return 0.0

        return (
            len(self.errors)
            / len(self.predictions)
        )

    def low_confidence_errors(
        self,
        threshold: float = 0.60,
    ) -> list[PredictionRecord]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0.0 and 1.0."
            )

        return [
            prediction
            for prediction in self.errors
            if prediction.confidence < threshold
        ]

    def high_confidence_errors(
        self,
        threshold: float = 0.80,
    ) -> list[PredictionRecord]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0.0 and 1.0."
            )

        return [
            prediction
            for prediction in self.errors
            if prediction.confidence >= threshold
        ]

    def confusion_pairs(
        self,
    ) -> list[dict[str, Any]]:
        pair_counts: dict[
            tuple[str, str],
            int,
        ] = {}

        for prediction in self.errors:
            pair = (
                prediction.true_behavior,
                prediction.predicted_behavior,
            )

            pair_counts[pair] = (
                pair_counts.get(pair, 0) + 1
            )

        results = [
            {
                "true_behavior": true_behavior,
                "predicted_behavior": predicted_behavior,
                "count": count,
            }
            for (
                true_behavior,
                predicted_behavior,
            ), count in pair_counts.items()
        ]

        results.sort(
            key=lambda item: item["count"],
            reverse=True,
        )

        return results

    def error_rate_by_true_behavior(
        self,
    ) -> list[dict[str, Any]]:
        totals: dict[str, int] = {}
        errors: dict[str, int] = {}

        for prediction in self.predictions:
            behavior = prediction.true_behavior

            totals[behavior] = (
                totals.get(behavior, 0) + 1
            )

            if prediction.is_error:
                errors[behavior] = (
                    errors.get(behavior, 0) + 1
                )

        results = []

        for behavior, total in totals.items():
            error_count = errors.get(
                behavior,
                0,
            )

            results.append(
                {
                    "behavior": behavior,
                    "total": total,
                    "errors": error_count,
                    "error_rate": (
                        error_count / total
                    ),
                }
            )

        results.sort(
            key=lambda item: item["error_rate"],
            reverse=True,
        )

        return results

    def summary(self) -> dict[str, Any]:
        return {
            "num_predictions": len(
                self.predictions
            ),
            "num_correct": len(
                self.correct_predictions
            ),
            "num_errors": len(
                self.errors
            ),
            "accuracy": self.accuracy(),
            "error_rate": self.error_rate(),
            "num_low_confidence_errors": len(
                self.low_confidence_errors()
            ),
            "num_high_confidence_errors": len(
                self.high_confidence_errors()
            ),
            "confusion_pairs": self.confusion_pairs(),
            "error_rate_by_true_behavior": (
                self.error_rate_by_true_behavior()
            ),
        }

    def save_json(
        self,
        path: str | Path,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "predictions": [
                prediction.to_dict()
                for prediction in self.predictions
            ],
            "summary": self.summary(),
        }

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
            )

    @classmethod
    def load_json(
        cls,
        path: str | Path,
    ) -> "PredictionErrorAnalyzer":
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Prediction file not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if "predictions" not in data:
            raise ValueError(
                "JSON must contain a 'predictions' field."
            )

        predictions = []

        for item in data["predictions"]:
            predictions.append(
                PredictionRecord(
                    sample_id=str(
                        item["sample_id"]
                    ),
                    video_id=str(
                        item["video_id"]
                    ),
                    track_id=int(
                        item["track_id"]
                    ),
                    start_frame=int(
                        item["start_frame"]
                    ),
                    end_frame=int(
                        item["end_frame"]
                    ),
                    true_behavior=str(
                        item["true_behavior"]
                    ),
                    predicted_behavior=str(
                        item["predicted_behavior"]
                    ),
                    confidence=float(
                        item["confidence"]
                    ),
                )
            )

        return cls(predictions)
