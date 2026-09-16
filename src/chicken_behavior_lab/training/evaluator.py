from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True, slots=True)
class RawPredictionRecord:
    """
    Raw per-sample prediction.

    Behavior names are intentionally not stored here.
    Class-index to behavior-ID conversion belongs to the
    analysis layer.
    """

    sample_id: str
    video_id: str
    track_id: int
    start_frame: int
    end_frame: int
    true_index: int
    predicted_index: int
    confidence: float

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

        if self.true_index < 0:
            raise ValueError(
                "true_index cannot be negative."
            )

        if self.predicted_index < 0:
            raise ValueError(
                "predicted_index cannot be negative."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0.0 and 1.0."
            )


@dataclass(slots=True)
class EvaluationResult:
    metrics: dict[str, Any]
    y_true: list[int]
    y_pred: list[int]
    sample_ids: list[str]
    video_ids: list[str]
    track_ids: list[int]
    prediction_records: list[
        RawPredictionRecord
    ]

    def validate(self) -> None:
        lengths = {
            len(self.y_true),
            len(self.y_pred),
            len(self.sample_ids),
            len(self.video_ids),
            len(self.track_ids),
            len(self.prediction_records),
        }

        if len(lengths) != 1:
            raise ValueError(
                "All evaluation result arrays must "
                "have the same length."
            )

        for record in self.prediction_records:
            record.validate()

    @property
    def num_samples(self) -> int:
        return len(self.y_true)


class Evaluator:
    """
    Evaluate a trained classification model.

    The evaluator is responsible for:
    - model inference,
    - class indices,
    - confidence,
    - source-sample metadata.

    It does not convert class indices into behavior IDs.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int,
    ) -> None:
        self.model = model
        self.device = device
        self.num_classes = num_classes

    def evaluate(
        self,
        data_loader: DataLoader,
    ) -> EvaluationResult:
        self.model.eval()

        y_true: list[int] = []
        y_pred: list[int] = []

        sample_ids: list[str] = []
        video_ids: list[str] = []
        track_ids: list[int] = []

        prediction_records: list[
            RawPredictionRecord
        ] = []

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(
                    self.device
                )

                logits = self.model(batch)

                probabilities = torch.softmax(
                    logits,
                    dim=-1,
                )

                predictions = torch.argmax(
                    probabilities,
                    dim=-1,
                )

                confidences = torch.max(
                    probabilities,
                    dim=-1,
                ).values

                targets = batch.y.view(-1)

                batch_predictions = (
                    predictions
                    .detach()
                    .cpu()
                    .tolist()
                )

                batch_targets = (
                    targets
                    .detach()
                    .cpu()
                    .tolist()
                )

                batch_confidences = (
                    confidences
                    .detach()
                    .cpu()
                    .tolist()
                )

                batch_sample_ids = (
                    self._normalize_metadata(
                        getattr(
                            batch,
                            "sample_id",
                            None,
                        ),
                        len(batch_targets),
                        "sample_id",
                    )
                )

                batch_video_ids = (
                    self._normalize_metadata(
                        getattr(
                            batch,
                            "video_id",
                            None,
                        ),
                        len(batch_targets),
                        "video_id",
                    )
                )

                batch_track_ids = (
                    self._normalize_metadata(
                        getattr(
                            batch,
                            "track_id",
                            None,
                        ),
                        len(batch_targets),
                        "track_id",
                    )
                )

                batch_start_frames = (
                    self._normalize_metadata(
                        getattr(
                            batch,
                            "start_frame",
                            None,
                        ),
                        len(batch_targets),
                        "start_frame",
                    )
                )

                batch_end_frames = (
                    self._normalize_metadata(
                        getattr(
                            batch,
                            "end_frame",
                            None,
                        ),
                        len(batch_targets),
                        "end_frame",
                    )
                )

                for index in range(
                    len(batch_targets)
                ):
                    true_index = int(
                        batch_targets[index]
                    )

                    predicted_index = int(
                        batch_predictions[index]
                    )

                    record = RawPredictionRecord(
                        sample_id=str(
                            batch_sample_ids[index]
                        ),
                        video_id=str(
                            batch_video_ids[index]
                        ),
                        track_id=int(
                            batch_track_ids[index]
                        ),
                        start_frame=int(
                            batch_start_frames[index]
                        ),
                        end_frame=int(
                            batch_end_frames[index]
                        ),
                        true_index=true_index,
                        predicted_index=(
                            predicted_index
                        ),
                        confidence=float(
                            batch_confidences[index]
                        ),
                    )

                    record.validate()

                    prediction_records.append(
                        record
                    )

                y_true.extend(
                    int(value)
                    for value in batch_targets
                )

                y_pred.extend(
                    int(value)
                    for value in batch_predictions
                )

                sample_ids.extend(
                    str(value)
                    for value in batch_sample_ids
                )

                video_ids.extend(
                    str(value)
                    for value in batch_video_ids
                )

                track_ids.extend(
                    int(value)
                    for value in batch_track_ids
                )

        metrics = self._compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
        )

        result = EvaluationResult(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sample_ids=sample_ids,
            video_ids=video_ids,
            track_ids=track_ids,
            prediction_records=prediction_records,
        )

        result.validate()

        return result

    @staticmethod
    def _normalize_metadata(
        values: Any,
        expected_length: int,
        field_name: str,
    ) -> list[Any]:
        """
        Normalize PyG-batched metadata into a list.
        """

        if values is None:
            raise ValueError(
                f"Batch is missing required field: "
                f"{field_name}"
            )

        if torch.is_tensor(values):
            values = (
                values.detach()
                .cpu()
                .tolist()
            )

        elif isinstance(values, tuple):
            values = list(values)

        elif not isinstance(values, list):
            values = [values]

        if len(values) != expected_length:
            raise ValueError(
                f"Metadata field '{field_name}' "
                f"contains {len(values)} values, "
                f"but batch contains "
                f"{expected_length} samples."
            )

        return values

    def _compute_metrics(
        self,
        y_true: list[int],
        y_pred: list[int],
    ) -> dict[str, Any]:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            precision_recall_fscore_support,
        )

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=list(
                    range(self.num_classes)
                ),
                average="macro",
                zero_division=0,
            )
        )

        _, _, weighted_f1, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=list(
                    range(self.num_classes)
                ),
                average="weighted",
                zero_division=0,
            )
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=list(
                range(self.num_classes)
            ),
        )

        return {
            "accuracy": float(accuracy),
            "macro_precision": float(
                precision
            ),
            "macro_recall": float(
                recall
            ),
            "macro_f1": float(f1),
            "weighted_f1": float(
                weighted_f1
            ),
            "confusion_matrix": (
                matrix.tolist()
            ),
        }
