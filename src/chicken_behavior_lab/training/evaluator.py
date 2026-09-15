from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from chicken_behavior_lab.analysis.error_analysis import (
    PredictionRecord,
)


@dataclass(slots=True)
class EvaluationResult:
    metrics: dict[str, Any]
    y_true: list[int]
    y_pred: list[int]
    sample_ids: list[str]
    video_ids: list[str]
    track_ids: list[int]
    prediction_records: list[PredictionRecord]

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
                "All evaluation result arrays must have the same length."
            )

    @property
    def num_samples(self) -> int:
        return len(self.y_true)


class Evaluator:
    """
    Evaluate a trained classification model and preserve
    per-sample prediction information for error analysis.
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
            PredictionRecord
        ] = []

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)

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
                    predictions.detach()
                    .cpu()
                    .tolist()
                )

                batch_targets = (
                    targets.detach()
                    .cpu()
                    .tolist()
                )

                batch_confidences = (
                    confidences.detach()
                    .cpu()
                    .tolist()
                )

                batch_sample_ids = self._get_batch_metadata(
                    batch,
                    "sample_id",
                )

                batch_video_ids = self._get_batch_metadata(
                    batch,
                    "video_id",
                )

                batch_track_ids = self._get_batch_metadata(
                    batch,
                    "track_id",
                )

                batch_metadata = self._get_batch_metadata(
                    batch,
                    "metadata",
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

                for index in range(
                    len(batch_predictions)
                ):
                    metadata = batch_metadata[index]

                    start_frame = int(
                        metadata.get(
                            "start_frame",
                            0,
                        )
                    )

                    end_frame = int(
                        metadata.get(
                            "end_frame",
                            start_frame,
                        )
                    )

                    prediction_records.append(
                        PredictionRecord(
                            sample_id=str(
                                batch_sample_ids[index]
                            ),
                            video_id=str(
                                batch_video_ids[index]
                            ),
                            track_id=int(
                                batch_track_ids[index]
                            ),
                            start_frame=start_frame,
                            end_frame=end_frame,
                            true_behavior=str(
                                batch_targets[index]
                            ),
                            predicted_behavior=str(
                                batch_predictions[index]
                            ),
                            confidence=float(
                                batch_confidences[index]
                            ),
                        )
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
    def _get_batch_metadata(
        batch: Any,
        field_name: str,
    ) -> list[Any]:
        """
        Extract metadata from a PyG batch.

        PyG may represent metadata differently depending
        on how individual samples were constructed.
        """

        if not hasattr(batch, field_name):
            raise ValueError(
                f"Batch is missing required field: {field_name}"
            )

        values = getattr(batch, field_name)

        if isinstance(values, list):
            return values

        if isinstance(values, tuple):
            return list(values)

        if torch.is_tensor(values):
            return values.detach().cpu().tolist()

        return [values]

    def _compute_metrics(
        self,
        y_true: list[int],
        y_pred: list[int],
    ) -> dict[str, Any]:
        """
        Compute classification metrics.

        Keep the metric implementation compatible with
        the existing experiment result format.
        """

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
                labels=list(range(self.num_classes)),
                average="macro",
                zero_division=0,
            )
        )

        _, _, weighted_f1, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=list(range(self.num_classes)),
                average="weighted",
                zero_division=0,
            )
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(self.num_classes)),
        )

        return {
            "accuracy": float(accuracy),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "weighted_f1": float(weighted_f1),
            "confusion_matrix": matrix.tolist(),
        }
