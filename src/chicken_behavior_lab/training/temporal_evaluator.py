from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from chicken_behavior_lab.dataset.temporal_batch import TemporalBatch


@dataclass(slots=True)
class TemporalPredictionRecord:
    sample_id: str
    video_id: str
    track_id: int
    start_frame: int
    end_frame: int
    true_index: int
    predicted_index: int
    confidence: float


@dataclass(slots=True)
class TemporalEvaluationResult:
    metrics: dict[str, Any]
    y_true: list[int]
    y_pred: list[int]
    sample_ids: list[str]
    video_ids: list[str]
    track_ids: list[int]
    prediction_records: list[TemporalPredictionRecord]


class TemporalEvaluator:
    """
    Evaluate TemporalBehaviorGNN models.

    The evaluator expects a TemporalBatch and a model that returns:

        logits: [B, num_classes]

    Metadata from TemporalBatch is preserved so that predictions can later
    be connected back to videos, tracks, and temporal windows.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str,
        index_to_label: dict[int, str] | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.index_to_label = index_to_label or {}

    @torch.no_grad()
    def evaluate(
        self,
        data_loader,
    ) -> TemporalEvaluationResult:
        self.model.eval()

        y_true: list[int] = []
        y_pred: list[int] = []

        sample_ids: list[str] = []
        video_ids: list[str] = []
        track_ids: list[int] = []

        prediction_records: list[TemporalPredictionRecord] = []

        for batch in data_loader:
            if not isinstance(batch, TemporalBatch):
                raise TypeError(
                    "TemporalEvaluator expects batches of type TemporalBatch."
                )

            batch = batch.to(self.device)

            logits = self.model(batch)

            if logits.ndim != 2:
                raise ValueError(
                    "Model output must have shape [B, num_classes]."
                )

            probabilities = torch.softmax(logits, dim=-1)

            predictions = torch.argmax(
                probabilities,
                dim=-1,
            )

            confidence = torch.max(
                probabilities,
                dim=-1,
            ).values

            batch_true = batch.y.detach().cpu().tolist()
            batch_pred = predictions.detach().cpu().tolist()
            batch_confidence = confidence.detach().cpu().tolist()

            y_true.extend(int(value) for value in batch_true)
            y_pred.extend(int(value) for value in batch_pred)

            sample_ids.extend(batch.sample_id)
            video_ids.extend(batch.video_id)

            track_ids.extend(
                int(value)
                for value in batch.track_id.detach().cpu().tolist()
            )

            for index in range(batch.batch_size):
                prediction_records.append(
                    TemporalPredictionRecord(
                        sample_id=batch.sample_id[index],
                        video_id=batch.video_id[index],
                        track_id=int(
                            batch.track_id[index].item()
                        ),
                        start_frame=int(
                            batch.start_frame[index].item()
                        ),
                        end_frame=int(
                            batch.end_frame[index].item()
                        ),
                        true_index=int(batch_true[index]),
                        predicted_index=int(batch_pred[index]),
                        confidence=float(
                            batch_confidence[index]
                        ),
                    )
                )

        metrics = self._compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
        )

        return TemporalEvaluationResult(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sample_ids=sample_ids,
            video_ids=video_ids,
            track_ids=track_ids,
            prediction_records=prediction_records,
        )

    def _compute_metrics(
        self,
        y_true: list[int],
        y_pred: list[int],
    ) -> dict[str, Any]:
        if not y_true:
            raise ValueError(
                "Cannot evaluate an empty temporal dataset."
            )

        labels = sorted(
            set(y_true) | set(y_pred)
        )

        metrics: dict[str, Any] = {
            "accuracy": float(
                accuracy_score(y_true, y_pred)
            ),
            "macro_precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "confusion_matrix": confusion_matrix(
                y_true,
                y_pred,
                labels=labels,
            ).tolist(),
            "labels": labels,
            "num_samples": len(y_true),
        }

        if self.index_to_label:
            metrics["label_names"] = [
                self.index_to_label.get(
                    index,
                    str(index),
                )
                for index in labels
            ]

        return metrics
