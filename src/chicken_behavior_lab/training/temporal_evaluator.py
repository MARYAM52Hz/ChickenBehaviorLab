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

from chicken_behavior_lab.dataset.temporal_batch import (
    TemporalBatch,
)


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
    prediction_records: list[
        TemporalPredictionRecord
    ]


class TemporalEvaluator:

    def __init__(
        self,
        model,
        device: torch.device | str,
        index_to_label: dict[int, str] | None = None,
        label_to_index: dict[str, int] | None = None,
    ) -> None:

        self.model = model
        self.device = torch.device(device)

        if (
            index_to_label is not None
            and label_to_index is not None
        ):
            generated = {
                index: label
                for label, index
                in label_to_index.items()
            }

            if generated != index_to_label:
                raise ValueError(
                    "index_to_label and label_to_index "
                    "describe different mappings."
                )

        if label_to_index is not None:
            self.label_to_index = dict(
                label_to_index
            )
            self.index_to_label = {
                index: label
                for label, index
                in label_to_index.items()
            }

        else:
            self.index_to_label = dict(
                index_to_label or {}
            )

            self.label_to_index = {
                label: index
                for index, label
                in self.index_to_label.items()
            }

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

        prediction_records: list[
            TemporalPredictionRecord
        ] = []

        for batch in data_loader:

            if not isinstance(
                batch,
                TemporalBatch,
            ):
                raise TypeError(
                    "TemporalEvaluator expects "
                    "TemporalBatch instances."
                )

            batch = batch.to(
                self.device
            )

            logits = self.model(batch)

            if logits.ndim != 2:
                raise ValueError(
                    "Model output must have shape [B, C]."
                )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=-1,
            )

            confidence = torch.max(
                probabilities,
                dim=-1,
            ).values

            true_values = batch.y.long()

            for index in range(
                batch.batch_size
            ):

                true_index = int(
                    true_values[index].item()
                )

                predicted_index = int(
                    predictions[index].item()
                )

                sample_id = batch.sample_id[
                    index
                ]

                video_id = batch.video_id[
                    index
                ]

                track_id = int(
                    batch.track_id[index].item()
                )

                start_frame = int(
                    batch.start_frame[index].item()
                )

                end_frame = int(
                    batch.end_frame[index].item()
                )

                conf = float(
                    confidence[index].item()
                )

                y_true.append(
                    true_index
                )

                y_pred.append(
                    predicted_index
                )

                sample_ids.append(
                    sample_id
                )

                video_ids.append(
                    video_id
                )

                track_ids.append(
                    track_id
                )

                prediction_records.append(
                    TemporalPredictionRecord(
                        sample_id=sample_id,
                        video_id=video_id,
                        track_id=track_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        true_index=true_index,
                        predicted_index=predicted_index,
                        confidence=conf,
                    )
                )

        metrics = self._compute_metrics(
            y_true,
            y_pred,
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
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "macro_precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=labels,
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
