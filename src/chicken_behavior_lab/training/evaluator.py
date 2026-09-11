from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from chicken_behavior_lab.training.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
)


@dataclass(slots=True)
class EvaluationResult:
    """
    Complete result of model evaluation.
    """

    metrics: ClassificationMetrics

    y_true: torch.Tensor

    y_pred: torch.Tensor

    sample_ids: list[str]

    video_ids: list[str]

    track_ids: list[str]


class Evaluator:
    """
    Evaluate a trained graph classification model.
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

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
    ) -> EvaluationResult:

        self.model.to(
            self.device
        )

        self.model.eval()

        all_true = []

        all_pred = []

        sample_ids: list[str] = []

        video_ids: list[str] = []

        track_ids: list[str] = []

        for batch in data_loader:

            batch = batch.to(
                self.device
            )

            logits = self.model(
                batch
            )

            predictions = (
                torch.argmax(
                    logits,
                    dim=-1,
                )
            )

            true_labels = batch.y.view(
                -1
            )

            all_true.append(
                true_labels.detach().cpu()
            )

            all_pred.append(
                predictions.detach().cpu()
            )

            if hasattr(
                batch,
                "sample_id",
            ):
                sample_ids.extend(
                    list(
                        batch.sample_id
                    )
                )

            if hasattr(
                batch,
                "video_id",
            ):
                video_ids.extend(
                    list(
                        batch.video_id
                    )
                )

            if hasattr(
                batch,
                "track_id",
            ):
                track_ids.extend(
                    list(
                        batch.track_id
                    )
                )

        if not all_true:
            raise RuntimeError(
                "Evaluation DataLoader "
                "returned no batches."
            )

        y_true = torch.cat(
            all_true
        )

        y_pred = torch.cat(
            all_pred
        )

        metrics = (
            compute_classification_metrics(
                y_true=y_true,
                y_pred=y_pred,
                num_classes=self.num_classes,
            )
        )

        return EvaluationResult(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sample_ids=sample_ids,
            video_ids=video_ids,
            track_ids=track_ids,
        )
