from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """
    Classification metrics for one evaluation pass.
    """

    accuracy: float

    macro_precision: float

    macro_recall: float

    macro_f1: float

    confusion_matrix: Tensor


def build_confusion_matrix(
    predictions: Tensor,
    targets: Tensor,
    num_classes: int,
) -> Tensor:
    """
    Construct a multiclass confusion matrix.

    Rows:
        true class

    Columns:
        predicted class
    """

    if predictions.ndim != 1:
        raise ValueError(
            "predictions must have shape (N,)."
        )

    if targets.ndim != 1:
        raise ValueError(
            "targets must have shape (N,)."
        )

    if predictions.shape[0] != targets.shape[0]:
        raise ValueError(
            "predictions and targets must "
            "have the same length."
        )

    if num_classes <= 1:
        raise ValueError(
            "num_classes must be greater than 1."
        )

    predictions = predictions.to(
        dtype=torch.long
    )

    targets = targets.to(
        dtype=torch.long
    )

    if predictions.numel() == 0:
        return torch.zeros(
            (
                num_classes,
                num_classes,
            ),
            dtype=torch.long,
        )

    if torch.any(predictions < 0):
        raise ValueError(
            "predictions cannot contain "
            "negative class indices."
        )

    if torch.any(targets < 0):
        raise ValueError(
            "targets cannot contain "
            "negative class indices."
        )

    if torch.any(
        predictions >= num_classes
    ):
        raise ValueError(
            "prediction index exceeds "
            "num_classes."
        )

    if torch.any(
        targets >= num_classes
    ):
        raise ValueError(
            "target index exceeds "
            "num_classes."
        )

    flat_indices = (
        targets * num_classes
        + predictions
    )

    counts = torch.bincount(
        flat_indices,
        minlength=(
            num_classes
            * num_classes
        ),
    )

    return counts.reshape(
        num_classes,
        num_classes,
    )


def compute_classification_metrics(
    predictions: Tensor,
    targets: Tensor,
    num_classes: int,
) -> ClassificationMetrics:
    """
    Compute multiclass classification metrics.
    """

    confusion_matrix = (
        build_confusion_matrix(
            predictions=predictions,
            targets=targets,
            num_classes=num_classes,
        )
    )

    matrix = confusion_matrix.to(
        dtype=torch.float32
    )

    true_positive = torch.diag(
        matrix
    )

    predicted_positive = matrix.sum(
        dim=0
    )

    actual_positive = matrix.sum(
        dim=1
    )

    precision = torch.where(
        predicted_positive > 0,
        true_positive
        / predicted_positive,
        torch.zeros_like(
            true_positive
        ),
    )

    recall = torch.where(
        actual_positive > 0,
        true_positive
        / actual_positive,
        torch.zeros_like(
            true_positive
        ),
    )

    denominator = (
        precision + recall
    )

    f1 = torch.where(
        denominator > 0,
        2.0
        * precision
        * recall
        / denominator,
        torch.zeros_like(
            denominator
        ),
    )

    total = matrix.sum()

    if total > 0:
        accuracy = (
            true_positive.sum()
            / total
        ).item()
    else:
        accuracy = 0.0

    return ClassificationMetrics(
        accuracy=float(
            accuracy
        ),
        macro_precision=float(
            precision.mean().item()
        ),
        macro_recall=float(
            recall.mean().item()
        ),
        macro_f1=float(
            f1.mean().item()
        ),
        confusion_matrix=(
            confusion_matrix
        ),
    )
