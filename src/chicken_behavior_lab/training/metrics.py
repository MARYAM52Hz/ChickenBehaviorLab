from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class ClassificationMetrics:
    """
    Classification metrics for a multi-class classifier.
    """

    accuracy: float

    macro_precision: float

    macro_recall: float

    macro_f1: float

    weighted_f1: float

    confusion_matrix: np.ndarray

    precision_per_class: np.ndarray

    recall_per_class: np.ndarray

    f1_per_class: np.ndarray

    support_per_class: np.ndarray


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    """
    Element-wise division with zero protection.
    """

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(
            numerator,
            dtype=float,
        ),
        where=denominator != 0,
    )


def compute_classification_metrics(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    num_classes: int,
) -> ClassificationMetrics:
    """
    Compute multi-class classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth class indices.

    y_pred:
        Predicted class indices.

    num_classes:
        Number of classes.
    """

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(
        y_true,
        dtype=int,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    ).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have "
            "the same shape."
        )

    if num_classes <= 0:
        raise ValueError(
            "num_classes must be > 0."
        )

    confusion_matrix = np.zeros(
        (
            num_classes,
            num_classes,
        ),
        dtype=int,
    )

    for true_label, predicted_label in zip(
        y_true,
        y_pred,
    ):
        if not (
            0 <= true_label < num_classes
        ):
            raise ValueError(
                f"Invalid true label: "
                f"{true_label}"
            )

        if not (
            0 <= predicted_label < num_classes
        ):
            raise ValueError(
                f"Invalid predicted label: "
                f"{predicted_label}"
            )

        confusion_matrix[
            true_label,
            predicted_label,
        ] += 1

    total = confusion_matrix.sum()

    if total == 0:
        raise ValueError(
            "Cannot compute metrics "
            "from an empty dataset."
        )

    accuracy = (
        np.trace(confusion_matrix)
        / total
    )

    true_positive = np.diag(
        confusion_matrix
    ).astype(float)

    predicted_positive = (
        confusion_matrix.sum(
            axis=0
        ).astype(float)
    )

    actual_positive = (
        confusion_matrix.sum(
            axis=1
        ).astype(float)
    )

    precision = _safe_divide(
        true_positive,
        predicted_positive,
    )

    recall = _safe_divide(
        true_positive,
        actual_positive,
    )

    f1 = _safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    support = actual_positive.astype(
        int
    )

    macro_precision = float(
        precision.mean()
    )

    macro_recall = float(
        recall.mean()
    )

    macro_f1 = float(
        f1.mean()
    )

    total_support = support.sum()

    if total_support == 0:
        weighted_f1 = 0.0

    else:
        weighted_f1 = float(
            np.sum(
                f1 * support
            )
            / total_support
        )

    return ClassificationMetrics(
        accuracy=float(accuracy),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        confusion_matrix=confusion_matrix,
        precision_per_class=precision,
        recall_per_class=recall,
        f1_per_class=f1,
        support_per_class=support,
    )


def format_classification_report(
    metrics: ClassificationMetrics,
    class_names: list[str],
) -> str:
    """
    Convert classification metrics into
    a human-readable text report.
    """

    if len(class_names) != len(
        metrics.f1_per_class
    ):
        raise ValueError(
            "Number of class names does not "
            "match number of classes."
        )

    lines = []

    lines.append(
        "Classification Report"
    )

    lines.append(
        "=" * 78
    )

    lines.append(
        f"{'Class':<20}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>12}"
    )

    lines.append(
        "-" * 78
    )

    for index, class_name in enumerate(
        class_names
    ):

        lines.append(
            f"{class_name:<20}"
            f"{metrics.precision_per_class[index]:>12.4f}"
            f"{metrics.recall_per_class[index]:>12.4f}"
            f"{metrics.f1_per_class[index]:>12.4f}"
            f"{metrics.support_per_class[index]:>12d}"
        )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"{'Accuracy':<20}"
        f"{metrics.accuracy:>12.4f}"
    )

    lines.append(
        f"{'Macro Precision':<20}"
        f"{metrics.macro_precision:>12.4f}"
    )

    lines.append(
        f"{'Macro Recall':<20}"
        f"{metrics.macro_recall:>12.4f}"
    )

    lines.append(
        f"{'Macro F1':<20}"
        f"{metrics.macro_f1:>12.4f}"
    )

    lines.append(
        f"{'Weighted F1':<20}"
        f"{metrics.weighted_f1:>12.4f}"
    )

    return "\n".join(
        lines
    )
