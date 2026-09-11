import numpy as np
import torch

from chicken_behavior_lab.training.metrics import (
    compute_classification_metrics,
)


def test_perfect_classification():

    y_true = torch.tensor(
        [0, 1, 2, 0, 1, 2]
    )

    y_pred = torch.tensor(
        [0, 1, 2, 0, 1, 2]
    )

    metrics = (
        compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=3,
        )
    )

    assert metrics.accuracy == 1.0

    assert metrics.macro_f1 == 1.0

    assert np.all(
        metrics.f1_per_class
        == 1.0
    )


def test_confusion_matrix():

    y_true = torch.tensor(
        [0, 0, 1, 1]
    )

    y_pred = torch.tensor(
        [0, 1, 1, 1]
    )

    metrics = (
        compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=2,
        )
    )

    expected = np.array(
        [
            [1, 1],
            [0, 2],
        ]
    )

    assert np.array_equal(
        metrics.confusion_matrix,
        expected,
    )


def test_macro_f1_is_bounded():

    y_true = torch.tensor(
        [0, 0, 0, 1]
    )

    y_pred = torch.tensor(
        [0, 0, 1, 1]
    )

    metrics = (
        compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=2,
        )
    )

    assert (
        0.0
        <= metrics.macro_f1
        <= 1.0
    )
