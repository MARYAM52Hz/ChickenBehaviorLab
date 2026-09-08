import torch

from chicken_behavior_lab.training import (
    build_confusion_matrix,
    compute_classification_metrics,
)


def test_perfect_predictions():

    predictions = torch.tensor(
        [
            0,
            1,
            2,
            0,
            1,
            2,
        ]
    )

    targets = torch.tensor(
        [
            0,
            1,
            2,
            0,
            1,
            2,
        ]
    )

    metrics = (
        compute_classification_metrics(
            predictions=predictions,
            targets=targets,
            num_classes=3,
        )
    )

    assert metrics.accuracy == 1.0
    assert metrics.macro_precision == 1.0
    assert metrics.macro_recall == 1.0
    assert metrics.macro_f1 == 1.0


def test_confusion_matrix():

    predictions = torch.tensor(
        [
            0,
            1,
            1,
            2,
        ]
    )

    targets = torch.tensor(
        [
            0,
            1,
            2,
            2,
        ]
    )

    matrix = build_confusion_matrix(
        predictions=predictions,
        targets=targets,
        num_classes=3,
    )

    expected = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
        ]
    )

    assert torch.equal(
        matrix,
        expected,
    )


def test_metrics_are_bounded():

    predictions = torch.tensor(
        [
            0,
            0,
            1,
            2,
            2,
        ]
    )

    targets = torch.tensor(
        [
            0,
            1,
            1,
            2,
            1,
        ]
    )

    metrics = (
        compute_classification_metrics(
            predictions=predictions,
            targets=targets,
            num_classes=3,
        )
    )

    assert (
        0.0
        <= metrics.accuracy
        <= 1.0
    )

    assert (
        0.0
        <= metrics.macro_precision
        <= 1.0
    )

    assert (
        0.0
        <= metrics.macro_recall
        <= 1.0
    )

    assert (
        0.0
        <= metrics.macro_f1
        <= 1.0
    )


def test_missing_predicted_class():

    predictions = torch.tensor(
        [
            0,
            0,
            0,
        ]
    )

    targets = torch.tensor(
        [
            0,
            1,
            2,
        ]
    )

    metrics = (
        compute_classification_metrics(
            predictions=predictions,
            targets=targets,
            num_classes=3,
        )
    )

    assert torch.isfinite(
        torch.tensor(
            metrics.macro_f1
        )
    )
