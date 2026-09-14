import numpy as np

from chicken_behavior_lab.analysis import (
    ConfusionMatrixAnalyzer,
)


def test_confusion_matrix_summary():

    matrix = np.array(
        [
            [8, 2, 0],
            [1, 7, 2],
            [0, 1, 9],
        ]
    )

    analyzer = (
        ConfusionMatrixAnalyzer(
            matrix=matrix,
            class_names=[
                "feeding",
                "walking",
                "standing",
            ],
        )
    )

    assert (
        analyzer.num_classes
        == 3
    )

    assert (
        analyzer.accuracy()
        == 24 / 30
    )


def test_most_confused_pairs():

    matrix = np.array(
        [
            [8, 2, 0],
            [1, 7, 2],
            [0, 1, 9],
        ]
    )

    analyzer = (
        ConfusionMatrixAnalyzer(
            matrix=matrix,
            class_names=[
                "feeding",
                "walking",
                "standing",
            ],
        )
    )

    pairs = (
        analyzer.most_confused_pairs(
            top_k=2
        )
    )

    assert len(pairs) == 2

    assert (
        pairs[0]["count"]
        == 2
    )
