from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ConfusionMatrixAnalyzer:
    """
    Analyze a multiclass confusion matrix.
    """

    matrix: np.ndarray
    class_names: Sequence[str]

    def __post_init__(self) -> None:
        matrix = np.asarray(
            self.matrix
        )

        if matrix.ndim != 2:
            raise ValueError(
                "Confusion matrix must be 2-dimensional."
            )

        if (
            matrix.shape[0]
            != matrix.shape[1]
        ):
            raise ValueError(
                "Confusion matrix must be square."
            )

        if matrix.shape[0] != len(
            self.class_names
        ):
            raise ValueError(
                "Number of class names must "
                "match matrix dimensions."
            )

    @property
    def num_classes(self) -> int:
        return self.matrix.shape[0]

    def per_class_summary(
        self,
    ) -> list[dict[str, object]]:
        """
        Return TP, FP, FN, and support for
        every class.
        """

        matrix = np.asarray(
            self.matrix
        )

        summaries = []

        for index, class_name in enumerate(
            self.class_names
        ):

            true_positive = int(
                matrix[index, index]
            )

            false_positive = int(
                matrix[:, index].sum()
                - true_positive
            )

            false_negative = int(
                matrix[index, :].sum()
                - true_positive
            )

            support = int(
                matrix[index, :].sum()
            )

            summaries.append(
                {
                    "class": class_name,
                    "true_positive": (
                        true_positive
                    ),
                    "false_positive": (
                        false_positive
                    ),
                    "false_negative": (
                        false_negative
                    ),
                    "support": support,
                }
            )

        return summaries

    def most_confused_pairs(
        self,
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        """
        Return the most frequent off-diagonal
        confusion pairs.
        """

        if top_k < 1:
            raise ValueError(
                "top_k must be >= 1."
            )

        matrix = np.asarray(
            self.matrix
        )

        pairs = []

        for true_index in range(
            self.num_classes
        ):

            for predicted_index in range(
                self.num_classes
            ):

                if (
                    true_index
                    == predicted_index
                ):
                    continue

                count = int(
                    matrix[
                        true_index,
                        predicted_index,
                    ]
                )

                if count == 0:
                    continue

                pairs.append(
                    {
                        "true_class": (
                            self.class_names[
                                true_index
                            ]
                        ),
                        "predicted_class": (
                            self.class_names[
                                predicted_index
                            ]
                        ),
                        "count": count,
                    }
                )

        pairs.sort(
            key=lambda item: item[
                "count"
            ],
            reverse=True,
        )

        return pairs[:top_k]

    def accuracy(self) -> float:
        """
        Compute accuracy directly from
        the confusion matrix.
        """

        total = float(
            self.matrix.sum()
        )

        if total == 0:
            return 0.0

        return float(
            np.trace(self.matrix)
            / total
        )
