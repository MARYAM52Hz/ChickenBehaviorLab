"""
ChickenBehaviorLab Graph Dataset
================================

Dataset container for temporal skeleton graphs.

Important design principle
--------------------------

Dataset splitting must happen at the recording/video level,
not at the frame level.

This prevents temporal leakage between training and testing.
"""

from __future__ import annotations

from collections.abc import Iterator

from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)


class GraphDataset:
    """
    Collection of GraphSample objects.

    Parameters
    ----------
    samples:
        Iterable containing graph samples.
    """

    def __init__(
        self,
        samples: list[GraphSample],
    ) -> None:

        self.samples = list(
            samples
        )

        self._validate()

    def _validate(self) -> None:
        """
        Validate all dataset samples.
        """

        sample_ids: set[str] = set()

        for sample in self.samples:

            if not isinstance(
                sample,
                GraphSample,
            ):

                raise TypeError(
                    "Every dataset item "
                    "must be a GraphSample."
                )

            sample.validate()

            if (
                sample.sample_id
                in sample_ids
            ):

                raise ValueError(
                    "Duplicate sample_id: "
                    f"{sample.sample_id}"
                )

            sample_ids.add(
                sample.sample_id
            )

    def __len__(self) -> int:
        """
        Return number of samples.
        """

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ) -> GraphSample:
        """
        Return a dataset sample.
        """

        return self.samples[index]

    def __iter__(
        self,
    ) -> Iterator[GraphSample]:
        """
        Iterate over dataset samples.
        """

        return iter(
            self.samples
        )

    @property
    def labels(self) -> list[int]:
        """
        Return all sample labels.
        """

        return [
            sample.label
            for sample in self.samples
        ]

    @property
    def sample_ids(self) -> list[str]:
        """
        Return all sample identifiers.
        """

        return [
            sample.sample_id
            for sample in self.samples
        ]
