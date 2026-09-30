from __future__ import annotations

from collections.abc import Iterator, Sequence

from chicken_behavior_lab.dataset.temporal_sample import (
    TemporalGraphSample,
)


class TemporalGraphDataset:
    """
    In-memory dataset of temporal graph samples.
    """

    def __init__(
        self,
        samples: Sequence[TemporalGraphSample],
    ) -> None:

        self.samples = list(
            samples
        )

        self._validate()

    def _validate(self) -> None:
        sample_ids: set[str] = set()

        for sample in self.samples:
            if not isinstance(
                sample,
                TemporalGraphSample,
            ):
                raise TypeError(
                    "Every dataset item must be a "
                    "TemporalGraphSample."
                )

            sample.validate()

            if sample.sample_id in sample_ids:
                raise ValueError(
                    "Duplicate sample_id: "
                    f"{sample.sample_id}"
                )

            sample_ids.add(
                sample.sample_id
            )

    def __len__(self) -> int:
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ) -> TemporalGraphSample:
        return self.samples[index]

    def __iter__(
        self,
    ) -> Iterator[TemporalGraphSample]:
        return iter(
            self.samples
        )

    @property
    def labels(self) -> list[int]:
        return [
            sample.label
            for sample in self.samples
        ]

    @property
    def behavior_ids(self) -> list[str]:
        return [
            sample.behavior_id
            for sample in self.samples
        ]

    @property
    def sample_ids(self) -> list[str]:
        return [
            sample.sample_id
            for sample in self.samples
        ]

    @property
    def video_ids(self) -> list[str | None]:
        return [
            sample.video_id
            for sample in self.samples
        ]

    @property
    def track_ids(self) -> list[int | None]:
        return [
            sample.track_id
            for sample in self.samples
        ]
