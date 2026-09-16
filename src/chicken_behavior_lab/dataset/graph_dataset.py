from __future__ import annotations

from collections.abc import Iterator

from chicken_behavior_lab.dataset.sample import GraphSample


class GraphDataset:
    """
    Collection of validated GraphSample objects.
    """

    def __init__(
        self,
        samples: list[GraphSample],
    ) -> None:
        self.samples = list(samples)
        self._validate()

    def _validate(self) -> None:
        sample_ids: set[str] = set()

        for sample in self.samples:
            if not isinstance(
                sample,
                GraphSample,
            ):
                raise TypeError(
                    "Every dataset item must be a GraphSample."
                )

            sample.validate()

            if sample.sample_id in sample_ids:
                raise ValueError(
                    f"Duplicate sample_id: "
                    f"{sample.sample_id}"
                )

            sample_ids.add(sample.sample_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> GraphSample:
        return self.samples[index]

    def __iter__(self) -> Iterator[GraphSample]:
        return iter(self.samples)

    @property
    def labels(self) -> list[int]:
        return [
            sample.label
            for sample in self.samples
        ]

    @property
    def sample_ids(self) -> list[str]:
        return [
            sample.sample_id
            for sample in self.samples
        ]

    @property
    def behavior_ids(self) -> list[str]:
        return [
            sample.behavior_id
            for sample in self.samples
        ]
