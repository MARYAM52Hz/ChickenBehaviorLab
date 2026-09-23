from __future__ import annotations

from collections.abc import Sequence


class GroupSubset:
    """
    Dataset subset selected by group IDs.

    This class does not copy the underlying data.
    """

    def __init__(
        self,
        dataset,
        indices: Sequence[int],
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)

        if not self.indices:
            raise ValueError(
                "GroupSubset cannot be empty."
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self,
        index: int,
    ):
        dataset_index = self.indices[
            index
        ]

        return self.dataset[
            dataset_index
        ]

    def original_index(
        self,
        index: int,
    ) -> int:
        return self.indices[index]
