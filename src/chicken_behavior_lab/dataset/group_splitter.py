from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.dataset.group_split import (
    SplitGroups,
    build_group_id,
    split_group_ids,
)

from chicken_behavior_lab.dataset.group_subset import (
    GroupSubset,
)


@dataclass(slots=True)
class DatasetSplits:
    train: GroupSubset
    validation: GroupSubset
    test: GroupSubset

    groups: SplitGroups

    def validate(self) -> None:
        self.groups.validate()

        if len(self.train) == 0:
            raise ValueError(
                "Train dataset is empty."
            )

        if len(self.validation) == 0:
            raise ValueError(
                "Validation dataset is empty."
            )

        if len(self.test) == 0:
            raise ValueError(
                "Test dataset is empty."
            )


def split_dataset_by_group(
    dataset,
    group_by: str = "video",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetSplits:
    """
    Split a dataset by video/track group.

    The split is performed on the ORIGINAL dataset items.

    Temporal windows must be generated only after this split
    if the underlying samples are raw frame/graph samples.
    """

    if len(dataset) < 1:
        raise ValueError(
            "Cannot split an empty dataset."
        )

    sample_groups: list[str] = []

    for index in range(
        len(dataset)
    ):
        sample = dataset[index]

        sample_groups.append(
            build_group_id(
                sample,
                group_by=group_by,
            )
        )

    unique_groups = sorted(
        set(sample_groups)
    )

    split_groups = split_group_ids(
        unique_groups,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_set = set(
        split_groups.train
    )

    validation_set = set(
        split_groups.validation
    )

    test_set = set(
        split_groups.test
    )

    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []

    for index, group in enumerate(
        sample_groups
    ):
        if group in train_set:
            train_indices.append(index)

        elif group in validation_set:
            validation_indices.append(index)

        elif group in test_set:
            test_indices.append(index)

        else:
            raise RuntimeError(
                "Encountered a group that does not "
                "belong to any split."
            )

    splits = DatasetSplits(
        train=GroupSubset(
            dataset,
            train_indices,
        ),
        validation=GroupSubset(
            dataset,
            validation_indices,
        ),
        test=GroupSubset(
            dataset,
            test_indices,
        ),
        groups=split_groups,
    )

    splits.validate()

    return splits
