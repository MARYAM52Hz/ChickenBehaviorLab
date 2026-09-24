from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from chicken_behavior_lab.dataset.group_splitter import (
    DatasetSplits,
)

from chicken_behavior_lab.dataset.temporal_builder import (
    TemporalSequenceBuilder,
    TemporalWindow,
)


@dataclass(slots=True)
class TemporalDatasetSplits:
    """
    Temporal windows generated independently for
    train/validation/test.
    """

    train: list[TemporalWindow]
    validation: list[TemporalWindow]
    test: list[TemporalWindow]

    def validate(self) -> None:

        if not self.train:
            raise ValueError(
                "Temporal train split is empty."
            )

        if not self.validation:
            raise ValueError(
                "Temporal validation split is empty."
            )

        if not self.test:
            raise ValueError(
                "Temporal test split is empty."
            )

        train_groups = {
            (
                window.video_id,
                window.track_id,
            )
            for window in self.train
        }

        validation_groups = {
            (
                window.video_id,
                window.track_id,
            )
            for window in self.validation
        }

        test_groups = {
            (
                window.video_id,
                window.track_id,
            )
            for window in self.test
        }

        if train_groups & validation_groups:
            raise ValueError(
                "Temporal train and validation "
                "groups overlap."
            )

        if train_groups & test_groups:
            raise ValueError(
                "Temporal train and test "
                "groups overlap."
            )

        if validation_groups & test_groups:
            raise ValueError(
                "Temporal validation and test "
                "groups overlap."
            )


def build_temporal_splits(
    dataset_splits: DatasetSplits,
    sequence_length: int = 16,
    sequence_stride: int = 4,
    require_same_behavior: bool = False,
) -> TemporalDatasetSplits:
    """
    Build temporal windows independently inside
    train/validation/test subsets.
    """

    builder = TemporalSequenceBuilder(
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        require_same_behavior=(
            require_same_behavior
        ),
    )

    train_samples = [
        dataset_splits.train[index]
        for index in range(
            len(dataset_splits.train)
        )
    ]

    validation_samples = [
        dataset_splits.validation[index]
        for index in range(
            len(dataset_splits.validation)
        )
    ]

    test_samples = [
        dataset_splits.test[index]
        for index in range(
            len(dataset_splits.test)
        )
    ]

    train_windows = builder.build(
        train_samples
    )

    validation_windows = builder.build(
        validation_samples
    )

    test_windows = builder.build(
        test_samples
    )

    result = TemporalDatasetSplits(
        train=train_windows,
        validation=validation_windows,
        test=test_windows,
    )

    result.validate()

    return result
