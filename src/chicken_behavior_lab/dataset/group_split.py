from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from chicken_behavior_lab.dataset.sample import GraphSample


GroupKey = Literal["video_id", "track_id"]


@dataclass(slots=True)
class GroupSplitResult:
    train: list[GraphSample]
    validation: list[GraphSample]
    test: list[GraphSample]

    train_groups: list[str]
    validation_groups: list[str]
    test_groups: list[str]

    def validate(self) -> None:
        train_set = set(self.train_groups)
        validation_set = set(self.validation_groups)
        test_set = set(self.test_groups)

        if train_set & validation_set:
            raise ValueError(
                "Train and validation groups overlap."
            )

        if train_set & test_set:
            raise ValueError(
                "Train and test groups overlap."
            )

        if validation_set & test_set:
            raise ValueError(
                "Validation and test groups overlap."
            )

        sample_ids: set[str] = set()

        for samples in (
            self.train,
            self.validation,
            self.test,
        ):
            for sample in samples:
                if sample.sample_id in sample_ids:
                    raise ValueError(
                        "A sample appears in multiple splits: "
                        f"{sample.sample_id}"
                    )

                sample_ids.add(sample.sample_id)


class GroupAwareSplitter:
    """
    Split graph samples by group.

    The split is performed before temporal windows are created.

    Supported groups:
        - video_id
        - track_id

    This prevents overlapping temporal windows from the same
    recording/track from appearing in different splits.
    """

    def __init__(
        self,
        *,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        group_key: GroupKey = "video_id",
    ) -> None:

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.group_key = group_key

        self._validate_ratios()

    def _validate_ratios(self) -> None:
        ratios = (
            self.train_ratio,
            self.validation_ratio,
            self.test_ratio,
        )

        if any(ratio <= 0.0 for ratio in ratios):
            raise ValueError(
                "All split ratios must be greater than zero."
            )

        total = sum(ratios)

        if abs(total - 1.0) > 1e-8:
            raise ValueError(
                "train_ratio + validation_ratio + "
                "test_ratio must equal 1.0."
            )

    def split(
        self,
        samples: Sequence[GraphSample],
        *,
        seed: int = 42,
        shuffle_groups: bool = True,
    ) -> GroupSplitResult:

        samples = list(samples)

        if not samples:
            raise ValueError(
                "Cannot split an empty dataset."
            )

        groups = self._collect_groups(
            samples
        )

        if len(groups) < 3:
            raise ValueError(
                "At least three distinct groups are required "
                "for train/validation/test splitting."
            )

        ordered_groups = sorted(
            groups.keys()
        )

        if shuffle_groups:
            import random

            rng = random.Random(seed)
            rng.shuffle(ordered_groups)

        train_count, validation_count = (
            self._calculate_group_counts(
                len(ordered_groups)
            )
        )

        train_groups = ordered_groups[
            :train_count
        ]

        validation_groups = ordered_groups[
            train_count:
            train_count + validation_count
        ]

        test_groups = ordered_groups[
            train_count + validation_count:
        ]

        train_group_set = set(
            train_groups
        )

        validation_group_set = set(
            validation_groups
        )

        test_group_set = set(
            test_groups
        )

        train_samples: list[GraphSample] = []
        validation_samples: list[GraphSample] = []
        test_samples: list[GraphSample] = []

        for sample in samples:

            group = self._get_group(
                sample
            )

            if group in train_group_set:
                train_samples.append(
                    sample
                )

            elif group in validation_group_set:
                validation_samples.append(
                    sample
                )

            elif group in test_group_set:
                test_samples.append(
                    sample
                )

            else:
                raise RuntimeError(
                    f"Group '{group}' was not assigned "
                    "to any split."
                )

        result = GroupSplitResult(
            train=train_samples,
            validation=validation_samples,
            test=test_samples,
            train_groups=sorted(
                train_group_set
            ),
            validation_groups=sorted(
                validation_group_set
            ),
            test_groups=sorted(
                test_group_set
            ),
        )

        result.validate()

        self._validate_non_empty_splits(
            result
        )

        return result

    def _collect_groups(
        self,
        samples: Sequence[GraphSample],
    ) -> dict[str, list[GraphSample]]:

        groups: dict[
            str,
            list[GraphSample],
        ] = {}

        for sample in samples:

            group = self._get_group(
                sample
            )

            groups.setdefault(
                group,
                [],
            ).append(sample)

        return groups

    def _get_group(
        self,
        sample: GraphSample,
    ) -> str:

        metadata = sample.metadata or {}

        if self.group_key == "video_id":

            video_id = metadata.get(
                "video_id"
            )

            if not video_id:
                raise ValueError(
                    "Cannot perform video-level split because "
                    f"sample '{sample.sample_id}' has no "
                    "metadata.video_id."
                )

            return str(video_id)

        if self.group_key == "track_id":

            track_id = metadata.get(
                "track_id"
            )

            if track_id is None:
                raise ValueError(
                    "Cannot perform track-level split because "
                    f"sample '{sample.sample_id}' has no "
                    "metadata.track_id."
                )

            video_id = metadata.get(
                "video_id"
            )

            if video_id:
                return (
                    f"{video_id}::track::{track_id}"
                )

            return f"track::{track_id}"

        raise ValueError(
            f"Unsupported group_key: {self.group_key}"
        )

    def _calculate_group_counts(
        self,
        number_of_groups: int,
    ) -> tuple[int, int]:

        if number_of_groups < 3:
            raise ValueError(
                "At least three groups are required."
            )

        train_count = int(
            number_of_groups
            * self.train_ratio
        )

        validation_count = int(
            number_of_groups
            * self.validation_ratio
        )

        # Guarantee that every split receives
        # at least one group.
        train_count = max(
            train_count,
            1,
        )

        validation_count = max(
            validation_count,
            1,
        )

        if (
            train_count
            + validation_count
            >= number_of_groups
        ):
            validation_count = (
                number_of_groups
                - train_count
                - 1
            )

        if validation_count < 1:
            train_count = max(
                number_of_groups - 2,
                1,
            )

            validation_count = 1

        return (
            train_count,
            validation_count,
        )

    @staticmethod
    def _validate_non_empty_splits(
        result: GroupSplitResult,
    ) -> None:

        if not result.train:
            raise ValueError(
                "Train split is empty."
            )

        if not result.validation:
            raise ValueError(
                "Validation split is empty."
            )

        if not result.test:
            raise ValueError(
                "Test split is empty."
            )
