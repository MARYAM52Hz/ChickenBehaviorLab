from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class SplitGroups:
    """
    Group-level train/validation/test split.

    The values stored here are group identifiers, not
    individual samples.
    """

    train: list[str]
    validation: list[str]
    test: list[str]

    def validate(self) -> None:
        train = set(self.train)
        validation = set(self.validation)
        test = set(self.test)

        if train & validation:
            raise ValueError(
                "Train and validation groups overlap."
            )

        if train & test:
            raise ValueError(
                "Train and test groups overlap."
            )

        if validation & test:
            raise ValueError(
                "Validation and test groups overlap."
            )

        if not train:
            raise ValueError(
                "Train split cannot be empty."
            )

        if not validation:
            raise ValueError(
                "Validation split cannot be empty."
            )

        if not test:
            raise ValueError(
                "Test split cannot be empty."
            )


def normalize_group_id(
    value: object,
) -> str:
    """
    Convert a group identifier to a stable string.
    """

    if value is None:
        raise ValueError(
            "Group identifier cannot be None."
        )

    value = str(value).strip()

    if not value:
        raise ValueError(
            "Group identifier cannot be empty."
        )

    return value


def collect_groups(
    samples: Iterable,
    group_key: str,
) -> list[str]:
    """
    Collect unique group IDs from dataset samples.

    Supported sample metadata layouts:

        sample.metadata[group_key]

    or:

        sample.<group_key>
    """

    groups: set[str] = set()

    for sample in samples:
        metadata = getattr(
            sample,
            "metadata",
            None,
        )

        value = None

        if isinstance(metadata, dict):
            value = metadata.get(
                group_key
            )

        if value is None:
            value = getattr(
                sample,
                group_key,
                None,
            )

        if value is None:
            raise ValueError(
                f"Could not find group key "
                f"'{group_key}' in sample."
            )

        groups.add(
            normalize_group_id(value)
        )

    return sorted(groups)


def build_group_id(
    sample,
    group_by: str,
) -> str:
    """
    Build a group identifier for one sample.

    Supported values:

        video
        track
        video_track
    """

    metadata = getattr(
        sample,
        "metadata",
        None,
    )

    if not isinstance(metadata, dict):
        metadata = {}

    video_id = metadata.get(
        "video_id",
        getattr(
            sample,
            "video_id",
            None,
        ),
    )

    track_id = metadata.get(
        "track_id",
        getattr(
            sample,
            "track_id",
            None,
        ),
    )

    if group_by == "video":
        return normalize_group_id(
            video_id
        )

    if group_by == "track":
        return normalize_group_id(
            track_id
        )

    if group_by == "video_track":
        video = normalize_group_id(
            video_id
        )

        track = normalize_group_id(
            track_id
        )

        return f"{video}::track_{track}"

    raise ValueError(
        "group_by must be one of: "
        "'video', 'track', 'video_track'."
    )


def collect_group_ids(
    samples: Iterable,
    group_by: str,
) -> list[str]:
    """
    Collect unique group IDs using the requested grouping rule.
    """

    groups = set()

    for sample in samples:
        groups.add(
            build_group_id(
                sample,
                group_by=group_by,
            )
        )

    return sorted(groups)


def validate_group_split(
    split: SplitGroups,
) -> None:
    """
    Validate that no group appears in more than one split.
    """

    split.validate()


def split_group_ids(
    groups: list[str],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitGroups:
    """
    Randomly split group IDs.

    IMPORTANT:
    This function splits GROUPS, not individual samples.
    """

    import random

    if not groups:
        raise ValueError(
            "Cannot split an empty group list."
        )

    if train_ratio <= 0:
        raise ValueError(
            "train_ratio must be > 0."
        )

    if validation_ratio <= 0:
        raise ValueError(
            "validation_ratio must be > 0."
        )

    if test_ratio <= 0:
        raise ValueError(
            "test_ratio must be > 0."
        )

    total_ratio = (
        train_ratio
        + validation_ratio
        + test_ratio
    )

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            "train_ratio + validation_ratio + "
            "test_ratio must equal 1."
        )

    groups = list(
        dict.fromkeys(groups)
    )

    rng = random.Random(seed)

    rng.shuffle(groups)

    total = len(groups)

    train_count = max(
        1,
        round(
            total * train_ratio
        ),
    )

    validation_count = max(
        1,
        round(
            total * validation_ratio
        ),
    )

    # Guarantee at least one test group.
    test_count = (
        total
        - train_count
        - validation_count
    )

    if test_count < 1:
        test_count = 1

        if train_count > validation_count:
            train_count -= 1
        else:
            validation_count -= 1

    if train_count < 1:
        raise ValueError(
            "Not enough groups for a train split."
        )

    if validation_count < 1:
        raise ValueError(
            "Not enough groups for a validation split."
        )

    if test_count < 1:
        raise ValueError(
            "Not enough groups for a test split."
        )

    train = groups[
        :train_count
    ]

    validation = groups[
        train_count:
        train_count + validation_count
    ]

    test = groups[
        train_count + validation_count:
    ]

    split = SplitGroups(
        train=train,
        validation=validation,
        test=test,
    )

    split.validate()

    return split
