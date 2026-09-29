from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import random


@dataclass(slots=True, frozen=True)
class GroupSplit:
    """
    Group-level train/validation/test split.

    Groups are kept completely isolated between splits.
    """

    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    test_groups: tuple[str, ...]

    def validate(self) -> None:
        train = set(self.train_groups)
        validation = set(self.validation_groups)
        test = set(self.test_groups)

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
                "Training group split cannot be empty."
            )

        if not validation:
            raise ValueError(
                "Validation group split cannot be empty."
            )

        if not test:
            raise ValueError(
                "Test group split cannot be empty."
            )


def group_train_validation_test_split(
    groups: Sequence[str],
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> GroupSplit:
    """
    Split unique groups into train/validation/test sets.

    This function must be called BEFORE temporal windows are generated.

    Parameters
    ----------
    groups:
        Group identifiers, typically video_id or track_id.

    validation_fraction:
        Fraction of unique groups assigned to validation.

    test_fraction:
        Fraction of unique groups assigned to test.

    seed:
        Random seed.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    if not 0.0 < test_fraction < 1.0:
        raise ValueError(
            "test_fraction must be between 0 and 1."
        )

    if validation_fraction + test_fraction >= 1.0:
        raise ValueError(
            "validation_fraction + test_fraction must be < 1."
        )

    unique_groups = sorted(
        set(str(group) for group in groups)
    )

    if len(unique_groups) < 3:
        raise ValueError(
            "At least 3 unique groups are required."
        )

    rng = random.Random(seed)

    shuffled = list(unique_groups)
    rng.shuffle(shuffled)

    n_groups = len(shuffled)

    n_test = max(
        1,
        round(
            n_groups * test_fraction
        ),
    )

    n_validation = max(
        1,
        round(
            n_groups * validation_fraction
        ),
    )

    if (
        n_test + n_validation
        >= n_groups
    ):
        raise ValueError(
            "Fractions produce an empty training split."
        )

    test_groups = tuple(
        sorted(
            shuffled[:n_test]
        )
    )

    validation_groups = tuple(
        sorted(
            shuffled[
                n_test:
                n_test + n_validation
            ]
        )
    )

    train_groups = tuple(
        sorted(
            shuffled[
                n_test + n_validation:
            ]
        )
    )

    result = GroupSplit(
        train_groups=train_groups,
        validation_groups=validation_groups,
        test_groups=test_groups,
    )

    result.validate()

    return result


def group_from_sample(
    *,
    video_id: str,
    track_id: int,
    split_group: str,
) -> str:
    """
    Build a stable group identifier.

    split_group:
        "video" -> video-level isolation
        "track" -> track-level isolation
    """

    if split_group == "video":
        if not video_id:
            raise ValueError(
                "video_id cannot be empty."
            )

        return str(video_id)

    if split_group == "track":
        if not video_id:
            raise ValueError(
                "video_id cannot be empty."
            )

        if track_id < 0:
            raise ValueError(
                "track_id cannot be negative."
            )

        return f"{video_id}::track::{track_id}"

    raise ValueError(
        "split_group must be either "
        "'video' or 'track'."
    )
