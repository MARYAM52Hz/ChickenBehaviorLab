from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)


@dataclass(slots=True)
class DatasetSplit:
    """
    Container for train, validation and test samples.
    """

    train: list[GraphSample]

    validation: list[GraphSample]

    test: list[GraphSample]


def _group_samples(
    samples: Sequence[GraphSample],
    group_by: str,
) -> dict[str, list[GraphSample]]:
    """
    Group samples by video_id or track_id.
    """

    groups: dict[
        str,
        list[GraphSample],
    ] = {}

    for sample in samples:

        if group_by == "video":

            group_id = sample.video_id

        elif group_by == "track":

            group_id = (
                f"{sample.video_id}::"
                f"{sample.track_id}"
            )

        else:

            raise ValueError(
                "group_by must be either "
                "'video' or 'track'."
            )

        groups.setdefault(
            group_id,
            [],
        ).append(sample)

    return groups


def group_train_validation_test_split(
    samples: Sequence[GraphSample],
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    group_by: str = "video",
    random_seed: int = 42,
) -> DatasetSplit:
    """
    Split samples without allowing samples from
    the same group to appear in multiple splits.

    Parameters
    ----------
    samples:
        Input graph samples.

    validation_fraction:
        Fraction of groups assigned to validation.

    test_fraction:
        Fraction of groups assigned to test.

    group_by:
        'video' or 'track'.

    random_seed:
        Reproducibility seed.
    """

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be "
            "between 0 and 1."
        )

    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(
            "test_fraction must be "
            "between 0 and 1."
        )

    if (
        validation_fraction
        + test_fraction
        >= 1.0
    ):
        raise ValueError(
            "validation_fraction + "
            "test_fraction must be < 1."
        )

    if len(samples) == 0:
        raise ValueError(
            "Cannot split an empty dataset."
        )

    groups = _group_samples(
        samples,
        group_by=group_by,
    )

    group_ids = list(
        groups.keys()
    )

    rng = random.Random(
        random_seed
    )

    rng.shuffle(group_ids)

    num_groups = len(group_ids)

    if num_groups < 3:
        raise ValueError(
            "At least 3 groups are required "
            "for train/validation/test splitting."
        )

    num_test = max(
        1,
        round(
            num_groups
            * test_fraction
        ),
    )

    num_validation = max(
        1,
        round(
            num_groups
            * validation_fraction
        ),
    )

    # Make sure train remains non-empty.
    while (
        num_test
        + num_validation
        >= num_groups
    ):

        if num_validation > 1:
            num_validation -= 1

        elif num_test > 1:
            num_test -= 1

        else:
            raise ValueError(
                "Unable to create a non-empty "
                "training split."
            )

    test_groups = group_ids[
        :num_test
    ]

    validation_groups = group_ids[
        num_test:
        num_test + num_validation
    ]

    train_groups = group_ids[
        num_test + num_validation:
    ]

    def flatten(
        selected_groups: list[str],
    ) -> list[GraphSample]:

        result: list[
            GraphSample
        ] = []

        for group_id in selected_groups:

            result.extend(
                groups[group_id]
            )

        return result

    train_samples = flatten(
        train_groups
    )

    validation_samples = flatten(
        validation_groups
    )

    test_samples = flatten(
        test_groups
    )

    return DatasetSplit(
        train=train_samples,
        validation=validation_samples,
        test=test_samples,
    )
