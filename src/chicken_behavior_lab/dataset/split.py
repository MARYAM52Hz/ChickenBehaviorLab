from __future__ import annotations

import torch

from torch.utils.data import (
    Dataset,
    random_split,
)


def train_validation_split(
    dataset: Dataset,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
):
    """
    Deterministically split a dataset into
    training and validation subsets.
    """

    if len(dataset) < 2:
        raise ValueError(
            "Dataset must contain at least "
            "two samples."
        )

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be "
            "between 0 and 1."
        )

    validation_size = max(
        1,
        int(
            len(dataset)
            * validation_fraction
        ),
    )

    train_size = (
        len(dataset)
        - validation_size
    )

    if train_size <= 0:
        raise ValueError(
            "Training split would be empty."
        )

    generator = (
        torch.Generator()
        .manual_seed(
            random_seed
        )
    )

    return random_split(
        dataset,
        [
            train_size,
            validation_size,
        ],
        generator=generator,
    )
