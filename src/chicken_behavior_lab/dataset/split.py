from __future__ import annotations

import torch

from torch.utils.data import (
    Dataset,
    Subset,
    random_split,
)

from chicken_behavior_lab.dataset.group_split import (
    build_group_id,
    stratified_group_assignment,
)


def train_validation_split(
    dataset: Dataset,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
):
    """
    Deterministically split independent samples into training and validation.

    This sample-level split is only appropriate when samples are independent.
    For samples originating from shared videos or tracks, use
    group_train_validation_split instead.
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



def group_train_validation_split(
    dataset,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
    group_by: str = "video",
):
    """Split a graph dataset without placing related groups in both subsets.

    The dataset must expose its source GraphSamples through a samples
    attribute. Group identities use the same video/track rules as the
    project's other group-splitting utilities.

    At least two distinct groups are required. The validation fraction is
    applied to the number of groups, not the number of samples.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise TypeError(
            "Group-aware splitting requires a dataset exposing source "
            "GraphSamples through a 'samples' attribute."
        )

    if len(samples) != len(dataset):
        raise ValueError(
            "Dataset samples and indexed items must have the same length."
        )

    group_to_indices: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        group_id = build_group_id(sample, group_by=group_by)
        group_to_indices.setdefault(group_id, []).append(index)

    groups = sorted(group_to_indices)
    if len(groups) < 2:
        raise ValueError(
            "At least two distinct groups are required for a "
            "train/validation group split."
        )

    group_labels = {
        group_id: [
            str(getattr(samples[index], "behavior_id", samples[index].label))
            for index in indices
        ]
        for group_id, indices in group_to_indices.items()
    }
    assignment = stratified_group_assignment(
        group_labels,
        (1.0 - validation_fraction, validation_fraction),
        seed=random_seed,
    )
    train_groups = {
        group_id for group_id, split_index in assignment.items()
        if split_index == 0
    }
    validation_groups = {
        group_id for group_id, split_index in assignment.items()
        if split_index == 1
    }

    train_indices: list[int] = []
    validation_indices: list[int] = []
    for group_id, indices in group_to_indices.items():
        destination = (
            validation_indices
            if group_id in validation_groups
            else train_indices
        )
        destination.extend(indices)

    if not train_indices or not validation_indices:
        raise RuntimeError(
            "Group split unexpectedly produced an empty subset."
        )

    if train_groups & validation_groups:
        raise RuntimeError(
            "Group-aware split produced overlapping group assignments."
        )

    return (
        Subset(dataset, sorted(train_indices)),
        Subset(dataset, sorted(validation_indices)),
    )
