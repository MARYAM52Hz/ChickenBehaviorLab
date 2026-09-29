from __future__ import annotations

import pytest

from chicken_behavior_lab.training.temporal_split import (
    GroupSplit,
    group_from_sample,
    group_train_validation_test_split,
)


def test_group_split_has_no_overlap() -> None:
    groups = [
        "video_001",
        "video_002",
        "video_003",
        "video_004",
        "video_005",
        "video_006",
        "video_007",
        "video_008",
    ]

    split = group_train_validation_test_split(
        groups,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=42,
    )

    assert not (
        set(split.train_groups)
        & set(split.validation_groups)
    )

    assert not (
        set(split.train_groups)
        & set(split.test_groups)
    )

    assert not (
        set(split.validation_groups)
        & set(split.test_groups)
    )


def test_group_split_is_reproducible() -> None:
    groups = [
        f"video_{index:03d}"
        for index in range(12)
    ]

    first = group_train_validation_test_split(
        groups,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=123,
    )

    second = group_train_validation_test_split(
        groups,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=123,
    )

    assert first == second


def test_group_split_changes_with_seed() -> None:
    groups = [
        f"video_{index:03d}"
        for index in range(12)
    ]

    first = group_train_validation_test_split(
        groups,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=1,
    )

    second = group_train_validation_test_split(
        groups,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=999,
    )

    assert first != second


def test_group_from_sample_video() -> None:
    group = group_from_sample(
        video_id="video_001",
        track_id=7,
        split_group="video",
    )

    assert group == "video_001"


def test_group_from_sample_track() -> None:
    group = group_from_sample(
        video_id="video_001",
        track_id=7,
        split_group="track",
    )

    assert group == "video_001::track::7"


def test_invalid_split_group() -> None:
    with pytest.raises(ValueError):
        group_from_sample(
            video_id="video_001",
            track_id=7,
            split_group="frame",
        )


def test_too_few_groups() -> None:
    with pytest.raises(ValueError):
        group_train_validation_test_split(
            [
                "video_001",
                "video_002",
            ]
        )


def test_invalid_fractions() -> None:
    with pytest.raises(ValueError):
        group_train_validation_test_split(
            [
                "v1",
                "v2",
                "v3",
                "v4",
            ],
            validation_fraction=0.6,
            test_fraction=0.5,
        )


def test_groups_are_sorted() -> None:
    groups = [
        "video_c",
        "video_a",
        "video_f",
        "video_b",
        "video_e",
        "video_d",
    ]

    split = group_train_validation_test_split(
        groups,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=42,
    )

    assert list(split.train_groups) == sorted(
        split.train_groups
    )

    assert list(split.validation_groups) == sorted(
        split.validation_groups
    )

    assert list(split.test_groups) == sorted(
        split.test_groups
    )


def test_group_split_validation_rejects_overlap() -> None:
    split = GroupSplit(
        train_groups=("a", "b"),
        validation_groups=("b", "c"),
        test_groups=("d",),
    )

    with pytest.raises(ValueError):
        split.validate()
