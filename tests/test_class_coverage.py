
from __future__ import annotations

import pytest

from chicken_behavior_lab.dataset.preflight import (
    inspect_class_coverage,
    validate_class_coverage,
)
from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.graph.graph import TemporalSkeletonGraph


def make_sample(
    sample_id: str,
    behavior_id: str,
    label: int = 0,
) -> GraphSample:
    """Create a valid minimal graph sample for unit tests."""

    graph = TemporalSkeletonGraph(
        node_features=[
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 1.0, 0.0],
        ],
        edge_index=[
            [0, 1],
            [1, 2],
        ],
        edge_features=[
            [1.0],
            [1.0],
        ],
    )

    return GraphSample(
        graph=graph,
        label=label,
        behavior_id=behavior_id,
        sample_id=sample_id,
        metadata={
            "video_id": f"video_{sample_id}",
            "track_id": 1,
            "start_frame": 0,
            "end_frame": 0,
        },
    )


def test_valid_class_coverage() -> None:
    """All validation/test classes are present in training."""

    train = [
        make_sample("train_1", "feeding", 0),
        make_sample("train_2", "walking", 1),
    ]

    validation = [
        make_sample("validation_1", "feeding", 0),
    ]

    test = [
        make_sample("test_1", "walking", 1),
    ]

    report = validate_class_coverage(
        train,
        validation,
        test,
    )

    assert report.is_valid
    assert report.train_classes == {"feeding", "walking"}
    assert report.validation_only_classes == set()
    assert report.test_only_classes == set()

    assert report.train_counts == {
        "feeding": 1,
        "walking": 1,
    }

    assert report.validation_counts == {"feeding": 1}
    assert report.test_counts == {"walking": 1}


def test_validation_only_class_is_rejected() -> None:
    """A validation-only class must not be silently added to the mapping."""

    train = [
        make_sample("train_1", "feeding", 0),
    ]

    validation = [
        make_sample("validation_1", "walking", 1),
    ]

    test = [
        make_sample("test_1", "feeding", 0),
    ]

    report = inspect_class_coverage(
        train,
        validation,
        test,
    )

    assert report.validation_only_classes == {"walking"}
    assert report.is_valid is False

    with pytest.raises(
        ValueError,
        match="Validation split contains behavior classes absent",
    ):
        validate_class_coverage(
            train,
            validation,
            test,
        )


def test_test_only_class_is_rejected() -> None:
    """A test-only class must be reported explicitly."""

    train = [
        make_sample("train_1", "feeding", 0),
    ]

    validation = [
        make_sample("validation_1", "feeding", 0),
    ]

    test = [
        make_sample("test_1", "aggression", 1),
    ]

    report = inspect_class_coverage(
        train,
        validation,
        test,
    )

    assert report.test_only_classes == {"aggression"}
    assert report.is_valid is False

    with pytest.raises(
        ValueError,
        match="Test split contains behavior classes absent",
    ):
        validate_class_coverage(
            train,
            validation,
            test,
        )


@pytest.mark.parametrize(
    ("train", "validation", "test", "empty_split"),
    [
        ([], [make_sample("v1", "feeding")],
         [make_sample("t1", "feeding")], "Training"),
        ([make_sample("tr1", "feeding")], [],
         [make_sample("t1", "feeding")], "Validation"),
        ([make_sample("tr1", "feeding")],
         [make_sample("v1", "feeding")], [], "Test"),
    ],
)
def test_empty_split_is_rejected(
    train: list[GraphSample],
    validation: list[GraphSample],
    test: list[GraphSample],
    empty_split: str,
) -> None:
    """Every split must contain samples before coverage is checked."""

    with pytest.raises(
        ValueError,
        match=f"{empty_split} split is empty",
    ):
        inspect_class_coverage(
            train,
            validation,
            test,
        )


def test_report_can_be_serialized() -> None:
    """The report should be suitable for JSON experiment manifests."""

    report = inspect_class_coverage(
        train=[
            make_sample("tr1", "feeding"),
            make_sample("tr2", "walking"),
        ],
        validation=[
            make_sample("v1", "feeding"),
        ],
        test=[
            make_sample("t1", "walking"),
        ],
    )

    serialized = report.to_dict()

    assert serialized["is_valid"] is True
    assert serialized["train_counts"] == {
        "feeding": 1,
        "walking": 1,
    }
