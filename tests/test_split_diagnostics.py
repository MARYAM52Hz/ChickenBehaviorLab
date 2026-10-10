
import pytest

from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.dataset.split_diagnostics import (
    inspect_split_quality,
)
from chicken_behavior_lab.graph.graph import TemporalSkeletonGraph


def make_sample(
    sample_id: str,
    behavior_id: str,
    video_id: str,
    track_id: int = 1,
) -> GraphSample:
    graph = TemporalSkeletonGraph(
        node_features=[
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 1.0, 0.0],
        ],
        edge_index=[[0, 1], [1, 2]],
        edge_features=[[1.0], [1.0]],
    )

    return GraphSample(
        graph=graph,
        label=0,
        behavior_id=behavior_id,
        sample_id=sample_id,
        metadata={
            "video_id": video_id,
            "track_id": track_id,
            "start_frame": 0,
            "end_frame": 10,
        },
    )


def test_valid_split_has_no_group_leakage():
    train = [
        make_sample("tr1", "feeding", "video_a"),
        make_sample("tr2", "walking", "video_b"),
    ]
    validation = [
        make_sample("va1", "feeding", "video_c"),
    ]
    test = [
        make_sample("te1", "walking", "video_d"),
    ]

    report = inspect_split_quality(train, validation, test)

    assert report.is_valid
    assert not report.has_group_leakage
    assert report.group_counts == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }
    assert report.sample_counts == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }
    assert report.sample_ratios["train"] == pytest.approx(0.5)


def test_same_video_in_multiple_splits_is_detected():
    train = [make_sample("tr1", "feeding", "video_a")]
    validation = [make_sample("va1", "feeding", "video_a")]
    test = [make_sample("te1", "feeding", "video_c")]

    report = inspect_split_quality(train, validation, test)

    assert report.has_group_leakage
    assert report.group_overlaps["train_validation"] == ["video_a"]
    assert not report.is_valid


def test_class_missing_from_training_is_reported():
    train = [make_sample("tr1", "feeding", "video_a")]
    validation = [make_sample("va1", "walking", "video_b")]
    test = [make_sample("te1", "feeding", "video_c")]

    report = inspect_split_quality(train, validation, test)

    assert report.has_unknown_classes
    assert report.validation_only_classes == ["walking"]
    assert not report.is_valid


def test_repeated_sample_id_is_detected():
    train = [make_sample("same_id", "feeding", "video_a")]
    validation = [make_sample("same_id", "feeding", "video_b")]
    test = [make_sample("te1", "feeding", "video_c")]

    report = inspect_split_quality(train, validation, test)

    assert report.duplicate_sample_ids == ["same_id"]
    assert not report.is_valid


def test_track_group_identity_includes_video_id():
    train = [make_sample("tr1", "feeding", "video_a", track_id=1)]
    validation = [make_sample("va1", "feeding", "video_b", track_id=1)]
    test = [make_sample("te1", "feeding", "video_c", track_id=2)]

    report = inspect_split_quality(
        train,
        validation,
        test,
        group_key="track_id",
    )

    # Track 1 in different videos represents different tracked individuals.
    assert not report.has_group_leakage
    assert report.is_valid


def test_report_is_json_compatible():
    train = [make_sample("tr1", "feeding", "video_a")]
    validation = [make_sample("va1", "feeding", "video_b")]
    test = [make_sample("te1", "feeding", "video_c")]

    report = inspect_split_quality(train, validation, test)
    payload = report.to_dict()

    assert payload["is_valid"] is True
    assert payload["class_counts"]["train"] == {"feeding": 1}
    assert payload["sample_counts"]["test"] == 1
