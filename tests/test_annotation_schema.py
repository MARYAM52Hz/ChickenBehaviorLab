"""Tests for the video-scoped version 1.0 annotation schema."""

import pytest

from chicken_behavior_lab.annotations.schema import (
    AnnotationSet,
    AnnotationValidationError,
    BehaviorAnnotation,
)


def make_annotation(**overrides):
    values = {
        "annotation_id": "ann_001",
        "track_id": 1,
        "behavior_id": "feeding",
        "start_frame": 100,
        "end_frame": 120,
        "annotator_id": "annotator_01",
        "confidence": 0.9,
        "attributes": {"note": "near feeder"},
    }
    values.update(overrides)
    return BehaviorAnnotation(**values)


def test_behavior_annotation_uses_inclusive_frame_intervals():
    annotation = make_annotation()
    assert annotation.duration_frames == 21
    assert annotation.contains_frame(100)
    assert annotation.contains_frame(120)
    assert not annotation.contains_frame(121)


def test_invalid_frame_range_is_rejected_at_construction():
    with pytest.raises(AnnotationValidationError, match="end_frame"):
        make_annotation(start_frame=120, end_frame=100)


def test_negative_track_id_is_rejected():
    with pytest.raises(AnnotationValidationError, match="non-negative"):
        make_annotation(track_id=-1)


def test_invalid_confidence_is_rejected():
    with pytest.raises(AnnotationValidationError, match="confidence"):
        make_annotation(confidence=1.5)


def test_annotation_set_owns_video_id_and_queries_by_track():
    annotations = (
        make_annotation(annotation_id="ann_001", track_id=1),
        make_annotation(annotation_id="ann_002", track_id="track_1", behavior_id="walking"),
        make_annotation(annotation_id="ann_003", track_id=2, behavior_id="standing"),
    )
    annotation_set = AnnotationSet(video_id="video_001", annotations=annotations)
    assert annotation_set.video_id == "video_001"
    assert len(annotation_set.annotations) == 3
    assert len(annotation_set.for_track(1)) == 2
    assert len(annotation_set.for_track("track_1")) == 2
    assert len(annotation_set.at_frame(110, track_id="track_1")) == 2


def test_duplicate_annotation_ids_are_rejected_at_construction():
    annotations = (
        make_annotation(annotation_id="ann_001", track_id=1),
        make_annotation(annotation_id="ann_001", track_id=2, behavior_id="walking"),
    )
    with pytest.raises(AnnotationValidationError, match="unique"):
        AnnotationSet(video_id="video_001", annotations=annotations)


def test_annotation_set_round_trips_through_dict():
    original = AnnotationSet(
        video_id="video_001",
        annotations=(make_annotation(),),
        metadata={"source": "cvat-export"},
    )
    restored = AnnotationSet.from_dict(original.to_dict())
    assert restored.video_id == original.video_id
    assert restored.annotations == original.annotations
    assert restored.metadata == original.metadata
