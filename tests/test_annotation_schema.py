"""
Tests for ChickenBehaviorLab annotation schema.
"""

import pytest

from chicken_behavior_lab.annotations import (
    AnnotationSet,
    BehaviorAnnotation,
)


def test_behavior_annotation():

    annotation = BehaviorAnnotation(
        annotation_id="ann_001",
        video_id="video_001",
        track_id=1,
        behavior_id="feeding",
        start_frame=100,
        end_frame=120,
    )

    annotation.validate()

    assert (
        annotation.num_frames
        == 21
    )


def test_invalid_frame_range():

    annotation = BehaviorAnnotation(
        annotation_id="ann_001",
        video_id="video_001",
        track_id=1,
        behavior_id="feeding",
        start_frame=120,
        end_frame=100,
    )

    with pytest.raises(
        ValueError
    ):

        annotation.validate()


def test_negative_track_id():

    annotation = BehaviorAnnotation(
        annotation_id="ann_001",
        video_id="video_001",
        track_id=-1,
        behavior_id="feeding",
        start_frame=100,
        end_frame=120,
    )

    with pytest.raises(
        ValueError
    ):

        annotation.validate()


def test_invalid_confidence():

    annotation = BehaviorAnnotation(
        annotation_id="ann_001",
        video_id="video_001",
        track_id=1,
        behavior_id="feeding",
        start_frame=100,
        end_frame=120,
        confidence=1.5,
    )

    with pytest.raises(
        ValueError
    ):

        annotation.validate()


def test_annotation_set():

    annotations = [
        BehaviorAnnotation(
            annotation_id="ann_001",
            video_id="video_001",
            track_id=1,
            behavior_id="feeding",
            start_frame=100,
            end_frame=120,
        ),
        BehaviorAnnotation(
            annotation_id="ann_002",
            video_id="video_001",
            track_id=1,
            behavior_id="walking",
            start_frame=121,
            end_frame=150,
        ),
    ]

    annotation_set = AnnotationSet(
        annotations
    )

    annotation_set.validate()

    assert len(
        annotation_set
    ) == 2


def test_duplicate_annotation_ids():

    annotations = [
        BehaviorAnnotation(
            annotation_id="ann_001",
            video_id="video_001",
            track_id=1,
            behavior_id="feeding",
            start_frame=100,
            end_frame=120,
        ),
        BehaviorAnnotation(
            annotation_id="ann_001",
            video_id="video_001",
            track_id=2,
            behavior_id="walking",
            start_frame=121,
            end_frame=150,
        ),
    ]

    annotation_set = AnnotationSet(
        annotations
    )

    with pytest.raises(
        ValueError,
        match="Duplicate annotation_id",
    ):

        annotation_set.validate()
