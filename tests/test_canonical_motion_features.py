"""Regression tests for canonical ordering and explicit motion validity masks."""

import numpy as np
import pytest

from chicken_behavior_lab.core.enums.keypoints import KeypointType
from chicken_behavior_lab.features.motion import MotionFeatureExtractor
from chicken_behavior_lab.features.position import PositionFeatureExtractor
from chicken_behavior_lab.features.temporal_features import TemporalFeatureBuilder
from chicken_behavior_lab.features.velocity import VelocityFeatureExtractor
from chicken_behavior_lab.models.keypoint import Keypoint
from chicken_behavior_lab.models.skeleton import Skeleton
from chicken_behavior_lab.models.tracked_skeleton import TrackedSkeleton


def make_observation(frame_id, points):
    keypoints = [
        Keypoint(keypoint_type=kind, x=x, y=y, confidence=confidence)
        for kind, x, y, confidence in points
    ]
    skeleton = Skeleton(
        skeleton_id=f"track_1_{frame_id}",
        frame_id=str(frame_id),
        keypoints=keypoints,
        edges=[],
        track_id="track_1",
        confidence=0.9,
    )
    return TrackedSkeleton(track_id="track_1", frame_id=str(frame_id), skeleton=skeleton)


def test_position_features_use_canonical_order_not_input_order():
    observation = make_observation(
        "1",
        [(KeypointType.HEAD, 10.0, 20.0, 0.9), (KeypointType.BEAK, 5.0, 8.0, 0.8)],
    )
    result = PositionFeatureExtractor().extract(observation)
    assert result.keypoint_types == tuple(KeypointType)
    assert result.coordinates[0].tolist() == [5.0, 8.0]
    assert result.coordinates[1].tolist() == [10.0, 20.0]
    assert result.valid_mask[:2].tolist() == [True, True]
    assert not result.valid_mask[2:].any()


def test_velocity_matches_by_type_when_input_order_changes():
    previous = make_observation(
        "1", [(KeypointType.BEAK, 1.0, 2.0, 0.9), (KeypointType.HEAD, 10.0, 20.0, 0.9)]
    )
    current = make_observation(
        "2", [(KeypointType.HEAD, 12.0, 24.0, 0.9), (KeypointType.BEAK, 4.0, 8.0, 0.9)]
    )
    result = VelocityFeatureExtractor().extract(previous, current, delta_time=1.0)
    assert result.velocity[0].tolist() == [3.0, 6.0]
    assert result.velocity[1].tolist() == [2.0, 4.0]
    assert result.valid_mask[:2].tolist() == [True, True]


def test_missing_keypoint_is_zero_filled_and_masked():
    observation = make_observation("1", [(KeypointType.HEAD, 10.0, 20.0, 0.9)])
    result = PositionFeatureExtractor().extract(observation)
    assert result.coordinates[0].tolist() == [0.0, 0.0]
    assert not result.valid_mask[0]
    assert result.coordinates[1].tolist() == [10.0, 20.0]
    assert result.valid_mask[1]


def test_duplicate_keypoint_type_is_rejected():
    observation = make_observation(
        "1", [(KeypointType.HEAD, 1.0, 2.0, 0.9), (KeypointType.HEAD, 3.0, 4.0, 0.9)]
    )
    with pytest.raises(ValueError, match="Duplicate keypoint"):
        PositionFeatureExtractor().extract(observation)


def test_low_confidence_keypoint_is_masked():
    observation = make_observation("1", [(KeypointType.HEAD, 10.0, 20.0, 0.1)])
    result = PositionFeatureExtractor(confidence_threshold=0.25).extract(observation)
    assert not result.valid_mask[1]
    assert result.coordinates[1].tolist() == [0.0, 0.0]


def test_non_finite_coordinate_is_masked():
    observation = make_observation("1", [(KeypointType.HEAD, float("nan"), 20.0, 0.9)])
    result = PositionFeatureExtractor().extract(observation)
    assert not result.valid_mask[1]
    assert result.coordinates[1].tolist() == [0.0, 0.0]


def test_motion_mask_requires_position_velocity_and_acceleration():
    first = make_observation(
        "1", [(KeypointType.BEAK, 1.0, 2.0, 0.9), (KeypointType.HEAD, 10.0, 20.0, 0.9)]
    )
    second = make_observation("2", [(KeypointType.HEAD, 11.0, 21.0, 0.9)])
    third = make_observation("3", [(KeypointType.HEAD, 12.0, 22.0, 0.9)])
    frame = MotionFeatureExtractor().extract(first, second, third, delta_time=1.0)
    assert frame.keypoint_mask is not None
    assert not frame.keypoint_mask[0]
    assert frame.keypoint_mask[1]


def test_temporal_builder_preserves_explicit_node_mask():
    frame = MotionFeatureExtractor().extract(
        make_observation("1", [(KeypointType.HEAD, 10.0, 20.0, 0.9)]),
        make_observation("2", [(KeypointType.HEAD, 11.0, 21.0, 0.9)]),
        make_observation("3", [(KeypointType.HEAD, 12.0, 22.0, 0.9)]),
        delta_time=1.0,
    )
    sequence = TemporalFeatureBuilder(sequence_length=2).build([frame], frame_ids=["3"])
    assert sequence.keypoint_mask.shape == (2, len(KeypointType))
    assert not sequence.keypoint_mask[0].any()
    assert sequence.keypoint_mask[1, 1]
    assert not sequence.keypoint_mask[1, 0]
    assert sequence.frame_mask.tolist() == [False, True]


def test_feature_extractors_reject_non_finite_threshold_and_delta_time():
    with pytest.raises(ValueError):
        PositionFeatureExtractor(confidence_threshold=float("nan"))
    previous = make_observation("1", [(KeypointType.HEAD, 1.0, 1.0, 0.9)])
    current = make_observation("2", [(KeypointType.HEAD, 2.0, 2.0, 0.9)])
    with pytest.raises(ValueError, match="delta_time"):
        VelocityFeatureExtractor().extract(previous, current, float("inf"))
