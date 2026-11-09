"""Tests for canonical velocity feature extraction."""

import numpy as np

from chicken_behavior_lab.core.enums.keypoints import KeypointType
from chicken_behavior_lab.features.velocity import VelocityFeatureExtractor
from chicken_behavior_lab.models.keypoint import Keypoint
from chicken_behavior_lab.models.skeleton import Skeleton
from chicken_behavior_lab.models.tracked_skeleton import TrackedSkeleton


def create_observation(frame_id, head_x, head_y, confidence=0.95):
    keypoints = [Keypoint(keypoint_type=KeypointType.HEAD, x=head_x, y=head_y, confidence=confidence)]
    skeleton = Skeleton(
        skeleton_id=f"track_1_{frame_id}", frame_id=frame_id, keypoints=keypoints,
        edges=[], track_id="track_1", confidence=confidence,
    )
    return TrackedSkeleton(track_id="track_1", frame_id=frame_id, skeleton=skeleton)


def test_velocity_extraction_uses_canonical_order():
    previous = create_observation("frame_1", 100.0, 50.0)
    current = create_observation("frame_2", 110.0, 55.0)
    features = VelocityFeatureExtractor().extract(previous, current, 0.1)
    head_index = tuple(KeypointType).index(KeypointType.HEAD)
    assert features.num_keypoints == len(KeypointType)
    assert features.feature_dimension == 2
    assert np.allclose(features.velocity[head_index], [100.0, 50.0])
    assert np.isclose(features.speed[head_index], np.sqrt(100.0**2 + 50.0**2))
    assert features.valid_mask[head_index]
    assert not features.valid_mask[0]


def test_low_confidence_keypoint_is_invalid():
    previous = create_observation("frame_1", 100.0, 50.0, confidence=0.2)
    current = create_observation("frame_2", 110.0, 55.0, confidence=0.9)
    features = VelocityFeatureExtractor(confidence_threshold=0.3).extract(previous, current, 0.1)
    head_index = tuple(KeypointType).index(KeypointType.HEAD)
    assert not features.valid_mask[head_index]
    assert np.allclose(features.velocity[head_index], [0.0, 0.0])


def test_track_mismatch_raises_error():
    previous = create_observation("frame_1", 100.0, 50.0)
    current = create_observation("frame_2", 110.0, 55.0)
    current.track_id = "track_2"
    try:
        VelocityFeatureExtractor().extract(previous, current, 0.1)
        assert False, "Expected a track mismatch to raise ValueError."
    except ValueError as error:
        assert "Track ID mismatch" in str(error)


def test_invalid_delta_time_raises_error():
    previous = create_observation("frame_1", 100.0, 50.0)
    current = create_observation("frame_2", 110.0, 55.0)
    try:
        VelocityFeatureExtractor().extract(previous, current, 0.0)
        assert False, "Expected an invalid delta_time to raise ValueError."
    except ValueError as error:
        assert "delta_time" in str(error)
