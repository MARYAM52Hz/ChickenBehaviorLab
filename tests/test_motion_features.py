"""Tests for unified canonical motion feature extraction."""

import numpy as np

from chicken_behavior_lab.core.enums.keypoints import KeypointType
from chicken_behavior_lab.features.motion import MotionFeatureExtractor
from chicken_behavior_lab.models.keypoint import Keypoint
from chicken_behavior_lab.models.skeleton import Skeleton
from chicken_behavior_lab.models.tracked_skeleton import TrackedSkeleton


def create_observation(frame_id, x, y):
    keypoints = [
        Keypoint(keypoint_type=KeypointType.HEAD, x=x, y=y, confidence=0.95),
        Keypoint(keypoint_type=KeypointType.BODY_CENTER, x=x + 20.0, y=y + 10.0, confidence=0.95),
    ]
    skeleton = Skeleton(
        skeleton_id=f"track_1_{frame_id}", frame_id=frame_id, keypoints=keypoints,
        edges=[], track_id="track_1", confidence=0.95,
    )
    return TrackedSkeleton(track_id="track_1", frame_id=frame_id, skeleton=skeleton)


def test_motion_feature_extraction():
    previous_previous = create_observation("frame_1", 0.0, 0.0)
    previous = create_observation("frame_2", 1.0, 0.0)
    current = create_observation("frame_3", 3.0, 0.0)
    result = MotionFeatureExtractor().extract(previous_previous, previous, current, 1.0)

    head_index = tuple(KeypointType).index(KeypointType.HEAD)
    assert result.num_keypoints == len(KeypointType)
    assert result.feature_dimension == 7
    assert result.feature_names == ("x", "y", "vx", "vy", "ax", "ay", "confidence")
    assert np.isclose(result.features[head_index, 0], 3.0)
    assert np.isclose(result.features[head_index, 2], 2.0)
    assert np.isclose(result.features[head_index, 4], 1.0)
    assert np.isclose(result.features[head_index, 6], 0.95)
    assert result.keypoint_mask[head_index]


def test_different_tracks_raise_error():
    first = create_observation("frame_1", 0.0, 0.0)
    second = create_observation("frame_2", 1.0, 0.0)
    third = create_observation("frame_3", 2.0, 0.0)
    third.track_id = "track_2"
    try:
        MotionFeatureExtractor().extract(first, second, third, 1.0)
        assert False, "Expected observations from different tracks to raise ValueError."
    except ValueError as error:
        assert "same track" in str(error)
