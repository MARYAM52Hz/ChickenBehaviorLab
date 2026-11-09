"""Tests for canonical position feature extraction."""

import numpy as np

from chicken_behavior_lab.core.enums.keypoints import KeypointType
from chicken_behavior_lab.features.position import PositionFeatureExtractor
from chicken_behavior_lab.models.keypoint import Keypoint
from chicken_behavior_lab.models.skeleton import Skeleton
from chicken_behavior_lab.models.tracked_skeleton import TrackedSkeleton


def create_observation():
    keypoints = [
        Keypoint(keypoint_type=KeypointType.HEAD, x=100.0, y=50.0, confidence=0.95),
        Keypoint(keypoint_type=KeypointType.NECK, x=90.0, y=70.0, confidence=0.90),
    ]
    skeleton = Skeleton(
        skeleton_id="track_1_frame_1", frame_id="frame_1", keypoints=keypoints,
        edges=[], track_id="track_1", confidence=0.92,
    )
    return TrackedSkeleton(track_id="track_1", frame_id="frame_1", skeleton=skeleton)


def test_position_extraction_uses_fixed_canonical_order():
    features = PositionFeatureExtractor().extract(create_observation())
    assert features.num_keypoints == len(KeypointType)
    assert features.feature_dimension == 2
    assert features.keypoint_types == tuple(KeypointType)
    assert np.allclose(features.coordinates[1], [100.0, 50.0])
    assert np.allclose(features.coordinates[2], [90.0, 70.0])
    assert features.valid_mask[:3].tolist() == [False, True, True]
    assert np.allclose(features.confidence[1:3], [0.95, 0.90])
