"""
Tests for velocity feature extraction.
"""

import numpy as np

from chicken_behavior_lab.core.enums.keypoints import KeypointType
from chicken_behavior_lab.features.velocity import VelocityFeatureExtractor
from chicken_behavior_lab.models.keypoint import Keypoint
from chicken_behavior_lab.models.skeleton import Skeleton
from chicken_behavior_lab.models.tracked_skeleton import TrackedSkeleton


def create_observation(
    frame_id: str,
    head_x: float,
    head_y: float,
    confidence: float = 0.95,
) -> TrackedSkeleton:
    keypoints = [
        Keypoint(
            keypoint_type=KeypointType.HEAD,
            x=head_x,
            y=head_y,
            confidence=confidence,
        ),
    ]

    skeleton = Skeleton(
        skeleton_id=f"track_1_{frame_id}",
        frame_id=frame_id,
        keypoints=keypoints,
        edges=[],
        track_id="track_1",
        confidence=confidence,
    )

    return TrackedSkeleton(
        track_id="track_1",
        frame_id=frame_id,
        skeleton=skeleton,
    )


def test_velocity_extraction():
    previous = create_observation("frame_1", 100.0, 50.0)
    current = create_observation("frame_2", 110.0, 55.0)

    features = VelocityFeatureExtractor(
        confidence_threshold=0.3
    ).extract(previous=previous, current=current, delta_time=0.1)

    assert features.num_keypoints == 1
    assert features.feature_dimension == 2
    assert np.allclose(
        features.velocity[0],
        np.array([100.0, 50.0], dtype=np.float32),
    )

    expected_speed = np.sqrt(100.0**2 + 50.0**2)
    assert np.isclose(features.speed[0], expected_speed)
    assert features.valid_mask[0]


def test_low_confidence_keypoint_is_invalid():
    previous = create_observation("frame_1", 100.0, 50.0, confidence=0.2)
    current = create_observation("frame_2", 110.0, 55.0, confidence=0.9)

    features = VelocityFeatureExtractor(
        confidence_threshold=0.3
    ).extract(previous, current, 0.1)

    assert not features.valid_mask[0]
    assert np.allclose(features.velocity[0], [0.0, 0.0])


def test_track_mismatch_raises_error():
    previous = create_observation("frame_1", 100.0, 50.0)
    current = create_observation("frame_2", 110.0, 55.0)
    current.track_id = "track_2"

    extractor = VelocityFeatureExtractor()

    try:
        extractor.extract(previous, current, 0.1)
        assert False, "Expected a track mismatch to raise ValueError."
    except ValueError as error:
        assert "Track ID mismatch" in str(error)


def test_invalid_delta_time_raises_error():
    previous = create_observation("frame_1", 100.0, 50.0)
    current = create_observation("frame_2", 110.0, 55.0)

    extractor = VelocityFeatureExtractor()

    try:
        extractor.extract(previous, current, 0.0)
        assert False, "Expected an invalid delta_time to raise ValueError."
    except ValueError as error:
        assert "delta_time" in str(error)
