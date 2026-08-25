"""
Tests for velocity feature extraction.
"""

import numpy as np

from chicken_behavior_lab.features.velocity import (
    VelocityFeatureExtractor,
)

from chicken_behavior_lab.models.keypoint import (
    Keypoint,
)

from chicken_behavior_lab.models.skeleton import (
    Skeleton,
)

from chicken_behavior_lab.models.tracked_skeleton import (
    TrackedSkeleton,
)


def create_observation(
    frame_id: str,
    head_x: float,
    head_y: float,
    confidence: float = 0.95,
) -> TrackedSkeleton:

    keypoints = [
        Keypoint(
            name="head",
            x=head_x,
            y=head_y,
            confidence=confidence,
        ),
    ]

    skeleton = Skeleton(
        skeleton_id=(
            f"track_1_{frame_id}"
        ),
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

    previous = create_observation(
        frame_id="frame_1",
        head_x=100.0,
        head_y=50.0,
    )

    current = create_observation(
        frame_id="frame_2",
        head_x=110.0,
        head_y=55.0,
    )

    extractor = (
        VelocityFeatureExtractor(
            confidence_threshold=0.3
        )
    )

    features = extractor.extract(
        previous=previous,
        current=current,
        delta_time=0.1,
    )

    assert features.num_keypoints == 1

    assert features.feature_dimension == 2

    assert np.allclose(
        features.velocity[0],
        np.array(
            [100.0, 50.0],
            dtype=np.float32,
        ),
    )

    expected_speed = np.sqrt(
        100.0**2 + 50.0**2
    )

    assert np.isclose(
        features.speed[0],
        expected_speed,
    )

    assert features.valid_mask[0]


def test_low_confidence_keypoint_is_invalid():

    previous = create_observation(
        frame_id="frame_1",
        head_x=100.0,
        head_y=50.0,
        confidence=0.2,
    )

    current = create_observation(
        frame_id="frame_2",
        head_x=110.0,
        head_y=55.0,
        confidence=0.9,
    )

    extractor = (
        VelocityFeatureExtractor(
            confidence_threshold=0.3
        )
    )

    features = extractor.extract(
        previous=previous,
        current=current,
        delta_time=0.1,
    )

    assert not features.valid_mask[0]

    assert np.allclose(
        features.velocity[0],
        np.array(
            [0.0, 0.0],
            dtype=np.float32,
        ),
    )


def test_track_mismatch_raises_error():

    previous = create_observation(
        frame_id="frame_1",
        head_x=100.0,
        head_y=50.0,
    )

    current = create_observation(
        frame_id="frame_2",
        head_x=110.0,
        head_y=55.0,
    )

    current.track_id = "track_2"

    extractor = (
        VelocityFeatureExtractor()
    )

    try:
        extractor.extract(
            previous,
            current,
            0.1,
        )

        assert False

    except ValueError as error:

        assert "Track ID mismatch" in str(
            error
        )


def test_invalid_delta_time_raises_error():

    previous = create_observation(
        frame_id="frame_1",
        head_x=100.0,
        head_y=50.0,
    )

    current = create_observation(
        frame_id="frame_2",
        head_x=110.0,
        head_y=55.0,
    )

    extractor = (
        VelocityFeatureExtractor()
    )

    try:
        extractor.extract(
            previous,
            current,
            0.0,
        )

        assert False

    except ValueError as error:

        assert "delta_time" in str(
            error
        )
