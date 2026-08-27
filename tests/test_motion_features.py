"""
Tests for unified motion feature extraction.
"""

import numpy as np

from chicken_behavior_lab.features.motion import (
    MotionFeatureExtractor,
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
    x: float,
    y: float,
) -> TrackedSkeleton:

    keypoints = [
        Keypoint(
            name="head",
            x=x,
            y=y,
            confidence=0.95,
        ),
        Keypoint(
            name="body",
            x=x + 20.0,
            y=y + 10.0,
            confidence=0.95,
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
        confidence=0.95,
    )

    return TrackedSkeleton(
        track_id="track_1",
        frame_id=frame_id,
        skeleton=skeleton,
    )


def test_motion_feature_extraction():

    observation_t_minus_2 = (
        create_observation(
            frame_id="frame_1",
            x=0.0,
            y=0.0,
        )
    )

    observation_t_minus_1 = (
        create_observation(
            frame_id="frame_2",
            x=1.0,
            y=0.0,
        )
    )

    observation_t = (
        create_observation(
            frame_id="frame_3",
            x=3.0,
            y=0.0,
        )
    )

    extractor = (
        MotionFeatureExtractor()
    )

    features = extractor.extract(
        previous_previous=(
            observation_t_minus_2
        ),
        previous=(
            observation_t_minus_1
        ),
        current=observation_t,
        delta_time=1.0,
    )

    assert features.num_keypoints == 2

    assert features.feature_dimension == 7

    assert features.feature_names == (
        "x",
        "y",
        "vx",
        "vy",
        "ax",
        "ay",
        "confidence",
    )

    # Head position
    assert np.isclose(
        features.features[0, 0],
        3.0,
    )

    # Head velocity
    assert np.isclose(
        features.features[0, 2],
        2.0,
    )

    # Head acceleration
    #
    # Previous velocity = 1
    # Current velocity = 2
    # Acceleration = 1
    assert np.isclose(
        features.features[0, 4],
        1.0,
    )

    # Confidence
    assert np.isclose(
        features.features[0, 6],
        0.95,
    )


def test_different_tracks_raise_error():

    observation_1 = create_observation(
        frame_id="frame_1",
        x=0.0,
        y=0.0,
    )

    observation_2 = create_observation(
        frame_id="frame_2",
        x=1.0,
        y=0.0,
    )

    observation_3 = create_observation(
        frame_id="frame_3",
        x=2.0,
        y=0.0,
    )

    observation_3.track_id = "track_2"

    extractor = (
        MotionFeatureExtractor()
    )

    try:

        extractor.extract(
            previous_previous=observation_1,
            previous=observation_2,
            current=observation_3,
            delta_time=1.0,
        )

        assert False

    except ValueError as error:

        assert (
            "same track"
            in str(error)
        )
