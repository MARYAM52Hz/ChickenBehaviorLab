"""
Tests for position feature extraction.
"""

import numpy as np

from chicken_behavior_lab.features.position import (
    PositionFeatureExtractor,
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


def create_observation() -> TrackedSkeleton:

    keypoints = [
        Keypoint(
            name="head",
            x=100.0,
            y=50.0,
            confidence=0.95,
        ),
        Keypoint(
            name="neck",
            x=90.0,
            y=70.0,
            confidence=0.90,
        ),
    ]

    skeleton = Skeleton(
        skeleton_id="track_1_frame_1",
        frame_id="frame_1",
        keypoints=keypoints,
        edges=[],
        track_id="track_1",
        confidence=0.92,
    )

    return TrackedSkeleton(
        track_id="track_1",
        frame_id="frame_1",
        skeleton=skeleton,
    )


def test_position_extraction():

    observation = create_observation()

    extractor = (
        PositionFeatureExtractor()
    )

    features = extractor.extract(
        observation
    )

    assert features.num_keypoints == 2

    assert features.feature_dimension == 2

    assert isinstance(
        features.coordinates,
        np.ndarray,
    )

    assert np.allclose(
        features.coordinates,
        np.array(
            [
                [100.0, 50.0],
                [90.0, 70.0],
            ],
            dtype=np.float32,
        ),
    )

    assert np.allclose(
        features.confidence,
        np.array(
            [0.95, 0.90],
            dtype=np.float32,
        ),
    )
