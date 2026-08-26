"""
Tests for joint-angle feature extraction.
"""

import numpy as np

from chicken_behavior_lab.features.angles import (
    JointAngleDefinition,
    JointAngleFeatureExtractor,
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
            x=0.0,
            y=1.0,
            confidence=0.95,
        ),
        Keypoint(
            name="neck",
            x=0.0,
            y=0.0,
            confidence=0.95,
        ),
        Keypoint(
            name="body",
            x=1.0,
            y=0.0,
            confidence=0.95,
        ),
    ]

    skeleton = Skeleton(
        skeleton_id="track_1_frame_1",
        frame_id="frame_1",
        keypoints=keypoints,
        edges=[],
        track_id="track_1",
        confidence=0.95,
    )

    return TrackedSkeleton(
        track_id="track_1",
        frame_id="frame_1",
        skeleton=skeleton,
    )


def test_right_angle():

    observation = create_observation()

    definitions = [
        JointAngleDefinition(
            name="neck_angle",
            first_keypoint="head",
            vertex_keypoint="neck",
            third_keypoint="body",
        )
    ]

    extractor = (
        JointAngleFeatureExtractor(
            definitions=definitions,
            confidence_threshold=0.3,
        )
    )

    features = extractor.extract(
        observation
    )

    assert features.num_angles == 1

    assert features.valid_mask[0]

    assert np.isclose(
        features.angles[0],
        np.pi / 2,
    )


def test_low_confidence_keypoint_is_invalid():

    keypoints = [
        Keypoint(
            name="head",
            x=0.0,
            y=1.0,
            confidence=0.95,
        ),
        Keypoint(
            name="neck",
            x=0.0,
            y=0.0,
            confidence=0.20,
        ),
        Keypoint(
            name="body",
            x=1.0,
            y=0.0,
            confidence=0.95,
        ),
    ]

    skeleton = Skeleton(
        skeleton_id="track_1_frame_1",
        frame_id="frame_1",
        keypoints=keypoints,
        edges=[],
        track_id="track_1",
        confidence=0.90,
    )

    observation = TrackedSkeleton(
        track_id="track_1",
        frame_id="frame_1",
        skeleton=skeleton,
    )

    definitions = [
        JointAngleDefinition(
            name="neck_angle",
            first_keypoint="head",
            vertex_keypoint="neck",
            third_keypoint="body",
        )
    ]

    extractor = (
        JointAngleFeatureExtractor(
            definitions=definitions
        )
    )

    features = extractor.extract(
        observation
    )

    assert not features.valid_mask[0]

    assert np.isclose(
        features.angles[0],
        0.0,
    )
