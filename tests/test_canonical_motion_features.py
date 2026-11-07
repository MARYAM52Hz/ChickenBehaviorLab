"""Regression tests for canonical keypoint ordering and validity masks."""

import numpy as np

from chicken_behavior_lab.core.enums.keypoints import KeypointType
from chicken_behavior_lab.models.keypoint import Keypoint
from chicken_behavior_lab.models.skeleton import Skeleton
from chicken_behavior_lab.models.tracked_skeleton import TrackedSkeleton
from chicken_behavior_lab.features.position import PositionFeatureExtractor
from chicken_behavior_lab.features.velocity import VelocityFeatureExtractor
from chicken_behavior_lab.features.motion import MotionFeatureExtractor
from chicken_behavior_lab.features.temporal_features import TemporalFeatureBuilder


def make_observation(frame_id, points):
    keypoints = [
        Keypoint(
            keypoint_type=keypoint_type,
            x=x,
            y=y,
            confidence=confidence,
        )
        for keypoint_type, x, y, confidence in points
    ]
    skeleton = Skeleton(
        skeleton_id=f"track_1_{frame_id}",
        frame_id=frame_id,
        keypoints=keypoints,
        edges=[],
        track_id="track_1",
        confidence=0.9,
    )
    return TrackedSkeleton(
        track_id="track_1",
        frame_id=frame_id,
        skeleton=skeleton,
    )


def test_position_features_use_enum_order_not_input_order():
    observation = make_observation(
        "frame_1",
        [
            (KeypointType.HEAD, 10.0, 20.0, 0.9),
            (KeypointType.BEAK, 5.0, 8.0, 0.8),
        ],
    )

    result = PositionFeatureExtractor().extract(observation)

    assert result.keypoint_types == tuple(KeypointType)
    assert result.coordinates[0].tolist() == [5.0, 8.0]
    assert result.coordinates[1].tolist() == [10.0, 20.0]
    assert result.valid_mask[:2].tolist() == [True, True]
    assert result.valid_mask[2:].sum() == 0


def test_velocity_matches_keypoints_by_type_when_input_order_changes():
    previous = make_observation(
        "frame_1",
        [
            (KeypointType.BEAK, 1.0, 2.0, 0.9),
            (KeypointType.HEAD, 10.0, 20.0, 0.9),
        ],
    )
    current = make_observation(
        "frame_2",
        [
            # Deliberately reversed order.
            (KeypointType.HEAD, 12.0, 24.0, 0.9),
            (KeypointType.BEAK, 4.0, 8.0, 0.9),
        ],
    )

    result = VelocityFeatureExtractor().extract(
        previous,
        current,
        delta_time=1.0,
    )

    assert result.velocity[0].tolist() == [3.0, 6.0]
    assert result.velocity[1].tolist() == [2.0, 4.0]
    assert result.valid_mask[:2].tolist() == [True, True]


def test_missing_keypoint_is_zero_filled_and_masked():
    observation = make_observation(
        "frame_1",
        [(KeypointType.HEAD, 10.0, 20.0, 0.9)],
    )

    result = PositionFeatureExtractor().extract(observation)

    assert result.coordinates[0].tolist() == [0.0, 0.0]
    assert result.valid_mask[0] is np.False_
    assert result.coordinates[1].tolist() == [10.0, 20.0]
    assert result.valid_mask[1] is np.True_


def test_motion_and_temporal_builder_preserve_explicit_keypoint_mask():
    first = make_observation(
        "frame_1",
        [
            (KeypointType.BEAK, 1.0, 2.0, 0.9),
            (KeypointType.HEAD, 10.0, 20.0, 0.9),
        ],
    )
    second = make_observation(
        "frame_2",
        [
            (KeypointType.HEAD, 11.0, 21.0, 0.9),
            # BEAK absent in this observation.
        ],
    )
    third = make_observation(
        "frame_3",
        [
            (KeypointType.BEAK, 3.0, 4.0, 0.9),
            (KeypointType.HEAD, 12.0, 22.0, 0.9),
        ],
    )

    extractor = MotionFeatureExtractor()
    frames = [
        extractor.extract(first, second, third, delta_time=1.0),
    ]

    sequence = TemporalFeatureBuilder(sequence_length=2).build(
        frames,
        frame_ids=["frame_3"],
    )

    assert sequence.keypoint_mask.shape == (2, len(KeypointType))
    assert not sequence.keypoint_mask[1, 0]  # BEAK was absent in current frame.
    assert sequence.keypoint_mask[1, 1]      # HEAD was observed.
    assert not sequence.frame_mask[0]
    assert sequence.frame_mask[1]
