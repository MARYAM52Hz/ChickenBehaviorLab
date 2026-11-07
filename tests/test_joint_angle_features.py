"""Regression tests for joint-angle extraction."""

import numpy as np
import pytest

from chicken_behavior_lab.core.enums.annotation import KeypointVisibility
from chicken_behavior_lab.core.enums.keypoints import KeypointType
from chicken_behavior_lab.features.angles import (
    JointAngleDefinition,
    JointAngleFeatureExtractor,
)
from chicken_behavior_lab.models.keypoint import Keypoint
from chicken_behavior_lab.models.skeleton import Skeleton
from chicken_behavior_lab.models.tracked_skeleton import TrackedSkeleton


def make_observation(points):
    keypoints = [
        Keypoint(
            keypoint_type=keypoint_type,
            x=x,
            y=y,
            confidence=confidence,
            visibility=visibility,
        )
        for keypoint_type, x, y, confidence, visibility in points
    ]
    skeleton = Skeleton(
        skeleton_id="skeleton_1",
        frame_id="frame_1",
        keypoints=keypoints,
        edges=[],
        track_id="track_1",
        confidence=0.9,
    )
    return TrackedSkeleton(
        track_id="track_1",
        frame_id="frame_1",
        skeleton=skeleton,
    )


def neck_definition():
    return JointAngleDefinition(
        name="neck_angle",
        first_keypoint="head",
        vertex_keypoint="neck",
        third_keypoint="body_center",
    )


def test_joint_angle_uses_canonical_keypoint_type_and_returns_radians():
    observation = make_observation([
        (KeypointType.BODY_CENTER, 0.0, 1.0, 0.9, KeypointVisibility.VISIBLE),
        (KeypointType.HEAD, 1.0, 0.0, 0.9, KeypointVisibility.VISIBLE),
        (KeypointType.NECK, 0.0, 0.0, 0.9, KeypointVisibility.VISIBLE),
    ])

    result = JointAngleFeatureExtractor([neck_definition()]).extract(observation)

    assert result.names == ("neck_angle",)
    assert result.valid_mask.tolist() == [True]
    assert np.isclose(result.angles[0], np.pi / 2)


def test_duplicate_keypoint_type_is_rejected():
    observation = make_observation([
        (KeypointType.HEAD, 1.0, 0.0, 0.9, KeypointVisibility.VISIBLE),
        (KeypointType.HEAD, 2.0, 0.0, 0.9, KeypointVisibility.VISIBLE),
        (KeypointType.NECK, 0.0, 0.0, 0.9, KeypointVisibility.VISIBLE),
        (KeypointType.BODY_CENTER, 0.0, 1.0, 0.9, KeypointVisibility.VISIBLE),
    ])

    with pytest.raises(ValueError, match="Duplicate keypoint"):
        JointAngleFeatureExtractor([neck_definition()]).extract(observation)


def test_out_of_frame_keypoint_invalidates_angle():
    observation = make_observation([
        (KeypointType.HEAD, 1.0, 0.0, 0.9, KeypointVisibility.OUT_OF_FRAME),
        (KeypointType.NECK, 0.0, 0.0, 0.9, KeypointVisibility.VISIBLE),
        (KeypointType.BODY_CENTER, 0.0, 1.0, 0.9, KeypointVisibility.VISIBLE),
    ])

    result = JointAngleFeatureExtractor([neck_definition()]).extract(observation)

    assert result.valid_mask.tolist() == [False]
    assert result.angles.tolist() == [0.0]


def test_non_finite_coordinates_invalidate_angle():
    observation = make_observation([
        (KeypointType.HEAD, float("nan"), 0.0, 0.9, KeypointVisibility.VISIBLE),
        (KeypointType.NECK, 0.0, 0.0, 0.9, KeypointVisibility.VISIBLE),
        (KeypointType.BODY_CENTER, 0.0, 1.0, 0.9, KeypointVisibility.VISIBLE),
    ])

    result = JointAngleFeatureExtractor([neck_definition()]).extract(observation)

    assert result.valid_mask.tolist() == [False]


def test_invalid_keypoint_name_in_definition_is_rejected():
    definition = JointAngleDefinition(
        name="bad_angle",
        first_keypoint="head",
        vertex_keypoint="not_a_chicken_keypoint",
        third_keypoint="body_center",
    )

    with pytest.raises(ValueError, match="unknown canonical keypoint"):
        JointAngleFeatureExtractor([definition])
