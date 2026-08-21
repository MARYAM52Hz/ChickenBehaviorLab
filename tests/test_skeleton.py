"""
Tests for ChickenBehaviorLab Skeleton Builder.
"""

from chicken_behavior_lab.core.enums.keypoints import (
    KeypointType,
)

from chicken_behavior_lab.models.keypoint import (
    Keypoint,
)

from chicken_behavior_lab.models.pose import (
    Pose,
)

from chicken_behavior_lab.skeleton.builder import (
    SkeletonBuilder,
)


def test_skeleton_builder():

    keypoints = [

        Keypoint(
            keypoint_type=KeypointType.BEAK,
            x=100.0,
            y=100.0,
            confidence=0.95,
        ),

        Keypoint(
            keypoint_type=KeypointType.HEAD,
            x=110.0,
            y=110.0,
            confidence=0.95,
        ),

        Keypoint(
            keypoint_type=KeypointType.NECK,
            x=120.0,
            y=130.0,
            confidence=0.90,
        ),

        Keypoint(
            keypoint_type=KeypointType.BODY_CENTER,
            x=150.0,
            y=180.0,
            confidence=0.92,
        ),

        Keypoint(
            keypoint_type=KeypointType.TAIL,
            x=200.0,
            y=190.0,
            confidence=0.91,
        ),
    ]

    pose = Pose(
        pose_id="test_pose_001",
        frame_id="frame_001",
        keypoints=keypoints,
        confidence=0.92,
    )

    builder = SkeletonBuilder()

    skeleton = builder.build(pose)

    assert skeleton is not None

    assert skeleton.num_nodes == 5

    assert skeleton.num_edges > 0

    assert skeleton.is_valid()
