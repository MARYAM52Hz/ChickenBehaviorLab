"""
ChickenBehaviorLab Skeleton Angle Definitions
=============================================
"""

from chicken_behavior_lab.features.angles import (
    JointAngleDefinition,
)


CHICKEN_ANGLE_DEFINITIONS = [
    JointAngleDefinition(
        name="neck_angle",
        first_keypoint="head",
        vertex_keypoint="neck",
        third_keypoint="body",
    ),
]
