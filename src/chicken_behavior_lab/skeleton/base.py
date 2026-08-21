"""
ChickenBehaviorLab Skeleton Definitions
========================================

Canonical anatomical graph definitions for chicken pose.
"""

from __future__ import annotations

from chicken_behavior_lab.core.enums.keypoints import (
    KeypointType,
)


CBAS_SKELETON_CONNECTIONS: tuple[
    tuple[KeypointType, KeypointType],
    ...
] = (

    # Head
    (
        KeypointType.BEAK,
        KeypointType.HEAD,
    ),

    (
        KeypointType.HEAD,
        KeypointType.NECK,
    ),

    # Main body axis
    (
        KeypointType.NECK,
        KeypointType.BODY_CENTER,
    ),

    (
        KeypointType.BODY_CENTER,
        KeypointType.TAIL,
    ),

    # Wings
    (
        KeypointType.BODY_CENTER,
        KeypointType.LEFT_WING,
    ),

    (
        KeypointType.BODY_CENTER,
        KeypointType.RIGHT_WING,
    ),

    # Left leg
    (
        KeypointType.BODY_CENTER,
        KeypointType.LEFT_HIP,
    ),

    (
        KeypointType.LEFT_HIP,
        KeypointType.LEFT_KNEE,
    ),

    (
        KeypointType.LEFT_KNEE,
        KeypointType.LEFT_FOOT,
    ),

    # Right leg
    (
        KeypointType.BODY_CENTER,
        KeypointType.RIGHT_HIP,
    ),

    (
        KeypointType.RIGHT_HIP,
        KeypointType.RIGHT_KNEE,
    ),

    (
        KeypointType.RIGHT_KNEE,
        KeypointType.RIGHT_FOOT,
    ),
)
