"""
ChickenBehaviorLab Keypoint Enums
=================================
"""

from __future__ import annotations

from enum import Enum


class KeypointType(str, Enum):
    """Canonical anatomical keypoints."""

    BEAK = "beak"

    HEAD = "head"

    NECK = "neck"

    BODY_CENTER = "body_center"

    LEFT_WING = "left_wing"

    RIGHT_WING = "right_wing"

    TAIL = "tail"

    LEFT_HIP = "left_hip"

    RIGHT_HIP = "right_hip"

    LEFT_KNEE = "left_knee"

    RIGHT_KNEE = "right_knee"

    LEFT_FOOT = "left_foot"

    RIGHT_FOOT = "right_foot"
