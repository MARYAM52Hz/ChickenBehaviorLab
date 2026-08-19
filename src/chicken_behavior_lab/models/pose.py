"""
ChickenBehaviorLab Pose Model
=============================
"""

from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.models.keypoint import (
    Keypoint,
)


@dataclass(slots=True)
class Pose:
    """
    Represents a chicken pose estimated from a frame.
    """

    pose_id: str

    track_id: str | None

    frame_id: str

    keypoints: list[Keypoint]

    confidence: float
