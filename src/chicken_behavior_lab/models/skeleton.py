"""
ChickenBehaviorLab Skeleton Model
=================================
"""

from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.core.metadata import Metadata
from chicken_behavior_lab.models.keypoint import (
    KeypointConnection,
    KeypointSet,
)


@dataclass(slots=True)
class Skeleton:
    """
    Graph representation of a chicken pose.
    """

    skeleton_id: str

    pose_id: str

    keypoints: KeypointSet

    connections: list[KeypointConnection]

    metadata: Metadata | None = None
