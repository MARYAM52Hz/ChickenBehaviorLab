"""
ChickenBehaviorLab Pose Model
=============================

Canonical pose representation.
"""

from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.core.metadata import Metadata
from chicken_behavior_lab.models.keypoint import KeypointSet


@dataclass(slots=True)
class Pose:
    """
    Pose estimated for one tracked chicken.
    """

    pose_id: str

    track_id: str

    keypoints: KeypointSet

    confidence: float

    metadata: Metadata | None = None
