"""
ChickenBehaviorLab Keypoint Models
==================================

Canonical keypoint data structures used for chicken pose estimation.

Author:
    ChickenBehaviorLab Contributors

License:
    MIT License
"""

from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.models.geometry import Point2D


# ==========================================================
# Keypoint
# ==========================================================

@dataclass(slots=True)
class Keypoint:
    """
    Represents a single anatomical landmark.
    """

    name: str

    location: Point2D

    confidence: float

    visible: bool = True


# ==========================================================
# KeypointConnection
# ==========================================================

@dataclass(slots=True)
class KeypointConnection:
    """
    Defines a connection between two anatomical landmarks.
    """

    source: str

    target: str


# ==========================================================
# KeypointSet
# ==========================================================

@dataclass(slots=True)
class KeypointSet:
    """
    Collection of anatomical landmarks for one chicken.
    """

    keypoints: list[Keypoint]
