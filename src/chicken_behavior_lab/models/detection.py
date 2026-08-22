"""
ChickenBehaviorLab Detection Model
===================================
"""

from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.models.geometry import (
    BoundingBox,
)


@dataclass(slots=True)
class Detection:
    """
    Represents one detected chicken in one frame.
    """

    detection_id: str

    frame_id: str

    bbox: BoundingBox

    confidence: float

    class_name: str = "chicken"
