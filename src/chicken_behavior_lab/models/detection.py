"""
ChickenBehaviorLab Detection Model
==================================
"""

from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.core.metadata import Metadata
from chicken_behavior_lab.models.geometry import BoundingBox


@dataclass(slots=True)
class Detection:
    """
    Represents a single detected chicken.
    """

    detection_id: str

    frame_id: str

    bbox: BoundingBox

    confidence: float

    class_name: str = "chicken"

    metadata: Metadata | None = None
