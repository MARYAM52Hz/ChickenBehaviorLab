"""
ChickenBehaviorLab Behavior Model
=================================
"""

from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.core.enums.behavior import (
    BehaviorType,
)

from chicken_behavior_lab.core.metadata import Metadata


@dataclass(slots=True)
class Behavior:
    """
    Represents one recognized behavior.
    """

    behavior_id: str

    track_id: str

    behavior_type: BehaviorType

    confidence: float

    start_frame: int

    end_frame: int

    metadata: Metadata | None = None
