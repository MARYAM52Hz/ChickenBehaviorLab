"""
ChickenBehaviorLab Behavior Event Model
=======================================
"""

from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.core.metadata import Metadata
from chicken_behavior_lab.core.enums.behavior import BehaviorType
from chicken_behavior_lab.core.enums.event import EventType


@dataclass(slots=True)
class BehaviorEvent:
    """
    Represents one behavior transition.
    """

    event_id: str

    track_id: str

    event_type: EventType

    previous_behavior: BehaviorType

    current_behavior: BehaviorType

    frame_number: int

    confidence: float

    metadata: Metadata | None = None
