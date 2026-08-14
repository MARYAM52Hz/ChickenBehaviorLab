"""
ChickenBehaviorLab Behavior Event Model
=======================================
"""

from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.core.enums.behavior import (
    BehaviorType,
)

from chicken_behavior_lab.core.enums.event import (
    EventType,
)

from chicken_behavior_lab.core.metadata import Metadata


@dataclass(slots=True)
class BehaviorEvent:
    """
    Represents one temporal behavior event.
    """

    event_id: str

    track_id: str

    event_type: EventType

    previous_behavior: BehaviorType

    current_behavior: BehaviorType

    frame_number: int

    confidence: float

    metadata: Metadata | None = None
