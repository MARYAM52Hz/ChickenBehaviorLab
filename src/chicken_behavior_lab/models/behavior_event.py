"""
ChickenBehaviorLab Behavior Event Model
=======================================
"""

from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.core.enums.behavior import BehaviorType
from chicken_behavior_lab.core.enums.event import EventType
from chicken_behavior_lab.core.identifiers import normalize_track_id
from chicken_behavior_lab.core.metadata import Metadata


@dataclass(slots=True)
class BehaviorEvent:
    """A frame-level behavior transition or event."""

    event_id: str
    track_id: str | int
    event_type: EventType
    previous_behavior: BehaviorType
    current_behavior: BehaviorType
    frame_number: int
    confidence: float
    metadata: Metadata | None = None

    def __post_init__(self) -> None:
        # Accept legacy integer IDs at the boundary, but keep the model layer
        # on the same canonical string contract as the rest of the pipeline.
        self.track_id = normalize_track_id(self.track_id)
