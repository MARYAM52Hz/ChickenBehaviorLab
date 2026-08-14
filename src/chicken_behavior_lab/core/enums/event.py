"""
ChickenBehaviorLab Event Enums
==============================
"""

from __future__ import annotations

from enum import Enum


class EventType(str, Enum):
    """Types of temporal behavior events."""

    BEHAVIOR_STARTED = "behavior_started"

    BEHAVIOR_ENDED = "behavior_ended"

    BEHAVIOR_CHANGED = "behavior_changed"

    INTERACTION_STARTED = "interaction_started"

    INTERACTION_ENDED = "interaction_ended"

    ANOMALY_DETECTED = "anomaly_detected"

    MORTALITY_EVENT = "mortality_event"

    UNKNOWN = "unknown"
