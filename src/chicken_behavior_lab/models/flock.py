"""
ChickenBehaviorLab Flock Model
==============================
"""

from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.core.metadata import Metadata


@dataclass(slots=True)
class Flock:
    """
    Represents one monitored flock.
    """

    flock_id: str

    farm_id: str

    active_tracks: int

    active_behaviors: int

    observation_start: float

    observation_end: float

    metadata: Metadata | None = None
