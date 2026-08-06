"""
ChickenBehaviorLab Track Model
==============================

Canonical object tracking model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chicken_behavior_lab.core.metadata import Metadata


@dataclass(slots=True)
class Track:
    """
    Represents one tracked chicken across multiple frames.
    """

    track_id: str

    detection_ids: list[str] = field(default_factory=list)

    frame_ids: list[str] = field(default_factory=list)

    first_frame: int = 0

    last_frame: int = 0

    is_active: bool = True

    metadata: Metadata | None = None
