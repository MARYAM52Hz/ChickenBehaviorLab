"""
ChickenBehaviorLab Frame Model
==============================

Canonical frame representation used throughout the framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from chicken_behavior_lab.core.metadata import Metadata
from chicken_behavior_lab.models.geometry import ImageSize


@dataclass(slots=True)
class Frame:
    """
    Represents a single video frame.
    """

    frame_id: str

    video_id: str

    frame_number: int

    timestamp: float

    image_size: ImageSize

    image_path: Path | None = None

    captured_at: datetime | None = None

    metadata: Metadata | None = None
