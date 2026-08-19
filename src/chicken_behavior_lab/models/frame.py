"""
ChickenBehaviorLab Frame Model
==============================
"""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path


@dataclass(slots=True)
class Frame:
    """
    Represents one frame from a video sequence.
    """

    frame_id: str

    image_path: str | Path

    frame_number: int

    timestamp: float | None = None

    width: int | None = None

    height: int | None = None
