"""
ChickenBehaviorLab Track Model
===============================
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chicken_behavior_lab.models.detection import (
    Detection,
)


@dataclass(slots=True)
class Track:
    """
    Represents the temporal identity of one chicken.
    """

    track_id: int

    detection: Detection

    age: int = 1

    hits: int = 1

    missed_frames: int = 0

    history: list[Detection] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        """
        Initialize history with the first detection.
        """

        if not self.history:
            self.history.append(
                self.detection
            )

    # =====================================================
    # Update
    # =====================================================

    def update(
        self,
        detection: Detection,
    ) -> None:
        """
        Update the track with a new detection.
        """

        self.detection = detection

        self.age += 1

        self.hits += 1

        self.missed_frames = 0

        self.history.append(
            detection
        )

    # =====================================================
    # Miss
    # =====================================================

    def mark_missed(self) -> None:
        """
        Mark this track as not detected in
        the current frame.
        """

        self.age += 1

        self.missed_frames += 1

    # =====================================================
    # State
    # =====================================================

    @property
    def is_active(self) -> bool:
        """
        Return whether the track is still active.
        """

        return self.missed_frames == 0
