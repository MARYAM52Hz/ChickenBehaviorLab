"""
ChickenBehaviorLab Track Model
===============================

Represents the persistent temporal identity of a
detected chicken across consecutive video frames.
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

    A Track connects detections belonging to the same
    chicken across multiple video frames.

    Attributes
    ----------
    track_id:
        Persistent identifier assigned to the chicken.

    detection:
        Most recent detection associated with this track.

    age:
        Number of frames since the track was created.

    hits:
        Number of successful detection-to-track matches.

    missed_frames:
        Number of consecutive frames in which the chicken
        was not successfully detected.

    history:
        Ordered history of detections associated with
        this track.
    """

    track_id: str

    detection: Detection

    age: int = 1

    hits: int = 1

    missed_frames: int = 0

    history: list[Detection] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        """
        Initialize the detection history.

        The first detection is automatically stored when
        a new Track is created.
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

        Parameters
        ----------
        detection:
            New detection associated with this track.
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
        Mark the track as missed for the current frame.

        The track is not immediately deleted because a
        chicken may temporarily disappear due to occlusion,
        detection failure, or overlap with another chicken.
        """

        self.age += 1

        self.missed_frames += 1

    # =====================================================
    # State
    # =====================================================

    @property
    def is_active(self) -> bool:
        """
        Return True when the track was detected
        in the current frame.
        """

        return self.missed_frames == 0

    @property
    def last_detection(self) -> Detection:
        """
        Return the most recent detection.

        This is equivalent to ``detection`` but provides
        a semantically explicit interface for downstream
        temporal processing.
        """

        return self.detection

    @property
    def history_length(self) -> int:
        """
        Return the number of detections stored
        in the track history.
        """

        return len(self.history)
