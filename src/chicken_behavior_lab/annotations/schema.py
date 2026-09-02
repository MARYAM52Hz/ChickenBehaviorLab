"""
ChickenBehaviorLab Annotation Schema
====================================

Initial annotation schema for chicken behavior recognition.

This is an MVP schema and is intentionally designed to be
simple, explicit, and extensible.

The schema can later be aligned directly with CBAS/CBO.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BehaviorAnnotation:
    """
    Represents one behavior annotation.

    Parameters
    ----------
    annotation_id:
        Unique identifier for the annotation.

    video_id:
        Identifier of the source video.

    track_id:
        Identifier of the tracked chicken.

    behavior_id:
        Canonical behavior identifier.

    start_frame:
        First frame of the behavior interval.

    end_frame:
        Last frame of the behavior interval.

    annotator:
        Optional annotator identifier.

    confidence:
        Optional annotation confidence.

    notes:
        Optional annotation notes.
    """

    annotation_id: str

    video_id: str

    track_id: int

    behavior_id: str

    start_frame: int

    end_frame: int

    annotator: str | None = None

    confidence: float | None = None

    notes: str | None = None

    def validate(self) -> None:
        """
        Validate annotation fields.
        """

        if not self.annotation_id:
            raise ValueError(
                "annotation_id cannot be empty."
            )

        if not self.video_id:
            raise ValueError(
                "video_id cannot be empty."
            )

        if self.track_id < 0:
            raise ValueError(
                "track_id cannot be negative."
            )

        if not self.behavior_id:
            raise ValueError(
                "behavior_id cannot be empty."
            )

        if self.start_frame < 0:
            raise ValueError(
                "start_frame cannot be negative."
            )

        if self.end_frame < self.start_frame:
            raise ValueError(
                "end_frame must be greater than "
                "or equal to start_frame."
            )

        if self.confidence is not None:

            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    "confidence must be between "
                    "0.0 and 1.0."
                )

    @property
    def num_frames(self) -> int:
        """
        Return the number of annotated frames.

        The interval is inclusive.
        """

        return (
            self.end_frame
            - self.start_frame
            + 1
        )
