"""
ChickenBehaviorLab Temporal Buffer
===================================
"""

from __future__ import annotations

from chicken_behavior_lab.models.tracked_skeleton import (
    TrackedSkeleton,
)

from chicken_behavior_lab.temporal.sequence import (
    TemporalSkeletonSequence,
)


class TemporalSkeletonBuffer:
    """
    Maintains temporal skeleton sequences
    for multiple tracked chickens.
    """

    def __init__(
        self,
        max_length: int = 60,
    ) -> None:

        if max_length <= 0:
            raise ValueError(
                "max_length must be positive."
            )

        self.max_length = max_length

        self.sequences: dict[
            str,
            TemporalSkeletonSequence,
        ] = {}

    # =====================================================
    # Add
    # =====================================================

    def add(
        self,
        observation: TrackedSkeleton,
    ) -> None:
        """
        Add an observation to the appropriate
        chicken sequence.
        """

        track_id = observation.track_id

        if track_id not in self.sequences:

            self.sequences[track_id] = (
                TemporalSkeletonSequence(
                    track_id=track_id,
                    max_length=self.max_length,
                )
            )

        self.sequences[
            track_id
        ].add(
            observation
        )

    # =====================================================
    # Get
    # =====================================================

    def get(
        self,
        track_id: str,
    ) -> TemporalSkeletonSequence | None:
        """
        Return the sequence for a track.
        """

        return self.sequences.get(
            track_id
        )

    # =====================================================
    # Remove
    # =====================================================

    def remove(
        self,
        track_id: str,
    ) -> None:
        """
        Remove a track sequence.
        """

        self.sequences.pop(
            track_id,
            None,
        )

    # =====================================================
    # Clear
    # =====================================================

    def clear(self) -> None:
        """
        Remove all temporal sequences.
        """

        self.sequences.clear()

    # =====================================================
    # Active Tracks
    # =====================================================

    @property
    def active_track_ids(
        self,
    ) -> list[str]:
        """
        Return all currently stored Track IDs.
        """

        return list(
            self.sequences.keys()
        )
