"""
ChickenBehaviorLab Temporal Skeleton Sequence
==============================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chicken_behavior_lab.models.tracked_skeleton import (
    TrackedSkeleton,
)


@dataclass(slots=True)
class TemporalSkeletonSequence:
    """
    Ordered sequence of skeleton observations
    belonging to one chicken.
    """

    track_id: str

    observations: list[
        TrackedSkeleton
    ] = field(
        default_factory=list
    )

    max_length: int | None = None

    # =====================================================
    # Add Observation
    # =====================================================

    def add(
        self,
        observation: TrackedSkeleton,
    ) -> None:
        """
        Add a new skeleton observation.
        """

        if (
            observation.track_id
            != self.track_id
        ):
            raise ValueError(
                "Track ID mismatch: "
                f"expected {self.track_id}, "
                f"received "
                f"{observation.track_id}"
            )

        self.observations.append(
            observation
        )

        self._trim()

    # =====================================================
    # Trim
    # =====================================================

    def _trim(self) -> None:
        """
        Keep only the most recent observations
        if max_length is defined.
        """

        if self.max_length is None:
            return

        if self.max_length <= 0:
            return

        if (
            len(self.observations)
            > self.max_length
        ):

            self.observations = (
                self.observations[
                    -self.max_length:
                ]
            )

    # =====================================================
    # Properties
    # =====================================================

    @property
    def length(self) -> int:
        """
        Number of observations.
        """

        return len(
            self.observations
        )

    @property
    def is_empty(self) -> bool:
        """
        Whether the sequence is empty.
        """

        return len(
            self.observations
        ) == 0

    @property
    def latest(
        self,
    ) -> TrackedSkeleton | None:
        """
        Return the most recent observation.
        """

        if self.is_empty:
            return None

        return self.observations[-1]

    # =====================================================
    # Clear
    # =====================================================

    def clear(self) -> None:
        """
        Remove all observations.
        """

        self.observations.clear()
