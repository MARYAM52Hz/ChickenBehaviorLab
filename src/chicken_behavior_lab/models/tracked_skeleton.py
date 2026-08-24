"""
ChickenBehaviorLab Tracked Skeleton
===================================

Connects a skeleton with a persistent temporal Track ID.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chicken_behavior_lab.models.skeleton import Skeleton


@dataclass(slots=True)
class TrackedSkeleton:
    """
    A skeleton associated with a persistent chicken identity.

    A TrackedSkeleton represents one chicken
    in one frame.
    """

    track_id: str

    frame_id: str

    skeleton: Skeleton

    timestamp: float | None = None

    metadata: dict = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """
        Ensure the underlying skeleton carries
        the same Track ID.
        """

        self.skeleton.track_id = self.track_id

    # =====================================================
    # Convenience Properties
    # =====================================================

    @property
    def num_keypoints(self) -> int:
        """
        Number of keypoints in this skeleton.
        """

        return self.skeleton.num_nodes

    @property
    def confidence(self) -> float:
        """
        Skeleton confidence.
        """

        return self.skeleton.confidence
