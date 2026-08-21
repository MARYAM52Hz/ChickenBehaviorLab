"""
ChickenBehaviorLab Skeleton Model
==================================

Graph representation of a chicken anatomical skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chicken_behavior_lab.models.keypoint import (
    Keypoint,
)


@dataclass(frozen=True, slots=True)
class SkeletonEdge:
    """
    Represents an anatomical connection between
    two keypoints.
    """

    source: str

    target: str

    weight: float = 1.0


@dataclass(slots=True)
class Skeleton:
    """
    Graph-based representation of a chicken skeleton.

    Nodes:
        Anatomical keypoints.

    Edges:
        Anatomical connections between keypoints.
    """

    skeleton_id: str

    frame_id: str

    keypoints: list[Keypoint]

    edges: list[SkeletonEdge]

    track_id: str | None = None

    confidence: float = 0.0

    metadata: dict = field(
        default_factory=dict
    )

    # =====================================================
    # Keypoint Access
    # =====================================================

    def get_keypoint(
        self,
        name: str,
    ) -> Keypoint | None:
        """
        Return a keypoint by its name.
        """

        for keypoint in self.keypoints:

            if keypoint.keypoint_type.value == name:
                return keypoint

        return None

    # =====================================================
    # Graph Information
    # =====================================================

    @property
    def num_nodes(self) -> int:
        """
        Number of keypoints in the skeleton.
        """

        return len(self.keypoints)

    @property
    def num_edges(self) -> int:
        """
        Number of anatomical connections.
        """

        return len(self.edges)

    # =====================================================
    # Graph Validation
    # =====================================================

    def is_valid(self) -> bool:
        """
        Basic structural validation.
        """

        if not self.keypoints:
            return False

        if not self.edges:
            return False

        return True
