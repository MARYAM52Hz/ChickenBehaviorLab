"""
ChickenBehaviorLab Skeleton Builder
====================================

Builds graph-based chicken skeletons from estimated poses.
"""

from __future__ import annotations

from chicken_behavior_lab.core.enums.keypoints import (
    KeypointType,
)

from chicken_behavior_lab.models.pose import (
    Pose,
)

from chicken_behavior_lab.models.skeleton import (
    Skeleton,
    SkeletonEdge,
)

from chicken_behavior_lab.skeleton.base import (
    CBAS_SKELETON_CONNECTIONS,
)


class SkeletonBuilder:
    """
    Converts Pose objects into canonical Skeleton graphs.
    """

    def __init__(
        self,
        connections=CBAS_SKELETON_CONNECTIONS,
    ) -> None:

        self.connections = connections

    # =====================================================
    # Build Skeleton
    # =====================================================

    def build(
        self,
        pose: Pose,
    ) -> Skeleton:
        """
        Build a skeleton graph from a pose.
        """

        edges: list[SkeletonEdge] = []

        available_keypoints = {
            keypoint.keypoint_type
            for keypoint in pose.keypoints
        }

        for source, target in self.connections:

            if source not in available_keypoints:
                continue

            if target not in available_keypoints:
                continue

            edges.append(
                SkeletonEdge(
                    source=source.value,
                    target=target.value,
                )
            )

        return Skeleton(
            skeleton_id=(
                f"{pose.pose_id}_skeleton"
            ),
            frame_id=pose.frame_id,
            keypoints=pose.keypoints,
            edges=edges,
            track_id=pose.track_id,
            confidence=pose.confidence,
        )
