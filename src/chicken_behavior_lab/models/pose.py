"""
ChickenBehaviorLab Pose Model
=============================

Represents the complete estimated pose of one chicken
in one video frame.
"""

from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.models.keypoint import (
    Keypoint,
)


@dataclass(slots=True)
class Pose:
    """
    Represents a chicken pose estimated from a frame.
    """

    pose_id: str

    frame_id: str

    keypoints: list[Keypoint]

    confidence: float

    track_id: str | None = None

    def get_keypoint(
        self,
        keypoint_type,
    ) -> Keypoint | None:
        """
        Return a keypoint by its anatomical type.
        """

        for keypoint in self.keypoints:

            if keypoint.keypoint_type == keypoint_type:
                return keypoint

        return None

    def valid_keypoints(
        self,
        confidence_threshold: float = 0.25,
    ) -> list[Keypoint]:
        """
        Return only valid keypoints.
        """

        return [
            keypoint
            for keypoint in self.keypoints
            if keypoint.is_valid(
                confidence_threshold
            )
        ]
