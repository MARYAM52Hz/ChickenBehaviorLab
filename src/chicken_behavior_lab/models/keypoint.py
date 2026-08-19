"""
ChickenBehaviorLab Keypoint Model
==================================

Canonical representation of an anatomical keypoint
estimated from an image or video frame.
"""

from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.core.enums.annotation import (
    KeypointVisibility,
)

from chicken_behavior_lab.core.enums.keypoints import (
    KeypointType,
)


@dataclass(slots=True)
class Keypoint:
    """
    Represents one anatomical keypoint of a chicken.
    """

    keypoint_type: KeypointType

    x: float

    y: float

    confidence: float

    visibility: KeypointVisibility = (
        KeypointVisibility.VISIBLE
    )

    def is_valid(
        self,
        confidence_threshold: float = 0.25,
    ) -> bool:
        """
        Determine whether the keypoint is considered valid.
        """

        return (
            self.visibility
            != KeypointVisibility.OUT_OF_FRAME
            and self.confidence
            >= confidence_threshold
        )
