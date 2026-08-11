"""
Pose Validators
===============

Validation utilities for chicken pose estimates.
"""

from __future__ import annotations

from chicken_behavior_lab.core.exceptions import ValidationError
from chicken_behavior_lab.models.pose import Pose
from chicken_behavior_lab.validators.base import BaseValidator
from chicken_behavior_lab.validators.keypoint_validator import (
    KeypointSetValidator,
)


class PoseValidator(BaseValidator[Pose]):
    """
    Validate a chicken pose.
    """

    def validate(self, data: Pose) -> bool:
        if not data.pose_id:
            return False

        if not data.track_id:
            return False

        if not 0.0 <= data.confidence <= 1.0:
            return False

        return KeypointSetValidator().validate(
            data.keypoints
        )

    def validate_or_raise(self, data: Pose) -> None:
        if not self.validate(data):
            raise ValidationError(
                f"Invalid pose: {data.pose_id}"
            )
