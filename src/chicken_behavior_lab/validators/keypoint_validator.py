"""
Keypoint Validators
===================

Validation utilities for chicken pose keypoints.
"""

from __future__ import annotations

from chicken_behavior_lab.core.exceptions import ValidationError
from chicken_behavior_lab.models.keypoint import (
    Keypoint,
    KeypointSet,
)
from chicken_behavior_lab.validators.base import BaseValidator


class KeypointValidator(BaseValidator[Keypoint]):
    """
    Validate an individual chicken keypoint.
    """

    def validate(self, data: Keypoint) -> bool:
        return (
            0.0 <= data.confidence <= 1.0
        )

    def validate_or_raise(self, data: Keypoint) -> None:
        if not self.validate(data):
            raise ValidationError(
                f"Invalid keypoint confidence: "
                f"{data.confidence}"
            )


class KeypointSetValidator(BaseValidator[KeypointSet]):
    """
    Validate a collection of chicken keypoints.
    """

    def validate(self, data: KeypointSet) -> bool:
        if not data.keypoints:
            return False

        return all(
            KeypointValidator().validate(keypoint)
            for keypoint in data.keypoints
        )

    def validate_or_raise(self, data: KeypointSet) -> None:
        if not self.validate(data):
            raise ValidationError(
                "KeypointSet contains invalid keypoints."
            )
