"""
Detection Validators
====================

Validation utilities for object detections.
"""

from __future__ import annotations

from chicken_behavior_lab.core.exceptions import ValidationError
from chicken_behavior_lab.models.detection import Detection
from chicken_behavior_lab.validators.base import BaseValidator


class DetectionValidator(BaseValidator[Detection]):
    """
    Validate a chicken detection.
    """

    def validate(self, data: Detection) -> bool:
        return (
            bool(data.detection_id)
            and bool(data.frame_id)
            and data.confidence >= 0.0
            and data.confidence <= 1.0
            and data.bbox.width >= 0.0
            and data.bbox.height >= 0.0
        )

    def validate_or_raise(self, data: Detection) -> None:
        if not self.validate(data):
            raise ValidationError(
                f"Invalid detection: {data.detection_id}"
            )
