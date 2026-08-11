"""
Behavior Validators
===================

Validation utilities for recognized chicken behaviors.
"""

from __future__ import annotations

from chicken_behavior_lab.core.exceptions import ValidationError
from chicken_behavior_lab.models.behavior import Behavior
from chicken_behavior_lab.validators.base import BaseValidator


class BehaviorValidator(BaseValidator[Behavior]):
    """
    Validate a recognized chicken behavior.
    """

    def validate(self, data: Behavior) -> bool:
        if not data.behavior_id:
            return False

        if not data.track_id:
            return False

        if not 0.0 <= data.confidence <= 1.0:
            return False

        if data.start_frame < 0:
            return False

        if data.end_frame < data.start_frame:
            return False

        return True

    def validate_or_raise(self, data: Behavior) -> None:
        if not self.validate(data):
            raise ValidationError(
                f"Invalid behavior: {data.behavior_id}"
            )
