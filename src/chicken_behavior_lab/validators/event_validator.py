"""
Behavior Event Validators
=========================

Validation utilities for behavior transitions and events.
"""

from __future__ import annotations

from chicken_behavior_lab.core.exceptions import ValidationError
from chicken_behavior_lab.models.behavior_event import (
    BehaviorEvent,
)
from chicken_behavior_lab.validators.base import BaseValidator


class BehaviorEventValidator(
    BaseValidator[BehaviorEvent]
):
    """
    Validate behavior events.
    """

    def validate(self, data: BehaviorEvent) -> bool:
        return (
            bool(data.event_id)
            and bool(data.track_id)
            and data.frame_number >= 0
            and 0.0 <= data.confidence <= 1.0
        )

    def validate_or_raise(
        self,
        data: BehaviorEvent,
    ) -> None:

        if not self.validate(data):
            raise ValidationError(
                f"Invalid behavior event: "
                f"{data.event_id}"
            )
