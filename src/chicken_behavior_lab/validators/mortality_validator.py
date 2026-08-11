"""
Mortality Prediction Validators
================================

Validation utilities for mortality risk predictions.
"""

from __future__ import annotations

from chicken_behavior_lab.core.exceptions import ValidationError
from chicken_behavior_lab.models.mortality import (
    MortalityPrediction,
)
from chicken_behavior_lab.validators.base import BaseValidator


class MortalityPredictionValidator(
    BaseValidator[MortalityPrediction]
):
    """
    Validate mortality predictions.
    """

    def validate(
        self,
        data: MortalityPrediction,
    ) -> bool:

        if not data.prediction_id:
            return False

        if not data.flock_id:
            return False

        if not 0.0 <= data.probability <= 1.0:
            return False

        if data.prediction_horizon_hours <= 0:
            return False

        return True

    def validate_or_raise(
        self,
        data: MortalityPrediction,
    ) -> None:

        if not self.validate(data):
            raise ValidationError(
                f"Invalid mortality prediction: "
                f"{data.prediction_id}"
            )
