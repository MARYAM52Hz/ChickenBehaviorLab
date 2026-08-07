"""
ChickenBehaviorLab Mortality Prediction Model
=============================================
"""

from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.core.metadata import Metadata
from chicken_behavior_lab.core.enums.prediction import RiskLevel


@dataclass(slots=True)
class MortalityPrediction:
    """
    Represents mortality prediction for a flock.
    """

    prediction_id: str

    flock_id: str

    risk_level: RiskLevel

    probability: float

    prediction_horizon_hours: int

    metadata: Metadata | None = None
