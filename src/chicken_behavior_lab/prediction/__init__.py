
"""Prediction interfaces for ChickenBehaviorLab."""

from .mortality_predictor import (
    MortalityPrediction,
    MortalityPredictionError,
    MortalityPredictor,
)

__all__ = [
    "MortalityPrediction",
    "MortalityPredictionError",
    "MortalityPredictor",
]
