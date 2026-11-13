"""Compute acceleration from two consecutive velocity observations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


@dataclass(slots=True)
class AccelerationFeatures:
    """Acceleration vectors, magnitudes, and per-keypoint validity."""

    acceleration: np.ndarray
    magnitude: np.ndarray
    valid_mask: np.ndarray

    @property
    def num_keypoints(self) -> int:
        return self.acceleration.shape[0]

    @property
    def feature_dimension(self) -> int:
        return self.acceleration.shape[1]


class AccelerationFeatureExtractor:
    """Extract acceleration from consecutive velocity observations."""

    def __init__(self, confidence_threshold: float = 0.3) -> None:
        if (
            isinstance(confidence_threshold, bool)
            or not isinstance(confidence_threshold, Real)
            or not math.isfinite(float(confidence_threshold))
            or not 0.0 <= float(confidence_threshold) <= 1.0
        ):
            raise ValueError("confidence_threshold must be a finite number in [0, 1].")
        self.confidence_threshold = float(confidence_threshold)

    def extract(
        self,
        previous_velocity: np.ndarray,
        current_velocity: np.ndarray,
        delta_time: float,
        previous_valid_mask: np.ndarray | None = None,
        current_valid_mask: np.ndarray | None = None,
    ) -> AccelerationFeatures:
        """Calculate acceleration; non-finite or masked rows become zero and invalid."""
        dt = self._validate_delta_time(delta_time)
        previous = self._as_velocity_array(previous_velocity, "previous_velocity")
        current = self._as_velocity_array(current_velocity, "current_velocity")
        if previous.shape != current.shape:
            raise ValueError(
                f"Velocity shape mismatch: previous={previous.shape}, current={current.shape}"
            )

        num_keypoints = current.shape[0]
        valid_mask = np.all(np.isfinite(previous), axis=1) & np.all(np.isfinite(current), axis=1)
        if previous_valid_mask is not None:
            valid_mask &= self._as_mask(previous_valid_mask, "previous_valid_mask", num_keypoints)
        if current_valid_mask is not None:
            valid_mask &= self._as_mask(current_valid_mask, "current_valid_mask", num_keypoints)

        acceleration = np.zeros((num_keypoints, 2), dtype=np.float32)
        computed = (current - previous) / dt
        acceleration[valid_mask] = computed[valid_mask].astype(np.float32)
        magnitude = np.linalg.norm(acceleration, axis=1).astype(np.float32)
        return AccelerationFeatures(
            acceleration=acceleration,
            magnitude=magnitude,
            valid_mask=valid_mask,
        )

    @staticmethod
    def _validate_delta_time(delta_time: float) -> float:
        if (
            isinstance(delta_time, bool)
            or not isinstance(delta_time, Real)
            or not math.isfinite(float(delta_time))
            or float(delta_time) <= 0.0
        ):
            raise ValueError("delta_time must be a finite positive number.")
        return float(delta_time)

    @staticmethod
    def _as_velocity_array(value: np.ndarray, name: str) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric array.") from exc
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"{name} must have shape (num_keypoints, 2).")
        return array

    @staticmethod
    def _as_mask(value: np.ndarray, name: str, num_keypoints: int) -> np.ndarray:
        mask = np.asarray(value)
        if mask.shape != (num_keypoints,):
            raise ValueError(f"{name} must have shape (num_keypoints,).")
        if mask.dtype != np.bool_:
            raise TypeError(f"{name} must contain boolean values.")
        return mask
