"""
ChickenBehaviorLab Velocity Features
=====================================

Extract temporal velocity features using anatomical keypoint identity,
not list position.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from chicken_behavior_lab.core.enums.keypoints import KeypointType
from chicken_behavior_lab.models.tracked_skeleton import TrackedSkeleton


@dataclass(slots=True)
class VelocityFeatures:
    """Velocity vectors and their validity in canonical keypoint order."""

    velocity: np.ndarray
    speed: np.ndarray
    valid_mask: np.ndarray
    keypoint_types: tuple[KeypointType, ...]

    @property
    def num_keypoints(self) -> int:
        return self.velocity.shape[0]

    @property
    def feature_dimension(self) -> int:
        return self.velocity.shape[1]


class VelocityFeatureExtractor:
    """Estimate velocity after matching points by KeypointType."""

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        keypoint_order: tuple[KeypointType, ...] | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1].")

        self.confidence_threshold = float(confidence_threshold)
        self.keypoint_order = (
            tuple(KeypointType)
            if keypoint_order is None
            else tuple(keypoint_order)
        )

        if not self.keypoint_order:
            raise ValueError("keypoint_order cannot be empty.")
        if len(set(self.keypoint_order)) != len(self.keypoint_order):
            raise ValueError("keypoint_order must contain unique keypoint types.")
        if any(not isinstance(item, KeypointType) for item in self.keypoint_order):
            raise TypeError("keypoint_order must contain KeypointType values.")

    def extract(
        self,
        previous: TrackedSkeleton,
        current: TrackedSkeleton,
        delta_time: float,
    ) -> VelocityFeatures:
        """Calculate per-keypoint velocity in the configured stable order."""

        self._validate_inputs(previous, current, delta_time)

        previous_by_type = self._index_keypoints(previous)
        current_by_type = self._index_keypoints(current)

        unexpected = (
            set(previous_by_type) | set(current_by_type)
        ) - set(self.keypoint_order)
        if unexpected:
            names = sorted(item.value for item in unexpected)
            raise ValueError(f"Unexpected keypoint types: {names}.")

        velocity = np.zeros((len(self.keypoint_order), 2), dtype=np.float32)
        valid_mask = np.zeros((len(self.keypoint_order),), dtype=bool)

        for index, keypoint_type in enumerate(self.keypoint_order):
            previous_point = previous_by_type.get(keypoint_type)
            current_point = current_by_type.get(keypoint_type)

            if previous_point is None or current_point is None:
                continue

            if (
                not previous_point.is_valid(self.confidence_threshold)
                or not current_point.is_valid(self.confidence_threshold)
            ):
                continue

            velocity[index] = (
                (current_point.x - previous_point.x) / delta_time,
                (current_point.y - previous_point.y) / delta_time,
            )
            valid_mask[index] = True

        speed = np.linalg.norm(velocity, axis=1).astype(np.float32)

        return VelocityFeatures(
            velocity=velocity,
            speed=speed,
            valid_mask=valid_mask,
            keypoint_types=self.keypoint_order,
        )

    @staticmethod
    def _index_keypoints(observation: TrackedSkeleton):
        result = {}
        for keypoint in observation.skeleton.keypoints:
            if keypoint.keypoint_type in result:
                raise ValueError(
                    f"Duplicate keypoint type: {keypoint.keypoint_type.value}."
                )
            result[keypoint.keypoint_type] = keypoint
        return result

    @staticmethod
    def _validate_inputs(
        previous: TrackedSkeleton,
        current: TrackedSkeleton,
        delta_time: float,
    ) -> None:
        if previous.track_id != current.track_id:
            raise ValueError(
                "Track ID mismatch: "
                f"previous={previous.track_id}, current={current.track_id}"
            )

        if isinstance(delta_time, bool) or not isinstance(delta_time, (int, float)):
            raise ValueError("delta_time must be a finite positive number.")
        if not math.isfinite(float(delta_time)) or delta_time <= 0.0:
            raise ValueError("delta_time must be a finite positive number.")
