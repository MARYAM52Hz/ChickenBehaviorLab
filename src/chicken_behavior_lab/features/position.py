"""Extract spatial features in a stable anatomical keypoint order."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from chicken_behavior_lab.core.enums.keypoints import KeypointType
from chicken_behavior_lab.models.tracked_skeleton import TrackedSkeleton


@dataclass(slots=True)
class PositionFeatures:
    """Spatial features in canonical keypoint order."""

    coordinates: np.ndarray
    confidence: np.ndarray
    valid_mask: np.ndarray
    keypoint_types: tuple[KeypointType, ...]

    @property
    def num_keypoints(self) -> int:
        return self.coordinates.shape[0]

    @property
    def feature_dimension(self) -> int:
        return self.coordinates.shape[1]


class PositionFeatureExtractor:
    """Extract x/y, confidence, and validity in a stable node order."""

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        keypoint_order: tuple[KeypointType, ...] | None = None,
    ) -> None:
        if (
            isinstance(confidence_threshold, bool)
            or not isinstance(confidence_threshold, Real)
            or not math.isfinite(float(confidence_threshold))
            or not 0.0 <= float(confidence_threshold) <= 1.0
        ):
            raise ValueError("confidence_threshold must be a finite number in [0, 1].")

        self.confidence_threshold = float(confidence_threshold)
        self.keypoint_order = tuple(KeypointType) if keypoint_order is None else tuple(keypoint_order)

        if not self.keypoint_order:
            raise ValueError("keypoint_order cannot be empty.")
        if any(not isinstance(item, KeypointType) for item in self.keypoint_order):
            raise TypeError("keypoint_order must contain KeypointType values.")
        if len(set(self.keypoint_order)) != len(self.keypoint_order):
            raise ValueError("keypoint_order must contain unique keypoint types.")

    def extract(self, observation: TrackedSkeleton) -> PositionFeatures:
        """Return fixed-size arrays; absent/invalid nodes are zero-filled and masked."""
        by_type = {}
        for keypoint in observation.skeleton.keypoints:
            if keypoint.keypoint_type in by_type:
                raise ValueError(f"Duplicate keypoint type: {keypoint.keypoint_type.value}.")
            by_type[keypoint.keypoint_type] = keypoint

        unknown = set(by_type) - set(self.keypoint_order)
        if unknown:
            names = sorted(item.value for item in unknown)
            raise ValueError(f"Unexpected keypoint types: {names}.")

        coordinates = np.zeros((len(self.keypoint_order), 2), dtype=np.float32)
        confidence = np.zeros((len(self.keypoint_order),), dtype=np.float32)
        valid_mask = np.zeros((len(self.keypoint_order),), dtype=bool)

        for index, keypoint_type in enumerate(self.keypoint_order):
            keypoint = by_type.get(keypoint_type)
            if keypoint is None:
                continue

            point_confidence = float(keypoint.confidence)
            if not (
                math.isfinite(point_confidence)
                and math.isfinite(float(keypoint.x))
                and math.isfinite(float(keypoint.y))
            ):
                continue

            confidence[index] = point_confidence
            if keypoint.is_valid(self.confidence_threshold):
                coordinates[index] = (keypoint.x, keypoint.y)
                valid_mask[index] = True

        return PositionFeatures(
            coordinates=coordinates,
            confidence=confidence,
            valid_mask=valid_mask,
            keypoint_types=self.keypoint_order,
        )
