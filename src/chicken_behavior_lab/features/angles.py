"""
ChickenBehaviorLab Joint Angle Features
=======================================

Extract configured 2D joint angles from canonical chicken keypoints.
Angles are returned in radians.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from chicken_behavior_lab.core.enums.keypoints import KeypointType
from chicken_behavior_lab.models.tracked_skeleton import TrackedSkeleton


@dataclass(frozen=True, slots=True)
class JointAngleDefinition:
    """Definition of an angle measured at the middle keypoint."""

    name: str
    first_keypoint: str
    vertex_keypoint: str
    third_keypoint: str


@dataclass(slots=True)
class JointAngleFeatures:
    """Angle values in radians and a corresponding validity mask."""

    names: tuple[str, ...]
    angles: np.ndarray
    valid_mask: np.ndarray

    @property
    def num_angles(self) -> int:
        return len(self.names)

    @property
    def feature_dimension(self) -> int:
        return self.angles.shape[0]


class JointAngleFeatureExtractor:
    """Extract configured joint angles from a TrackedSkeleton."""

    def __init__(
        self,
        definitions: list[JointAngleDefinition],
        confidence_threshold: float = 0.3,
    ) -> None:
        if (
            isinstance(confidence_threshold, bool)
            or not isinstance(confidence_threshold, Real)
            or not math.isfinite(float(confidence_threshold))
            or not 0.0 <= float(confidence_threshold) <= 1.0
        ):
            raise ValueError("confidence_threshold must be a finite number in [0, 1].")

        self.definitions = tuple(definitions)
        if any(not isinstance(item, JointAngleDefinition) for item in self.definitions):
            raise TypeError("definitions must contain JointAngleDefinition instances.")

        names = [item.name for item in self.definitions]
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("Each angle definition must have a non-empty name.")
        if len(set(names)) != len(names):
            raise ValueError("Angle definition names must be unique.")

        valid_keypoint_names = {item.value for item in KeypointType}
        for definition in self.definitions:
            referenced = (
                definition.first_keypoint,
                definition.vertex_keypoint,
                definition.third_keypoint,
            )
            if any(name not in valid_keypoint_names for name in referenced):
                raise ValueError(
                    f"Angle definition {definition.name!r} references an unknown "
                    "canonical keypoint."
                )

        self.confidence_threshold = float(confidence_threshold)

    def extract(self, observation: TrackedSkeleton) -> JointAngleFeatures:
        keypoints = {}
        for keypoint in observation.skeleton.keypoints:
            name = keypoint.keypoint_type.value
            if name in keypoints:
                raise ValueError(f"Duplicate keypoint type in skeleton: {name}.")
            keypoints[name] = keypoint

        angles = np.zeros(len(self.definitions), dtype=np.float32)
        valid_mask = np.zeros(len(self.definitions), dtype=bool)
        names = tuple(definition.name for definition in self.definitions)

        for index, definition in enumerate(self.definitions):
            points = (
                keypoints.get(definition.first_keypoint),
                keypoints.get(definition.vertex_keypoint),
                keypoints.get(definition.third_keypoint),
            )
            if any(point is None for point in points):
                continue

            if not all(
                point.is_valid(self.confidence_threshold)
                for point in points
            ):
                continue

            coordinates = [
                np.asarray([point.x, point.y], dtype=np.float64)
                for point in points
            ]
            angle = self._calculate_angle(*coordinates)
            if angle is None:
                continue

            angles[index] = angle
            valid_mask[index] = True

        return JointAngleFeatures(
            names=names,
            angles=angles,
            valid_mask=valid_mask,
        )

    @staticmethod
    def _calculate_angle(
        first_point: np.ndarray,
        vertex_point: np.ndarray,
        third_point: np.ndarray,
    ) -> float | None:
        """Return the unsigned angle at vertex_point, or None if undefined."""
        if not all(
            np.all(np.isfinite(point))
            for point in (first_point, vertex_point, third_point)
        ):
            return None

        vector_a = first_point - vertex_point
        vector_b = third_point - vertex_point
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        if (
            not np.isfinite(norm_a)
            or not np.isfinite(norm_b)
            or norm_a < 1e-8
            or norm_b < 1e-8
        ):
            return None

        cosine = np.dot(vector_a, vector_b) / (norm_a * norm_b)
        cosine = np.clip(cosine, -1.0, 1.0)
        return float(np.arccos(cosine))
