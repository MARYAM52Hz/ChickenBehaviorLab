"""
ChickenBehaviorLab Joint Angle Features
=======================================

Generic joint-angle extraction from canonical chicken keypoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1].")
        self.definitions = tuple(definitions)
        self.confidence_threshold = confidence_threshold

    def extract(self, observation: TrackedSkeleton) -> JointAngleFeatures:
        keypoints = {
            keypoint.keypoint_type.value: keypoint
            for keypoint in observation.skeleton.keypoints
        }

        angle_values = np.zeros(len(self.definitions), dtype=np.float32)
        valid_mask = np.zeros(len(self.definitions), dtype=bool)
        names = tuple(definition.name for definition in self.definitions)

        for index, definition in enumerate(self.definitions):
            first = keypoints.get(definition.first_keypoint)
            vertex = keypoints.get(definition.vertex_keypoint)
            third = keypoints.get(definition.third_keypoint)

            if first is None or vertex is None or third is None:
                continue

            if not all(
                point.is_valid(self.confidence_threshold)
                for point in (first, vertex, third)
            ):
                continue

            angle = self._calculate_angle(
                np.array([first.x, first.y], dtype=np.float32),
                np.array([vertex.x, vertex.y], dtype=np.float32),
                np.array([third.x, third.y], dtype=np.float32),
            )
            if angle is None:
                continue

            angle_values[index] = angle
            valid_mask[index] = True

        return JointAngleFeatures(
            names=names,
            angles=angle_values,
            valid_mask=valid_mask,
        )

    @staticmethod
    def _calculate_angle(
        first_point: np.ndarray,
        vertex_point: np.ndarray,
        third_point: np.ndarray,
    ) -> float | None:
        vector_a = first_point - vertex_point
        vector_b = third_point - vertex_point
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return None

        cosine = np.dot(vector_a, vector_b) / (norm_a * norm_b)
        cosine = np.clip(cosine, -1.0, 1.0)
        return float(np.arccos(cosine))
