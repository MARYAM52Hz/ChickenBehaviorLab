"""
ChickenBehaviorLab Joint Angle Features
=======================================

Generic utilities for extracting joint-angle features
from chicken skeletons.

The angle extractor is intentionally independent from
the chicken skeleton topology. Joint definitions such as
neck angle, body angle, or leg angle should be provided
by the skeleton configuration rather than hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chicken_behavior_lab.models.tracked_skeleton import (
    TrackedSkeleton,
)


@dataclass(frozen=True, slots=True)
class JointAngleDefinition:
    """
    Definition of one joint angle.

    The angle is measured at the middle keypoint.

    Example
    -------
    For:

        A ---- B ---- C

    the calculated angle is:

        angle(A, B, C)

    Attributes
    ----------
    name:
        Unique semantic name of the angle.

    first_keypoint:
        Name of the first keypoint.

    vertex_keypoint:
        Name of the middle keypoint where the angle
        is measured.

    third_keypoint:
        Name of the third keypoint.
    """

    name: str

    first_keypoint: str

    vertex_keypoint: str

    third_keypoint: str


@dataclass(slots=True)
class JointAngleFeatures:
    """
    Joint-angle features for one skeleton observation.

    Attributes
    ----------
    names:
        Ordered names of the calculated angles.

    angles:
        Angle values in radians.

        Shape:

            (num_angles,)

    valid_mask:
        Indicates whether each angle is valid.

        Shape:

            (num_angles,)
    """

    names: tuple[str, ...]

    angles: np.ndarray

    valid_mask: np.ndarray

    @property
    def num_angles(self) -> int:
        """
        Return the number of defined angles.
        """

        return len(self.names)

    @property
    def feature_dimension(self) -> int:
        """
        Return the number of angle features.
        """

        return self.angles.shape[0]


class JointAngleFeatureExtractor:
    """
    Extract joint angles from a TrackedSkeleton.

    Angles are returned in radians.
    """

    def __init__(
        self,
        definitions: list[JointAngleDefinition],
        confidence_threshold: float = 0.3,
    ) -> None:
        """
        Parameters
        ----------
        definitions:
            List of joint-angle definitions.

        confidence_threshold:
            Minimum keypoint confidence required
            for a valid angle.
        """

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must "
                "be between 0.0 and 1.0."
            )

        self.definitions = tuple(
            definitions
        )

        self.confidence_threshold = (
            confidence_threshold
        )

    # =====================================================
    # Main extraction
    # =====================================================

    def extract(
        self,
        observation: TrackedSkeleton,
    ) -> JointAngleFeatures:
        """
        Extract all configured joint angles.

        Parameters
        ----------
        observation:
            Tracked skeleton observation.

        Returns
        -------
        JointAngleFeatures
            Calculated joint angles and validity masks.
        """

        keypoints = {
            keypoint.name: keypoint
            for keypoint
            in observation.skeleton.keypoints
        }

        angle_values = np.zeros(
            len(self.definitions),
            dtype=np.float32,
        )

        valid_mask = np.zeros(
            len(self.definitions),
            dtype=bool,
        )

        names = tuple(
            definition.name
            for definition
            in self.definitions
        )

        for index, definition in enumerate(
            self.definitions
        ):

            first = keypoints.get(
                definition.first_keypoint
            )

            vertex = keypoints.get(
                definition.vertex_keypoint
            )

            third = keypoints.get(
                definition.third_keypoint
            )

            if (
                first is None
                or vertex is None
                or third is None
            ):
                continue

            if (
                first.confidence
                < self.confidence_threshold
                or vertex.confidence
                < self.confidence_threshold
                or third.confidence
                < self.confidence_threshold
            ):
                continue

            first_point = np.array(
                [
                    first.x,
                    first.y,
                ],
                dtype=np.float32,
            )

            vertex_point = np.array(
                [
                    vertex.x,
                    vertex.y,
                ],
                dtype=np.float32,
            )

            third_point = np.array(
                [
                    third.x,
                    third.y,
                ],
                dtype=np.float32,
            )

            angle = self._calculate_angle(
                first_point,
                vertex_point,
                third_point,
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

    # =====================================================
    # Angle calculation
    # =====================================================

    @staticmethod
    def _calculate_angle(
        first_point: np.ndarray,
        vertex_point: np.ndarray,
        third_point: np.ndarray,
    ) -> float | None:
        """
        Calculate the angle formed by three points.

        The angle is measured at ``vertex_point``.

        Returns
        -------
        float | None
            Angle in radians, or None when the angle
            cannot be reliably calculated.
        """

        vector_a = (
            first_point - vertex_point
        )

        vector_b = (
            third_point - vertex_point
        )

        norm_a = np.linalg.norm(
            vector_a
        )

        norm_b = np.linalg.norm(
            vector_b
        )

        if norm_a < 1e-8 or norm_b < 1e-8:
            return None

        cosine = np.dot(
            vector_a,
            vector_b,
        ) / (
            norm_a * norm_b
        )

        cosine = np.clip(
            cosine,
            -1.0,
            1.0,
        )

        return float(
            np.arccos(cosine)
        )
