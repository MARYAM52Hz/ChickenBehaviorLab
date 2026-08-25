"""
ChickenBehaviorLab Velocity Features
=====================================

Utilities for extracting temporal velocity features
from tracked chicken skeletons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chicken_behavior_lab.models.tracked_skeleton import (
    TrackedSkeleton,
)


@dataclass(slots=True)
class VelocityFeatures:
    """
    Velocity features for one temporal transition.

    Attributes
    ----------
    velocity:
        Velocity vectors for each keypoint.

        Shape:
            (num_keypoints, 2)

        The last dimension contains:

            [vx, vy]

    speed:
        Scalar speed for each keypoint.

        Shape:
            (num_keypoints,)

    valid_mask:
        Boolean mask indicating which keypoints
        have sufficient confidence for velocity
        estimation.

        Shape:
            (num_keypoints,)
    """

    velocity: np.ndarray

    speed: np.ndarray

    valid_mask: np.ndarray

    @property
    def num_keypoints(self) -> int:
        """
        Return the number of keypoints.
        """

        return self.velocity.shape[0]

    @property
    def feature_dimension(self) -> int:
        """
        Return the number of velocity components
        per keypoint.
        """

        return self.velocity.shape[1]


class VelocityFeatureExtractor:
    """
    Extract temporal velocity features from
    consecutive tracked skeletons.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
    ) -> None:
        """
        Parameters
        ----------
        confidence_threshold:
            Minimum confidence required for a keypoint
            to participate in velocity estimation.
        """

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must "
                "be between 0.0 and 1.0."
            )

        self.confidence_threshold = (
            confidence_threshold
        )

    # =====================================================
    # Main Extraction
    # =====================================================

    def extract(
        self,
        previous: TrackedSkeleton,
        current: TrackedSkeleton,
        delta_time: float,
    ) -> VelocityFeatures:
        """
        Calculate keypoint velocity between two
        consecutive observations.

        Parameters
        ----------
        previous:
            Previous skeleton observation.

        current:
            Current skeleton observation.

        delta_time:
            Time difference between observations
            in seconds.

        Returns
        -------
        VelocityFeatures
            Velocity and speed for every keypoint.

        Raises
        ------
        ValueError
            If Track IDs do not match, delta_time is
            invalid, or skeleton sizes differ.
        """

        self._validate_inputs(
            previous,
            current,
            delta_time,
        )

        previous_keypoints = (
            previous.skeleton.keypoints
        )

        current_keypoints = (
            current.skeleton.keypoints
        )

        num_keypoints = len(
            current_keypoints
        )

        velocity = np.zeros(
            (num_keypoints, 2),
            dtype=np.float32,
        )

        valid_mask = np.zeros(
            num_keypoints,
            dtype=bool,
        )

        for index, (
            previous_keypoint,
            current_keypoint,
        ) in enumerate(
            zip(
                previous_keypoints,
                current_keypoints,
            )
        ):

            previous_confidence = (
                previous_keypoint.confidence
            )

            current_confidence = (
                current_keypoint.confidence
            )

            if (
                previous_confidence
                < self.confidence_threshold
                or current_confidence
                < self.confidence_threshold
            ):
                continue

            dx = (
                current_keypoint.x
                - previous_keypoint.x
            )

            dy = (
                current_keypoint.y
                - previous_keypoint.y
            )

            velocity[index, 0] = (
                dx / delta_time
            )

            velocity[index, 1] = (
                dy / delta_time
            )

            valid_mask[index] = True

        speed = np.linalg.norm(
            velocity,
            axis=1,
        ).astype(
            np.float32
        )

        return VelocityFeatures(
            velocity=velocity,
            speed=speed,
            valid_mask=valid_mask,
        )

    # =====================================================
    # Validation
    # =====================================================

    def _validate_inputs(
        self,
        previous: TrackedSkeleton,
        current: TrackedSkeleton,
        delta_time: float,
    ) -> None:
        """
        Validate temporal velocity inputs.
        """

        if previous.track_id != current.track_id:
            raise ValueError(
                "Track ID mismatch: "
                f"previous={previous.track_id}, "
                f"current={current.track_id}"
            )

        if delta_time <= 0.0:
            raise ValueError(
                "delta_time must be greater than zero."
            )

        previous_count = len(
            previous.skeleton.keypoints
        )

        current_count = len(
            current.skeleton.keypoints
        )

        if previous_count != current_count:
            raise ValueError(
                "Skeleton keypoint count mismatch: "
                f"previous={previous_count}, "
                f"current={current_count}"
            )
