"""
ChickenBehaviorLab Acceleration Features
=========================================

Utilities for extracting temporal acceleration
features from tracked chicken skeletons.

Acceleration is computed from two consecutive
velocity observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AccelerationFeatures:
    """
    Acceleration features for one temporal transition.

    Attributes
    ----------
    acceleration:
        Acceleration vectors for each keypoint.

        Shape:
            (num_keypoints, 2)

        Last dimension:

            [ax, ay]

    magnitude:
        Scalar acceleration magnitude for each keypoint.

        Shape:
            (num_keypoints,)

    valid_mask:
        Boolean mask indicating whether the acceleration
        estimate is considered valid.
    """

    acceleration: np.ndarray

    magnitude: np.ndarray

    valid_mask: np.ndarray

    @property
    def num_keypoints(self) -> int:
        """
        Return the number of keypoints.
        """

        return self.acceleration.shape[0]

    @property
    def feature_dimension(self) -> int:
        """
        Return the number of acceleration components
        per keypoint.
        """

        return self.acceleration.shape[1]


class AccelerationFeatureExtractor:
    """
    Extract acceleration features from consecutive
    velocity observations.
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
            to contribute to acceleration estimation.
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
        previous_velocity: np.ndarray,
        current_velocity: np.ndarray,
        delta_time: float,
        previous_valid_mask: np.ndarray | None = None,
        current_valid_mask: np.ndarray | None = None,
    ) -> AccelerationFeatures:
        """
        Calculate acceleration between two consecutive
        velocity observations.

        Parameters
        ----------
        previous_velocity:
            Velocity from the previous temporal step.

            Shape:
                (num_keypoints, 2)

        current_velocity:
            Velocity from the current temporal step.

            Shape:
                (num_keypoints, 2)

        delta_time:
            Time difference between the two velocity
            observations, in seconds.

        previous_valid_mask:
            Optional validity mask for the previous
            velocity observation.

        current_valid_mask:
            Optional validity mask for the current
            velocity observation.

        Returns
        -------
        AccelerationFeatures
            Acceleration vectors, magnitudes and
            validity mask.

        Raises
        ------
        ValueError
            If inputs have incompatible shapes or
            delta_time is invalid.
        """

        self._validate_inputs(
            previous_velocity,
            current_velocity,
            delta_time,
            previous_valid_mask,
            current_valid_mask,
        )

        num_keypoints = (
            current_velocity.shape[0]
        )

        acceleration = np.zeros(
            (num_keypoints, 2),
            dtype=np.float32,
        )

        valid_mask = np.ones(
            num_keypoints,
            dtype=bool,
        )

        # -------------------------------------------------
        # Apply velocity validity masks
        # -------------------------------------------------

        if previous_valid_mask is not None:

            valid_mask &= (
                previous_valid_mask
            )

        if current_valid_mask is not None:

            valid_mask &= (
                current_valid_mask
            )

        # -------------------------------------------------
        # Calculate acceleration
        # -------------------------------------------------

        acceleration = (
            current_velocity
            - previous_velocity
        ) / delta_time

        acceleration = acceleration.astype(
            np.float32
        )

        # -------------------------------------------------
        # Invalid keypoints
        # -------------------------------------------------

        acceleration[
            ~valid_mask
        ] = 0.0

        # -------------------------------------------------
        # Acceleration magnitude
        # -------------------------------------------------

        magnitude = np.linalg.norm(
            acceleration,
            axis=1,
        ).astype(
            np.float32
        )

        return AccelerationFeatures(
            acceleration=acceleration,
            magnitude=magnitude,
            valid_mask=valid_mask,
        )

    # =====================================================
    # Validation
    # =====================================================

    def _validate_inputs(
        self,
        previous_velocity: np.ndarray,
        current_velocity: np.ndarray,
        delta_time: float,
        previous_valid_mask: np.ndarray | None,
        current_valid_mask: np.ndarray | None,
    ) -> None:
        """
        Validate acceleration inputs.
        """

        if delta_time <= 0.0:
            raise ValueError(
                "delta_time must be greater than zero."
            )

        if previous_velocity.ndim != 2:
            raise ValueError(
                "previous_velocity must have shape "
                "(num_keypoints, 2)."
            )

        if current_velocity.ndim != 2:
            raise ValueError(
                "current_velocity must have shape "
                "(num_keypoints, 2)."
            )

        if previous_velocity.shape != (
            current_velocity.shape
        ):
            raise ValueError(
                "Velocity shape mismatch: "
                f"previous="
                f"{previous_velocity.shape}, "
                f"current="
                f"{current_velocity.shape}"
            )

        if previous_velocity.shape[1] != 2:
            raise ValueError(
                "Velocity arrays must contain "
                "exactly two components: vx and vy."
            )

        num_keypoints = (
            current_velocity.shape[0]
        )

        if previous_valid_mask is not None:

            if previous_valid_mask.shape != (
                num_keypoints,
            ):
                raise ValueError(
                    "previous_valid_mask must have "
                    "shape (num_keypoints,)."
                )

        if current_valid_mask is not None:

            if current_valid_mask.shape != (
                num_keypoints,
            ):
                raise ValueError(
                    "current_valid_mask must have "
                    "shape (num_keypoints,)."
                )
