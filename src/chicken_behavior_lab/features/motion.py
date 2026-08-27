"""
ChickenBehaviorLab Motion Features
===================================

Combines spatial, temporal, and geometric features
into a unified motion representation.

The resulting feature tensor is designed to be used
by downstream behavior-recognition and graph-based
models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chicken_behavior_lab.features.acceleration import (
    AccelerationFeatureExtractor,
)

from chicken_behavior_lab.features.angles import (
    JointAngleFeatureExtractor,
)

from chicken_behavior_lab.features.position import (
    PositionFeatureExtractor,
)

from chicken_behavior_lab.features.velocity import (
    VelocityFeatureExtractor,
)

from chicken_behavior_lab.models.tracked_skeleton import (
    TrackedSkeleton,
)


@dataclass(slots=True)
class MotionFeatureFrame:
    """
    Unified motion representation for one frame.

    Attributes
    ----------
    features:
        Per-keypoint feature matrix.

        Shape:

            (num_keypoints, feature_dimension)

        Feature order:

            [x, y,
             vx, vy,
             ax, ay,
             confidence]

    angles:
        Joint-angle features.

        Shape:

            (num_angles,)

    angle_valid_mask:
        Validity mask for angle features.

    feature_names:
        Names of per-keypoint features.
    """

    features: np.ndarray

    angles: np.ndarray

    angle_valid_mask: np.ndarray

    feature_names: tuple[str, ...]

    @property
    def num_keypoints(self) -> int:
        """
        Return the number of keypoints.
        """

        return self.features.shape[0]

    @property
    def feature_dimension(self) -> int:
        """
        Return the number of per-keypoint features.
        """

        return self.features.shape[1]

    @property
    def num_angles(self) -> int:
        """
        Return the number of angle features.
        """

        return self.angles.shape[0]


class MotionFeatureExtractor:
    """
    Build a unified motion representation from
    consecutive tracked skeleton observations.
    """

    FEATURE_NAMES = (
        "x",
        "y",
        "vx",
        "vy",
        "ax",
        "ay",
        "confidence",
    )

    def __init__(
        self,
        position_extractor: PositionFeatureExtractor | None = None,
        velocity_extractor: VelocityFeatureExtractor | None = None,
        acceleration_extractor: AccelerationFeatureExtractor | None = None,
        angle_extractor: JointAngleFeatureExtractor | None = None,
    ) -> None:
        """
        Initialize the motion feature extractor.

        Individual extractors can be injected for testing,
        configuration, or future customization.
        """

        self.position_extractor = (
            position_extractor
            or PositionFeatureExtractor()
        )

        self.velocity_extractor = (
            velocity_extractor
            or VelocityFeatureExtractor()
        )

        self.acceleration_extractor = (
            acceleration_extractor
            or AccelerationFeatureExtractor()
        )

        self.angle_extractor = (
            angle_extractor
        )

    # =====================================================
    # Main extraction
    # =====================================================

    def extract(
        self,
        previous_previous: TrackedSkeleton,
        previous: TrackedSkeleton,
        current: TrackedSkeleton,
        delta_time: float,
    ) -> MotionFeatureFrame:
        """
        Extract unified motion features.

        Three consecutive observations are required:

            t-2
            t-1
             t

        because acceleration requires two velocity
        observations.

        Parameters
        ----------
        previous_previous:
            Observation at t-2.

        previous:
            Observation at t-1.

        current:
            Observation at t.

        delta_time:
            Time interval between consecutive observations
            in seconds.

        Returns
        -------
        MotionFeatureFrame
            Unified per-keypoint motion representation.
        """

        self._validate_tracks(
            previous_previous,
            previous,
            current,
        )

        # -------------------------------------------------
        # Position
        # -------------------------------------------------

        position = (
            self.position_extractor.extract(
                current
            )
        )

        # -------------------------------------------------
        # Velocity
        # -------------------------------------------------

        previous_velocity = (
            self.velocity_extractor.extract(
                previous_previous,
                previous,
                delta_time,
            )
        )

        current_velocity = (
            self.velocity_extractor.extract(
                previous,
                current,
                delta_time,
            )
        )

        # -------------------------------------------------
        # Acceleration
        # -------------------------------------------------

        acceleration = (
            self.acceleration_extractor.extract(
                previous_velocity=(
                    previous_velocity.velocity
                ),
                current_velocity=(
                    current_velocity.velocity
                ),
                delta_time=delta_time,
                previous_valid_mask=(
                    previous_velocity.valid_mask
                ),
                current_valid_mask=(
                    current_velocity.valid_mask
                ),
            )
        )

        # -------------------------------------------------
        # Per-keypoint feature matrix
        # -------------------------------------------------

        feature_matrix = np.column_stack(
            [
                position.coordinates,
                current_velocity.velocity,
                acceleration.acceleration,
                position.confidence,
            ]
        ).astype(
            np.float32
        )

        # -------------------------------------------------
        # Joint angles
        # -------------------------------------------------

        if self.angle_extractor is not None:

            angle_features = (
                self.angle_extractor.extract(
                    current
                )
            )

            angles = (
                angle_features.angles
            )

            angle_valid_mask = (
                angle_features.valid_mask
            )

        else:

            angles = np.empty(
                (0,),
                dtype=np.float32,
            )

            angle_valid_mask = np.empty(
                (0,),
                dtype=bool,
            )

        return MotionFeatureFrame(
            features=feature_matrix,
            angles=angles,
            angle_valid_mask=angle_valid_mask,
            feature_names=self.FEATURE_NAMES,
        )

    # =====================================================
    # Validation
    # =====================================================

    @staticmethod
    def _validate_tracks(
        previous_previous: TrackedSkeleton,
        previous: TrackedSkeleton,
        current: TrackedSkeleton,
    ) -> None:
        """
        Ensure that all observations belong to the
        same tracked chicken.
        """

        track_ids = {
            previous_previous.track_id,
            previous.track_id,
            current.track_id,
        }

        if len(track_ids) != 1:

            raise ValueError(
                "All observations must belong to "
                "the same track."
            )
