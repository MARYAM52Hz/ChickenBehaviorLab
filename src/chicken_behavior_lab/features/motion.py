"""Combine spatial, temporal, and geometric features into one frame representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chicken_behavior_lab.features.acceleration import AccelerationFeatureExtractor
from chicken_behavior_lab.features.angles import JointAngleFeatureExtractor
from chicken_behavior_lab.features.position import PositionFeatureExtractor
from chicken_behavior_lab.features.velocity import VelocityFeatureExtractor
from chicken_behavior_lab.models.tracked_skeleton import TrackedSkeleton


@dataclass(slots=True)
class MotionFeatureFrame:
    """Per-keypoint motion features for one frame."""

    features: np.ndarray
    angles: np.ndarray
    angle_valid_mask: np.ndarray
    feature_names: tuple[str, ...]
    keypoint_mask: np.ndarray | None = None

    @property
    def num_keypoints(self) -> int:
        return self.features.shape[0]

    @property
    def feature_dimension(self) -> int:
        return self.features.shape[1]

    @property
    def num_angles(self) -> int:
        return self.angles.shape[0]


class MotionFeatureExtractor:
    """Build canonical per-keypoint motion features from three observations."""

    FEATURE_NAMES = ("x", "y", "vx", "vy", "ax", "ay", "confidence")

    def __init__(
        self,
        position_extractor: PositionFeatureExtractor | None = None,
        velocity_extractor: VelocityFeatureExtractor | None = None,
        acceleration_extractor: AccelerationFeatureExtractor | None = None,
        angle_extractor: JointAngleFeatureExtractor | None = None,
    ) -> None:
        self.position_extractor = position_extractor or PositionFeatureExtractor()
        self.velocity_extractor = velocity_extractor or VelocityFeatureExtractor(
            keypoint_order=self.position_extractor.keypoint_order
        )
        self.acceleration_extractor = acceleration_extractor or AccelerationFeatureExtractor()
        self.angle_extractor = angle_extractor

        if self.velocity_extractor.keypoint_order != self.position_extractor.keypoint_order:
            raise ValueError("Position and velocity extractors must use the same keypoint order.")

    def extract(
        self,
        previous_previous: TrackedSkeleton,
        previous: TrackedSkeleton,
        current: TrackedSkeleton,
        delta_time: float,
    ) -> MotionFeatureFrame:
        self._validate_tracks(previous_previous, previous, current)
        position = self.position_extractor.extract(current)
        previous_velocity = self.velocity_extractor.extract(
            previous_previous, previous, delta_time
        )
        current_velocity = self.velocity_extractor.extract(previous, current, delta_time)

        acceleration = self.acceleration_extractor.extract(
            previous_velocity=previous_velocity.velocity,
            current_velocity=current_velocity.velocity,
            delta_time=delta_time,
            previous_valid_mask=previous_velocity.valid_mask,
            current_valid_mask=current_velocity.valid_mask,
        )

        feature_matrix = np.column_stack(
            [
                position.coordinates,
                current_velocity.velocity,
                acceleration.acceleration,
                position.confidence,
            ]
        ).astype(np.float32)

        if self.angle_extractor is not None:
            angle_features = self.angle_extractor.extract(current)
            angles = angle_features.angles
            angle_valid_mask = angle_features.valid_mask
        else:
            angles = np.empty((0,), dtype=np.float32)
            angle_valid_mask = np.empty((0,), dtype=bool)

        # A node is valid only when every component in its seven-value row is
        # supported by the observations used to calculate position/velocity/acceleration.
        row_valid_mask = (
            position.valid_mask
            & current_velocity.valid_mask
            & acceleration.valid_mask
        )

        return MotionFeatureFrame(
            features=feature_matrix,
            angles=angles,
            angle_valid_mask=angle_valid_mask,
            feature_names=self.FEATURE_NAMES,
            keypoint_mask=row_valid_mask,
        )

    @staticmethod
    def _validate_tracks(
        previous_previous: TrackedSkeleton,
        previous: TrackedSkeleton,
        current: TrackedSkeleton,
    ) -> None:
        track_ids = {previous_previous.track_id, previous.track_id, current.track_id}
        if len(track_ids) != 1:
            raise ValueError("All observations must belong to the same track.")
