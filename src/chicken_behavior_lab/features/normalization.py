"""Normalize pose-based motion features while preserving missing-data masks.

Input representation is T × V × F with feature order
[x, y, vx, vy, ax, ay, confidence]. Statistics must be fitted on training data only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from chicken_behavior_lab.core.enums.keypoints import KeypointType
from chicken_behavior_lab.features.temporal_features import TemporalFeatureSequence


@dataclass(slots=True)
class NormalizationStatistics:
    """Training-set statistics for velocity and acceleration."""

    velocity_mean: np.ndarray
    velocity_std: np.ndarray
    acceleration_mean: np.ndarray
    acceleration_std: np.ndarray


class MotionFeatureNormalizer:
    """Body-center and body-scale normalize motion features in canonical node order."""

    FEATURE_X = 0
    FEATURE_Y = 1
    FEATURE_VX = 2
    FEATURE_VY = 3
    FEATURE_AX = 4
    FEATURE_AY = 5
    FEATURE_CONFIDENCE = 6

    def __init__(
        self,
        reference_keypoint: int | KeypointType = KeypointType.BODY_CENTER,
        scale_keypoints: tuple[int | KeypointType, int | KeypointType] = (
            KeypointType.LEFT_HIP, KeypointType.RIGHT_HIP
        ),
        epsilon: float = 1e-6,
        keypoint_order: tuple[KeypointType, ...] | None = None,
    ) -> None:
        if isinstance(epsilon, bool) or not isinstance(epsilon, Real) or not math.isfinite(float(epsilon)) or epsilon <= 0:
            raise ValueError("epsilon must be a finite positive number.")
        self.keypoint_order = tuple(KeypointType) if keypoint_order is None else tuple(keypoint_order)
        if not self.keypoint_order or any(not isinstance(item, KeypointType) for item in self.keypoint_order):
            raise ValueError("keypoint_order must contain KeypointType values.")
        if len(set(self.keypoint_order)) != len(self.keypoint_order):
            raise ValueError("keypoint_order must contain unique keypoint types.")
        self.reference_keypoint = self._resolve_index(reference_keypoint)
        if len(scale_keypoints) != 2:
            raise ValueError("scale_keypoints must contain exactly two keypoints.")
        self.scale_keypoints = tuple(self._resolve_index(item) for item in scale_keypoints)
        if len({self.reference_keypoint, *self.scale_keypoints}) < 2:
            raise ValueError("Reference and scale keypoints must not all be identical.")
        self.epsilon = float(epsilon)
        self.statistics: NormalizationStatistics | None = None

    def _resolve_index(self, keypoint: int | KeypointType) -> int:
        if isinstance(keypoint, KeypointType):
            if keypoint not in self.keypoint_order:
                raise ValueError(f"{keypoint.value} is not present in keypoint_order.")
            return self.keypoint_order.index(keypoint)
        if isinstance(keypoint, bool) or not isinstance(keypoint, Integral):
            raise TypeError("Keypoint references must be KeypointType values or integer indices.")
        index = int(keypoint)
        if index < 0 or index >= len(self.keypoint_order):
            raise ValueError("Keypoint index is outside keypoint_order.")
        return index

    @staticmethod
    def _validate_sequence(sequence: TemporalFeatureSequence) -> None:
        if sequence.features.ndim != 3 or sequence.features.shape[2] < 7:
            raise ValueError("features must have shape (T, V, F) with F >= 7.")
        expected_mask_shape = sequence.features.shape[:2]
        if sequence.frame_mask.shape != (sequence.features.shape[0],):
            raise ValueError("frame_mask shape does not match the temporal dimension.")
        if sequence.keypoint_mask.shape != expected_mask_shape:
            raise ValueError("keypoint_mask shape does not match (T, V).")
        if len(sequence.frame_ids) != sequence.features.shape[0]:
            raise ValueError("frame_ids length does not match the temporal dimension.")

    def fit(self, sequences: list[TemporalFeatureSequence]) -> None:
        """Learn statistics from training sequences only; masked/non-finite rows are excluded."""
        if not sequences:
            raise ValueError("sequences cannot be empty.")

        velocity_rows = []
        acceleration_rows = []
        for sequence in sequences:
            self._validate_sequence(sequence)
            if sequence.num_keypoints != len(self.keypoint_order):
                raise ValueError("Sequence keypoint dimension does not match keypoint_order.")
            valid = sequence.keypoint_mask & sequence.frame_mask[:, None]
            valid &= np.all(np.isfinite(sequence.features[:, :, :7]), axis=2)
            reference = self.reference_keypoint
            point_a, point_b = self.scale_keypoints
            for frame_index in range(sequence.temporal_length):
                if not sequence.frame_mask[frame_index]:
                    valid[frame_index] = False
                    continue
                required = (reference, point_a, point_b)
                if not all(valid[frame_index, index] for index in required):
                    valid[frame_index] = False
                    continue
                scale = float(np.linalg.norm(
                    sequence.features[frame_index, point_a, :2]
                    - sequence.features[frame_index, point_b, :2]
                ))
                if not math.isfinite(scale) or scale <= self.epsilon:
                    valid[frame_index] = False
            if valid.any():
                velocity_rows.append(sequence.features[:, :, self.FEATURE_VX:self.FEATURE_VY + 1][valid])
                acceleration_rows.append(sequence.features[:, :, self.FEATURE_AX:self.FEATURE_AY + 1][valid])

        if not velocity_rows:
            raise ValueError("No valid motion feature rows are available for fitting.")

        velocity = np.concatenate(velocity_rows, axis=0).astype(np.float64)
        acceleration = np.concatenate(acceleration_rows, axis=0).astype(np.float64)
        velocity_mean = velocity.mean(axis=0)
        velocity_std = np.maximum(velocity.std(axis=0), self.epsilon)
        acceleration_mean = acceleration.mean(axis=0)
        acceleration_std = np.maximum(acceleration.std(axis=0), self.epsilon)
        self.statistics = NormalizationStatistics(
            velocity_mean=velocity_mean.astype(np.float32),
            velocity_std=velocity_std.astype(np.float32),
            acceleration_mean=acceleration_mean.astype(np.float32),
            acceleration_std=acceleration_std.astype(np.float32),
        )

    def transform(self, sequence: TemporalFeatureSequence) -> TemporalFeatureSequence:
        """Normalize one sequence; frames lacking required body landmarks are masked."""
        if self.statistics is None:
            raise RuntimeError("Normalizer must be fitted before calling transform().")
        self._validate_sequence(sequence)
        if sequence.num_keypoints != len(self.keypoint_order):
            raise ValueError("Sequence keypoint dimension does not match keypoint_order.")

        features = sequence.features.astype(np.float32, copy=True)
        valid_mask = sequence.keypoint_mask.copy() & sequence.frame_mask[:, None]
        valid_mask &= np.all(np.isfinite(features[:, :, :7]), axis=2)

        ref = self.reference_keypoint
        point_a, point_b = self.scale_keypoints
        for frame_index in range(sequence.temporal_length):
            if not sequence.frame_mask[frame_index]:
                valid_mask[frame_index] = False
                continue
            required = (ref, point_a, point_b)
            if not all(valid_mask[frame_index, index] for index in required):
                valid_mask[frame_index] = False
                continue
            reference_xy = features[frame_index, ref, :2].copy()
            scale = float(np.linalg.norm(
                features[frame_index, point_a, :2] - features[frame_index, point_b, :2]
            ))
            if not math.isfinite(scale) or scale <= self.epsilon:
                valid_mask[frame_index] = False
                continue
            features[frame_index, :, :2] = (features[frame_index, :, :2] - reference_xy) / scale

        stats = self.statistics
        features[:, :, self.FEATURE_VX:self.FEATURE_VY + 1] = (
            features[:, :, self.FEATURE_VX:self.FEATURE_VY + 1] - stats.velocity_mean
        ) / stats.velocity_std
        features[:, :, self.FEATURE_AX:self.FEATURE_AY + 1] = (
            features[:, :, self.FEATURE_AX:self.FEATURE_AY + 1] - stats.acceleration_mean
        ) / stats.acceleration_std
        features[:, :, self.FEATURE_CONFIDENCE] = np.clip(
            features[:, :, self.FEATURE_CONFIDENCE], 0.0, 1.0
        )
        features[~valid_mask] = 0.0

        return TemporalFeatureSequence(
            features=features,
            frame_mask=sequence.frame_mask.copy(),
            keypoint_mask=valid_mask,
            frame_ids=sequence.frame_ids,
        )

    def is_fitted(self) -> bool:
        return self.statistics is not None
