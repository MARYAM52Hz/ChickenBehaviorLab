"""
ChickenBehaviorLab Feature Normalization
========================================

Normalization utilities for pose-based motion features.

The normalization pipeline is designed to:

1. Reduce dependency on absolute image coordinates.
2. Normalize spatial features relative to the chicken body.
3. Normalize motion features using training-set statistics.
4. Preserve missing-data masks.
5. Prevent data leakage between training and validation/test sets.

Input representation:

    T × V × F

Expected feature order:

    [x, y, vx, vy, ax, ay, confidence]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chicken_behavior_lab.features.temporal_features import (
    TemporalFeatureSequence,
)


@dataclass(slots=True)
class NormalizationStatistics:
    """
    Statistics learned from the training dataset.

    Attributes
    ----------
    velocity_mean:
        Mean velocity for x/y dimensions.

    velocity_std:
        Standard deviation of velocity.

    acceleration_mean:
        Mean acceleration for x/y dimensions.

    acceleration_std:
        Standard deviation of acceleration.
    """

    velocity_mean: np.ndarray

    velocity_std: np.ndarray

    acceleration_mean: np.ndarray

    acceleration_std: np.ndarray


class MotionFeatureNormalizer:
    """
    Normalize temporal motion feature sequences.

    The normalizer separates geometric normalization
    from statistical normalization.

    Spatial coordinates are converted to a body-centered
    representation, while velocity and acceleration are
    standardized using statistics learned from training data.
    """

    FEATURE_X = 0
    FEATURE_Y = 1
    FEATURE_VX = 2
    FEATURE_VY = 3
    FEATURE_AX = 4
    FEATURE_AY = 5
    FEATURE_CONFIDENCE = 6

    def __init__(
        self,
        reference_keypoint: int = 0,
        scale_keypoints: tuple[int, int] = (0, 1),
        epsilon: float = 1e-6,
    ) -> None:
        """
        Parameters
        ----------
        reference_keypoint:
            Keypoint used as the body-centered reference.

        scale_keypoints:
            Two keypoints used to estimate body scale.

        epsilon:
            Small value used to avoid division by zero.
        """

        self.reference_keypoint = (
            reference_keypoint
        )

        self.scale_keypoints = (
            scale_keypoints
        )

        self.epsilon = epsilon

        self.statistics: (
            NormalizationStatistics | None
        ) = None

    # =====================================================
    # Fit
    # =====================================================

    def fit(
        self,
        sequences: list[
            TemporalFeatureSequence
        ],
    ) -> None:
        """
        Learn normalization statistics from training
        sequences only.

        This method must NEVER be fitted on validation
        or test data.
        """

        if not sequences:

            raise ValueError(
                "sequences cannot be empty."
            )

        velocities = []
        accelerations = []

        for sequence in sequences:

            features = sequence.features

            valid = (
                sequence.keypoint_mask
                & sequence.frame_mask[:, None]
            )

            velocity = features[
                :, :, self.FEATURE_VX:
                self.FEATURE_VY + 1
            ]

            acceleration = features[
                :, :, self.FEATURE_AX:
                self.FEATURE_AY + 1
            ]

            velocity_valid = valid[
                :, :, None
            ]

            acceleration_valid = valid[
                :, :, None
            ]

            velocity = np.where(
                velocity_valid,
                velocity,
                np.nan,
            )

            acceleration = np.where(
                acceleration_valid,
                acceleration,
                np.nan,
            )

            velocities.append(
                velocity.reshape(
                    -1,
                    2,
                )
            )

            accelerations.append(
                acceleration.reshape(
                    -1,
                    2,
                )
            )

        velocity_values = np.concatenate(
            velocities,
            axis=0,
        )

        acceleration_values = np.concatenate(
            accelerations,
            axis=0,
        )

        velocity_mean = np.nanmean(
            velocity_values,
            axis=0,
        )

        velocity_std = np.nanstd(
            velocity_values,
            axis=0,
        )

        acceleration_mean = np.nanmean(
            acceleration_values,
            axis=0,
        )

        acceleration_std = np.nanstd(
            acceleration_values,
            axis=0,
        )

        velocity_std = np.maximum(
            velocity_std,
            self.epsilon,
        )

        acceleration_std = np.maximum(
            acceleration_std,
            self.epsilon,
        )

        self.statistics = (
            NormalizationStatistics(
                velocity_mean=velocity_mean.astype(
                    np.float32
                ),
                velocity_std=velocity_std.astype(
                    np.float32
                ),
                acceleration_mean=(
                    acceleration_mean.astype(
                        np.float32
                    )
                ),
                acceleration_std=(
                    acceleration_std.astype(
                        np.float32
                    )
                ),
            )
        )

    # =====================================================
    # Transform
    # =====================================================

    def transform(
        self,
        sequence: TemporalFeatureSequence,
    ) -> TemporalFeatureSequence:
        """
        Normalize a temporal feature sequence.

        The normalizer must be fitted before calling this
        method.
        """

        if self.statistics is None:

            raise RuntimeError(
                "Normalizer must be fitted "
                "before calling transform()."
            )

        features = (
            sequence.features.copy()
        )

        valid_mask = (
            sequence.keypoint_mask
            & sequence.frame_mask[:, None]
        )

        # -------------------------------------------------
        # Spatial normalization
        # -------------------------------------------------

        features = (
            self._normalize_position(
                features,
                valid_mask,
            )
        )

        # -------------------------------------------------
        # Velocity normalization
        # -------------------------------------------------

        velocity = features[
            :, :, self.FEATURE_VX:
            self.FEATURE_VY + 1
        ]

        velocity = (
            velocity
            - self.statistics.velocity_mean
        )

        velocity = (
            velocity
            / self.statistics.velocity_std
        )

        features[
            :, :, self.FEATURE_VX:
            self.FEATURE_VY + 1
        ] = velocity

        # -------------------------------------------------
        # Acceleration normalization
        # -------------------------------------------------

        acceleration = features[
            :, :, self.FEATURE_AX:
            self.FEATURE_AY + 1
        ]

        acceleration = (
            acceleration
            - self.statistics.acceleration_mean
        )

        acceleration = (
            acceleration
            / self.statistics.acceleration_std
        )

        features[
            :, :, self.FEATURE_AX:
            self.FEATURE_AY + 1
        ] = acceleration

        # -------------------------------------------------
        # Confidence
        # -------------------------------------------------

        features[
            :, :, self.FEATURE_CONFIDENCE
        ] = np.clip(
            features[
                :, :, self.FEATURE_CONFIDENCE
            ],
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Restore padding
        # -------------------------------------------------

        features[
            ~valid_mask
        ] = 0.0

        return TemporalFeatureSequence(
            features=features.astype(
                np.float32
            ),
            frame_mask=sequence.frame_mask.copy(),
            keypoint_mask=sequence.keypoint_mask.copy(),
            frame_ids=sequence.frame_ids,
        )

    # =====================================================
    # Position normalization
    # =====================================================

    def _normalize_position(
        self,
        features: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Convert absolute image coordinates into
        body-centered and scale-normalized coordinates.
        """

        output = features.copy()

        reference = (
            self.reference_keypoint
        )

        keypoint_a, keypoint_b = (
            self.scale_keypoints
        )

        reference_xy = output[
            :,
            reference,
            self.FEATURE_X:
            self.FEATURE_Y + 1,
        ]

        point_a = output[
            :,
            keypoint_a,
            self.FEATURE_X:
            self.FEATURE_Y + 1,
        ]

        point_b = output[
            :,
            keypoint_b,
            self.FEATURE_X:
            self.FEATURE_Y + 1,
        ]

        scale = np.linalg.norm(
            point_a - point_b,
            axis=-1,
        )

        scale = np.maximum(
            scale,
            self.epsilon,
        )

        centered = (
            output[
                :,
                :,
                self.FEATURE_X:
                self.FEATURE_Y + 1,
            ]
            - reference_xy[:, None, :]
        )

        normalized = (
            centered
            / scale[:, None, None]
        )

        output[
            :,
            :,
            self.FEATURE_X:
            self.FEATURE_Y + 1,
        ] = normalized

        output[
            :,
            :,
            self.FEATURE_X:
            self.FEATURE_Y + 1,
        ] = np.where(
            valid_mask[:, :, None],
            output[
                :,
                :,
                self.FEATURE_X:
                self.FEATURE_Y + 1,
            ],
            0.0,
        )

        return output

    # =====================================================
    # Inverse transform
    # =====================================================

    def is_fitted(self) -> bool:
        """
        Return True when normalization statistics
        have been learned.
        """

        return self.statistics is not None
