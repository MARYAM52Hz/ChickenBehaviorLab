"""
ChickenBehaviorLab Temporal Feature Sequences
==============================================

Utilities for converting per-frame motion features
into fixed-length temporal sequences.

The main output representation is:

    T × V × F

where:

    T = number of temporal frames
    V = number of skeleton keypoints
    F = number of per-keypoint features
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chicken_behavior_lab.features.motion import (
    MotionFeatureFrame,
)


@dataclass(slots=True)
class TemporalFeatureSequence:
    """
    Fixed-length temporal feature sequence.

    Attributes
    ----------
    features:
        Feature tensor with shape:

            (T, V, F)

    frame_mask:
        Boolean mask indicating which temporal frames
        contain valid observations.

        Shape:

            (T,)

    keypoint_mask:
        Boolean mask indicating valid keypoints.

        Shape:

            (T, V)

    frame_ids:
        Ordered frame identifiers.

    """

    features: np.ndarray

    frame_mask: np.ndarray

    keypoint_mask: np.ndarray

    frame_ids: tuple[str, ...]

    @property
    def temporal_length(self) -> int:
        """
        Number of frames in the sequence.
        """

        return self.features.shape[0]

    @property
    def num_keypoints(self) -> int:
        """
        Number of keypoints.
        """

        return self.features.shape[1]

    @property
    def feature_dimension(self) -> int:
        """
        Number of features per keypoint.
        """

        return self.features.shape[2]

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Return the tensor shape.
        """

        return self.features.shape


class TemporalFeatureBuilder:
    """
    Build fixed-length temporal feature sequences.
    """

    def __init__(
        self,
        sequence_length: int = 60,
    ) -> None:
        """
        Parameters
        ----------
        sequence_length:
            Number of frames in each temporal window.
        """

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be "
                "greater than zero."
            )

        self.sequence_length = (
            sequence_length
        )

    # =====================================================
    # Main builder
    # =====================================================

    def build(
        self,
        frames: list[
            MotionFeatureFrame
        ],
        frame_ids: list[str] | None = None,
    ) -> TemporalFeatureSequence:
        """
        Convert motion feature frames into a fixed-length
        temporal sequence.

        Parameters
        ----------
        frames:
            Ordered list of MotionFeatureFrame objects.

        frame_ids:
            Optional frame identifiers.

        Returns
        -------
        TemporalFeatureSequence
            Fixed-length tensor of shape T × V × F.
        """

        if not frames:
            raise ValueError(
                "frames cannot be empty."
            )

        self._validate_frames(
            frames
        )

        num_keypoints = (
            frames[0].num_keypoints
        )

        feature_dimension = (
            frames[0].feature_dimension
        )

        if frame_ids is None:

            frame_ids = [
                str(index)
                for index in range(
                    len(frames)
                )
            ]

        if len(frame_ids) != len(frames):

            raise ValueError(
                "frame_ids must have the same "
                "length as frames."
            )

        # -------------------------------------------------
        # Keep only the most recent frames
        # -------------------------------------------------

        selected_frames = frames[
            -self.sequence_length :
        ]

        selected_ids = frame_ids[
            -self.sequence_length :
        ]

        actual_length = len(
            selected_frames
        )

        # -------------------------------------------------
        # Allocate padded tensor
        # -------------------------------------------------

        features = np.zeros(
            (
                self.sequence_length,
                num_keypoints,
                feature_dimension,
            ),
            dtype=np.float32,
        )

        frame_mask = np.zeros(
            self.sequence_length,
            dtype=bool,
        )

        keypoint_mask = np.zeros(
            (
                self.sequence_length,
                num_keypoints,
            ),
            dtype=bool,
        )

        # -------------------------------------------------
        # Right-align the temporal window
        # -------------------------------------------------

        start_index = (
            self.sequence_length
            - actual_length
        )

        for index, frame in enumerate(
            selected_frames
        ):

            target_index = (
                start_index + index
            )

            features[target_index] = (
                frame.features
            )

            frame_mask[
                target_index
            ] = True

            # A keypoint is considered valid when
            # its feature row contains finite values.
            keypoint_mask[
                target_index
            ] = np.all(
                np.isfinite(
                    frame.features
                ),
                axis=1,
            )

        padded_ids = [
            ""
        ] * start_index

        padded_ids.extend(
            selected_ids
        )

        return TemporalFeatureSequence(
            features=features,
            frame_mask=frame_mask,
            keypoint_mask=keypoint_mask,
            frame_ids=tuple(
                padded_ids
            ),
        )

    # =====================================================
    # Validation
    # =====================================================

    @staticmethod
    def _validate_frames(
        frames: list[
            MotionFeatureFrame
        ],
    ) -> None:
        """
        Ensure that all frames have compatible shapes.
        """

        first = frames[0]

        expected_keypoints = (
            first.num_keypoints
        )

        expected_features = (
            first.feature_dimension
        )

        for index, frame in enumerate(
            frames[1:],
            start=1,
        ):

            if (
                frame.num_keypoints
                != expected_keypoints
            ):

                raise ValueError(
                    "Inconsistent number of "
                    f"keypoints at frame {index}: "
                    f"expected "
                    f"{expected_keypoints}, "
                    f"got "
                    f"{frame.num_keypoints}."
                )

            if (
                frame.feature_dimension
                != expected_features
            ):

                raise ValueError(
                    "Inconsistent feature "
                    f"dimension at frame {index}: "
                    f"expected "
                    f"{expected_features}, "
                    f"got "
                    f"{frame.feature_dimension}."
                )
