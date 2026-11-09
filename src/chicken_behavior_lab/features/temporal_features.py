"""Build fixed-length T × V × F sequences with explicit validity masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chicken_behavior_lab.features.motion import MotionFeatureFrame


@dataclass(slots=True)
class TemporalFeatureSequence:
    """Fixed-length temporal feature sequence."""

    features: np.ndarray
    frame_mask: np.ndarray
    keypoint_mask: np.ndarray
    frame_ids: tuple[str, ...]

    @property
    def temporal_length(self) -> int:
        return self.features.shape[0]

    @property
    def num_keypoints(self) -> int:
        return self.features.shape[1]

    @property
    def feature_dimension(self) -> int:
        return self.features.shape[2]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.features.shape


class TemporalFeatureBuilder:
    """Build right-aligned, fixed-length temporal feature sequences."""

    def __init__(self, sequence_length: int = 60) -> None:
        if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer.")
        self.sequence_length = sequence_length

    def build(
        self,
        frames: list[MotionFeatureFrame],
        frame_ids: list[str] | None = None,
    ) -> TemporalFeatureSequence:
        if not frames:
            raise ValueError("frames cannot be empty.")
        self._validate_frames(frames)

        num_keypoints = frames[0].num_keypoints
        feature_dimension = frames[0].feature_dimension
        if frame_ids is None:
            frame_ids = [str(index) for index in range(len(frames))]
        if len(frame_ids) != len(frames):
            raise ValueError("frame_ids must have the same length as frames.")

        selected_frames = frames[-self.sequence_length:]
        selected_ids = frame_ids[-self.sequence_length:]
        actual_length = len(selected_frames)
        start_index = self.sequence_length - actual_length

        features = np.zeros(
            (self.sequence_length, num_keypoints, feature_dimension), dtype=np.float32
        )
        frame_mask = np.zeros((self.sequence_length,), dtype=bool)
        keypoint_mask = np.zeros((self.sequence_length, num_keypoints), dtype=bool)

        for index, frame in enumerate(selected_frames):
            target_index = start_index + index
            features[target_index] = frame.features
            frame_mask[target_index] = True

            if frame.keypoint_mask is None:
                mask = np.all(np.isfinite(frame.features), axis=1)
            else:
                mask = np.asarray(frame.keypoint_mask, dtype=bool)
                if mask.shape != (num_keypoints,):
                    raise ValueError("frame.keypoint_mask must have shape (num_keypoints,).")
                mask = mask & np.all(np.isfinite(frame.features), axis=1)
            keypoint_mask[target_index] = mask

        padded_ids = [""] * start_index + list(selected_ids)
        return TemporalFeatureSequence(
            features=features,
            frame_mask=frame_mask,
            keypoint_mask=keypoint_mask,
            frame_ids=tuple(str(item) for item in padded_ids),
        )

    @staticmethod
    def _validate_frames(frames: list[MotionFeatureFrame]) -> None:
        first = frames[0]
        if first.features.ndim != 2:
            raise ValueError("Each frame.features array must have shape (num_keypoints, feature_dimension).")
        expected_keypoints = first.num_keypoints
        expected_features = first.feature_dimension

        for index, frame in enumerate(frames):
            if frame.features.ndim != 2:
                raise ValueError(f"frame {index}.features must be a two-dimensional array.")
            if frame.num_keypoints != expected_keypoints:
                raise ValueError(
                    f"Inconsistent number of keypoints at frame {index}: "
                    f"expected {expected_keypoints}, got {frame.num_keypoints}."
                )
            if frame.feature_dimension != expected_features:
                raise ValueError(
                    f"Inconsistent feature dimension at frame {index}: "
                    f"expected {expected_features}, got {frame.feature_dimension}."
                )
            if frame.keypoint_mask is not None and np.asarray(frame.keypoint_mask).shape != (expected_keypoints,):
                raise ValueError(f"frame {index}.keypoint_mask must have shape (num_keypoints,).")
