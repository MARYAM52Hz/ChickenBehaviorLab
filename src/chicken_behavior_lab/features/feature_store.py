"""Persist temporal skeleton features as compressed NumPy archives."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from chicken_behavior_lab.features.temporal_features import TemporalFeatureSequence
from chicken_behavior_lab.core.identifiers import normalize_track_id


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class FeatureStore:
    """Store one temporal feature sequence per video and tracked individual."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_identifier(value: str, name: str) -> str:
        if not isinstance(value, str) or not _SAFE_ID.fullmatch(value) or value in {".", ".."}:
            raise ValueError(f"{name} must be a safe non-empty identifier.")
        return value

    @staticmethod
    def _normalize_track_id(track_id: str | int) -> str:
        return normalize_track_id(track_id)

    def _get_path(self, video_id: str, track_id: str | int) -> Path:
        safe_video_id = self._validate_identifier(video_id, "video_id")
        safe_track_id = self._normalize_track_id(track_id)
        return self.root / safe_video_id / f"{safe_track_id}.npz"

    @staticmethod
    def _validate_sequence(sequence: TemporalFeatureSequence) -> None:
        if not isinstance(sequence, TemporalFeatureSequence):
            raise TypeError("sequence must be a TemporalFeatureSequence.")
        if sequence.features.ndim != 3:
            raise ValueError("sequence.features must have shape (T, V, F).")
        temporal_length, num_keypoints, _ = sequence.features.shape
        if sequence.frame_mask.shape != (temporal_length,):
            raise ValueError("frame_mask must have shape (T,).")
        if sequence.keypoint_mask.shape != (temporal_length, num_keypoints):
            raise ValueError("keypoint_mask must have shape (T, V).")
        if len(sequence.frame_ids) != temporal_length:
            raise ValueError("frame_ids length must match the temporal dimension.")

    def save(
        self,
        video_id: str,
        track_id: str | int,
        sequence: TemporalFeatureSequence,
    ) -> Path:
        self._validate_sequence(sequence)
        path = self._get_path(video_id, track_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=sequence.features,
            frame_mask=sequence.frame_mask,
            keypoint_mask=sequence.keypoint_mask,
            frame_ids=np.asarray(sequence.frame_ids, dtype=str),
        )
        return path

    def load(
        self,
        video_id: str,
        track_id: str | int,
        start_frame: str | int | None = None,
        end_frame: str | int | None = None,
    ) -> TemporalFeatureSequence:
        path = self._get_path(video_id, track_id)
        if not path.exists():
            raise FileNotFoundError(f"Feature sequence not found: {path}")

        with np.load(path, allow_pickle=False) as data:
            sequence = TemporalFeatureSequence(
                features=data["features"],
                frame_mask=data["frame_mask"],
                keypoint_mask=data["keypoint_mask"],
                frame_ids=tuple(str(item) for item in data["frame_ids"]),
            )
        self._validate_sequence(sequence)

        if start_frame is None and end_frame is None:
            return sequence
        return self._slice_sequence(sequence, start_frame, end_frame)

    @staticmethod
    def _slice_sequence(
        sequence: TemporalFeatureSequence,
        start_frame: str | int | None,
        end_frame: str | int | None,
    ) -> TemporalFeatureSequence:
        if start_frame is None or end_frame is None:
            raise ValueError("Both start_frame and end_frame must be provided for slicing.")

        frame_ids = list(sequence.frame_ids)
        try:
            first_index = frame_ids.index(str(start_frame))
            last_index = frame_ids.index(str(end_frame))
        except ValueError as exc:
            raise ValueError("Requested frame range is not available in the feature sequence.") from exc
        if first_index > last_index:
            raise ValueError("Invalid frame ordering.")

        return TemporalFeatureSequence(
            features=sequence.features[first_index:last_index + 1],
            frame_mask=sequence.frame_mask[first_index:last_index + 1],
            keypoint_mask=sequence.keypoint_mask[first_index:last_index + 1],
            frame_ids=tuple(frame_ids[first_index:last_index + 1]),
        )

    def exists(self, video_id: str, track_id: str | int) -> bool:
        return self._get_path(video_id, track_id).exists()

    def delete(self, video_id: str, track_id: str | int) -> None:
        path = self._get_path(video_id, track_id)
        if path.exists():
            path.unlink()
