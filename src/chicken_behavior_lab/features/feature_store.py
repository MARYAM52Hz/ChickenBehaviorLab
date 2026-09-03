"""
ChickenBehaviorLab Feature Store
================================

Storage and retrieval of temporal skeleton features.

The initial implementation uses NumPy files (.npz).

The design intentionally keeps storage separate from:

- detection
- tracking
- pose estimation
- graph construction
- dataset construction
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chicken_behavior_lab.features.temporal_features import (
    TemporalFeatureSequence,
)


class FeatureStore:
    """
    Store and retrieve TemporalFeatureSequence objects.

    Directory structure
    -------------------

    feature_root/
        video_001/
            track_1.npz
            track_2.npz

        video_002/
            track_1.npz
            track_5.npz
    """

    def __init__(
        self,
        root: str | Path,
    ) -> None:

        self.root = Path(root)

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =====================================================
    # Path Handling
    # =====================================================

    def _get_path(
        self,
        video_id: str,
        track_id: int,
    ) -> Path:
        """
        Return the storage path for one track.
        """

        if not video_id:
            raise ValueError(
                "video_id cannot be empty."
            )

        if track_id < 0:
            raise ValueError(
                "track_id cannot be negative."
            )

        video_dir = (
            self.root / video_id
        )

        return (
            video_dir
            / f"track_{track_id}.npz"
        )

    # =====================================================
    # Save
    # =====================================================

    def save(
        self,
        video_id: str,
        track_id: int,
        sequence: TemporalFeatureSequence,
    ) -> Path:
        """
        Save a temporal feature sequence.

        Parameters
        ----------
        video_id:
            Source video identifier.

        track_id:
            Tracked chicken identifier.

        sequence:
            TemporalFeatureSequence to store.

        Returns
        -------
        Path
            Path of the saved feature file.
        """

        if not isinstance(
            sequence,
            TemporalFeatureSequence,
        ):
            raise TypeError(
                "sequence must be a "
                "TemporalFeatureSequence."
            )

        path = self._get_path(
            video_id,
            track_id,
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            path,
            features=sequence.features,
            frame_mask=sequence.frame_mask,
            keypoint_mask=sequence.keypoint_mask,
            frame_ids=np.asarray(
                sequence.frame_ids
            ),
        )

        return path

    # =====================================================
    # Load
    # =====================================================

    def load(
        self,
        video_id: str,
        track_id: int,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> TemporalFeatureSequence:
        """
        Load a temporal feature sequence.

        Optionally returns only a frame interval.

        Parameters
        ----------
        video_id:
            Source video identifier.

        track_id:
            Tracked chicken identifier.

        start_frame:
            Optional first frame.

        end_frame:
            Optional last frame.

        Returns
        -------
        TemporalFeatureSequence
        """

        path = self._get_path(
            video_id,
            track_id,
        )

        if not path.exists():
            raise FileNotFoundError(
                "Feature sequence not found: "
                f"{path}"
            )

        with np.load(
            path,
            allow_pickle=False,
        ) as data:

            features = data[
                "features"
            ]

            frame_mask = data[
                "frame_mask"
            ]

            keypoint_mask = data[
                "keypoint_mask"
            ]

            frame_ids_array = data[
                "frame_ids"
            ]

        frame_ids = tuple(
            str(frame_id)
            for frame_id in frame_ids_array
        )

        sequence = TemporalFeatureSequence(
            features=features,
            frame_mask=frame_mask,
            keypoint_mask=keypoint_mask,
            frame_ids=frame_ids,
        )

        if (
            start_frame is None
            and end_frame is None
        ):
            return sequence

        return self._slice_sequence(
            sequence,
            start_frame,
            end_frame,
        )

    # =====================================================
    # Slice
    # =====================================================

    @staticmethod
    def _slice_sequence(
        sequence: TemporalFeatureSequence,
        start_frame: int | None,
        end_frame: int | None,
    ) -> TemporalFeatureSequence:
        """
        Extract an inclusive temporal interval.

        Important:
        start_frame and end_frame refer to the frame IDs,
        not necessarily array indices.
        """

        if (
            start_frame is None
            or end_frame is None
        ):
            raise ValueError(
                "Both start_frame and end_frame "
                "must be provided for slicing."
            )

        if start_frame > end_frame:
            raise ValueError(
                "start_frame must be <= end_frame."
            )

        frame_ids = list(
            sequence.frame_ids
        )

        try:
            first_index = frame_ids.index(
                str(start_frame)
            )

            last_index = frame_ids.index(
                str(end_frame)
            )

        except ValueError as exc:
            raise ValueError(
                "Requested frame range is not "
                "available in the feature sequence."
            ) from exc

        if first_index > last_index:
            raise ValueError(
                "Invalid frame ordering."
            )

        return TemporalFeatureSequence(
            features=sequence.features[
                first_index:last_index + 1
            ],
            frame_mask=sequence.frame_mask[
                first_index:last_index + 1
            ],
            keypoint_mask=sequence.keypoint_mask[
                first_index:last_index + 1
            ],
            frame_ids=tuple(
                frame_ids[
                    first_index:last_index + 1
                ]
            ),
        )

    # =====================================================
    # Exists
    # =====================================================

    def exists(
        self,
        video_id: str,
        track_id: int,
    ) -> bool:
        """
        Check whether a feature sequence exists.
        """

        return self._get_path(
            video_id,
            track_id,
        ).exists()

    # =====================================================
    # Delete
    # =====================================================

    def delete(
        self,
        video_id: str,
        track_id: int,
    ) -> None:
        """
        Delete a stored feature sequence.
        """

        path = self._get_path(
            video_id,
            track_id,
        )

        if path.exists():
            path.unlink()
