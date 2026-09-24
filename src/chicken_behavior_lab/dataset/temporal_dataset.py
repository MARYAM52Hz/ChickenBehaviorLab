from __future__ import annotations

from typing import Sequence

from chicken_behavior_lab.dataset.temporal_builder import (
    TemporalSequenceBuilder,
    TemporalWindow,
)


class TemporalGraphDataset:
    """
    Dataset of temporal graph windows.

    This class does not create windows itself. Windows are created
    by TemporalSequenceBuilder and passed here.
    """

    def __init__(
        self,
        windows: Sequence[TemporalWindow],
    ) -> None:

        self.windows = list(windows)

        if not self.windows:
            raise ValueError(
                "TemporalGraphDataset cannot be empty."
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(
        self,
        index: int,
    ) -> TemporalWindow:

        return self.windows[index]

    @property
    def sample_ids(self) -> list[str]:
        return [
            window.sample_id
            for window in self.windows
        ]

    @property
    def video_ids(self) -> list[str]:
        return [
            window.video_id
            for window in self.windows
        ]

    @property
    def track_ids(self) -> list[int]:
        return [
            window.track_id
            for window in self.windows
        ]

    @property
    def behavior_ids(self) -> list[str]:
        return [
            window.behavior_id
            for window in self.windows
        ]

    @property
    def start_frames(self) -> list[int]:
        return [
            window.start_frame
            for window in self.windows
        ]

    @property
    def end_frames(self) -> list[int]:
        return [
            window.end_frame
            for window in self.windows
        ]
