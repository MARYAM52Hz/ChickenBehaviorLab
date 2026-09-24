from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(slots=True)
class TemporalWindow:
    """
    A temporal window belonging to exactly one video/track group.
    """

    samples: list
    sample_id: str
    video_id: str
    track_id: int
    start_frame: int
    end_frame: int
    behavior_id: str


class TemporalSequenceBuilder:
    """
    Build temporal windows from graph samples.

    IMPORTANT
    ---------
    This builder operates only on the samples supplied to it.

    Therefore train/validation/test splitting must happen BEFORE
    this builder is called.
    """

    def __init__(
        self,
        sequence_length: int = 16,
        sequence_stride: int = 4,
        require_same_behavior: bool = False,
    ) -> None:

        if sequence_length < 1:
            raise ValueError(
                "sequence_length must be >= 1."
            )

        if sequence_stride < 1:
            raise ValueError(
                "sequence_stride must be >= 1."
            )

        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.require_same_behavior = (
            require_same_behavior
        )

    def build(
        self,
        samples: Sequence,
    ) -> list[TemporalWindow]:
        """
        Build temporal windows.

        Samples are first grouped by video + track.

        Within each group, samples are sorted by frame.

        Windows are then created using sequence_length and
        sequence_stride.
        """

        grouped = self._group_samples(
            samples
        )

        windows: list[TemporalWindow] = []

        for _, group_samples in grouped.items():

            ordered = self._sort_by_frame(
                group_samples
            )

            group_windows = (
                self._build_group_windows(
                    ordered
                )
            )

            windows.extend(
                group_windows
            )

        return windows

    def _group_samples(
        self,
        samples: Iterable,
    ) -> dict[tuple[str, int], list]:

        grouped: dict[
            tuple[str, int],
            list,
        ] = defaultdict(list)

        for sample in samples:

            video_id = self._get_video_id(
                sample
            )

            track_id = self._get_track_id(
                sample
            )

            key = (
                video_id,
                track_id,
            )

            grouped[key].append(
                sample
            )

        return dict(grouped)

    def _sort_by_frame(
        self,
        samples: Sequence,
    ) -> list:

        return sorted(
            samples,
            key=self._get_frame_id,
        )

    def _build_group_windows(
        self,
        samples: Sequence,
    ) -> list[TemporalWindow]:

        if len(samples) < self.sequence_length:
            return []

        windows: list[
            TemporalWindow
        ] = []

        start = 0

        while (
            start + self.sequence_length
            <= len(samples)
        ):

            window_samples = list(
                samples[
                    start:
                    start
                    + self.sequence_length
                ]
            )

            if self.require_same_behavior:
                if not self._same_behavior(
                    window_samples
                ):
                    start += self.sequence_stride
                    continue

            first = window_samples[0]
            last = window_samples[-1]

            video_id = self._get_video_id(
                first
            )

            track_id = self._get_track_id(
                first
            )

            start_frame = (
                self._get_frame_id(
                    first
                )
            )

            end_frame = (
                self._get_frame_id(
                    last
                )
            )

            behavior_id = (
                self._get_window_behavior(
                    window_samples
                )
            )

            sample_id = (
                f"{video_id}"
                f"__track_{track_id}"
                f"__frames_{start_frame}"
                f"_{end_frame}"
            )

            windows.append(
                TemporalWindow(
                    samples=window_samples,
                    sample_id=sample_id,
                    video_id=video_id,
                    track_id=track_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    behavior_id=behavior_id,
                )
            )

            start += self.sequence_stride

        return windows

    def _same_behavior(
        self,
        samples: Sequence,
    ) -> bool:

        behaviors = {
            self._get_behavior_id(
                sample
            )
            for sample in samples
        }

        return len(behaviors) == 1

    def _get_window_behavior(
        self,
        samples: Sequence,
    ) -> str:
        """
        Determine the label of a temporal window.

        For now we use the final frame's behavior.

        This keeps the implementation compatible with
        sequence-to-one classification.

        Later we can replace this with:
            - majority voting
            - center-frame label
            - transition label
            - event label
        """

        return self._get_behavior_id(
            samples[-1]
        )

    @staticmethod
    def _get_video_id(
        sample,
    ) -> str:

        metadata = getattr(
            sample,
            "metadata",
            None,
        )

        if isinstance(metadata, dict):
            value = metadata.get(
                "video_id"
            )

            if value is not None:
                return str(value)

        value = getattr(
            sample,
            "video_id",
            None,
        )

        if value is None:
            raise ValueError(
                "Sample does not contain video_id."
            )

        return str(value)

    @staticmethod
    def _get_track_id(
        sample,
    ) -> int:

        metadata = getattr(
            sample,
            "metadata",
            None,
        )

        if isinstance(metadata, dict):
            value = metadata.get(
                "track_id"
            )

            if value is not None:
                return int(value)

        value = getattr(
            sample,
            "track_id",
            None,
        )

        if value is None:
            raise ValueError(
                "Sample does not contain track_id."
            )

        return int(value)

    @staticmethod
    def _get_frame_id(
        sample,
    ) -> int:

        metadata = getattr(
            sample,
            "metadata",
            None,
        )

        if isinstance(metadata, dict):
            for key in (
                "frame_id",
                "frame_index",
            ):
                value = metadata.get(
                    key
                )

                if value is not None:
                    return int(value)

        for key in (
            "frame_id",
            "frame_index",
        ):
            value = getattr(
                sample,
                key,
                None,
            )

            if value is not None:
                return int(value)

        raise ValueError(
            "Sample does not contain frame_id "
            "or frame_index."
        )

    @staticmethod
    def _get_behavior_id(
        sample,
    ) -> str:

        metadata = getattr(
            sample,
            "metadata",
            None,
        )

        if isinstance(metadata, dict):
            value = metadata.get(
                "behavior_id"
            )

            if value is not None:
                return str(value)

        value = getattr(
            sample,
            "behavior_id",
            None,
        )

        if value is None:
            raise ValueError(
                "Sample does not contain behavior_id."
            )

        return str(value)
