from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)

from chicken_behavior_lab.dataset.temporal_dataset import (
    TemporalGraphDataset,
)

from chicken_behavior_lab.dataset.temporal_sample import (
    TemporalGraphSample,
)


class TemporalSequenceBuilder:
    """
    Build fixed-length temporal windows from graph samples.

    Important
    ---------
    This builder assumes that group-level train/validation/test
    splitting has already happened.

    It therefore never performs a random split itself.

    Expected GraphSample metadata:

        video_id
        track_id
        start_frame
        end_frame

    """

    def __init__(
        self,
        *,
        sequence_length: int,
        sequence_stride: int = 1,
        require_contiguous_frames: bool = True,
    ) -> None:

        if sequence_length < 1:
            raise ValueError(
                "sequence_length must be >= 1."
            )

        if sequence_stride < 1:
            raise ValueError(
                "sequence_stride must be >= 1."
            )

        self.sequence_length = (
            sequence_length
        )

        self.sequence_stride = (
            sequence_stride
        )

        self.require_contiguous_frames = (
            require_contiguous_frames
        )

    def build(
        self,
        samples: Sequence[GraphSample],
    ) -> TemporalGraphDataset:
        """
        Build temporal windows from GraphSamples.

        Samples are first grouped by:

            (video_id, track_id)

        and then sorted by frame.

        Windows are generated independently inside each group.
        """

        if not samples:
            raise ValueError(
                "Cannot build temporal dataset from empty samples."
            )

        groups = self._group_samples(
            samples
        )

        temporal_samples: list[
            TemporalGraphSample
        ] = []

        for (
            video_id,
            track_id,
        ), group_samples in groups.items():

            group_samples.sort(
                key=self._frame_sort_key
            )

            windows = self._build_group_windows(
                group_samples
            )

            for window_index, window in enumerate(
                windows
            ):
                temporal_samples.append(
                    self._create_temporal_sample(
                        window=window,
                        video_id=video_id,
                        track_id=track_id,
                        window_index=window_index,
                    )
                )

        if not temporal_samples:
            raise ValueError(
                "No temporal windows could be created. "
                "Check sequence_length and input samples."
            )

        temporal_samples.sort(
            key=lambda sample: (
                sample.video_id or "",
                sample.track_id
                if sample.track_id is not None
                else -1,
                sample.first_frame,
            )
        )

        return TemporalGraphDataset(
            temporal_samples
        )

    def _group_samples(
        self,
        samples: Sequence[GraphSample],
    ) -> dict[
        tuple[str, int],
        list[GraphSample],
    ]:

        groups: dict[
            tuple[str, int],
            list[GraphSample],
        ] = defaultdict(list)

        for sample in samples:
            sample.validate()

            video_id = sample.get_metadata(
                "video_id"
            )

            track_id = sample.get_metadata(
                "track_id"
            )

            if video_id is None:
                raise ValueError(
                    "GraphSample requires metadata.video_id "
                    "for temporal sequence construction."
                )

            if track_id is None:
                raise ValueError(
                    "GraphSample requires metadata.track_id "
                    "for temporal sequence construction."
                )

            video_id = str(
                video_id
            )

            track_id = int(
                track_id
            )

            groups[
                (
                    video_id,
                    track_id,
                )
            ].append(
                sample
            )

        return dict(
            groups
        )

    @staticmethod
    def _frame_sort_key(
        sample: GraphSample,
    ) -> int:

        value = sample.get_metadata(
            "start_frame"
        )

        if value is None:
            raise ValueError(
                "GraphSample requires metadata.start_frame."
            )

        return int(value)

    def _build_group_windows(
        self,
        samples: list[GraphSample],
    ) -> list[list[GraphSample]]:

        windows: list[
            list[GraphSample]
        ] = []

        max_start = (
            len(samples)
            - self.sequence_length
            + 1
        )

        if max_start <= 0:
            return windows

        start_index = 0

        while start_index < max_start:

            end_index = (
                start_index
                + self.sequence_length
            )

            window = samples[
                start_index:end_index
            ]

            if (
                len(window)
                == self.sequence_length
            ):
                if (
                    not self.require_contiguous_frames
                    or self._is_contiguous(
                        window
                    )
                ):
                    windows.append(
                        list(window)
                    )

            start_index += (
                self.sequence_stride
            )

        return windows

    @staticmethod
    def _is_contiguous(
        window: Sequence[GraphSample],
    ) -> bool:

        frame_ranges: list[
            tuple[int, int]
        ] = []

        for sample in window:
            start_frame = sample.get_metadata(
                "start_frame"
            )

            end_frame = sample.get_metadata(
                "end_frame"
            )

            if (
                start_frame is None
                or end_frame is None
            ):
                return False

            frame_ranges.append(
                (
                    int(start_frame),
                    int(end_frame),
                )
            )

        for index in range(
            1,
            len(frame_ranges),
        ):
            previous_end = (
                frame_ranges[index - 1][1]
            )

            current_start = (
                frame_ranges[index][0]
            )

            if current_start != previous_end + 1:
                return False

        return True

    def _create_temporal_sample(
        self,
        *,
        window: Sequence[GraphSample],
        video_id: str,
        track_id: int,
        window_index: int,
    ) -> TemporalGraphSample:

        first_sample = window[0]
        last_sample = window[-1]

        labels = [
            sample.label
            for sample in window
        ]

        behavior_ids = [
            sample.behavior_id
            for sample in window
        ]

        label = self._resolve_label(
            labels
        )

        behavior_id = self._resolve_behavior_id(
            behavior_ids
        )

        start_frame = int(
            first_sample.get_metadata(
                "start_frame"
            )
        )

        end_frame = int(
            last_sample.get_metadata(
                "end_frame"
            )
        )

        sample_id = (
            f"{video_id}"
            f"__track_{track_id}"
            f"__frames_{start_frame}_{end_frame}"
            f"__window_{window_index:06d}"
        )

        metadata = {
            "video_id": video_id,
            "track_id": track_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "sequence_length": len(window),
            "sequence_stride": self.sequence_stride,
            "label_resolution": "majority",
        }

        temporal_sample = TemporalGraphSample(
            graphs=list(window),
            label=label,
            behavior_id=behavior_id,
            sample_id=sample_id,
            metadata=metadata,
        )

        temporal_sample.validate()

        return temporal_sample

    @staticmethod
    def _resolve_label(
        labels: Sequence[int],
    ) -> int:

        if not labels:
            raise ValueError(
                "Cannot resolve label from empty sequence."
            )

        counts: dict[int, int] = {}

        for label in labels:
            counts[label] = (
                counts.get(label, 0)
                + 1
            )

        # Deterministic majority vote:
        # in case of a tie, choose the smaller label index.
        return min(
            counts,
            key=lambda label: (
                -counts[label],
                label,
            ),
        )

    @staticmethod
    def _resolve_behavior_id(
        behavior_ids: Sequence[str],
    ) -> str:

        if not behavior_ids:
            raise ValueError(
                "Cannot resolve behavior_id from empty sequence."
            )

        counts: dict[str, int] = {}

        for behavior_id in behavior_ids:
            counts[behavior_id] = (
                counts.get(behavior_id, 0)
                + 1
            )

        return min(
            counts,
            key=lambda behavior_id: (
                -counts[behavior_id],
                behavior_id,
            ),
        )

    @staticmethod
    def filter_by_groups(
        samples: Sequence[GraphSample],
        allowed_groups: set[str],
        *,
        split_group: str = "video",
    ) -> list[GraphSample]:
        """
        Filter GraphSamples by a precomputed group split.

        This method is intentionally separate from `build()`.

        Correct workflow:

            all frame/graph samples
                    ↓
            group-level split
                    ↓
            filter samples
                    ↓
            TemporalSequenceBuilder.build()

        """

        if split_group not in {
            "video",
            "track",
        }:
            raise ValueError(
                "split_group must be either "
                "'video' or 'track'."
            )

        result: list[GraphSample] = []

        for sample in samples:
            video_id = sample.get_metadata(
                "video_id"
            )

            track_id = sample.get_metadata(
                "track_id"
            )

            if video_id is None:
                raise ValueError(
                    "GraphSample is missing video_id."
                )

            if track_id is None:
                raise ValueError(
                    "GraphSample is missing track_id."
                )

            if split_group == "video":
                group = str(
                    video_id
                )
            else:
                group = (
                    f"{video_id}"
                    f"::track::{int(track_id)}"
                )

            if group in allowed_groups:
                result.append(
                    sample
                )

        return result
