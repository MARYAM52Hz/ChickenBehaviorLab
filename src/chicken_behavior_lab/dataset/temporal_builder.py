from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.dataset.temporal_sample import (
    TemporalGraphSample,
)


class TemporalSequenceBuilder:
    """
    Build temporal graph sequences from graph samples.

    Graph samples are grouped by video and track, then ordered
    temporally before being assembled into fixed-length
    sequences.
    """

    def __init__(
        self,
        sequence_length: int,
        stride: int | None = None,
    ) -> None:
        if sequence_length < 1:
            raise ValueError(
                "sequence_length must be >= 1."
            )

        if stride is None:
            stride = sequence_length

        if stride < 1:
            raise ValueError(
                "stride must be >= 1."
            )

        self.sequence_length = sequence_length
        self.stride = stride

    def build(
        self,
        samples: Iterable[GraphSample],
    ) -> list[TemporalGraphSample]:
        samples = list(samples)

        if not samples:
            return []

        grouped: dict[
            tuple[str, int],
            list[GraphSample],
        ] = defaultdict(list)

        for sample in samples:
            video_id = sample.get_metadata(
                "video_id"
            )

            track_id = sample.get_metadata(
                "track_id"
            )

            if video_id is None:
                raise ValueError(
                    f"Sample '{sample.sample_id}' "
                    "is missing metadata.video_id."
                )

            if track_id is None:
                raise ValueError(
                    f"Sample '{sample.sample_id}' "
                    "is missing metadata.track_id."
                )

            grouped[
                (str(video_id), int(track_id))
            ].append(sample)

        sequences = []

        for (
            video_id,
            track_id,
        ), track_samples in grouped.items():

            track_samples.sort(
                key=self._start_frame
            )

            sequences.extend(
                self._build_track_sequences(
                    video_id=video_id,
                    track_id=track_id,
                    samples=track_samples,
                )
            )

        return sequences

    def _build_track_sequences(
        self,
        video_id: str,
        track_id: int,
        samples: list[GraphSample],
    ) -> list[TemporalGraphSample]:
        sequences = []

        if len(samples) < self.sequence_length:
            return sequences

        for start in range(
            0,
            len(samples)
            - self.sequence_length
            + 1,
            self.stride,
        ):
            window = samples[
                start:
                start + self.sequence_length
            ]

            if not self._is_contiguous(
                window
            ):
                continue

            labels = {
                sample.label
                for sample in window
            }

            behavior_ids = {
                sample.behavior_id
                for sample in window
            }

            # A sequence should represent one behavior.
            if len(labels) != 1:
                continue

            if len(behavior_ids) != 1:
                continue

            first = window[0]
            last = window[-1]

            first_frame = first.get_metadata(
                "start_frame"
            )

            last_frame = last.get_metadata(
                "end_frame"
            )

            if first_frame is None:
                first_frame = 0

            if last_frame is None:
                last_frame = first_frame

            sample_id = (
                f"{video_id}_"
                f"track_{track_id}_"
                f"{first_frame}_"
                f"{last_frame}"
            )

            temporal_sample = TemporalGraphSample(
                graphs=window,
                label=first.label,
                behavior_id=(
                    first.behavior_id
                ),
                sample_id=sample_id,
                metadata={
                    "video_id": video_id,
                    "track_id": track_id,
                    "start_frame": first_frame,
                    "end_frame": last_frame,
                    "sequence_length": (
                        self.sequence_length
                    ),
                },
            )

            temporal_sample.validate()

            sequences.append(
                temporal_sample
            )

        return sequences

    @staticmethod
    def _start_frame(
        sample: GraphSample,
    ) -> int:
        value = sample.get_metadata(
            "start_frame"
        )

        if value is None:
            return 0

        return int(value)

    @staticmethod
    def _is_contiguous(
        samples: list[GraphSample],
    ) -> bool:
        if len(samples) < 2:
            return True

        for previous, current in zip(
            samples,
            samples[1:],
        ):
            previous_end = previous.get_metadata(
                "end_frame"
            )

            current_start = current.get_metadata(
                "start_frame"
            )

            if (
                previous_end is None
                or current_start is None
            ):
                return False

            if current_start <= previous_end:
                return False

        return True
