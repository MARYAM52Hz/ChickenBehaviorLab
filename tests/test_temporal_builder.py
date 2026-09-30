from __future__ import annotations

import numpy as np
import pytest

from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)

from chicken_behavior_lab.dataset.temporal_builder import (
    TemporalSequenceBuilder,
)


class DummyGraph:
    def __init__(self) -> None:
        self.node_features = np.zeros(
            (3, 4),
            dtype=np.float32,
        )

        self.edge_index = np.array(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=np.int64,
        )

        self.edge_features = None

    def validate(self) -> None:
        pass


def make_sample(
    frame: int,
    *,
    video_id: str = "video_001",
    track_id: int = 1,
    label: int = 0,
    behavior_id: str = "feeding",
) -> GraphSample:

    return GraphSample(
        graph=DummyGraph(),
        label=label,
        behavior_id=behavior_id,
        sample_id=(
            f"{video_id}_{track_id}_{frame}"
        ),
        metadata={
            "video_id": video_id,
            "track_id": track_id,
            "start_frame": frame,
            "end_frame": frame,
        },
    )


def test_builder_creates_overlapping_windows() -> None:
    samples = [
        make_sample(frame)
        for frame in range(10)
    ]

    builder = TemporalSequenceBuilder(
        sequence_length=4,
        sequence_stride=2,
    )

    dataset = builder.build(
        samples
    )

    # Starts:
    # 0, 2, 4, 6
    assert len(dataset) == 4

    assert dataset[0].first_frame == 0
    assert dataset[0].last_frame == 3

    assert dataset[1].first_frame == 2
    assert dataset[1].last_frame == 5

    assert dataset[2].first_frame == 4
    assert dataset[2].last_frame == 7

    assert dataset[3].first_frame == 6
    assert dataset[3].last_frame == 9


def test_builder_does_not_cross_tracks() -> None:
    samples = []

    for frame in range(6):
        samples.append(
            make_sample(
                frame,
                track_id=1,
            )
        )

    for frame in range(6):
        samples.append(
            make_sample(
                frame,
                track_id=2,
            )
        )

    builder = TemporalSequenceBuilder(
        sequence_length=3,
        sequence_stride=3,
    )

    dataset = builder.build(
        samples
    )

    assert len(dataset) == 4

    for sample in dataset:
        assert sample.track_id in {
            1,
            2,
        }


def test_builder_requires_contiguous_frames() -> None:
    samples = [
        make_sample(0),
        make_sample(1),
        make_sample(3),
        make_sample(4),
    ]

    builder = TemporalSequenceBuilder(
        sequence_length=3,
        sequence_stride=1,
        require_contiguous_frames=True,
    )

    dataset = builder.build(
        samples
    )

    assert len(dataset) == 0


def test_builder_can_allow_noncontiguous_frames() -> None:
    samples = [
        make_sample(0),
        make_sample(1),
        make_sample(3),
        make_sample(4),
    ]

    builder = TemporalSequenceBuilder(
        sequence_length=3,
        sequence_stride=1,
        require_contiguous_frames=False,
    )

    dataset = builder.build(
        samples
    )

    assert len(dataset) == 2


def test_builder_majority_label() -> None:
    samples = [
        make_sample(
            0,
            label=0,
            behavior_id="feeding",
        ),
        make_sample(
            1,
            label=0,
            behavior_id="feeding",
        ),
        make_sample(
            2,
            label=1,
            behavior_id="walking",
        ),
    ]

    builder = TemporalSequenceBuilder(
        sequence_length=3,
    )

    dataset = builder.build(
        samples
    )

    assert len(dataset) == 1

    assert dataset[0].label == 0
    assert dataset[0].behavior_id == "feeding"


def test_builder_filter_by_video_group() -> None:
    samples = [
        make_sample(
            frame=0,
            video_id="video_001",
        ),
        make_sample(
            frame=1,
            video_id="video_001",
        ),
        make_sample(
            frame=0,
            video_id="video_002",
        ),
        make_sample(
            frame=1,
            video_id="video_002",
        ),
    ]

    filtered = (
        TemporalSequenceBuilder.filter_by_groups(
            samples,
            {"video_001"},
            split_group="video",
        )
    )

    assert len(filtered) == 2

    assert all(
        sample.get_metadata("video_id")
        == "video_001"
        for sample in filtered
    )


def test_builder_filter_by_track_group() -> None:
    samples = [
        make_sample(
            frame=0,
            video_id="video_001",
            track_id=1,
        ),
        make_sample(
            frame=1,
            video_id="video_001",
            track_id=1,
        ),
        make_sample(
            frame=0,
            video_id="video_001",
            track_id=2,
        ),
    ]

    filtered = (
        TemporalSequenceBuilder.filter_by_groups(
            samples,
            {
                "video_001::track::1"
            },
            split_group="track",
        )
    )

    assert len(filtered) == 2

    assert all(
        sample.get_metadata("track_id")
        == 1
        for sample in filtered
    )
