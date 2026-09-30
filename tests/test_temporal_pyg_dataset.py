from __future__ import annotations

import numpy as np

from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)

from chicken_behavior_lab.dataset.temporal_builder import (
    TemporalSequenceBuilder,
)

from chicken_behavior_lab.dataset.temporal_pyg_dataset import (
    TemporalPyGDataset,
)


class DummyGraph:
    def __init__(
        self,
        value: float,
    ) -> None:

        self.node_features = np.full(
            (3, 4),
            value,
            dtype=np.float32,
        )

        self.edge_index = np.array(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=np.int64,
        )

        self.edge_features = np.full(
            (2, 2),
            value,
            dtype=np.float32,
        )

    def validate(self) -> None:
        pass


def make_sample(
    frame: int,
) -> GraphSample:

    return GraphSample(
        graph=DummyGraph(
            float(frame)
        ),
        label=0,
        behavior_id="feeding",
        sample_id=f"sample_{frame}",
        metadata={
            "video_id": "video_001",
            "track_id": 1,
            "start_frame": frame,
            "end_frame": frame,
        },
    )


def test_temporal_pyg_dataset_shapes() -> None:
    samples = [
        make_sample(frame)
        for frame in range(4)
    ]

    builder = TemporalSequenceBuilder(
        sequence_length=4,
        sequence_stride=1,
    )

    temporal_dataset = builder.build(
        samples
    )

    dataset = TemporalPyGDataset(
        temporal_dataset.samples
    )

    assert len(dataset) == 1

    data = dataset[0]

    assert tuple(data.x.shape) == (
        4,
        3,
        4,
    )

    assert tuple(
        data.edge_index.shape
    ) == (
        2,
        2,
    )

    assert tuple(
        data.edge_attr.shape
    ) == (
        4,
        2,
        2,
    )

    assert tuple(
        data.y.shape
    ) == (
        1,
    )


def test_temporal_pyg_metadata() -> None:
    samples = [
        make_sample(frame)
        for frame in range(4)
    ]

    builder = TemporalSequenceBuilder(
        sequence_length=4,
    )

    temporal_dataset = builder.build(
        samples
    )

    dataset = TemporalPyGDataset(
        temporal_dataset.samples
    )

    data = dataset[0]

    assert data.sample_id.startswith(
        "video_001__track_1"
    )

    assert data.video_id == (
        "video_001"
    )

    assert data.track_id == 1
    assert data.start_frame == 0
    assert data.end_frame == 3
    assert data.behavior_id == "feeding"


def test_temporal_label_mapping() -> None:
    samples = [
        make_sample(
            frame=0,
        ),
        make_sample(
            frame=1,
        ),
        make_sample(
            frame=2,
        ),
        make_sample(
            frame=3,
        ),
    ]

    builder = TemporalSequenceBuilder(
        sequence_length=2,
        sequence_stride=2,
    )

    temporal_dataset = builder.build(
        samples
    )

    dataset = TemporalPyGDataset(
        temporal_dataset.samples
    )

    assert dataset.label_to_index == {
        "feeding": 0,
    }
