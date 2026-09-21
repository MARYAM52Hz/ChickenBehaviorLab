import numpy as np
import torch

from types import SimpleNamespace

from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)

from chicken_behavior_lab.dataset.temporal_sample import (
    TemporalGraphSample,
)

from chicken_behavior_lab.dataset.temporal_pyg_dataset import (
    TemporalPyGDataset,
)

from chicken_behavior_lab.dataset.temporal_collate import (
    TemporalCollator,
)


def make_graph(
    offset: float,
):
    return SimpleNamespace(
        node_features=np.array(
            [
                [1.0 + offset, 2.0],
                [3.0 + offset, 4.0],
                [5.0 + offset, 6.0],
            ],
            dtype=np.float32,
        ),
        edge_index=np.array(
            [
                [0, 1, 1, 2],
                [1, 0, 2, 1],
            ],
            dtype=np.int64,
        ),
        edge_features=np.array(
            [
                [0.1, 0.2],
                [0.1, 0.2],
                [0.2, 0.3],
                [0.2, 0.3],
            ],
            dtype=np.float32,
        ),
        validate=lambda: None,
    )


def make_sequence(
    sample_id: str,
    offset: float,
):
    graphs = []

    for index in range(3):
        graphs.append(
            GraphSample(
                graph=make_graph(
                    offset + index
                ),
                label=0,
                behavior_id="feeding",
                sample_id=(
                    f"{sample_id}_graph_{index}"
                ),
                metadata={
                    "video_id": "video_001",
                    "track_id": 1,
                    "start_frame": (
                        100 + index * 10
                    ),
                    "end_frame": (
                        109 + index * 10
                    ),
                },
            )
        )

    return TemporalGraphSample(
        graphs=graphs,
        label=0,
        behavior_id="feeding",
        sample_id=sample_id,
        metadata={
            "video_id": "video_001",
            "track_id": 1,
        },
    )


def test_temporal_collator():
    temporal_samples = [
        make_sequence(
            "sequence_001",
            0.0,
        ),
        make_sequence(
            "sequence_002",
            10.0,
        ),
    ]

    dataset = TemporalPyGDataset(
        temporal_samples
    )

    data_list = [
        dataset[0],
        dataset[1],
    ]

    collator = TemporalCollator()

    batch = collator(
        data_list
    )

    assert batch.x.shape == (
        2,
        3,
        3,
        2,
    )

    assert batch.edge_index.shape == (
        2,
        4,
    )

    assert batch.edge_attr.shape == (
        2,
        3,
        4,
        2,
    )

    assert batch.y.shape == (
        2,
    )

    assert batch.sample_id == [
        "sequence_001",
        "sequence_002",
    ]

    assert batch.track_id.tolist() == [
        1,
        1,
    ]

    assert batch.start_frame.tolist() == [
        100,
        100,
    ]

    assert batch.end_frame.tolist() == [
        129,
        129,
    ]


def test_temporal_batch_to_device():
    temporal_samples = [
        make_sequence(
            "sequence_001",
            0.0,
        ),
    ]

    dataset = TemporalPyGDataset(
        temporal_samples
    )

    batch = TemporalCollator()(
        [dataset[0]]
    )

    moved = batch.to(
        torch.device("cpu")
    )

    assert moved.x.device.type == "cpu"
    assert moved.edge_index.device.type == "cpu"
    assert moved.y.device.type == "cpu"
