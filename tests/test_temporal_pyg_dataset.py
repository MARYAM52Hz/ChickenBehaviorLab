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


def make_graph(
    node_offset: float,
):
    return SimpleNamespace(
        node_features=np.array(
            [
                [
                    1.0 + node_offset,
                    2.0 + node_offset,
                ],
                [
                    3.0 + node_offset,
                    4.0 + node_offset,
                ],
                [
                    5.0 + node_offset,
                    6.0 + node_offset,
                ],
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


def make_graph_sample(
    sample_id: str,
    start_frame: int,
    end_frame: int,
    node_offset: float,
):
    return GraphSample(
        graph=make_graph(
            node_offset
        ),
        label=0,
        behavior_id="feeding",
        sample_id=sample_id,
        metadata={
            "video_id": "video_001",
            "track_id": 1,
            "start_frame": start_frame,
            "end_frame": end_frame,
        },
    )


def make_temporal_sample():
    graphs = [
        make_graph_sample(
            "graph_001",
            100,
            109,
            0.0,
        ),
        make_graph_sample(
            "graph_002",
            110,
            119,
            0.5,
        ),
        make_graph_sample(
            "graph_003",
            120,
            129,
            1.0,
        ),
    ]

    return TemporalGraphSample(
        graphs=graphs,
        label=0,
        behavior_id="feeding",
        sample_id="sequence_001",
        metadata={
            "video_id": "video_001",
            "track_id": 1,
        },
    )


def test_temporal_pyg_shape():
    dataset = TemporalPyGDataset(
        [make_temporal_sample()]
    )

    data = dataset[0]

    assert data.x.shape == (
        3,
        3,
        2,
    )

    assert data.edge_index.shape == (
        2,
        4,
    )

    assert data.edge_attr.shape == (
        3,
        4,
        2,
    )

    assert data.y.tolist() == [0]


def test_temporal_metadata():
    dataset = TemporalPyGDataset(
        [make_temporal_sample()]
    )

    data = dataset[0]

    assert data.sample_id == (
        "sequence_001"
    )

    assert data.video_id == (
        "video_001"
    )

    assert data.track_id == 1
    assert data.start_frame == 100
    assert data.end_frame == 129
    assert data.sequence_length == 3


def test_temporal_tensor_dtype():
    dataset = TemporalPyGDataset(
        [make_temporal_sample()]
    )

    data = dataset[0]

    assert data.x.dtype == (
        torch.float32
    )

    assert data.edge_index.dtype == (
        torch.long
    )

    assert data.edge_attr.dtype == (
        torch.float32
    )

    assert data.y.dtype == (
        torch.long
    )
