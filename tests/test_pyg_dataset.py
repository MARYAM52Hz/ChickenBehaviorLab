import numpy as np
import torch

from chicken_behavior_lab.dataset import (
    FrameGraphDataset,
    GraphSample,
    PyGGraphDataset,
)

from chicken_behavior_lab.graphs import (
    TemporalSkeletonGraph,
)


def create_graph() -> TemporalSkeletonGraph:
    """
    Create a small temporal skeleton graph
    for testing.
    """

    node_features = np.array(
        [
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 3.0],
            ],
            [
                [0.5, 0.0],
                [1.5, 1.0],
                [2.5, 3.0],
            ],
        ],
        dtype=np.float32,
    )

    edge_index = np.array(
        [
            [0, 1],
            [1, 2],
        ],
        dtype=np.int64,
    )

    edge_features = np.array(
        [
            [
                [1.0, 1.0],
                [1.0, 2.0],
            ],
            [
                [1.0, 1.0],
                [1.0, 2.0],
            ],
        ],
        dtype=np.float32,
    )

    graph = TemporalSkeletonGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        frame_ids=(
            "100",
            "101",
        ),
    )

    graph.validate()

    return graph


def create_sample(sample_id: str = "video_001_track_1_ann_001") -> GraphSample:
    """
    Create a GraphSample for testing.
    """

    graph = create_graph()

    return GraphSample(
        graph=graph,
        label=2,
        behavior_id="standing",
        sample_id=sample_id,
        metadata={
            "video_id": "video_001",
            "track_id": 1,
        },
    )


def test_pyg_dataset_length():

    samples = [
        create_sample("video_001_track_1_ann_001"),
        create_sample("video_001_track_1_ann_002"),
    ]

    dataset = PyGGraphDataset(
        samples
    )

    assert len(dataset) == 2


def test_pyg_dataset_item():

    dataset = PyGGraphDataset(
        [create_sample()]
    )

    data = dataset[0]

    assert data.x.shape == (
        2,
        3,
        2,
    )

    assert data.edge_index.shape == (
        2,
        2,
    )

    assert data.edge_attr.shape == (
        2,
        2,
        2,
    )


def test_pyg_dataset_label():

    dataset = PyGGraphDataset(
        [create_sample()]
    )

    data = dataset[0]

    assert isinstance(
        data.y,
        torch.Tensor,
    )

    assert data.y.dtype == torch.long

    assert data.y.item() == 2


def test_pyg_dataset_metadata():

    dataset = PyGGraphDataset(
        [create_sample()]
    )

    data = dataset[0]

    assert (
        data.sample_id
        == "video_001_track_1_ann_001"
    )

    assert (
        data.behavior_id
        == "standing"
    )

    assert data.frame_ids == ("100", "101")
    assert data.video_id == "video_001"
    assert data.track_id == "track_1"


def test_pyg_dataset_preserves_values():

    dataset = PyGGraphDataset(
        [create_sample()]
    )

    data = dataset[0]

    expected_x = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 3.0],
            ],
            [
                [0.5, 0.0],
                [1.5, 1.0],
                [2.5, 3.0],
            ],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(
        data.x,
        expected_x,
    )



def test_frame_graph_dataset_squeezes_single_frame_graph():
    graph = create_graph()
    single_frame_graph = TemporalSkeletonGraph(
        node_features=graph.node_features[:1],
        edge_index=graph.edge_index,
        edge_features=graph.edge_features[:1],
        frame_ids=("100",),
    )
    sample = GraphSample(
        graph=single_frame_graph,
        label=2,
        behavior_id="standing",
        sample_id="single_frame",
        metadata={"video_id": "video_001", "track_id": 1},
    )

    data = FrameGraphDataset([sample])[0]

    assert data.x.shape == (3, 2)
    assert data.edge_attr.shape == (2, 2)


def test_frame_graph_dataset_rejects_multi_frame_graph():
    import pytest

    with pytest.raises(ValueError, match="exactly one frame"):
        FrameGraphDataset([create_sample()])[0]
