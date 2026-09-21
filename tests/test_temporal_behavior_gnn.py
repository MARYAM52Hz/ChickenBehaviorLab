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

from chicken_behavior_lab.models.temporal_behavior_gnn import (
    TemporalBehaviorGNN,
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
    label: int,
    behavior_id: str,
    offset: float,
):
    graphs = []

    for index in range(3):
        graphs.append(
            GraphSample(
                graph=make_graph(
                    offset + index
                ),
                label=label,
                behavior_id=behavior_id,
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
        label=label,
        behavior_id=behavior_id,
        sample_id=sample_id,
        metadata={
            "video_id": "video_001",
            "track_id": 1,
        },
    )


def test_temporal_behavior_gnn_output_shape():
    temporal_samples = [
        make_sequence(
            "sequence_001",
            0,
            "feeding",
            0.0,
        ),
        make_sequence(
            "sequence_002",
            1,
            "walking",
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

    batch = TemporalCollator()(
        data_list
    )

    model = TemporalBehaviorGNN(
        node_feature_dim=2,
        edge_feature_dim=2,
        spatial_hidden_dim=8,
        temporal_hidden_dim=16,
        num_classes=2,
        num_gnn_layers=2,
        num_gru_layers=1,
        dropout=0.0,
    )

    logits = model(batch)

    assert logits.shape == (
        2,
        2,
    )


def test_temporal_behavior_gnn_backward():
    temporal_samples = [
        make_sequence(
            "sequence_001",
            0,
            "feeding",
            0.0,
        ),
        make_sequence(
            "sequence_002",
            1,
            "walking",
            10.0,
        ),
    ]

    dataset = TemporalPyGDataset(
        temporal_samples
    )

    batch = TemporalCollator()(
        [
            dataset[0],
            dataset[1],
        ]
    )

    model = TemporalBehaviorGNN(
        node_feature_dim=2,
        edge_feature_dim=2,
        spatial_hidden_dim=8,
        temporal_hidden_dim=16,
        num_classes=2,
    )

    logits = model(batch)

    loss = torch.nn.functional.cross_entropy(
        logits,
        batch.y,
    )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert any(
        gradient is not None
        for gradient in gradients
    )
