import numpy as np
import torch

from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)
from chicken_behavior_lab.dataset.temporal_builder import (
    TemporalSequenceBuilder,
)
from chicken_behavior_lab.dataset.temporal_pyg_dataset import (
    TemporalPyGDataset,
)
from chicken_behavior_lab.graph.graph import (
    TemporalSkeletonGraph,
)
from chicken_behavior_lab.models.factory import (
    TemporalModelConfig,
    build_temporal_model,
)


def make_sample(
    sample_id: str,
    frame: int,
    behavior_id: str,
    label: int,
) -> GraphSample:

    graph = TemporalSkeletonGraph(
        node_features=np.zeros(
            (13, 4),
            dtype=np.float32,
        ),
        edge_index=np.array(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=np.int64,
        ),
        edge_features=None,
    )

    return GraphSample(
        graph=graph,
        label=label,
        behavior_id=behavior_id,
        sample_id=sample_id,
        metadata={
            "video_id": "video_001",
            "track_id": 1,
            "start_frame": frame,
            "end_frame": frame,
        },
    )


def test_build_temporal_model() -> None:
    config = TemporalModelConfig(
        node_feature_dim=4,
        edge_feature_dim=None,
        spatial_hidden_dim=16,
        temporal_hidden_dim=16,
    )

    model = build_temporal_model(
        model_config=config,
        num_classes=3,
    )

    assert isinstance(
        model,
        torch.nn.Module,
    )

    final_layer = model.classifier[-1]

    assert isinstance(
        final_layer,
        torch.nn.Linear,
    )

    assert final_layer.out_features == 3
