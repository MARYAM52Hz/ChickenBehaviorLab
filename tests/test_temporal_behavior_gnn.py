from __future__ import annotations

import torch

from chicken_behavior_lab.config.model_config import (
    ModelConfig,
)

from chicken_behavior_lab.dataset.temporal_builder import (
    TemporalSequenceBuilder,
)

from chicken_behavior_lab.dataset.temporal_pyg_dataset import (
    TemporalPyGDataset,
)

from chicken_behavior_lab.dataset.temporal_collate import (
    make_temporal_dataloader,
)

from chicken_behavior_lab.models.model_factory import (
    build_model,
)


class DummySample:

    def __init__(
        self,
        frame_id: int,
    ) -> None:

        self.x = torch.randn(
            13,
            8,
        )

        self.edge_index = torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
            ],
            dtype=torch.long,
        )

        self.edge_attr = torch.randn(
            4,
            4,
        )

        self.metadata = {
            "video_id": "video_001",
            "track_id": 1,
            "frame_id": frame_id,
            "behavior_id": 0,
        }


def test_temporal_behavior_gnn_forward():

    samples = [
        DummySample(i)
        for i in range(16)
    ]

    builder = TemporalSequenceBuilder(
        sequence_length=8,
        sequence_stride=4,
    )

    windows = builder.build(
        samples
    )

    dataset = TemporalPyGDataset(
        windows
    )

    loader = make_temporal_dataloader(
        dataset,
        batch_size=2,
    )

    batch = next(
        iter(loader)
    )

    config = ModelConfig(
        model_type="temporal",
        node_feature_dim=8,
        edge_feature_dim=4,
        spatial_hidden_dim=32,
        temporal_hidden_dim=64,
        num_classes=3,
        sequence_length=8,
    )

    model = build_model(
        config
    )

    logits = model(
        batch
    )

    assert logits.shape == (
        2,
        3,
    )

    assert torch.isfinite(
        logits
    ).all()


def test_temporal_behavior_gnn_backward():

    samples = [
        DummySample(i)
        for i in range(16)
    ]

    builder = TemporalSequenceBuilder(
        sequence_length=8,
        sequence_stride=4,
    )

    windows = builder.build(
        samples
    )

    dataset = TemporalPyGDataset(
        windows
    )

    loader = make_temporal_dataloader(
        dataset,
        batch_size=2,
    )

    batch = next(
        iter(loader)
    )

    config = ModelConfig(
        model_type="temporal",
        node_feature_dim=8,
        edge_feature_dim=4,
        spatial_hidden_dim=32,
        temporal_hidden_dim=64,
        num_classes=3,
        sequence_length=8,
    )

    model = build_model(
        config
    )

    logits = model(
        batch
    )

    loss = logits.mean()

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
