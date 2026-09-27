from __future__ import annotations

import torch

from chicken_behavior_lab.config.model_config import (
    ModelConfig,
)

from chicken_behavior_lab.models.model_factory import (
    build_model,
)

from chicken_behavior_lab.training.checkpoint import (
    CheckpointManager,
)


def test_checkpoint_roundtrip(
    tmp_path,
):

    config = ModelConfig(
        model_type="temporal",
        node_feature_dim=8,
        edge_feature_dim=4,
        spatial_hidden_dim=32,
        temporal_hidden_dim=64,
        num_classes=3,
        sequence_length=8,
        sequence_stride=4,
    )

    model = build_model(
        config
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    manager = CheckpointManager(
        tmp_path
    )

    path = manager.save(
        filename="test.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=5,
        model_config=config,
        label_mapping={
            "standing": 0,
            "walking": 1,
            "feeding": 2,
        },
        best_metric=0.85,
        train_metrics={
            "loss": 0.4,
            "accuracy": 0.82,
        },
        val_metrics={
            "loss": 0.5,
            "accuracy": 0.85,
        },
    )

    assert path.exists()

    checkpoint = manager.load(
        path
    )

    assert checkpoint[
        "epoch"
    ] == 5

    assert checkpoint[
        "model_type"
    ] == "temporal"

    assert checkpoint[
        "label_mapping"
    ] == {
        "standing": 0,
        "walking": 1,
        "feeding": 2,
    }

    restored_config = (
        manager.build_model_config(
            checkpoint
        )
    )

    assert (
        restored_config.node_feature_dim
        == 8
    )

    assert (
        restored_config.edge_feature_dim
        == 4
    )

    assert (
        restored_config.sequence_length
        == 8
    )
