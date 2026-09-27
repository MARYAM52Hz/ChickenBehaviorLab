```python
from __future__ import annotations

import torch

from chicken_behavior_lab.config.model_config import (
    ModelConfig,
)

from chicken_behavior_lab.models.model_factory import (
    build_model,
)

from chicken_behavior_lab.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManager,
)


def create_temporal_config() -> ModelConfig:
    """
    Create a small temporal model configuration suitable
    for unit tests.
    """

    return ModelConfig(
        model_type="temporal",
        node_feature_dim=8,
        edge_feature_dim=4,
        spatial_hidden_dim=32,
        temporal_hidden_dim=64,
        num_classes=3,
        num_gnn_layers=2,
        num_gru_layers=1,
        dropout=0.2,
        bidirectional_gru=False,
        sequence_length=8,
        sequence_stride=4,
    )


def create_label_mapping() -> dict[str, int]:
    """
    Create a deterministic behavior-label mapping.
    """

    return {
        "standing": 0,
        "walking": 1,
        "feeding": 2,
    }


def test_checkpoint_roundtrip(
    tmp_path,
):
    """
    Verify that a complete checkpoint can be saved
    and loaded successfully.
    """

    config = create_temporal_config()

    model = build_model(
        config
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    manager = CheckpointManager(
        tmp_path
    )

    label_mapping = (
        create_label_mapping()
    )

    train_metrics = {
        "loss": 0.40,
        "accuracy": 0.82,
    }

    val_metrics = {
        "loss": 0.50,
        "accuracy": 0.85,
    }

    checkpoint_path = (
        manager.save(
            filename="test.pt",
            model=model,
           
```
