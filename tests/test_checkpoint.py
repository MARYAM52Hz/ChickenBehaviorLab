from __future__ import annotations

from pathlib import Path

import torch

from chicken_behavior_lab.config.model_config import (
    ModelConfig,
)

from chicken_behavior_lab.models.model_factory import (
    build_model,
)

from chicken_behavior_lab.training.checkpoint import (
    CheckpointManager,
    CHECKPOINT_FORMAT_VERSION,
)


def create_temporal_config() -> ModelConfig:
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
    return {
        "feeding": 0,
        "walking": 1,
        "standing": 2,
    }


def test_checkpoint_roundtrip(
    tmp_path: Path,
) -> None:

    config = create_temporal_config()

    model = build_model(
        config
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    manager = CheckpointManager(
        tmp_path / "checkpoints"
    )

    path = manager.save(
        filename="test.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=5,
        model_config=config,
        label_mapping=create_label_mapping(),
        best_metric=0.91,
        train_metrics={
            "loss": 0.21,
            "accuracy": 0.92,
        },
        val_metrics={
            "loss": 0.24,
            "accuracy": 0.91,
        },
        metadata={
            "experiment": "unit_test",
        },
    )

    assert path.exists()

    checkpoint = manager.load(
        "test.pt"
    )

    assert (
        checkpoint["format_version"]
        == CHECKPOINT_FORMAT_VERSION
    )

    assert checkpoint["epoch"] == 5

    assert checkpoint["model_type"] == "temporal"

    assert checkpoint["best_metric"] == 0.91

    assert checkpoint["label_mapping"] == {
        "feeding": 0,
        "walking": 1,
        "standing": 2,
    }


def test_checkpoint_preserves_model_config(
    tmp_path: Path,
) -> None:

    config = create_temporal_config()

    model = build_model(
        config
    )

    manager = CheckpointManager(
        tmp_path / "checkpoints"
    )

    manager.save(
        filename="config.pt",
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=3,
        model_config=config,
    )

    checkpoint = manager.load(
        "config.pt"
    )

    restored_config = (
        manager.build_model_config(
            checkpoint
        )
    )

    assert restored_config == config


def test_model_can_be_reconstructed_from_checkpoint(
    tmp_path: Path,
) -> None:

    config = create_temporal_config()

    model = build_model(
        config
    )

    manager = CheckpointManager(
        tmp_path / "checkpoints"
    )

    manager.save(
        filename="reconstruct.pt",
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=1,
        model_config=config,
    )

    checkpoint = manager.load(
        "reconstruct.pt"
    )

    restored_config = (
        manager.build_model_config(
            checkpoint
        )
    )

    restored_model = build_model(
        restored_config
    )

    restored_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    restored_model.eval()

    assert isinstance(
        restored_model,
        torch.nn.Module,
    )


def test_checkpoint_restores_optimizer_state(
    tmp_path: Path,
) -> None:

    config = create_temporal_config()

    model = build_model(
        config
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    dummy_parameter = next(
        model.parameters()
    )

    loss = dummy_parameter.sum()

    loss.backward()

    optimizer.step()

    manager = CheckpointManager(
        tmp_path / "checkpoints"
    )

    manager.save(
        filename="optimizer.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=2,
        model_config=config,
    )

    checkpoint = manager.load(
        "optimizer.pt"
    )

    assert (
        checkpoint["optimizer_state_dict"]
        is not None
    )

    restored_model = build_model(
        config
    )

    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(),
        lr=1e-3,
    )

    restored_optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    assert (
        len(restored_optimizer.state)
        == len(optimizer.state)
    )


def test_checkpoint_model_state_is_identical(
    tmp_path: Path,
) -> None:

    config = create_temporal_config()

    model = build_model(
        config
    )

    manager = CheckpointManager(
        tmp_path / "checkpoints"
    )

    manager.save(
        filename="state.pt",
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=1,
        model_config=config,
    )

    checkpoint = manager.load(
        "state.pt"
    )

    saved_state = (
        checkpoint["model_state_dict"]
    )

    current_state = model.state_dict()

    assert saved_state.keys() == (
        current_state.keys()
    )

    for key in current_state:
        assert torch.equal(
            saved_state[key],
            current_state[key],
        )


def test_checkpoint_label_mapping_is_preserved(
    tmp_path: Path,
) -> None:

    config = create_temporal_config()

    model = build_model(
        config
    )

    label_mapping = {
        "feeding": 0,
        "walking": 1,
        "standing": 2,
    }

    manager = CheckpointManager(
        tmp_path / "checkpoints"
    )

    manager.save(
        filename="labels.pt",
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=1,
        model_config=config,
        label_mapping=label_mapping,
    )

    checkpoint = manager.load(
        "labels.pt"
    )

    assert (
        checkpoint["label_mapping"]
        == label_mapping
    )


def test_checkpoint_without_optimizer(
    tmp_path: Path,
) -> None:

    config = create_temporal_config()

    model = build_model(
        config
    )

    manager = CheckpointManager(
        tmp_path / "checkpoints"
    )

    manager.save(
        filename="no_optimizer.pt",
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=1,
        model_config=config,
    )

    checkpoint = manager.load(
        "no_optimizer.pt"
    )

    assert (
        checkpoint["optimizer_state_dict"]
        is None
    )

    assert (
        checkpoint["scheduler_state_dict"]
        is None
    )


def test_checkpoint_rejects_missing_required_keys(
    tmp_path: Path,
) -> None:

    manager = CheckpointManager(
        tmp_path / "checkpoints"
    )

    path = (
        tmp_path
        / "checkpoints"
        / "invalid.pt"
    )

    torch.save(
        {
            "format_version":
                CHECKPOINT_FORMAT_VERSION,
        },
        path,
    )

    try:
        manager.load(
            "invalid.pt"
        )
    except ValueError as exc:
        assert "missing required keys" in str(
            exc
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_checkpoint_path_is_created(
    tmp_path: Path,
) -> None:

    checkpoint_dir = (
        tmp_path
        / "nested"
        / "checkpoints"
    )

    manager = CheckpointManager(
        checkpoint_dir
    )

    assert checkpoint_dir.exists()

    config = create_temporal_config()

    model = build_model(
        config
    )

    path = manager.save(
        filename="nested.pt",
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=0,
        model_config=config,
    )

    assert path.exists()
