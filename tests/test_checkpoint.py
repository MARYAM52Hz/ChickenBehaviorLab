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

    checkpoint_path = manager.save(
        filename="test.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=5,
        model_config=config,
        label_mapping=label_mapping,
        best_metric=0.85,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        metadata={
            "experiment": "checkpoint_test",
        },
    )

    assert checkpoint_path.exists()

    checkpoint = manager.load(
        checkpoint_path
    )

    assert isinstance(
        checkpoint,
        dict,
    )

    assert (
        checkpoint["format_version"]
        == CHECKPOINT_FORMAT_VERSION
    )

    assert checkpoint["epoch"] == 5

    assert (
        checkpoint["model_type"]
        == "temporal"
    )

    assert (
        checkpoint["label_mapping"]
        == label_mapping
    )

    assert (
        checkpoint["best_metric"]
        == 0.85
    )

    assert (
        checkpoint["train_metrics"]
        == train_metrics
    )

    assert (
        checkpoint["val_metrics"]
        == val_metrics
    )

    assert (
        checkpoint["metadata"]["experiment"]
        == "checkpoint_test"
    )

    assert (
        checkpoint["optimizer_state_dict"]
        is not None
    )

    assert (
        checkpoint["model_state_dict"]
        is not None
    )


def test_checkpoint_preserves_model_config(
    tmp_path,
):
    """
    Verify that all temporal model configuration
    parameters survive the checkpoint roundtrip.
    """

    config = create_temporal_config()

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

    manager.save(
        filename="config_test.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=3,
        model_config=config,
        label_mapping=create_label_mapping(),
    )

    checkpoint = manager.load(
        "config_test.pt"
    )

    restored_config = (
        manager.build_model_config(
            checkpoint
        )
    )

    assert (
        restored_config.model_type
        == "temporal"
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
        restored_config.spatial_hidden_dim
        == 32
    )

    assert (
        restored_config.temporal_hidden_dim
        == 64
    )

    assert (
        restored_config.num_classes
        == 3
    )

    assert (
        restored_config.num_gnn_layers
        == 2
    )

    assert (
        restored_config.num_gru_layers
        == 1
    )

    assert (
        restored_config.dropout
        == 0.2
    )

    assert (
        restored_config.bidirectional_gru
        is False
    )

    assert (
        restored_config.sequence_length
        == 8
    )

    assert (
        restored_config.sequence_stride
        == 4
    )


def test_model_can_be_reconstructed_from_checkpoint(
    tmp_path,
):
    """
    Verify that the model can be reconstructed exclusively
    from the configuration stored in the checkpoint.
    """

    original_config = (
        create_temporal_config()
    )

    original_model = build_model(
        original_config
    )

    optimizer = torch.optim.AdamW(
        original_model.parameters(),
        lr=1e-3,
    )

    manager = CheckpointManager(
        tmp_path
    )

    manager.save(
        filename="model.pt",
        model=original_model,
        optimizer=optimizer,
        scheduler=None,
        epoch=3,
        model_config=original_config,
        label_mapping=create_label_mapping(),
    )

    checkpoint = manager.load(
        "model.pt"
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
        checkpoint[
            "model_state_dict"
        ]
    )

    original_state = (
        original_model.state_dict()
    )

    restored_state = (
        restored_model.state_dict()
    )

    assert (
        original_state.keys()
        == restored_state.keys()
    )

    for key in original_state:
        assert torch.equal(
            original_state[key],
            restored_state[key],
        )


def test_checkpoint_restores_optimizer_state(
    tmp_path,
):
    """
    Verify that optimizer state is preserved.

    A single optimization step is performed first so that
    AdamW has non-empty internal state.
    """

    config = create_temporal_config()

    model = build_model(
        config
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    # Create optimizer state.
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    loss = sum(
        parameter.square().mean()
        for parameter in trainable_parameters
    )

    loss.backward()

    optimizer.step()

    manager = CheckpointManager(
        tmp_path
    )

    manager.save(
        filename="optimizer_test.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=1,
        model_config=config,
        label_mapping=create_label_mapping(),
    )

    checkpoint = manager.load(
        "optimizer_test.pt"
    )

    optimizer_state = (
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    assert optimizer_state is not None

    assert (
        "state"
        in optimizer_state
    )

    assert (
        "param_groups"
        in optimizer_state
    )

    assert len(
        optimizer_state["state"]
    ) > 0


def test_checkpoint_model_state_is_identical(
    tmp_path,
):
    """
    Verify that model parameters stored in the checkpoint
    exactly match the parameters of the original model.
    """

    config = create_temporal_config()

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

    manager.save(
        filename="state_test.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=2,
        model_config=config,
        label_mapping=create_label_mapping(),
    )

    checkpoint = manager.load(
        "state_test.pt"
    )

    saved_state = (
        checkpoint[
            "model_state_dict"
        ]
    )

    current_state = (
        model.state_dict()
    )

    assert (
        saved_state.keys()
        == current_state.keys()
    )

    for key in current_state:

        assert torch.equal(
            saved_state[key],
            current_state[key],
        )


def test_checkpoint_label_mapping_is_preserved(
    tmp_path,
):
    """
    Verify that behavior label mappings are preserved
    exactly and without reordering.
    """

    config = create_temporal_config()

    model = build_model(
        config
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    label_mapping = {
        "feeding": 0,
        "walking": 1,
        "standing": 2,
    }

    manager = CheckpointManager(
        tmp_path
    )

    manager.save(
        filename="labels_test.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=1,
        model_config=config,
        label_mapping=label_mapping,
    )

    checkpoint = manager.load(
        "labels_test.pt"
    )

    assert (
        checkpoint["label_mapping"]
        == label_mapping
    )

    assert (
        checkpoint["label_mapping"]["feeding"]
        == 0
    )

    assert (
        checkpoint["label_mapping"]["walking"]
        == 1
    )

    assert (
        checkpoint["label_mapping"]["standing"]
        == 2
    )


def test_checkpoint_without_optimizer(
    tmp_path,
):
    """
    Verify that checkpoints can also be created without
    an optimizer, which is useful for inference-only models.
    """

    config = create_temporal_config()

    model = build_model(
        config
    )

    manager = CheckpointManager(
        tmp_path
    )

    manager.save(
        filename="inference.pt",
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=0,
        model_config=config,
        label_mapping=create_label_mapping(),
    )

    checkpoint = manager.load(
        "inference.pt"
    )

    assert (
        checkpoint[
            "optimizer_state_dict"
        ]
        is None
    )

    assert (
        checkpoint[
            "scheduler_state_dict"
        ]
        is None
    )

    assert (
        checkpoint[
            "model_state_dict"
        ]
        is not None
    )


def test_checkpoint_rejects_missing_required_keys(
    tmp_path,
):
    """
    Verify that malformed checkpoints are rejected.
    """

    manager = CheckpointManager(
        tmp_path
    )

    invalid_checkpoint = {
        "format_version": 1,
        "epoch": 1,
    }

    checkpoint_path = (
        tmp_path
        / "invalid.pt"
    )

    torch.save(
        invalid_checkpoint,
        checkpoint_path,
    )

    try:
        manager.load(
            checkpoint_path
        )
    except ValueError as error:
        message = str(error)

        assert (
            "missing required keys"
            in message
        )

    else:
        raise AssertionError(
            "Invalid checkpoint was accepted."
        )


def test_checkpoint_path_is_created(
    tmp_path,
):
    """
    Verify that CheckpointManager creates its directory
    automatically.
    """

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

    manager.save(
        filename="nested_test.pt",
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=0,
        model_config=config,
        label_mapping=create_label_mapping(),
    )

    assert (
        checkpoint_dir
        / "nested_test.pt"
    ).exists()
