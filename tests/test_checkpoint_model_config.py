import torch
from torch import nn

from chicken_behavior_lab.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_contains_model_configuration(
    tmp_path,
):

    model = nn.Linear(
        8,
        4,
    )

    checkpoint_path = (
        tmp_path
        / "model.pt"
    )

    model_config = {
        "node_feature_dim": 32,
        "edge_feature_dim": 8,
        "hidden_dim": 64,
        "num_classes": 4,
        "num_layers": 3,
        "dropout": 0.2,
    }

    training_config = {
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 1e-3,
    }

    label_to_index = {
        "feeding": 0,
        "standing": 1,
        "walking": 2,
        "pecking": 3,
    }

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        model_config=model_config,
        training_config=training_config,
        label_to_index=label_to_index,
    )

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    assert (
        checkpoint["model_config"]
        == model_config
    )

    assert (
        checkpoint["training_config"]
        == training_config
    )

    assert (
        checkpoint["label_to_index"]
        == label_to_index
    )
