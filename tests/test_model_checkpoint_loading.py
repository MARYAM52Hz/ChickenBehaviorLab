import torch

from chicken_behavior_lab.models import (
    ChickenBehaviorGNN,
)

from chicken_behavior_lab.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


def test_model_can_be_reconstructed_from_checkpoint(
    tmp_path,
):

    checkpoint_path = (
        tmp_path
        / "gnn.pt"
    )

    model_config = {
        "node_feature_dim": 32,
        "edge_feature_dim": 8,
        "hidden_dim": 64,
        "num_classes": 4,
        "num_layers": 3,
        "dropout": 0.2,
    }

    model = ChickenBehaviorGNN(
        node_feature_dim=(
            model_config[
                "node_feature_dim"
            ]
        ),
        edge_feature_dim=(
            model_config[
                "edge_feature_dim"
            ]
        ),
        hidden_dim=(
            model_config[
                "hidden_dim"
            ]
        ),
        num_classes=(
            model_config[
                "num_classes"
            ]
        ),
        num_layers=(
            model_config[
                "num_layers"
            ]
        ),
        dropout=(
            model_config[
                "dropout"
            ]
        ),
    )

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        model_config=model_config,
        training_config={
            "epochs": 10,
        },
        label_to_index={
            "feeding": 0,
            "standing": 1,
            "walking": 2,
            "pecking": 3,
        },
    )

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    loaded_config = checkpoint[
        "model_config"
    ]

    reconstructed_model = (
        ChickenBehaviorGNN(
            node_feature_dim=(
                loaded_config[
                    "node_feature_dim"
                ]
            ),
            edge_feature_dim=(
                loaded_config[
                    "edge_feature_dim"
                ]
            ),
            hidden_dim=(
                loaded_config[
                    "hidden_dim"
                ]
            ),
            num_classes=(
                loaded_config[
                    "num_classes"
                ]
            ),
            num_layers=(
                loaded_config[
                    "num_layers"
                ]
            ),
            dropout=(
                loaded_config[
                    "dropout"
                ]
            ),
        )
    )

    reconstructed_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    original_state = (
        model.state_dict()
    )

    reconstructed_state = (
        reconstructed_model.state_dict()
    )

    assert (
        original_state.keys()
        == reconstructed_state.keys()
    )

    for key in original_state:

        assert torch.equal(
            original_state[key],
            reconstructed_state[key],
        )
