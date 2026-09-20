import torch

from chicken_behavior_lab.models.spatial_encoder import (
    SpatialGraphEncoder,
)


def test_spatial_encoder_output_shape():
    encoder = SpatialGraphEncoder(
        node_feature_dim=4,
        edge_feature_dim=2,
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
    )

    x = torch.randn(
        5,
        4,
    )

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ],
        dtype=torch.long,
    )

    edge_attr = torch.randn(
        6,
        2,
    )

    output = encoder(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    assert output.shape == (
        5,
        8,
    )
