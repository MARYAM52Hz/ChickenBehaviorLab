import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from chicken_behavior_lab.models import (
    ChickenBehaviorGNN,
)


def create_graph():

    num_nodes = 6
    num_edges = 10

    node_feature_dim = 4
    edge_feature_dim = 5

    x = torch.randn(
        num_nodes,
        node_feature_dim,
    )

    edge_index = torch.tensor(
        [
            [
                0, 1, 2, 3, 4,
                1, 0, 3, 2, 5,
            ],
            [
                1, 0, 3, 2, 5,
                0, 1, 2, 3, 4,
            ],
        ],
        dtype=torch.long,
    )

    edge_attr = torch.randn(
        num_edges,
        edge_feature_dim,
    )

    y = torch.tensor(
        2,
        dtype=torch.long,
    )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )


def test_model_output_shape():

    data = create_graph()

    model = ChickenBehaviorGNN(
        node_feature_dim=4,
        edge_feature_dim=5,
        hidden_dim=32,
        num_classes=6,
    )

    logits = model(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )

    assert logits.shape == (
        1,
        6,
    )


def test_model_batching():

    data_1 = create_graph()
    data_2 = create_graph()

    loader = DataLoader(
        [
            data_1,
            data_2,
        ],
        batch_size=2,
    )

    batch = next(
        iter(loader)
    )

    model = ChickenBehaviorGNN(
        node_feature_dim=4,
        edge_feature_dim=5,
        hidden_dim=32,
        num_classes=6,
    )

    logits = model(
        x=batch.x,
        edge_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        batch=batch.batch,
    )

    assert logits.shape == (
        2,
        6,
    )


def test_model_backward():

    data = create_graph()

    model = ChickenBehaviorGNN(
        node_feature_dim=4,
        edge_feature_dim=5,
        hidden_dim=32,
        num_classes=6,
    )

    logits = model(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )

    target = torch.tensor(
        [2],
        dtype=torch.long,
    )

    loss = torch.nn.functional.cross_entropy(
        logits,
        target,
    )

    loss.backward()

    has_gradient = False

    for parameter in model.parameters():

        if parameter.grad is not None:

            has_gradient = True

            break

    assert has_gradient
