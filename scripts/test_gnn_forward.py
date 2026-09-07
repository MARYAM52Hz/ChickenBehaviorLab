import torch

from chicken_behavior_lab.models import (
    ChickenBehaviorGNN,
)


NUM_NODES = 390
NUM_EDGES = 1474

NODE_FEATURE_DIM = 7
EDGE_FEATURE_DIM = 8

NUM_CLASSES = 6


def main():

    x = torch.randn(
        NUM_NODES,
        NODE_FEATURE_DIM,
    )

    edge_index = torch.randint(
        0,
        NUM_NODES,
        (
            2,
            NUM_EDGES,
        ),
    )

    edge_attr = torch.randn(
        NUM_EDGES,
        EDGE_FEATURE_DIM,
    )

    model = ChickenBehaviorGNN(
        node_feature_dim=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        hidden_dim=64,
        num_classes=NUM_CLASSES,
    )

    logits = model(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    print(
        "Node features:",
        x.shape,
    )

    print(
        "Edge index:",
        edge_index.shape,
    )

    print(
        "Edge features:",
        edge_attr.shape,
    )

    print(
        "Logits:",
        logits.shape,
    )

    print(
        "Predicted class:",
        logits.argmax(
            dim=1
        ).item(),
    )


if __name__ == "__main__":
    main()
