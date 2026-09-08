from __future__ import annotations

import torch

from torch_geometric.data import (
    Data,
)

from torch_geometric.loader import (
    DataLoader,
)

from chicken_behavior_lab.models import (
    ChickenBehaviorGNN,
)

from chicken_behavior_lab.training import (
    Trainer,
)


NUM_CLASSES = 5

NODE_FEATURE_DIM = 7

EDGE_FEATURE_DIM = 8


def create_demo_graph(
    label: int,
) -> Data:

    num_frames = 10

    num_keypoints = 13

    num_nodes = (
        num_frames
        * num_keypoints
    )

    num_edges = 400

    x = torch.randn(
        num_nodes,
        NODE_FEATURE_DIM,
    )

    edge_index = torch.randint(
        low=0,
        high=num_nodes,
        size=(
            2,
            num_edges,
        ),
        dtype=torch.long,
    )

    edge_attr = torch.randn(
        num_edges,
        EDGE_FEATURE_DIM,
    )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(
            label,
            dtype=torch.long,
        ),
    )


def main() -> None:

    torch.manual_seed(
        42
    )

    dataset = [
        create_demo_graph(
            index % NUM_CLASSES
        )
        for index in range(
            50
        )
    ]

    train_dataset = (
        dataset[:40]
    )

    validation_dataset = (
        dataset[40:]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=8,
        shuffle=False,
    )

    model = ChickenBehaviorGNN(
        node_feature_dim=(
            NODE_FEATURE_DIM
        ),
        edge_feature_dim=(
            EDGE_FEATURE_DIM
        ),
        hidden_dim=64,
        num_classes=NUM_CLASSES,
        num_layers=3,
        dropout=0.2,
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
    )

    loss_fn = (
        torch.nn.CrossEntropyLoss()
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_classes=NUM_CLASSES,
        device=device,
        checkpoint_path=(
            "checkpoints/"
            "demo_best.pt"
        ),
    )

    history = trainer.fit(
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        epochs=5,
    )

    print()

    print(
        "Training finished."
    )

    print(
        "Best epoch:",
        history.best_epoch,
    )

    print(
        "Best validation Macro-F1:",
        history.best_validation_f1,
    )


if __name__ == "__main__":
    main()
