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


def create_graph(
    label: int,
) -> Data:

    node_feature_dim = 4

    edge_feature_dim = 5

    num_nodes = 6

    edge_index = torch.tensor(
        [
            [
                0, 1,
                1, 2,
                3, 4,
                4, 5,
            ],
            [
                1, 0,
                2, 1,
                4, 3,
                5, 4,
            ],
        ],
        dtype=torch.long,
    )

    x = torch.randn(
        num_nodes,
        node_feature_dim,
    )

    edge_attr = torch.randn(
        edge_index.shape[1],
        edge_feature_dim,
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


def create_loader():

    graphs = [
        create_graph(0),
        create_graph(1),
        create_graph(2),
        create_graph(0),
        create_graph(1),
        create_graph(2),
    ]

    return DataLoader(
        graphs,
        batch_size=2,
        shuffle=False,
    )


def create_trainer(
    checkpoint_path=None,
):

    model = ChickenBehaviorGNN(
        node_feature_dim=4,
        edge_feature_dim=5,
        hidden_dim=16,
        num_classes=3,
        num_layers=2,
        dropout=0.1,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    loss_fn = (
        torch.nn.CrossEntropyLoss()
    )

    return Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_classes=3,
        device="cpu",
        checkpoint_path=(
            checkpoint_path
        ),
    )


def test_train_epoch():

    trainer = create_trainer()

    loader = create_loader()

    result = trainer.train_epoch(
        loader
    )

    assert result.loss >= 0

    assert (
        0.0
        <= result.accuracy
        <= 1.0
    )

    assert (
        0.0
        <= result.macro_f1
        <= 1.0
    )


def test_evaluate():

    trainer = create_trainer()

    loader = create_loader()

    result = trainer.evaluate(
        loader
    )

    assert result.loss >= 0

    assert (
        0.0
        <= result.accuracy
        <= 1.0
    )


def test_fit():

    trainer = create_trainer()

    train_loader = create_loader()

    validation_loader = (
        create_loader()
    )

    history = trainer.fit(
        train_loader=(
            train_loader
        ),
        validation_loader=(
            validation_loader
        ),
        epochs=2,
    )

    assert len(
        history.train
    ) == 2

    assert len(
        history.validation
    ) == 2

    assert (
        history.best_epoch
        is not None
    )

    assert (
        history.best_validation_f1
        is not None
    )


def test_checkpoint_saved(
    tmp_path,
):

    checkpoint_path = (
        tmp_path
        / "model.pt"
    )

    trainer = create_trainer(
        checkpoint_path
    )

    train_loader = create_loader()

    validation_loader = (
        create_loader()
    )

    trainer.fit(
        train_loader=(
            train_loader
        ),
        validation_loader=(
            validation_loader
        ),
        epochs=1,
    )

    assert (
        checkpoint_path.exists()
    )


def test_checkpoint_load(
    tmp_path,
):

    checkpoint_path = (
        tmp_path
        / "model.pt"
    )

    trainer = create_trainer(
        checkpoint_path
    )

    loader = create_loader()

    trainer.fit(
        train_loader=loader,
        validation_loader=loader,
        epochs=1,
    )

    checkpoint = (
        trainer.load_checkpoint()
    )

    assert (
        checkpoint["epoch"]
        == 1
    )

    assert (
        "model_state_dict"
        in checkpoint
    )

    assert (
        "optimizer_state_dict"
        in checkpoint
    )
