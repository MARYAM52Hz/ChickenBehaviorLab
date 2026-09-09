from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from torch_geometric.loader import (
    DataLoader,
)

from chicken_behavior_lab.dataset import (
    DatasetFactory,
    PyGGraphDataset,
    train_validation_split,
)

from chicken_behavior_lab.models import (
    ChickenBehaviorGNN,
)

from chicken_behavior_lab.training import (
    TrainingConfig,
    Trainer,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train ChickenBehaviorGNN."
        )
    )

    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--features",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "checkpoints/"
            "chicken_behavior_gnn_best.pt"
        ),
    )

    return parser.parse_args()


def set_seed(
    seed: int,
) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def choose_device():

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    if (
        hasattr(
            torch.backends,
            "mps",
        )
        and torch.backends.mps.is_available()
    ):

        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


def main():

    args = parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=(
            args.learning_rate
        ),
        validation_fraction=(
            args.validation_fraction
        ),
        checkpoint_path=(
            args.checkpoint
        ),
    )

    config.validate()

    set_seed(
        config.random_seed
    )

    device = choose_device()

    print(
        f"Device: {device}"
    )

    # =====================================================
    # Build dataset
    # =====================================================

    factory = DatasetFactory(
        annotation_file=args.annotations,
        feature_root=args.features,
    )

    samples = factory.build()

    print(
        f"Total samples: "
        f"{len(samples)}"
    )

    if len(samples) < 2:
        raise RuntimeError(
            "At least two samples are "
            "required for training."
        )

    dataset = PyGGraphDataset(
        samples
    )

    print(
        f"Classes: {dataset.labels}"
    )

    # =====================================================
    # Split
    # =====================================================

    train_dataset, validation_dataset = (
        train_validation_split(
            dataset,
            validation_fraction=(
                config.validation_fraction
            ),
            random_seed=(
                config.random_seed
            ),
        )
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    # =====================================================
    # DataLoaders
    # =====================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # =====================================================
    # Infer feature dimensions
    # =====================================================

    first_sample = dataset[0]

    node_feature_dim = (
        first_sample.x.shape[1]
    )

    edge_feature_dim = (
        first_sample.edge_attr.shape[1]
    )

    num_classes = len(
        dataset.labels
    )

    print(
        f"Node feature dim: "
        f"{node_feature_dim}"
    )

    print(
        f"Edge feature dim: "
        f"{edge_feature_dim}"
    )

    print(
        f"Number of classes: "
        f"{num_classes}"
    )

    # =====================================================
    # Model
    # =====================================================

    model = ChickenBehaviorGNN(
        node_feature_dim=(
            node_feature_dim
        ),
        edge_feature_dim=(
            edge_feature_dim
        ),
        hidden_dim=config.hidden_dim,
        num_classes=num_classes,
        num_layers=(
            config.num_gnn_layers
        ),
        dropout=config.dropout,
    )

    # =====================================================
    # Optimizer
    # =====================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # =====================================================
    # Loss
    # =====================================================

    loss_fn = (
        torch.nn.CrossEntropyLoss()
    )

    # =====================================================
    # Trainer
    # =====================================================

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_classes=num_classes,
        device=device,
        checkpoint_path=(
            config.checkpoint_path
        ),
    )

    # =====================================================
    # Training
    # =====================================================

    history = trainer.fit(
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        epochs=config.epochs,
    )

    # =====================================================
    # Summary
    # =====================================================

    print()

    print(
        "================================"
    )

    print(
        "Training completed"
    )

    print(
        "================================"
    )

    print(
        f"Best epoch: "
        f"{history.best_epoch}"
    )

    print(
        f"Best validation Macro-F1: "
        f"{history.best_validation_f1:.4f}"
    )

    print(
        f"Checkpoint: "
        f"{config.checkpoint_path}"
    )


if __name__ == "__main__":
    main()
