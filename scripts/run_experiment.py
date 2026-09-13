from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chicken_behavior_lab.training import (
    TrainingConfig,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run a complete ChickenBehaviorLab "
            "training experiment."
        )
    )

    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help=(
            "Path to the behavior annotation file."
        ),
    )

    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help=(
            "Path to the extracted feature directory."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay.",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension of the GNN.",
    )

    parser.add_argument(
        "--num-gnn-layers",
        type=int,
        default=3,
        help="Number of GNN layers.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability.",
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Validation fraction.",
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Test fraction.",
    )

    parser.add_argument(
        "--split-group",
        type=str,
        choices=[
            "video",
            "track",
        ],
        default="video",
        help=(
            "Group level used for "
            "dataset splitting."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "checkpoints/"
            "chicken_behavior_gnn_best.pt"
        ),
        help="Checkpoint output path.",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=(
            "results/"
            "experiments"
        ),
        help="Directory for experiment results.",
    )

    return parser.parse_args()


def set_random_seed(
    seed: int,
) -> None:
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make CUDA behavior as deterministic as
    # practical for reproducible experiments.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device() -> torch.device:
    """
    Select the best available device.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(
            torch.backends,
            "mps",
        )
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def create_training_config(
    args: argparse.Namespace,
) -> TrainingConfig:
    """
    Build TrainingConfig from command-line arguments.
    """

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        dropout=args.dropout,
        validation_fraction=(
            args.validation_fraction
        ),
        test_fraction=(
            args.test_fraction
        ),
        split_group=args.split_group,
        random_seed=args.seed,
        checkpoint_path=args.checkpoint,
    )

    config.validate()

    return config


def create_experiment_directory(
    results_dir: str,
) -> Path:
    """
    Create a unique directory for the experiment.
    """

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    experiment_dir = (
        Path(results_dir)
        / timestamp
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return experiment_dir


def save_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    """
    Save a dictionary as formatted JSON.
    """

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def main() -> None:
    """
    Run one complete experiment.
    """

    args = parse_args()

    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------

    set_random_seed(
        args.seed
    )

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = choose_device()

    print()
    print("=" * 70)
    print("ChickenBehaviorLab Experiment")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    print(
        f"Random seed: {args.seed}"
    )

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------

    config = create_training_config(
        args
    )

    print()
    print("Training configuration:")

    for key, value in config.to_dict().items():

        print(
            f"  {key}: {value}"
        )

    # -----------------------------------------------------
    # Experiment directory
    # -----------------------------------------------------

    experiment_dir = (
        create_experiment_directory(
            args.results_dir
        )
    )

    print()
    print(
        f"Experiment directory:\n"
        f"  {experiment_dir}"
    )

    # -----------------------------------------------------
    # Save initial configuration
    # -----------------------------------------------------

    experiment_metadata = {
        "created_at": datetime.now().isoformat(),
        "device": str(device),
        "seed": args.seed,
        "annotations": str(
            Path(args.annotations)
        ),
        "features": str(
            Path(args.features)
        ),
        "checkpoint": str(
            Path(args.checkpoint)
        ),
        "training_config": config.to_dict(),
    }

    save_json(
        experiment_dir
        / "experiment_config.json",
        experiment_metadata,
    )

    # -----------------------------------------------------
    # Import project components
    # -----------------------------------------------------

    from chicken_behavior_lab.dataset import (
        DatasetFactory,
        PyGGraphDataset,
        group_train_validation_test_split,
    )

    from chicken_behavior_lab.models import (
        ChickenBehaviorGNN,
    )

    from chicken_behavior_lab.training import (
        ClassificationLoss,
        Trainer,
    )

    from torch_geometric.loader import DataLoader

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    print()
    print(
        "Building dataset..."
    )

    factory = DatasetFactory(
        annotation_file=args.annotations,
        feature_root=args.features,
    )

    samples = factory.build()

    if len(samples) < 3:
        raise RuntimeError(
            "Not enough samples/groups for "
            "train/validation/test splitting."
        )

    print(
        f"Total samples: {len(samples)}"
    )

    # -----------------------------------------------------
    # Group-aware split
    # -----------------------------------------------------

    print()
    print(
        "Creating group-aware split..."
    )

    split = (
        group_train_validation_test_split(
            samples,
            validation_fraction=(
                config.validation_fraction
            ),
            test_fraction=(
                config.test_fraction
            ),
            group_by=config.split_group,
            random_seed=config.random_seed,
        )
    )

    print(
        f"Train samples: "
        f"{len(split.train)}"
    )

    print(
        f"Validation samples: "
        f"{len(split.validation)}"
    )

    print(
        f"Test samples: "
        f"{len(split.test)}"
    )

    # -----------------------------------------------------
    # PyG datasets
    # -----------------------------------------------------

    print()
    print(
        "Creating PyTorch Geometric datasets..."
    )

    train_dataset = PyGGraphDataset(
        split.train
    )

    validation_dataset = PyGGraphDataset(
        split.validation,
        label_to_index=(
            train_dataset.label_to_index
        ),
    )

    test_dataset = PyGGraphDataset(
        split.test,
        label_to_index=(
            train_dataset.label_to_index
        ),
    )

    print(
        f"Number of classes: "
        f"{len(train_dataset.label_to_index)}"
    )

    print(
        "Label mapping:"
    )

    for (
        label,
        index,
    ) in train_dataset.label_to_index.items():

        print(
            f"  {index}: {label}"
        )

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # -----------------------------------------------------
    # Determine feature dimensions
    # -----------------------------------------------------

    first_sample = train_dataset[0]

    if not hasattr(
        first_sample,
        "x",
    ):
        raise AttributeError(
            "Graph sample does not contain "
            "node feature matrix 'x'."
        )

    node_feature_dim = int(
        first_sample.x.size(-1)
    )

    if (
        not hasattr(
            first_sample,
            "edge_attr",
        )
        or first_sample.edge_attr is None
    ):
        raise AttributeError(
            "Graph sample does not contain "
            "'edge_attr'."
        )

    edge_feature_dim = int(
        first_sample.edge_attr.size(-1)
    )

    num_classes = len(
        train_dataset.label_to_index
    )

    print()
    print(
        "Graph dimensions:"
    )

    print(
        f"  Node features: {node_feature_dim}"
    )

    print(
        f"  Edge features: {edge_feature_dim}"
    )

    print(
        f"  Classes: {num_classes}"
    )

    # -----------------------------------------------------
    # Model configuration
    # -----------------------------------------------------

    model_config = {
        "node_feature_dim": node_feature_dim,
        "edge_feature_dim": edge_feature_dim,
        "hidden_dim": config.hidden_dim,
        "num_classes": num_classes,
        "num_layers": config.num_gnn_layers,
        "dropout": config.dropout,
    }

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    print()
    print(
        "Building model..."
    )

    model = ChickenBehaviorGNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=config.hidden_dim,
        num_classes=num_classes,
        num_layers=config.num_gnn_layers,
        dropout=config.dropout,
    )

    model.to(device)

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    loss_function = ClassificationLoss()

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # -----------------------------------------------------
    # Trainer
    # -----------------------------------------------------

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        checkpoint_path=(
            config.checkpoint_path
        ),
        model_config=model_config,
        training_config=config.to_dict(),
        label_to_index=(
            train_dataset.label_to_index
        ),
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    history = trainer.fit(
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=config.epochs,
    )

    # -----------------------------------------------------
    # Save training history
    # -----------------------------------------------------

    history_data = {
        "train_loss": (
            history.train_loss
        ),
        "validation_loss": (
            history.validation_loss
        ),
        "validation_accuracy": (
            history.validation_accuracy
        ),
        "validation_macro_f1": (
            history.validation_macro_f1
        ),
    }

    save_json(
        experiment_dir
        / "training_history.json",
        history_data,
    )

    # -----------------------------------------------------
    # Load best checkpoint
    # -----------------------------------------------------

    from chicken_behavior_lab.training import (
        load_checkpoint,
    )

    print()
    print(
        "Loading best checkpoint..."
    )

    checkpoint = load_checkpoint(
        config.checkpoint_path,
        map_location=device,
    )

    # -----------------------------------------------------
    # Rebuild best model
    # -----------------------------------------------------

    checkpoint_model_config = (
        checkpoint["model_config"]
    )

    best_model = ChickenBehaviorGNN(
        node_feature_dim=int(
            checkpoint_model_config[
                "node_feature_dim"
            ]
        ),
        edge_feature_dim=int(
            checkpoint_model_config[
                "edge_feature_dim"
            ]
        ),
        hidden_dim=int(
            checkpoint_model_config[
                "hidden_dim"
            ]
        ),
        num_classes=int(
            checkpoint_model_config[
                "num_classes"
            ]
        ),
        num_layers=int(
            checkpoint_model_config[
                "num_layers"
            ]
        ),
        dropout=float(
            checkpoint_model_config[
                "dropout"
            ]
        ),
    )

    best_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    best_model.to(device)

    # -----------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------

    from chicken_behavior_lab.training import (
        Evaluator,
        format_classification_report,
    )

    print()
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    evaluator = Evaluator(
        model=best_model,
        device=device,
        num_classes=num_classes,
    )

    result = evaluator.evaluate(
        test_loader
    )

    # -----------------------------------------------------
    # Print report
    # -----------------------------------------------------

    report = (
        format_classification_report(
            metrics=result.metrics,
            class_names=(
                train_dataset.labels
            ),
        )
    )

    print()
    print(report)

    # -----------------------------------------------------
    # Save final metrics
    # -----------------------------------------------------

    final_metrics = {
        "accuracy": (
            result.metrics.accuracy
        ),
        "macro_precision": (
            result.metrics.macro_precision
        ),
        "macro_recall": (
            result.metrics.macro_recall
        ),
        "macro_f1": (
            result.metrics.macro_f1
        ),
        "weighted_f1": (
            result.metrics.weighted_f1
        ),
        "confusion_matrix": (
            result.metrics.confusion_matrix.tolist()
        ),
    }

    save_json(
        experiment_dir
        / "test_metrics.json",
        final_metrics,
    )

    # -----------------------------------------------------
    # Save final summary
    # -----------------------------------------------------

    summary = {
        "experiment_directory": str(
            experiment_dir
        ),
        "checkpoint": str(
            Path(config.checkpoint_path)
        ),
        "best_epoch": checkpoint.get(
            "epoch"
        ),
        "best_validation_f1": checkpoint.get(
            "validation_f1"
        ),
        "test_accuracy": (
            result.metrics.accuracy
        ),
        "test_macro_f1": (
            result.metrics.macro_f1
        ),
        "test_weighted_f1": (
            result.metrics.weighted_f1
        ),
    }

    save_json(
        experiment_dir
        / "summary.json",
        summary,
    )

    # -----------------------------------------------------
    # Complete
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print(
        "EXPERIMENT COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print()
    print(
        f"Results saved to:"
    )

    print(
        f"  {experiment_dir}"
    )

    print()
    print(
        f"Best validation F1: "
        f"{checkpoint.get('validation_f1', 'N/A')}"
    )

    print(
        f"Test accuracy: "
        f"{result.metrics.accuracy:.4f}"
    )

    print(
        f"Test Macro-F1: "
        f"{result.metrics.macro_f1:.4f}"
    )

    print()


if __name__ == "__main__":
    main()
