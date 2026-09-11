from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from chicken_behavior_lab.dataset import (
    DatasetFactory,
    PyGGraphDataset,
    group_train_validation_test_split,
)

from chicken_behavior_lab.models import (
    ChickenBehaviorGNN,
)

from chicken_behavior_lab.training import (
    Evaluator,
    format_classification_report,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ChickenBehaviorGNN "
            "on the held-out test set."
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
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--split-group",
        type=str,
        choices=[
            "video",
            "track",
        ],
        default="video",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


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

    device = choose_device()

    print(
        f"Evaluation device: {device}"
    )

    # =====================================================
    # Load samples
    # =====================================================

    factory = DatasetFactory(
        annotation_file=args.annotations,
        feature_root=args.features,
    )

    samples = factory.build()

    if len(samples) < 3:
        raise RuntimeError(
            "At least three groups are required "
            "for train/validation/test evaluation."
        )

    # =====================================================
    # Recreate exactly the same split
    # =====================================================

    split = (
        group_train_validation_test_split(
            samples,
            validation_fraction=(
                args.validation_fraction
            ),
            test_fraction=(
                args.test_fraction
            ),
            group_by=args.split_group,
            random_seed=args.seed,
        )
    )

    # =====================================================
    # Build training dataset first
    # =====================================================

    train_dataset = PyGGraphDataset(
        split.train
    )

    test_dataset = PyGGraphDataset(
        split.test,
        label_to_index=(
            train_dataset.label_to_index
        ),
    )

    print(
        f"Classes: "
        f"{train_dataset.labels}"
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    # =====================================================
    # DataLoader
    # =====================================================

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # =====================================================
    # Infer dimensions
    # =====================================================

    first_graph = train_dataset[0]

    node_feature_dim = (
        first_graph.x.shape[1]
    )

    edge_feature_dim = (
        first_graph.edge_attr.shape[1]
    )

    num_classes = len(
        train_dataset.labels
    )

    # =====================================================
    # Model
    # =====================================================

    model = ChickenBehaviorGNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=64,
        num_classes=num_classes,
        num_layers=3,
        dropout=0.2,
    )

    # =====================================================
    # Load checkpoint
    # =====================================================

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(
        checkpoint,
        dict,
    ) and "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    # =====================================================
    # Evaluation
    # =====================================================

    evaluator = Evaluator(
        model=model,
        device=device,
        num_classes=num_classes,
    )

    result = evaluator.evaluate(
        test_loader
    )

    # =====================================================
    # Report
    # =====================================================

    report = (
        format_classification_report(
            metrics=result.metrics,
            class_names=train_dataset.labels,
        )
    )

    print()

    print(report)

    print()

    print(
        "Confusion Matrix"
    )

    print(
        result.metrics.confusion_matrix
    )


if __name__ == "__main__":
    main()
