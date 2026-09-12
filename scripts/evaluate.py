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
    load_checkpoint,
)


def parse_args():
    """
    Parse command-line arguments.
    """

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
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to the trained model checkpoint."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Batch size used during test evaluation."
        ),
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of groups used for validation."
        ),
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of groups used for testing."
        ),
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
            "Group level used to split the dataset."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed used for dataset splitting."
        ),
    )

    return parser.parse_args()


def choose_device() -> torch.device:
    """
    Select the best available computation device.
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


def build_dataset_split(
    args,
):
    """
    Build the dataset and reproduce the
    train/validation/test split.

    The same split configuration used during
    training must be supplied during evaluation.
    """

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

    return split


def build_test_dataset(
    split,
):
    """
    Build the PyG datasets.

    The training dataset is constructed first because
    it defines the canonical label-to-index mapping.
    """

    train_dataset = PyGGraphDataset(
        split.train
    )

    test_dataset = PyGGraphDataset(
        split.test,
        label_to_index=(
            train_dataset.label_to_index
        ),
    )

    return (
        train_dataset,
        test_dataset,
    )


def build_model_from_checkpoint(
    checkpoint: dict,
    device: torch.device,
) -> ChickenBehaviorGNN:
    """
    Reconstruct ChickenBehaviorGNN using the
    architecture configuration stored in the checkpoint.
    """

    if "model_config" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain "
            "'model_config'. "
            "The model architecture cannot be "
            "reconstructed safely."
        )

    model_config = checkpoint[
        "model_config"
    ]

    required_keys = [
        "node_feature_dim",
        "edge_feature_dim",
        "hidden_dim",
        "num_classes",
        "num_layers",
        "dropout",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in model_config
    ]

    if missing_keys:
        raise KeyError(
            "The checkpoint model_config is missing "
            f"the following keys: {missing_keys}"
        )

    model = ChickenBehaviorGNN(
        node_feature_dim=int(
            model_config[
                "node_feature_dim"
            ]
        ),
        edge_feature_dim=int(
            model_config[
                "edge_feature_dim"
            ]
        ),
        hidden_dim=int(
            model_config[
                "hidden_dim"
            ]
        ),
        num_classes=int(
            model_config[
                "num_classes"
            ]
        ),
        num_layers=int(
            model_config[
                "num_layers"
            ]
        ),
        dropout=float(
            model_config[
                "dropout"
            ]
        ),
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(device)

    return model


def verify_label_mapping(
    checkpoint: dict,
    train_dataset: PyGGraphDataset,
) -> None:
    """
    Verify that the label mapping stored in the
    checkpoint is compatible with the current dataset.
    """

    if "label_to_index" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain "
            "'label_to_index'."
        )

    checkpoint_mapping = checkpoint[
        "label_to_index"
    ]

    current_mapping = (
        train_dataset.label_to_index
    )

    if checkpoint_mapping != current_mapping:
        raise ValueError(
            "The label mapping stored in the checkpoint "
            "does not match the current training dataset.\n\n"
            f"Checkpoint mapping:\n"
            f"{checkpoint_mapping}\n\n"
            f"Current mapping:\n"
            f"{current_mapping}\n\n"
            "Use the same dataset version and label "
            "configuration that were used during training."
        )


def print_checkpoint_information(
    checkpoint: dict,
) -> None:
    """
    Print metadata stored in the checkpoint.
    """

    print()
    print("=" * 70)
    print("CHECKPOINT INFORMATION")
    print("=" * 70)

    if "epoch" in checkpoint:
        print(
            f"Best epoch: "
            f"{checkpoint['epoch']}"
        )

    if "train_loss" in checkpoint:
        print(
            f"Train loss: "
            f"{checkpoint['train_loss']:.6f}"
        )

    if "validation_loss" in checkpoint:
        print(
            f"Validation loss: "
            f"{checkpoint['validation_loss']:.6f}"
        )

    if "validation_f1" in checkpoint:
        print(
            f"Validation Macro-F1: "
            f"{checkpoint['validation_f1']:.6f}"
        )

    if "model_config" in checkpoint:
        print()
        print("Model configuration:")

        for key, value in checkpoint[
            "model_config"
        ].items():

            print(
                f"  {key}: {value}"
            )

    if "training_config" in checkpoint:
        print()
        print("Training configuration:")

        for key, value in checkpoint[
            "training_config"
        ].items():

            print(
                f"  {key}: {value}"
            )

    print("=" * 70)
    print()


def main():
    """
    Main evaluation entry point.
    """

    args = parse_args()

    # =====================================================
    # Device
    # =====================================================

    device = choose_device()

    print(
        f"Evaluation device: {device}"
    )

    # =====================================================
    # Validate arguments
    # =====================================================

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be greater than zero."
        )

    if not (
        0.0
        <= args.validation_fraction
        < 1.0
    ):
        raise ValueError(
            "validation-fraction must be "
            "between 0 and 1."
        )

    if not (
        0.0
        <= args.test_fraction
        < 1.0
    ):
        raise ValueError(
            "test-fraction must be "
            "between 0 and 1."
        )

    if (
        args.validation_fraction
        + args.test_fraction
        >= 1.0
    ):
        raise ValueError(
            "validation-fraction + test-fraction "
            "must be less than 1."
        )

    # =====================================================
    # Check checkpoint path
    # =====================================================

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint file was not found:\n"
            f"{checkpoint_path}"
        )

    # =====================================================
    # Load checkpoint
    # =====================================================

    print(
        "Loading checkpoint..."
    )

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        map_location=device,
    )

    print(
        "Checkpoint loaded successfully."
    )

    print_checkpoint_information(
        checkpoint
    )

    # =====================================================
    # Recreate dataset split
    # =====================================================

    print(
        "Building dataset..."
    )

    split = build_dataset_split(
        args
    )

    print(
        "Dataset split created."
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

    # =====================================================
    # Build datasets
    # =====================================================

    (
        train_dataset,
        test_dataset,
    ) = build_test_dataset(
        split
    )

    print()
    print(
        "Classes:"
    )

    for (
        class_name,
        class_index,
    ) in train_dataset.label_to_index.items():

        print(
            f"  {class_index}: "
            f"{class_name}"
        )

    # =====================================================
    # Verify checkpoint label mapping
    # =====================================================

    verify_label_mapping(
        checkpoint=checkpoint,
        train_dataset=train_dataset,
    )

    # =====================================================
    # Build test DataLoader
    # =====================================================

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print()
    print(
        f"Test DataLoader created "
        f"with {len(test_dataset)} samples."
    )

    # =====================================================
    # Build model from checkpoint
    # =====================================================

    print()
    print(
        "Reconstructing model "
        "from checkpoint..."
    )

    model = build_model_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
    )

    print(
        "Model reconstructed successfully."
    )

    # =====================================================
    # Verify model output classes
    # =====================================================

    checkpoint_num_classes = int(
        checkpoint[
            "model_config"
        ][
            "num_classes"
        ]
    )

    dataset_num_classes = len(
        train_dataset.label_to_index
    )

    if (
        checkpoint_num_classes
        != dataset_num_classes
    ):
        raise ValueError(
            "The number of classes in the checkpoint "
            "does not match the current dataset.\n\n"
            f"Checkpoint: "
            f"{checkpoint_num_classes}\n"
            f"Dataset: "
            f"{dataset_num_classes}"
        )

    # =====================================================
    # Evaluate
    # =====================================================

    print()
    print(
        "=" * 70
    )

    print(
        "STARTING TEST EVALUATION"
    )

    print(
        "=" * 70
    )

    evaluator = Evaluator(
        model=model,
        device=device,
        num_classes=(
            dataset_num_classes
        ),
    )

    result = evaluator.evaluate(
        test_loader
    )

    # =====================================================
    # Classification report
    # =====================================================

    report = (
        format_classification_report(
            metrics=result.metrics,
            class_names=(
                train_dataset.labels
            ),
        )
    )

    print()
    print(
        report
    )

    # =====================================================
    # Confusion matrix
    # =====================================================

    print()
    print(
        "Confusion Matrix"
    )

    print(
        "=" * 70
    )

    print(
        result.metrics.confusion_matrix
    )

    print(
        "=" * 70
    )

    # =====================================================
    # Summary
    # =====================================================

    print()
    print(
        "TEST RESULTS SUMMARY"
    )

    print(
        f"Accuracy:          "
        f"{result.metrics.accuracy:.4f}"
    )

    print(
        f"Macro Precision:   "
        f"{result.metrics.macro_precision:.4f}"
    )

    print(
        f"Macro Recall:      "
        f"{result.metrics.macro_recall:.4f}"
    )

    print(
        f"Macro F1:          "
        f"{result.metrics.macro_f1:.4f}"
    )

    print(
        f"Weighted F1:       "
        f"{result.metrics.weighted_f1:.4f}"
    )

    print()
    print(
        "Evaluation completed successfully."
    )


if __name__ == "__main__":
    main()
