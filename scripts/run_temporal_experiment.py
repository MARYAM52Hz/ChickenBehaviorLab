from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chicken_behavior_lab.config.model_config import (
    ModelConfig,
)

from chicken_behavior_lab.models.model_factory import (
    build_model,
)

from chicken_behavior_lab.training.trainer_factory import (
    build_trainer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the temporal "
            "ChickenBehaviorLab behavior model."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--node-feature-dim",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--edge-feature-dim",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--spatial-hidden-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--temporal-hidden-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--num-gnn-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--num-gru-layers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--sequence-stride",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--bidirectional-gru",
        action="store_true",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/temporal"),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser


def resolve_device(
    requested: str,
) -> torch.device:

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    return torch.device(
        requested
    )


def set_seed(
    seed: int,
) -> None:

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def main() -> None:
    parser = build_parser()

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError(
            "--epochs must be >= 1."
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1."
        )

    set_seed(
        args.seed
    )

    device = resolve_device(
        args.device
    )

    model_config = ModelConfig(
        model_type="temporal",
        node_feature_dim=args.node_feature_dim,
        edge_feature_dim=args.edge_feature_dim,
        spatial_hidden_dim=args.spatial_hidden_dim,
        temporal_hidden_dim=args.temporal_hidden_dim,
        num_classes=args.num_classes,
        num_gnn_layers=args.num_gnn_layers,
        num_gru_layers=args.num_gru_layers,
        dropout=args.dropout,
        bidirectional_gru=args.bidirectional_gru,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
    )

    print(
        "ChickenBehaviorLab Temporal Experiment"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Epochs: {args.epochs}"
    )

    print(
        f"Batch size: {args.batch_size}"
    )

    print(
        f"Sequence length: "
        f"{args.sequence_length}"
    )

    print(
        f"Sequence stride: "
        f"{args.sequence_stride}"
    )

    print(
        f"Checkpoint directory: "
        f"{args.checkpoint_dir}"
    )

    # The actual dataset construction and group-level splitting
    # will be connected here once the temporal feature/annotation
    # ingestion API is finalized.
    #
    # This prevents the experiment script from inventing a data format
    # that may conflict with the existing DatasetFactory.


if __name__ == "__main__":
    main()
