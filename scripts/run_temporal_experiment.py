from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from chicken_behavior_lab.experiments.temporal_experiment import (
    TemporalExperiment,
    TemporalExperimentConfig,
)
from chicken_behavior_lab.io import (
    ExperimentOutputWriter,
    load_annotations,
    load_graph_samples,
    save_temporal_evaluation,
)
from chicken_behavior_lab.models.factory import (
    TemporalModelConfig,
    build_temporal_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a temporal graph behavior "
            "classification experiment."
        )
    )

    parser.add_argument(
        "--graphs",
        type=str,
        required=True,
        help="Path to graph JSON file or directory.",
    )

    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to annotation JSON file.",
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
        "--train-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--group-key",
        type=str,
        choices=[
            "video_id",
            "track_id",
        ],
        default="video_id",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
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
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/temporal",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/experiments",
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
        default=0.0,
    )

    parser.add_argument(
        "--bidirectional-gru",
        action="store_true",
    )

    return parser.parse_args()


def save_experiment_outputs(
    *,
    writer: ExperimentOutputWriter,
    experiment_dir: Path,
    experiment: TemporalExperiment,
    data,
    result,
) -> None:
    writer.save_experiment_config(
        experiment_dir,
        {
            "experiment_config": asdict(
                experiment.config
            ),
            "model_config": asdict(
                experiment.model_config
            ),
            "label_mapping": (
                data.label_mapping
            ),
        },
    )

    writer.save_training_history(
        experiment_dir,
        result.history,
    )

    writer.save_test_metrics(
        experiment_dir,
        result.test_metrics,
    )

    split_result = data.split_result

    split_manifest = {
        "train_groups": list(
            split_result.train_groups
        ),
        "validation_groups": list(
            split_result.validation_groups
        ),
        "test_groups": list(
            split_result.test_groups
        ),
        "train_samples": [
            {
                "sample_id": sample.sample_id,
                "video_id": sample.get_metadata(
                    "video_id"
                ),
                "track_id": sample.get_metadata(
                    "track_id"
                ),
                "behavior_id": sample.behavior_id,
                "label": int(sample.label),
            }
            for sample in data.train_samples
        ],
        "validation_samples": [
            {
                "sample_id": sample.sample_id,
                "video_id": sample.get_metadata(
                    "video_id"
                ),
                "track_id": sample.get_metadata(
                    "track_id"
                ),
                "behavior_id": sample.behavior_id,
                "label": int(sample.label),
            }
            for sample in data.validation_samples
        ],
        "test_samples": [
            {
                "sample_id": sample.sample_id,
                "video_id": sample.get_metadata(
                    "video_id"
                ),
                "track_id": sample.get_metadata(
                    "track_id"
                ),
                "behavior_id": sample.behavior_id,
                "label": int(sample.label),
            }
            for sample in data.test_samples
        ],
    }

    writer.save_split_manifest(
        experiment_dir,
        split_manifest,
    )

    summary = {
        "model_type": result.model_type,
        "model_dimensions": (
            result.model_dimensions
        ),
        "label_mapping": (
            result.label_mapping
        ),
        "train_size": result.train_size,
        "validation_size": (
            result.validation_size
        ),
        "test_size": result.test_size,
        "train_temporal_size": (
            result.train_temporal_size
        ),
        "validation_temporal_size": (
            result.validation_temporal_size
        ),
        "test_temporal_size": (
            result.test_temporal_size
        ),
        "checkpoint_path": (
            result.checkpoint_path
        ),
        "test_metrics": (
            result.test_metrics
        ),
    }

    writer.save_summary(
        experiment_dir,
        summary,
    )


def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------
    # Load raw data
    # ---------------------------------------------------------

    graph_samples = load_graph_samples(
        args.graphs
    )

    annotation_set = load_annotations(
        args.annotations
    )

    # ---------------------------------------------------------
    # Experiment configuration
    # ---------------------------------------------------------

    experiment_config = (
        TemporalExperimentConfig(
            sequence_length=(
                args.sequence_length
            ),
            sequence_stride=(
                args.sequence_stride
            ),
            train_ratio=(
                args.train_ratio
            ),
            validation_ratio=(
                args.validation_ratio
            ),
            test_ratio=(
                args.test_ratio
            ),
            group_key=args.group_key,
            split_seed=args.seed,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=(
                args.learning_rate
            ),
            weight_decay=(
                args.weight_decay
            ),
            device=args.device,
            checkpoint_dir=(
                args.checkpoint_dir
            ),
        )
    )

    # ---------------------------------------------------------
    # Initial model configuration
    #
    # node/edge dimensions are inferred after prepare_data().
    # ---------------------------------------------------------

    model_config = None

    # ---------------------------------------------------------
    # Create experiment
    # ---------------------------------------------------------

    experiment = TemporalExperiment(
        model_config=model_config,
        config=experiment_config,
    )

    # ---------------------------------------------------------
    # Prepare data
    # ---------------------------------------------------------

    data = experiment.prepare_data(
        graph_samples=graph_samples,
        annotation_set=annotation_set,
    )

    # ---------------------------------------------------------
    # Build model AFTER train-only label mapping
    # ---------------------------------------------------------

    dimensions = (
        experiment.model_dimensions
    )

    model_config = TemporalModelConfig(
        node_feature_dim=(
            dimensions["node_feature_dim"]
        ),
        edge_feature_dim=(
            dimensions["edge_feature_dim"]
        ),
        spatial_hidden_dim=(
            args.spatial_hidden_dim
        ),
        temporal_hidden_dim=(
            args.temporal_hidden_dim
        ),
        num_gnn_layers=(
            args.num_gnn_layers
        ),
        num_gru_layers=(
            args.num_gru_layers
        ),
        dropout=args.dropout,
        bidirectional_gru=(
            args.bidirectional_gru
        ),
    )

    experiment.model_config = model_config

    model = build_temporal_model(
        model_config=model_config,
        num_classes=len(
            data.label_mapping
        ),
    )

    experiment.build_model(
        model=model
    )

    # ---------------------------------------------------------
    # Train and evaluate
    # ---------------------------------------------------------

    result = experiment.train_and_evaluate(
        data
    )

    # ---------------------------------------------------------
    # Create experiment output directory
    # ---------------------------------------------------------

    writer = ExperimentOutputWriter(
        root_directory=args.results_dir
    )

    experiment_dir = (
        writer.create_experiment_directory()
    )

    # ---------------------------------------------------------
    # Save all outputs
    # ---------------------------------------------------------

    save_experiment_outputs(
        writer=writer,
        experiment_dir=experiment_dir,
        experiment=experiment,
        data=data,
        result=result,
    )

    # ---------------------------------------------------------
    # Save evaluation prediction records
    # ---------------------------------------------------------

    # Re-run evaluator to retain complete prediction records.
    from chicken_behavior_lab.dataset.temporal_collate import (
        make_temporal_dataloader,
    )
    from chicken_behavior_lab.training.temporal_evaluator import (
        TemporalEvaluator,
    )

    test_loader = make_temporal_dataloader(
        data.test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    evaluator = TemporalEvaluator(
        model=experiment.model,
        device=args.device,
        label_to_index=data.label_mapping,
    )

    evaluation = evaluator.evaluate(
        test_loader
    )

    save_temporal_evaluation(
        experiment_dir,
        evaluation,
    )

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("Temporal Experiment Completed")
    print("=" * 70)

    print(
        f"Experiment directory: {experiment_dir}"
    )

    print(
        f"Train samples: {result.train_size}"
    )

    print(
        f"Validation samples: "
        f"{result.validation_size}"
    )

    print(
        f"Test samples: {result.test_size}"
    )

    print(
        f"Train temporal windows: "
        f"{result.train_temporal_size}"
    )

    print(
        f"Validation temporal windows: "
        f"{result.validation_temporal_size}"
    )

    print(
        f"Test temporal windows: "
        f"{result.test_temporal_size}"
    )

    print(
        f"Label mapping: "
        f"{result.label_mapping}"
    )

    print(
        f"Test accuracy: "
        f"{result.test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test macro F1: "
        f"{result.test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Checkpoint: "
        f"{result.checkpoint_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
