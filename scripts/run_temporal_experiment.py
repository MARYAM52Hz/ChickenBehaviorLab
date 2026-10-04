from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch

from chicken_behavior_lab.annotations.schema import AnnotationSet
from chicken_behavior_lab.experiments import (
    TemporalExperiment,
    TemporalExperimentConfig,
)
from chicken_behavior_lab.io import (
    ExperimentOutputWriter,
    load_annotations,
    load_graph_samples,
    save_temporal_evaluation,
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run the ChickenBehaviorLab temporal "
            "graph behavior experiment."
        )
    )

    parser.add_argument(
        "--graphs",
        required=True,
        help="Path to graph JSON file or directory.",
    )

    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to annotation JSON file.",
    )

    parser.add_argument(
        "--model-factory",
        required=True,
        help=(
            "Python factory in module:function format. "
            "The factory receives label_mapping and "
            "returns (model, model_config)."
        ),
    )

    parser.add_argument(
        "--output-root",
        default="results/experiments",
        help="Root directory for experiment outputs.",
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
        default="cpu",
    )

    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/temporal",
    )

    return parser.parse_args()


def import_factory(
    specification: str,
) -> Callable[..., Any]:

    if ":" not in specification:
        raise ValueError(
            "--model-factory must use "
            "module:function format."
        )

    module_name, function_name = (
        specification.split(
            ":",
            maxsplit=1,
        )
    )

    module = importlib.import_module(
        module_name
    )

    factory = getattr(
        module,
        function_name,
        None,
    )

    if factory is None:
        raise AttributeError(
            f"Factory '{function_name}' was not "
            f"found in module '{module_name}'."
        )

    if not callable(factory):
        raise TypeError(
            "Specified model factory is not callable."
        )

    return factory


def set_seed(
    seed: int,
) -> None:

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(
    args: argparse.Namespace,
) -> TemporalExperimentConfig:

    return TemporalExperimentConfig(
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        group_key=args.group_key,
        split_seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )


def main() -> None:

    args = parse_args()

    set_seed(args.seed)

    config = build_config(args)

    graph_samples = load_graph_samples(
        args.graphs
    )

    annotation_set = load_annotations(
        args.annotations
    )

    if not isinstance(
        annotation_set,
        AnnotationSet,
    ):
        raise TypeError(
            "Annotation loader returned an invalid "
            "annotation object."
        )

    output_writer = ExperimentOutputWriter(
        args.output_root
    )

    experiment_dir = (
        output_writer
        .create_experiment_directory()
    )

    factory = import_factory(
        args.model_factory
    )

    # We first create the experiment with a temporary
    # model because the final number of classes is known
    # only after annotation alignment and train-only
    # label mapping.
    #
    # Therefore model construction is intentionally
    # deferred until after prepare_data().
    #
    # The factory itself is called after data preparation.

    temporary_model = None

    experiment_metadata = {
        "graphs": str(
            Path(args.graphs)
        ),
        "annotations": str(
            Path(args.annotations)
        ),
        "config": asdict(config),
        "seed": args.seed,
    }

    output_writer.save_experiment_config(
        experiment_dir,
        experiment_metadata,
    )

    # Build data first to establish the canonical
    # training-only label mapping.
    #
    # Model creation happens immediately afterwards.

    # The experiment object requires a model at
    # construction time, so a lightweight two-stage
    # factory contract is used below.
    #
    # The factory may return a model directly when
    # called with label_mapping.

    model_result = factory(
        label_mapping=None,
        config=config,
    )

    if isinstance(
        model_result,
        tuple,
    ):
        temporary_model = model_result[0]
        temporary_model_config = model_result[1]
    else:
        temporary_model = model_result
        temporary_model_config = None

    experiment = TemporalExperiment(
        model=temporary_model,
        model_config=temporary_model_config,
        config=config,
    )

    data = experiment.prepare_data(
        graph_samples,
        annotation_set,
    )

    # Rebuild the model now that the canonical
    # train-only label mapping is known.
    model_result = factory(
        label_mapping=data.label_mapping,
        config=config,
    )

    if isinstance(
        model_result,
        tuple,
    ):
        model = model_result[0]
        model_config = model_result[1]
    else:
        model = model_result
        model_config = None

    experiment = TemporalExperiment(
        model=model,
        model_config=model_config,
        config=config,
    )

    result = experiment.train_and_evaluate(
        data
    )

    output_writer.save_experiment_config(
        experiment_dir,
        {
            **experiment_metadata,
            "label_mapping": data.label_mapping,
            "model_config": model_config,
        },
    )

    output_writer.save_training_history(
        experiment_dir,
        result.history,
    )

    output_writer.save_test_metrics(
        experiment_dir,
        result.test_metrics,
    )

    output_writer.save_split_manifest(
        experiment_dir,
        train_samples=data.train_samples,
        validation_samples=data.validation_samples,
        test_samples=data.test_samples,
        group_key=config.group_key,
        train_groups=(
            data.split_result.train_groups
        ),
        validation_groups=(
            data.split_result.validation_groups
        ),
        test_groups=(
            data.split_result.test_groups
        ),
    )

    summary = {
        "label_mapping": result.label_mapping,
        "train_size": result.train_size,
        "validation_size": result.validation_size,
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
        "test_metrics": result.test_metrics,
        "checkpoint_path": result.checkpoint_path,
    }

    output_writer.save_summary(
        experiment_dir,
        summary,
    )

    print(
        json.dumps(
            {
                "experiment_dir": str(
                    experiment_dir
                ),
                "test_metrics": result.test_metrics,
                "label_mapping": result.label_mapping,
                "checkpoint_path": (
                    result.checkpoint_path
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
