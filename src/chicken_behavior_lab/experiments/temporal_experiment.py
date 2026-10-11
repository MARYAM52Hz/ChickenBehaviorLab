
"""End-to-end experiment workflow for temporal skeleton-graph behavior recognition.

This module coordinates annotation alignment, group-aware splitting, temporal
window construction, PyG dataset creation, model training, evaluation, and
experiment-manifest persistence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from chicken_behavior_lab.alignment.annotation_graph_aligner import (
    AnnotationGraphAligner,
)
from chicken_behavior_lab.annotations.schema import AnnotationSet
from chicken_behavior_lab.dataset.group_split import GroupAwareSplitter
from chicken_behavior_lab.dataset.preflight import (
    validate_class_coverage,
)
from chicken_behavior_lab.dataset.split_diagnostics import (
    SplitDiagnosticsReport,
    inspect_split_quality,
)
from chicken_behavior_lab.dataset.temporal_builder import (
    TemporalSequenceBuilder,
)
from chicken_behavior_lab.dataset.temporal_pyg_dataset import (
    TemporalPyGDataset,
)
from chicken_behavior_lab.dataset.temporal_collate import (
    make_temporal_dataloader,
)
from chicken_behavior_lab.experiments.manifest import (
    ExperimentManifest,
    set_reproducibility_seed,
)
from chicken_behavior_lab.io.manifest_io import ExperimentManifestIO
from chicken_behavior_lab.training.temporal_evaluator import TemporalEvaluator
from chicken_behavior_lab.training.trainer_factory import build_trainer
from chicken_behavior_lab.models.factory import infer_temporal_model_dimensions


@dataclass(slots=True)
class TemporalExperimentConfig:
    """Configuration for data preparation and temporal-model training."""

    # Temporal windowing
    sequence_length: int = 16
    sequence_stride: int = 4

    # Group-aware split
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    group_key: str = "video_id"
    split_seed: int = 42

    # Training reproducibility is kept separate from split reproducibility.
    training_seed: int = 42

    # Training
    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"

    # Checkpoint output
    checkpoint_dir: str = "checkpoints/temporal"

    def validate(self) -> None:
        """Validate the configuration before starting an experiment."""

        if self.sequence_length < 1:
            raise ValueError("sequence_length must be >= 1.")

        if self.sequence_stride < 1:
            raise ValueError("sequence_stride must be >= 1.")

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        if self.epochs < 1:
            raise ValueError("epochs must be >= 1.")

        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")

        if self.group_key not in {"video_id", "track_id"}:
            raise ValueError(
                "group_key must be either 'video_id' or 'track_id'."
            )

        ratios = (
            self.train_ratio,
            self.validation_ratio,
            self.test_ratio,
        )

        if any(ratio <= 0 or ratio >= 1 for ratio in ratios):
            raise ValueError("All split ratios must be between 0 and 1.")

        if abs(sum(ratios) - 1.0) > 1e-8:
            raise ValueError("Split ratios must sum to 1.0.")


@dataclass(slots=True)
class TemporalExperimentData:
    """Prepared samples, temporal windows, datasets, and split diagnostics."""

    train_samples: list[Any]
    validation_samples: list[Any]
    test_samples: list[Any]

    label_mapping: dict[str, int]

    train_temporal: list[Any]
    validation_temporal: list[Any]
    test_temporal: list[Any]

    train_dataset: TemporalPyGDataset
    validation_dataset: TemporalPyGDataset
    test_dataset: TemporalPyGDataset

    alignment_result: Any
    split_result: Any
    split_diagnostics: SplitDiagnosticsReport


@dataclass(slots=True)
class TemporalExperimentResult:
    """Summary of training and held-out test evaluation."""

    history: Any
    test_metrics: dict[str, Any]
    label_mapping: dict[str, int]

    train_size: int
    validation_size: int
    test_size: int

    train_temporal_size: int
    validation_temporal_size: int
    test_temporal_size: int

    checkpoint_path: str | None

    model_type: str | None = None
    model_dimensions: dict[str, Any] | None = None
    split_groups: dict[str, list[Any]] | None = None
    split_diagnostics: dict[str, Any] | None = None
    prediction_records: list[dict[str, Any]] | None = None


class TemporalExperiment:
    """Prepare, train, evaluate, and document a temporal graph experiment.

    Typical usage:

        experiment = TemporalExperiment(model_config=model_config)

        data = experiment.prepare_data(
            graph_samples=graph_samples,
            annotation_set=annotation_set,
        )

        set_reproducibility_seed(experiment.config.training_seed)
        experiment.build_model(model)

        result = experiment.train_and_evaluate(data=data)

        manifest_path = experiment.save_manifest(
            experiment_dir="results/experiments/run_001",
            experiment_name="temporal_baseline",
        )
    """

    def __init__(
        self,
        model_config: Any,
        config: TemporalExperimentConfig | None = None,
    ) -> None:
        self.config = config or TemporalExperimentConfig()
        self.config.validate()

        self.model_config = model_config
        self.model: nn.Module | None = None
        self._prepared_data: TemporalExperimentData | None = None

    @property
    def prepared_data(self) -> TemporalExperimentData:
        """Return prepared data or fail with an actionable message."""

        if self._prepared_data is None:
            raise RuntimeError(
                "Data has not been prepared. Call prepare_data() first."
            )

        return self._prepared_data

    @property
    def num_classes(self) -> int:
        """Number of behavior classes in the training-derived mapping."""

        return len(self.prepared_data.label_mapping)

    @property
    def model_dimensions(self) -> dict[str, Any]:
        """Infer model input dimensions from the prepared training dataset."""

        data = self.prepared_data

        dimensions = infer_temporal_model_dimensions(data.train_dataset)

        if is_dataclass(dimensions):
            return asdict(dimensions)

        if isinstance(dimensions, Mapping):
            return dict(dimensions)

        if hasattr(dimensions, "__dict__"):
            return dict(vars(dimensions))

        raise TypeError(
            "infer_temporal_model_dimensions() returned an unsupported "
            f"configuration type: {type(dimensions).__name__}."
        )

    @staticmethod
    def _validate_graph_samples(graph_samples: Sequence[Any]) -> None:
        if not graph_samples:
            raise ValueError("graph_samples cannot be empty.")

        sample_ids = [
            getattr(sample, "sample_id", None)
            for sample in graph_samples
        ]

        if any(not sample_id for sample_id in sample_ids):
            raise ValueError(
                "Every graph sample must have a non-empty sample_id."
            )

        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(
                "graph_samples contains duplicate sample_id values."
            )

        for sample in graph_samples:
            validator = getattr(sample, "validate", None)
            if callable(validator):
                validator()

    @staticmethod
    def _validate_split_nonempty(
        train_samples: Sequence[Any],
        validation_samples: Sequence[Any],
        test_samples: Sequence[Any],
    ) -> None:
        sizes = {
            "train": len(train_samples),
            "validation": len(validation_samples),
            "test": len(test_samples),
        }

        empty_splits = [
            name for name, size in sizes.items() if size == 0
        ]

        if empty_splits:
            raise ValueError(
                "Group-aware splitting produced empty split(s): "
                + ", ".join(empty_splits)
                + ". Check the number of independent groups and split ratios."
            )

    def prepare_data(
        self,
        graph_samples: Sequence[Any],
        annotation_set: AnnotationSet,
    ) -> TemporalExperimentData:
        """Align labels, split by group, then construct temporal windows.

        The group split deliberately occurs before window generation to
        prevent overlapping windows from the same group leaking across splits.
        """

        self._validate_graph_samples(graph_samples)

        if not isinstance(annotation_set, AnnotationSet):
            raise TypeError(
                "annotation_set must be an AnnotationSet instance."
            )

        annotation_set.validate()

        aligner = AnnotationGraphAligner(strict_conflicts=True)
        alignment_result = aligner.align(
            graph_samples=list(graph_samples),
            annotation_set=annotation_set,
        )

        labeled_samples = list(alignment_result.labeled_samples)

        if not labeled_samples:
            raise ValueError(
                "Annotation alignment produced no labeled graph samples. "
                "Check video IDs, track IDs, frame ranges, and annotations."
            )

        # Split labeled graph samples before building temporal windows.
        splitter = GroupAwareSplitter(
            train_ratio=self.config.train_ratio,
            validation_ratio=self.config.validation_ratio,
            test_ratio=self.config.test_ratio,
            group_key=self.config.group_key,
            seed=self.config.split_seed,
        )

        split_result = splitter.split(labeled_samples)

        train_samples = list(split_result.train)
        validation_samples = list(split_result.validation)
        test_samples = list(split_result.test)

        self._validate_split_nonempty(
            train_samples,
            validation_samples,
            test_samples,
        )

        # Inspect split quality before creating any windows.
        split_diagnostics = inspect_split_quality(
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            group_key=self.config.group_key,
        )

        if split_diagnostics.has_group_leakage:
            raise ValueError(
                "Group leakage detected across train, validation, and test."
            )

        if split_diagnostics.duplicate_sample_ids:
            raise ValueError(
                "Duplicate sample IDs were found across data splits."
            )

        # A validation/test class absent from training cannot be learned by
        # the current classifier. Fail instead of silently remapping labels.
        validate_class_coverage(
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
        )

        # Derive the canonical class mapping from training samples only.
        label_mapping = aligner.build_label_mapping(train_samples)

        train_samples = aligner.apply_label_mapping(
            train_samples,
            label_mapping,
        )
        validation_samples = aligner.apply_label_mapping(
            validation_samples,
            label_mapping,
        )
        test_samples = aligner.apply_label_mapping(
            test_samples,
            label_mapping,
        )

        # Temporal windows are generated independently inside each split.
        sequence_builder = TemporalSequenceBuilder(
            sequence_length=self.config.sequence_length,
            stride=self.config.sequence_stride,
        )

        train_temporal = list(sequence_builder.build(train_samples))
        validation_temporal = list(
            sequence_builder.build(validation_samples)
        )
        test_temporal = list(sequence_builder.build(test_samples))

        temporal_sizes = {
            "train": len(train_temporal),
            "validation": len(validation_temporal),
            "test": len(test_temporal),
        }

        empty_temporal_splits = [
            name for name, size in temporal_sizes.items() if size == 0
        ]

        if empty_temporal_splits:
            raise ValueError(
                "Temporal window construction produced no samples for: "
                + ", ".join(empty_temporal_splits)
                + ". Check sequence_length, sequence_stride, and the lengths "
                "of the individual tracks."
            )

        # All splits share exactly the same training-derived class mapping.
        train_dataset = TemporalPyGDataset(
            train_temporal,
            label_to_index=label_mapping,
        )
        validation_dataset = TemporalPyGDataset(
            validation_temporal,
            label_to_index=label_mapping,
        )
        test_dataset = TemporalPyGDataset(
            test_temporal,
            label_to_index=label_mapping,
        )

        prepared = TemporalExperimentData(
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            label_mapping=label_mapping,
            train_temporal=train_temporal,
            validation_temporal=validation_temporal,
            test_temporal=test_temporal,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            alignment_result=alignment_result,
            split_result=split_result,
            split_diagnostics=split_diagnostics,
        )

        self._prepared_data = prepared
        return prepared

    def build_model(self, model: nn.Module) -> nn.Module:
        """Attach a model and verify its classifier output dimensions."""

        if self._prepared_data is None:
            raise RuntimeError(
                "Call prepare_data() before build_model() so the number "
                "of behavior classes and input dimensions are known."
            )

        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module.")

        classifier = getattr(model, "classifier", None)

        if classifier is None:
            raise ValueError(
                "The model must expose a 'classifier' module so its output "
                "class count can be checked."
            )

        linear_layers = [
            module
            for module in classifier.modules()
            if isinstance(module, nn.Linear)
        ]

        if not linear_layers:
            raise ValueError(
                "The model classifier contains no nn.Linear layer."
            )

        output_classes = linear_layers[-1].out_features

        if output_classes != self.num_classes:
            raise ValueError(
                f"Model outputs {output_classes} classes, but the prepared "
                f"training data contains {self.num_classes} classes."
            )

        if not getattr(model, "model_type", None):
            model.model_type = "temporal"

        self.model = model
        return self.model

    def _build_loaders(
        self,
        data: TemporalExperimentData,
    ) -> tuple[Any, Any, Any]:
        """Create train, validation, and test data loaders."""

        train_loader = make_temporal_dataloader(
            data.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        validation_loader = make_temporal_dataloader(
            data.validation_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        test_loader = make_temporal_dataloader(
            data.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        return train_loader, validation_loader, test_loader

    @staticmethod
    def _history_to_serializable(history: Any) -> Any:
        """Convert common history representations to JSON-compatible data."""

        if is_dataclass(history):
            return asdict(history)

        if isinstance(history, Mapping):
            return dict(history)

        if hasattr(history, "to_dict") and callable(history.to_dict):
            return history.to_dict()

        if hasattr(history, "__dict__"):
            return dict(vars(history))

        return history

    def train_and_evaluate(
        self,
        data: TemporalExperimentData | None = None,
    ) -> TemporalExperimentResult:
        """Train the model and evaluate once on the held-out test split."""

        if self.model is None:
            raise RuntimeError(
                "No model has been attached. Call build_model(model) first."
            )

        prepared = data or self.prepared_data

        if self._prepared_data is not None and prepared is not self._prepared_data:
            raise ValueError(
                "The supplied data is not the experiment's prepared data. "
                "Use the object returned by prepare_data()."
            )

        if prepared is not self.prepared_data:
            raise RuntimeError(
                "Data must be prepared by this TemporalExperiment instance."
            )

        # Seed training independently of group splitting.
        set_reproducibility_seed(self.config.training_seed)

        train_loader, validation_loader, test_loader = (
            self._build_loaders(prepared)
        )

        device = torch.device(self.config.device)
        self.model.to(device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        trainer = build_trainer(
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            label_mapping=prepared.label_mapping,
            checkpoint_dir=self.config.checkpoint_dir,
        )

        history = trainer.fit(
            train_loader,
            validation_loader=validation_loader,
            epochs=self.config.epochs,
        )

        # Restore the best checkpoint when the trainer exposes its path.
        checkpoint_path: str | None = None
        checkpoint_manager = getattr(trainer, "checkpoint_manager", None)

        if checkpoint_manager is not None:
            candidate_path = getattr(
                checkpoint_manager,
                "checkpoint_path",
                None,
            )

            if candidate_path is not None:
                candidate_path = Path(candidate_path)

                if candidate_path.is_file():
                    trainer.load_checkpoint(candidate_path)
                    checkpoint_path = str(candidate_path)

        index_to_label = {
            index: behavior_id
            for behavior_id, index in prepared.label_mapping.items()
        }

        evaluator = TemporalEvaluator(
            model=self.model,
            device=device,
            index_to_label=index_to_label,
        )

        # Test data is used only for final evaluation, not model selection.
        evaluation_result = evaluator.evaluate(test_loader)

        test_metrics = dict(evaluation_result.metrics)

        split_groups = {
            "train": list(
                getattr(prepared.split_result, "train_groups", [])
            ),
            "validation": list(
                getattr(prepared.split_result, "validation_groups", [])
            ),
            "test": list(
                getattr(prepared.split_result, "test_groups", [])
            ),
        }

        model_type = getattr(
            self.model,
            "model_type",
            type(self.model).__name__,
        )

        return TemporalExperimentResult(
            history=self._history_to_serializable(history),
            test_metrics=test_metrics,
            label_mapping=dict(prepared.label_mapping),
            train_size=len(prepared.train_samples),
            validation_size=len(prepared.validation_samples),
            test_size=len(prepared.test_samples),
            train_temporal_size=len(prepared.train_temporal),
            validation_temporal_size=len(prepared.validation_temporal),
            test_temporal_size=len(prepared.test_temporal),
            checkpoint_path=checkpoint_path,
            model_type=str(model_type),
            model_dimensions=self.model_dimensions,
            split_groups=split_groups,
            split_diagnostics=prepared.split_diagnostics.to_dict(),
            prediction_records=list(
                getattr(evaluation_result, "prediction_records", [])
            ),
        )

    def create_manifest(
        self,
        experiment_name: str = "temporal_behavior_experiment",
        extra_metadata: dict[str, Any] | None = None,
    ) -> ExperimentManifest:
        """Create a manifest for the prepared data and experiment settings."""

        data = self.prepared_data

        split_group_ids = {
            "train": list(
                getattr(data.split_result, "train_groups", [])
            ),
            "validation": list(
                getattr(data.split_result, "validation_groups", [])
            ),
            "test": list(
                getattr(data.split_result, "test_groups", [])
            ),
        }

        return ExperimentManifest.create(
            experiment_name=experiment_name,
            experiment_config=self.config,
            model_config=self.model_config,
            label_mapping=data.label_mapping,
            train_samples=data.train_samples,
            validation_samples=data.validation_samples,
            test_samples=data.test_samples,
            split_diagnostics=data.split_diagnostics.to_dict(),
            split_group_ids=split_group_ids,
            seed=self.config.split_seed,
            extra_metadata={
                "training_seed": self.config.training_seed,
                **(extra_metadata or {}),
            },
        )

    def save_manifest(
        self,
        experiment_dir: str | Path,
        experiment_name: str = "temporal_behavior_experiment",
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Persist the manifest as experiment_manifest.json."""

        manifest = self.create_manifest(
            experiment_name=experiment_name,
            extra_metadata=extra_metadata,
        )

        return ExperimentManifestIO.save(
            manifest=manifest,
            experiment_dir=experiment_dir,
        )
