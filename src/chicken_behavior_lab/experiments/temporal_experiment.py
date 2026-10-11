# src/chicken_behavior_lab/experiments/temporal_experiment.py

"""End-to-end experiment orchestration for temporal skeleton-graph learning."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from chicken_behavior_lab.alignment.annotation_graph_aligner import (
    AnnotationGraphAligner,
)
from chicken_behavior_lab.annotations.schema import AnnotationSet
from chicken_behavior_lab.dataset.preflight import validate_class_coverage
from chicken_behavior_lab.dataset.split_diagnostics import (
    SplitDiagnosticsReport,
    inspect_split_quality,
)
from chicken_behavior_lab.dataset.group_split import GroupAwareSplitter
from chicken_behavior_lab.dataset.temporal_builder import TemporalSequenceBuilder
from chicken_behavior_lab.dataset.temporal_pyg_dataset import TemporalPyGDataset
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TemporalExperimentConfig:
    """Configuration for data preparation, training, and evaluation."""

    sequence_length: int = 16
    sequence_stride: int = 4

    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    group_key: str = "video_id"
    split_seed: int = 42
    training_seed: int = 42

    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 20

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    device: str = "cpu"
    checkpoint_dir: str = "checkpoints/temporal"

    def validate(self) -> None:
        """Validate experiment parameters before processing data."""

        if self.sequence_length < 1:
            raise ValueError("sequence_length must be >= 1.")

        if self.sequence_stride < 1:
            raise ValueError("sequence_stride must be >= 1.")

        ratios = (
            self.train_ratio,
            self.validation_ratio,
            self.test_ratio,
        )

        if any(ratio <= 0.0 or ratio >= 1.0 for ratio in ratios):
            raise ValueError("All split ratios must be between 0 and 1.")

        if not np.isclose(sum(ratios), 1.0, atol=1e-8):
            raise ValueError("Split ratios must sum to 1.0.")

        if self.group_key not in {"video_id", "track_id"}:
            raise ValueError("group_key must be 'video_id' or 'track_id'.")

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0.")

        if self.epochs < 1:
            raise ValueError("epochs must be >= 1.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")

        if not self.device:
            raise ValueError("device cannot be empty.")


# ---------------------------------------------------------------------------
# Prepared data and experiment results
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TemporalExperimentData:
    """Prepared sample splits and temporal datasets."""

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
    """Serializable summary of a completed experiment."""

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
    split_groups: dict[str, list[str]] | None = None
    split_diagnostics: dict[str, Any] | None = None
    prediction_records: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _to_serializable(value: Any) -> Any:
    """Convert common Python, NumPy, and PyTorch objects to JSON-compatible data."""

    if is_dataclass(value) and not isinstance(value, type):
        return _to_serializable(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): _to_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]

    if isinstance(value, set):
        return sorted(_to_serializable(item) for item in value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_serializable(value.to_dict())

    if hasattr(value, "__dict__"):
        return _to_serializable(vars(value))

    return str(value)


def _config_to_dict(config: Any) -> dict[str, Any]:
    """Convert a dataclass or mapping configuration into a dictionary."""

    if is_dataclass(config) and not isinstance(config, type):
        result = asdict(config)
    elif isinstance(config, Mapping):
        result = dict(config)
    elif hasattr(config, "to_dict") and callable(config.to_dict):
        result = config.to_dict()
    elif hasattr(config, "__dict__"):
        result = vars(config)
    else:
        raise TypeError(
            "Configuration must be a dataclass, mapping, or object "
            "with a serializable dictionary representation."
        )

    return _to_serializable(result)


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------


class TemporalExperiment:
    """Prepare temporal skeleton data, train a model, and evaluate its predictions."""

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
        """Return prepared data or raise if preparation has not run."""

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

        from chicken_behavior_lab.models.factory import (
            infer_temporal_model_dimensions,
        )

        return infer_temporal_model_dimensions(
            self.prepared_data.train_dataset
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        graph_samples: Sequence[Any],
        annotation_set: AnnotationSet,
    ) -> TemporalExperimentData:
        """Align labels, split by group, then construct temporal windows.

        Splitting occurs before temporal-window generation to prevent
        overlapping windows from the same group leaking across splits.
        """

        if not graph_samples:
            raise ValueError("graph_samples cannot be empty.")

        if not isinstance(annotation_set, AnnotationSet):
            raise TypeError("annotation_set must be an AnnotationSet.")

        annotation_set.validate()

        # Ensure each preparation starts with reproducible split randomness.
        random.seed(self.config.split_seed)
        np.random.seed(self.config.split_seed)

        aligner = AnnotationGraphAligner(strict_conflicts=True)
        alignment_result = aligner.align(
            graph_samples=list(graph_samples),
            annotation_set=annotation_set,
        )

        labeled_samples = list(alignment_result.labeled_samples)

        if not labeled_samples:
            raise ValueError(
                "No graph samples could be aligned with behavior annotations."
            )

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

        if not train_samples or not validation_samples or not test_samples:
            raise ValueError(
                "Group-aware splitting produced an empty split. "
                "Add more independent groups or adjust the split ratios."
            )

        # Check for group leakage, duplicate IDs, and class coverage.
        diagnostics = inspect_split_quality(
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            group_key=self.config.group_key,
        )

        if diagnostics.has_group_leakage:
            raise ValueError(
                "Group leakage detected across splits: "
                f"{diagnostics.group_overlaps}"
            )

        if diagnostics.duplicate_sample_ids:
            raise ValueError(
                "Duplicate sample IDs detected across splits: "
                f"{diagnostics.duplicate_sample_ids}"
            )

        validate_class_coverage(
            train_samples,
            validation_samples,
            test_samples,
        )

        # Build the canonical mapping from training samples only.
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

        # Temporal windows are constructed only after group splitting.
        sequence_builder = TemporalSequenceBuilder(
            sequence_length=self.config.sequence_length,
            stride=self.config.sequence_stride,
        )

        train_temporal = list(sequence_builder.build(train_samples))
        validation_temporal = list(sequence_builder.build(validation_samples))
        test_temporal = list(sequence_builder.build(test_samples))

        if not train_temporal:
            raise ValueError(
                "No training sequences were created. Check sequence_length "
                "and the number of graphs per track."
            )

        if not validation_temporal:
            raise ValueError(
                "No validation sequences were created. Check sequence_length "
                "and the number of graphs per track."
            )

        if not test_temporal:
            raise ValueError(
                "No test sequences were created. Check sequence_length "
                "and the number of graphs per track."
            )

        # Use the same behavior-to-index mapping for every split.
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
            split_diagnostics=diagnostics,
        )

        self._prepared_data = prepared
        return prepared

    # ------------------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------------------

    def build_model(self, model: nn.Module) -> nn.Module:
        """Attach a constructed model and validate its classifier output size."""

        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module.")

        classifier = getattr(model, "classifier", None)

        if classifier is None:
            raise ValueError(
                "The model must expose a 'classifier' module so its output "
                "dimension can be validated."
            )

        linear_layers = [
            layer
            for layer in classifier.modules()
            if isinstance(layer, nn.Linear)
        ]

        if not linear_layers:
            raise ValueError(
                "The model classifier must contain at least one nn.Linear layer."
            )

        output_classes = linear_layers[-1].out_features

        if output_classes != self.num_classes:
            raise ValueError(
                f"Model outputs {output_classes} classes, but the prepared "
                f"dataset contains {self.num_classes} classes."
            )

        self.model = model
        return model

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _build_loaders(
        self,
        data: TemporalExperimentData,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test loaders."""

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

    # ------------------------------------------------------------------
    # Training and evaluation
    # ------------------------------------------------------------------

    def train_and_evaluate(
        self,
        data: TemporalExperimentData | None = None,
    ) -> TemporalExperimentResult:
        """Train using train/validation data and evaluate once on the test set."""

        if self.model is None:
            raise RuntimeError(
                "No model has been attached. Construct a model and call "
                "build_model(model) before training."
            )

        if data is None:
            data = self.prepared_data
        elif self._prepared_data is not data:
            raise ValueError(
                "The supplied data object is not the one prepared by this "
                "TemporalExperiment instance."
            )

        set_reproducibility_seed(self.config.training_seed)

        device = torch.device(self.config.device)

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA is not available in this environment."
            )

        self.model.to(device)

        train_loader, validation_loader, test_loader = self._build_loaders(data)

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
            label_mapping=data.label_mapping,
            checkpoint_dir=self.config.checkpoint_dir,
        )

        history = trainer.fit(
            train_loader,
            validation_loader=validation_loader,
            epochs=self.config.epochs,
        )

        # Restore the best checkpoint before final test evaluation when one exists.
        checkpoint_path: str | None = None
        manager = getattr(trainer, "checkpoint_manager", None)

        if manager is not None:
            candidate = getattr(manager, "checkpoint_path", None)

            if candidate is not None:
                candidate_path = Path(candidate)

                if candidate_path.exists():
                    trainer.load_checkpoint(candidate_path)
                    checkpoint_path = str(candidate_path)

        index_to_label = {
            index: behavior
            for behavior, index in data.label_mapping.items()
        }

        evaluator = TemporalEvaluator(
            model=self.model,
            device=device,
            index_to_label=index_to_label,
        )

        evaluation = evaluator.evaluate(test_loader)

        metrics = _to_serializable(evaluation.metrics)
        predictions = _to_serializable(
            getattr(evaluation, "prediction_records", [])
        )

        split_groups = {
            "train": list(getattr(data.split_result, "train_groups", [])),
            "validation": list(
                getattr(data.split_result, "validation_groups", [])
            ),
            "test": list(getattr(data.split_result, "test_groups", [])),
        }

        model_type = getattr(
            self.model,
            "model_type",
            self.model.__class__.__name__,
        )

        return TemporalExperimentResult(
            history=_to_serializable(history),
            test_metrics=metrics,
            label_mapping=dict(data.label_mapping),
            train_size=len(data.train_samples),
            validation_size=len(data.validation_samples),
            test_size=len(data.test_samples),
            train_temporal_size=len(data.train_temporal),
            validation_temporal_size=len(data.validation_temporal),
            test_temporal_size=len(data.test_temporal),
            checkpoint_path=checkpoint_path,
            model_type=str(model_type),
            model_dimensions=self.model_dimensions,
            split_groups=split_groups,
            split_diagnostics=data.split_diagnostics.to_dict(),
            prediction_records=predictions,
        )

    # ------------------------------------------------------------------
    # Manifest support
    # ------------------------------------------------------------------

    def create_manifest(
        self,
        experiment_name: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ExperimentManifest:
        """Create a reproducibility manifest from the prepared data."""

        data = self.prepared_data

        split_groups = {
            "train": list(getattr(data.split_result, "train_groups", [])),
            "validation": list(
                getattr(data.split_result, "validation_groups", [])
            ),
            "test": list(getattr(data.split_result, "test_groups", [])),
        }

        metadata = dict(extra_metadata or {})
        metadata.setdefault("training_seed", self.config.training_seed)
        metadata.setdefault("group_key", self.config.group_key)

        return ExperimentManifest.create(
            experiment_name=experiment_name,
            experiment_config=self.config,
            model_config=self.model_config,
            label_mapping=data.label_mapping,
            train_samples=data.train_samples,
            validation_samples=data.validation_samples,
            test_samples=data.test_samples,
            split_diagnostics=data.split_diagnostics.to_dict(),
            split_group_ids=split_groups,
            seed=self.config.split_seed,
            extra_metadata=metadata,
        )

    def save_manifest(
        self,
        experiment_dir: str | Path,
        experiment_name: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Persist the reproducibility manifest in an experiment directory."""

        manifest = self.create_manifest(
            experiment_name=experiment_name,
            extra_metadata=extra_metadata,
        )

        return ExperimentManifestIO().save(
            manifest,
            Path(experiment_dir),
        )

    # ------------------------------------------------------------------
    # End-to-end run with persisted outputs
    # ------------------------------------------------------------------

    def run_and_save(
        self,
        experiment_name: str = "temporal_baseline",
        output_root: str | Path = "results/experiments",
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[TemporalExperimentResult, Path]:
        """Run training and evaluation while saving experiment artifacts.

        Expected artifacts include the experiment manifest, configuration,
        split manifest, training history, test metrics, predictions, and summary.
        """

        from chicken_behavior_lab.io.experiment_io import ExperimentOutputWriter

        data = self.prepared_data

        if self.model is None:
            raise RuntimeError(
                "No model has been attached. Call build_model(model) first."
            )

        writer = ExperimentOutputWriter(root=output_root)

        experiment_dir = writer.create_experiment_directory(
            experiment_name=experiment_name,
        )

        # Save checkpoints inside the current run directory.
        checkpoint_dir = experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir = str(checkpoint_dir)

        writer.save_experiment_config(
            experiment_dir,
            self.config,
        )
        writer.save_model_config(
            experiment_dir,
            self.model_config,
        )

        split_sample_ids = {
            "train": [sample.sample_id for sample in data.train_samples],
            "validation": [
                sample.sample_id for sample in data.validation_samples
            ],
            "test": [sample.sample_id for sample in data.test_samples],
        }

        split_group_ids = {
            "train": list(getattr(data.split_result, "train_groups", [])),
            "validation": list(
                getattr(data.split_result, "validation_groups", [])
            ),
            "test": list(getattr(data.split_result, "test_groups", [])),
        }

        writer.save_split_manifest(
            experiment_dir,
            split_sample_ids,
            split_group_ids,
            diagnostics=data.split_diagnostics.to_dict(),
        )

        self.save_manifest(
            experiment_dir=experiment_dir,
            experiment_name=experiment_name,
            extra_metadata=extra_metadata,
        )

        result = self.train_and_evaluate(data=data)

        writer.save_training_history(
            experiment_dir,
            result.history,
        )
        writer.save_test_metrics(
            experiment_dir,
            result.test_metrics,
        )
        writer.save_prediction_errors(
            experiment_dir,
            result.prediction_records or [],
        )
        writer.save_all_predictions(
            experiment_dir,
            result.prediction_records or [],
        )

        summary = {
            "experiment_name": experiment_name,
            "status": "completed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_type": result.model_type,
            "model_dimensions": result.model_dimensions,
            "label_mapping": result.label_mapping,
            "train_size": result.train_size,
            "validation_size": result.validation_size,
            "test_size": result.test_size,
            "train_temporal_size": result.train_temporal_size,
            "validation_temporal_size": result.validation_temporal_size,
            "test_temporal_size": result.test_temporal_size,
            "checkpoint_path": result.checkpoint_path,
            "test_metrics": result.test_metrics,
            "split_diagnostics": result.split_diagnostics,
            "extra_metadata": extra_metadata or {},
        }

        writer.save_summary(
            experiment_dir,
            _to_serializable(summary),
        )

        return result, experiment_dir
