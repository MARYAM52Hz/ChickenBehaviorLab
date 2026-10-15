
"""End-to-end experiment orchestration for temporal skeleton-graph behavior recognition."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from chicken_behavior_lab.alignment.annotation_graph_aligner import (
    AnnotationGraphAligner,
)
from chicken_behavior_lab.annotations.schema import AnnotationSet
from chicken_behavior_lab.dataset.preflight import validate_class_coverage
from chicken_behavior_lab.dataset.split_diagnostics import (
    inspect_split_quality,
)
from chicken_behavior_lab.dataset.group_split import GroupAwareSplitter
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
from chicken_behavior_lab.io.experiment_io import ExperimentOutputWriter
from chicken_behavior_lab.io.manifest_io import ExperimentManifestIO
from chicken_behavior_lab.models.factory import (
    TemporalModelConfig,
    build_temporal_model,
    infer_temporal_model_dimensions,
)
from chicken_behavior_lab.training.temporal_evaluator import (
    TemporalEvaluator,
)
from chicken_behavior_lab.training.trainer_factory import build_trainer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TemporalExperimentConfig:
    """Configuration for a reproducible temporal behavior experiment."""

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
        """Validate configuration values before data preparation or training."""
        integer_fields = {
            "sequence_length": self.sequence_length,
            "sequence_stride": self.sequence_stride,
            "split_seed": self.split_seed,
            "training_seed": self.training_seed,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "epochs": self.epochs,
        }
        for field_name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{field_name} must be an integer.")

        if self.sequence_length < 1:
            raise ValueError("sequence_length must be >= 1.")

        if self.sequence_stride < 1:
            raise ValueError("sequence_stride must be >= 1.")

        ratios = (
            self.train_ratio,
            self.validation_ratio,
            self.test_ratio,
        )

        if any(
            isinstance(ratio, bool) or not isinstance(ratio, Real)
            for ratio in ratios
        ):
            raise ValueError("Split ratios must be real numbers.")

        if any(not math.isfinite(ratio) for ratio in ratios):
            raise ValueError("Split ratios must be finite numbers.")

        if any(ratio <= 0.0 or ratio >= 1.0 for ratio in ratios):
            raise ValueError("All split ratios must be between 0 and 1.")

        if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("Split ratios must sum to 1.0.")

        if self.group_key not in {"video_id", "track_id"}:
            raise ValueError(
                "group_key must be either 'video_id' or 'track_id'."
            )

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        if self.epochs < 1:
            raise ValueError("epochs must be >= 1.")

        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, Real)
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be a finite positive number.")

        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, Real)
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be a finite non-negative number.")

        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string.")

        if not isinstance(self.checkpoint_dir, str) or not self.checkpoint_dir.strip():
            raise ValueError("checkpoint_dir must be a non-empty string.")


# ---------------------------------------------------------------------------
# Experiment data and result objects
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TemporalExperimentData:
    """Prepared samples, split diagnostics, mappings, and PyG datasets."""

    train_samples: list[Any]
    validation_samples: list[Any]
    test_samples: list[Any]

    label_mapping: dict[str, int]

    train_temporal_samples: list[Any]
    validation_temporal_samples: list[Any]
    test_temporal_samples: list[Any]

    train_dataset: TemporalPyGDataset
    validation_dataset: TemporalPyGDataset
    test_dataset: TemporalPyGDataset

    alignment_result: Any
    split_result: Any
    split_diagnostics: Any


@dataclass(slots=True)
class TemporalExperimentResult:
    """Summary of training and held-out test evaluation."""

    history: Any
    test_metrics: Any
    label_mapping: dict[str, int]

    num_train_samples: int
    num_validation_samples: int
    num_test_samples: int

    num_train_sequences: int
    num_validation_sequences: int
    num_test_sequences: int

    checkpoint_path: str | None
    model_type: str

    model_dimensions: dict[str, Any]
    split_groups: dict[str, list[str]]
    split_diagnostics: dict[str, Any]

    prediction_records: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _to_serializable(value: Any) -> Any:
    """Convert common Python and PyTorch objects to JSON-compatible values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        # JSON has no standard representation for NaN or infinity. Preserve
        # the record while representing undefined metrics as null.
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.device):
        return str(value)

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if is_dataclass(value) and not isinstance(value, type):
        return _to_serializable(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): _to_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, set):
        # Sets have no stable iteration order; sort their serialized values so
        # manifests and JSON artifacts remain reproducible across runs.
        serialized_items = [_to_serializable(item) for item in value]
        return sorted(
            serialized_items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
        )

    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_serializable(value.to_dict())

    if hasattr(value, "__dict__"):
        return _to_serializable(vars(value))

    return str(value)


def _config_to_dict(config: Any) -> dict[str, Any]:
    """Convert a configuration object to a dictionary."""
    if is_dataclass(config) and not isinstance(config, type):
        return _to_serializable(asdict(config))

    if isinstance(config, Mapping):
        return _to_serializable(dict(config))

    if hasattr(config, "to_dict") and callable(config.to_dict):
        return _to_serializable(config.to_dict())

    if hasattr(config, "__dict__"):
        return _to_serializable(vars(config))

    raise TypeError(
        "Configuration must be a dataclass, mapping, or object "
        "with a to_dict() method."
    )


def _utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _sample_ids(samples: Sequence[Any]) -> list[str]:
    """Extract sample IDs and reject missing or duplicate IDs."""
    identifiers: list[str] = []

    for sample in samples:
        sample_id = getattr(sample, "sample_id", None)

        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("Every sample must have a non-empty sample_id.")

        identifiers.append(sample_id)

    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate sample IDs were found within a split.")

    return identifiers


def _group_ids(samples: Sequence[Any], group_key: str) -> list[str]:
    """Extract unique split-group identifiers from sample metadata."""
    groups: set[str] = set()

    for sample in samples:
        metadata = getattr(sample, "metadata", None) or {}

        if group_key == "video_id":
            group_id = metadata.get("video_id")
        elif group_key == "track_id":
            track_id = metadata.get("track_id")
            video_id = metadata.get("video_id")

            if track_id is None:
                group_id = None
            elif video_id is not None:
                group_id = f"{video_id}::track::{track_id}"
            else:
                group_id = f"track::{track_id}"
        else:
            raise ValueError(f"Unsupported group_key: {group_key!r}")

        if group_id is None or not str(group_id).strip():
            raise ValueError(
                f"Sample {getattr(sample, 'sample_id', '<unknown>')!r} "
                f"has no usable {group_key!r} in metadata."
            )

        groups.add(str(group_id))

    return sorted(groups)


def _metrics_to_dict(metrics: Any) -> dict[str, Any]:
    """Normalize an evaluation result or metrics mapping."""
    if isinstance(metrics, Mapping):
        return _to_serializable(dict(metrics))

    if hasattr(metrics, "to_dict") and callable(metrics.to_dict):
        return _to_serializable(metrics.to_dict())

    if is_dataclass(metrics) and not isinstance(metrics, type):
        return _to_serializable(asdict(metrics))

    if hasattr(metrics, "__dict__"):
        return _to_serializable(vars(metrics))

    return {"value": _to_serializable(metrics)}


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------


class TemporalExperiment:
    """Prepare data, train a temporal model, evaluate it, and save artifacts."""

    def __init__(
        self,
        config: TemporalExperimentConfig,
        model: nn.Module | None = None,
        model_config: TemporalModelConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.config.validate()

        self.model = model
        self.model_config = (
            _config_to_dict(model_config)
            if model_config is not None
            else None
        )

        self._prepared_data: TemporalExperimentData | None = None
        self._last_result: TemporalExperimentResult | None = None

    @property
    def prepared_data(self) -> TemporalExperimentData:
        """Return prepared data or raise if prepare_data() has not run."""
        if self._prepared_data is None:
            raise RuntimeError(
                "Data has not been prepared. Call prepare_data() first."
            )

        return self._prepared_data

    @property
    def num_classes(self) -> int:
        """Number of behavior classes inferred from the training mapping."""
        return len(self.prepared_data.label_mapping)

    @property
    def model_dimensions(self) -> dict[str, Any]:
        """Infer input dimensions from the prepared training dataset."""
        data = self.prepared_data

        if len(data.train_dataset) == 0:
            raise RuntimeError("The training dataset is empty.")

        item = data.train_dataset[0]
        x = getattr(item, "x", None)
        edge_index = getattr(item, "edge_index", None)
        edge_attr = getattr(item, "edge_attr", None)

        if x is None or x.ndim != 3:
            raise ValueError(
                "Expected temporal node features with shape [T, N, F_node]."
            )

        return {
            "sequence_length": int(x.shape[0]),
            "num_nodes": int(x.shape[1]),
            "node_feature_dim": int(x.shape[2]),
            "num_edges": (
                int(edge_index.shape[1])
                if edge_index is not None
                else 0
            ),
            "edge_feature_dim": (
                int(edge_attr.shape[-1])
                if edge_attr is not None
                else 0
            ),
            "num_classes": self.num_classes,
        }

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        graph_samples: Sequence[Any],
        annotation_set: AnnotationSet,
    ) -> TemporalExperimentData:
        """Align annotations, split groups, validate splits, and build windows."""
        if not graph_samples:
            raise ValueError("graph_samples cannot be empty.")

        if annotation_set is None:
            raise ValueError("annotation_set cannot be None.")

        graph_samples = list(graph_samples)
        _sample_ids(graph_samples)

        # 1. Align graph samples with behavior annotations.
        aligner = AnnotationGraphAligner(strict_conflicts=True)
        alignment_result = aligner.align(graph_samples, annotation_set)

        labeled_samples = list(alignment_result.labeled_samples)

        if not labeled_samples:
            raise ValueError(
                "Annotation alignment produced no labeled graph samples."
            )

        _sample_ids(labeled_samples)

        # 2. Split labeled samples by group before creating temporal windows.
        splitter = GroupAwareSplitter(
            train_ratio=self.config.train_ratio,
            validation_ratio=self.config.validation_ratio,
            test_ratio=self.config.test_ratio,
            group_key=self.config.group_key,
        )
        split_result = splitter.split(
            labeled_samples,
            seed=self.config.split_seed,
        )
        split_result.validate()

        train_samples = list(split_result.train)
        validation_samples = list(split_result.validation)
        test_samples = list(split_result.test)

        if not train_samples or not validation_samples or not test_samples:
            raise ValueError(
                "Group-aware splitting must produce non-empty train, "
                "validation, and test splits."
            )

        # 3. Diagnose leakage and sample-ID collisions before class checks.
        split_diagnostics = inspect_split_quality(
            train_samples,
            validation_samples,
            test_samples,
            group_key=self.config.group_key,
        )

        if split_diagnostics.has_group_leakage:
            raise ValueError(
                "Group leakage detected between dataset splits: "
                f"{_to_serializable(split_diagnostics.to_dict())}"
            )

        if getattr(split_diagnostics, "duplicate_sample_ids", None):
            duplicates = split_diagnostics.duplicate_sample_ids
            if duplicates:
                raise ValueError(
                    f"Duplicate sample IDs detected across splits: {duplicates}"
                )

        # Explicitly check IDs across splits as a second guard.
        split_ids = {
            "train": set(_sample_ids(train_samples)),
            "validation": set(_sample_ids(validation_samples)),
            "test": set(_sample_ids(test_samples)),
        }

        if split_ids["train"] & split_ids["validation"]:
            raise ValueError("Sample-ID leakage between train and validation.")

        if split_ids["train"] & split_ids["test"]:
            raise ValueError("Sample-ID leakage between train and test.")

        if split_ids["validation"] & split_ids["test"]:
            raise ValueError("Sample-ID leakage between validation and test.")

        # 4. Ensure all evaluation classes are represented in training.
        validate_class_coverage(
            train_samples,
            validation_samples,
            test_samples,
        )

        # 5. Build the canonical label mapping from training data only.
        label_mapping = aligner.build_label_mapping(train_samples)

        if not label_mapping:
            raise ValueError("No behavior classes were found in training data.")

        # The classifier and CrossEntropyLoss require class targets to be
        # contiguous zero-based indices. Fail early if the aligner violates
        # that contract instead of surfacing a later tensor/index error.
        mapping_indices = list(label_mapping.values())
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for index in mapping_indices
        ):
            raise ValueError("Label mapping values must be integer class indices.")
        if sorted(mapping_indices) != list(range(len(mapping_indices))):
            raise ValueError(
                "Label mapping indices must be unique, contiguous, and start at 0."
            )

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

        # 6. Create temporal windows independently within each split.
        builder = TemporalSequenceBuilder(
            sequence_length=self.config.sequence_length,
            sequence_stride=self.config.sequence_stride,
        )

        train_temporal_samples = list(builder.build(train_samples))
        validation_temporal_samples = list(builder.build(validation_samples))
        test_temporal_samples = list(builder.build(test_samples))

        if not train_temporal_samples:
            raise ValueError(
                "No training temporal windows were generated. "
                "Check sequence_length and the number of samples per track."
            )

        if not validation_temporal_samples:
            raise ValueError(
                "No validation temporal windows were generated. "
                "Check sequence_length and the number of samples per track."
            )

        if not test_temporal_samples:
            raise ValueError(
                "No test temporal windows were generated. "
                "Check sequence_length and the number of samples per track."
            )

        # 7. Convert temporal samples to PyTorch Geometric datasets.
        train_dataset = TemporalPyGDataset(
            train_temporal_samples,
            label_to_index=label_mapping,
        )
        validation_dataset = TemporalPyGDataset(
            validation_temporal_samples,
            label_to_index=label_mapping,
        )
        test_dataset = TemporalPyGDataset(
            test_temporal_samples,
            label_to_index=label_mapping,
        )

        prepared = TemporalExperimentData(
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            label_mapping=dict(label_mapping),
            train_temporal_samples=train_temporal_samples,
            validation_temporal_samples=validation_temporal_samples,
            test_temporal_samples=test_temporal_samples,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            alignment_result=alignment_result,
            split_result=split_result,
            split_diagnostics=split_diagnostics,
        )

        self._prepared_data = prepared
        self._last_result = None

        return prepared

    # ------------------------------------------------------------------
    # Model and data loaders
    # ------------------------------------------------------------------

    def build_model(
        self,
        model: nn.Module | None = None,
        model_config: TemporalModelConfig | Mapping[str, Any] | None = None,
    ) -> nn.Module:
        """Attach a supplied model or construct a temporal GNN from the train data."""
        if self._prepared_data is None:
            raise RuntimeError(
                "Call prepare_data() before building the experiment model."
            )

        if model is None:
            dimensions = infer_temporal_model_dimensions(
                self.prepared_data.train_dataset
            )

            if model_config is None:
                raw_config = self.model_config or {}
                if isinstance(raw_config, Mapping):
                    allowed_keys = {
                        "spatial_hidden_dim",
                        "temporal_hidden_dim",
                        "num_gnn_layers",
                        "num_gru_layers",
                        "dropout",
                        "bidirectional_gru",
                    }
                    model_kwargs = {
                        key: value
                        for key, value in raw_config.items()
                        if key in allowed_keys
                    }
                else:
                    model_kwargs = {}

                model_config = TemporalModelConfig(
                    node_feature_dim=dimensions["node_feature_dim"],
                    edge_feature_dim=dimensions["edge_feature_dim"],
                    **model_kwargs,
                )
            elif isinstance(model_config, Mapping):
                allowed_keys = {
                    "spatial_hidden_dim",
                    "temporal_hidden_dim",
                    "num_gnn_layers",
                    "num_gru_layers",
                    "dropout",
                    "bidirectional_gru",
                }
                model_kwargs = {
                    key: value
                    for key, value in model_config.items()
                    if key in allowed_keys
                }
                model_config = TemporalModelConfig(
                    node_feature_dim=dimensions["node_feature_dim"],
                    edge_feature_dim=dimensions["edge_feature_dim"],
                    **model_kwargs,
                )
            else:
                if (
                    model_config.node_feature_dim
                    != dimensions["node_feature_dim"]
                ):
                    raise ValueError(
                        "model_config.node_feature_dim does not match the dataset."
                    )
                if (
                    model_config.edge_feature_dim
                    != dimensions["edge_feature_dim"]
                ):
                    raise ValueError(
                        "model_config.edge_feature_dim does not match the dataset."
                    )

            model_config.validate()

            # Seed before model initialization, not only before the training
            # loop, so repeated runs start from the same initial weights.
            set_reproducibility_seed(self.config.training_seed)

            model = build_temporal_model(
                model_config=model_config,
                num_classes=self.num_classes,
            )
            self.model_config = {
                **asdict(model_config),
                "model_class": (
                    f"{model.__class__.__module__}."
                    f"{model.__class__.__qualname__}"
                ),
                "num_classes": self.num_classes,
            }
        else:
            if not isinstance(model, nn.Module):
                raise TypeError("model must be a torch.nn.Module.")

            if self.model_config is None:
                self.model_config = {
                    "model_class": (
                        f"{model.__class__.__module__}."
                        f"{model.__class__.__qualname__}"
                    ),
                    "num_classes": self.num_classes,
                }

        linear_layers = [
            layer
            for layer in model.modules()
            if isinstance(layer, nn.Linear)
        ]

        if not linear_layers:
            raise ValueError(
                "The supplied model has no nn.Linear classifier layer."
            )

        output_features = linear_layers[-1].out_features

        if output_features != self.num_classes:
            raise ValueError(
                "Model output dimension does not match the number of "
                f"behavior classes: model={output_features}, "
                f"classes={self.num_classes}."
            )

        self.model = model
        return model

    def _build_loaders(self) -> tuple[Any, Any, Any]:
        """Create data loaders for the three prepared datasets."""
        data = self.prepared_data

        dataset_sizes = {
            "train": len(data.train_dataset),
            "validation": len(data.validation_dataset),
            "test": len(data.test_dataset),
        }
        empty_splits = [name for name, size in dataset_sizes.items() if size == 0]
        if empty_splits:
            details = ", ".join(
                f"{name}={dataset_sizes[name]}" for name in empty_splits
            )
            raise ValueError(
                "Temporal datasets must all contain at least one sequence; "
                f"empty split(s): {details}. Check group splitting, class "
                "coverage, and sequence_length/sequence_stride."
            )

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
        """Train the model and evaluate it once on the held-out test split."""
        if data is not None:
            self._prepared_data = data

        prepared = self.prepared_data

        if self.model is None:
            self.build_model()
        else:
            # Validate caller-supplied models and ensure manifest metadata exists.
            self.build_model(model=self.model)

        # Re-seed stochastic training operations (sampling/dropout) after
        # model initialization.
        set_reproducibility_seed(self.config.training_seed)

        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device {self.config.device!r}, "
                "but CUDA is not available."
            )

        device = torch.device(self.config.device)
        self.model.to(device)

        train_loader, validation_loader, test_loader = self._build_loaders()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        trainer = build_trainer(
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            label_mapping=prepared.label_mapping,
            checkpoint_dir=str(checkpoint_dir),
        )

        history = trainer.fit(
            train_loader,
            validation_loader=validation_loader,
            epochs=self.config.epochs,
        )

        # Restore the best available checkpoint before test evaluation.
        checkpoint_path: str | None = None
        checkpoint_manager = getattr(trainer, "checkpoint_manager", None)
        candidate_path = getattr(
            checkpoint_manager,
            "path",
            None,
        )

        if candidate_path is None:
            raise RuntimeError(
                "Trainer did not expose a checkpoint path; refusing to claim "
                "best-checkpoint evaluation without a verifiable checkpoint."
            )

        candidate = Path(candidate_path)
        if not candidate.is_file():
            raise RuntimeError(
                "Training finished but the expected checkpoint file was not "
                f"created: {candidate}. Check validation metrics and the "
                "trainer/checkpoint configuration."
            )

        trainer.load_checkpoint(path=str(candidate))
        checkpoint_path = str(candidate.resolve())

        # Evaluate only after restoring the selected checkpoint.
        evaluator = TemporalEvaluator(
            model=self.model,
            device=device,
            index_to_label={
                index: label
                for label, index in prepared.label_mapping.items()
            },
        )
        evaluation_result = evaluator.evaluate(test_loader)

        metrics = getattr(
            evaluation_result,
            "metrics",
            evaluation_result,
        )
        prediction_records = getattr(
            evaluation_result,
            "prediction_records",
            [],
        )

        split_groups = {
            "train": _group_ids(
                prepared.train_samples,
                self.config.group_key,
            ),
            "validation": _group_ids(
                prepared.validation_samples,
                self.config.group_key,
            ),
            "test": _group_ids(
                prepared.test_samples,
                self.config.group_key,
            ),
        }

        result = TemporalExperimentResult(
            history=history,
            test_metrics=metrics,
            label_mapping=dict(prepared.label_mapping),
            num_train_samples=len(prepared.train_samples),
            num_validation_samples=len(prepared.validation_samples),
            num_test_samples=len(prepared.test_samples),
            num_train_sequences=len(prepared.train_dataset),
            num_validation_sequences=len(prepared.validation_dataset),
            num_test_sequences=len(prepared.test_dataset),
            checkpoint_path=checkpoint_path,
            model_type=getattr(
                self.model,
                "model_type",
                self.model.__class__.__name__,
            ),
            model_dimensions=self.model_dimensions,
            split_groups=split_groups,
            split_diagnostics=_to_serializable(
                prepared.split_diagnostics.to_dict()
            ),
            prediction_records=_to_serializable(prediction_records),
        )

        self._last_result = result
        return result

    # ------------------------------------------------------------------
    # Experiment manifest
    # ------------------------------------------------------------------

    def create_manifest(
        self,
        experiment_name: str,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> ExperimentManifest:
        """Create a reproducibility manifest for the prepared experiment."""
        data = self.prepared_data

        if self.model_config is None:
            raise RuntimeError(
                "model_config is not set. Attach a model first or provide "
                "model_config to TemporalExperiment."
            )

        split_group_ids = {
            "train": _group_ids(data.train_samples, self.config.group_key),
            "validation": _group_ids(
                data.validation_samples,
                self.config.group_key,
            ),
            "test": _group_ids(data.test_samples, self.config.group_key),
        }

        return ExperimentManifest.create(
            experiment_name=experiment_name,
            experiment_config=_config_to_dict(self.config),
            model_config=self.model_config,
            label_mapping=data.label_mapping,
            train_samples=data.train_samples,
            validation_samples=data.validation_samples,
            test_samples=data.test_samples,
            split_diagnostics=_to_serializable(
                data.split_diagnostics.to_dict()
            ),
            split_group_ids=split_group_ids,
            seed=self.config.training_seed,
            extra_metadata=_to_serializable(
                dict(extra_metadata or {})
            ),
        )

    def save_manifest(
        self,
        manifest: ExperimentManifest,
        experiment_dir: str | Path,
    ) -> Path:
        """Persist the manifest using the atomic manifest writer."""
        destination = Path(experiment_dir)
        destination.mkdir(parents=True, exist_ok=True)

        return ExperimentManifestIO().save(
            manifest,
            destination,
        )

    # ------------------------------------------------------------------
    # Complete run with output and lifecycle tracking
    # ------------------------------------------------------------------

    def run_and_save(
        self,
        experiment_name: str,
        graph_samples: Sequence[Any] | None = None,
        annotation_set: AnnotationSet | None = None,
        output_root: str | Path = "results/experiments",
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[TemporalExperimentResult, Path]:
        """Run the prepared experiment and save outputs and lifecycle status.

        If graph_samples and annotation_set are supplied, prepare the data
        first. Otherwise, prepare_data() must already have been called.
        """
        if not experiment_name or not experiment_name.strip():
            raise ValueError("experiment_name cannot be empty.")

        if graph_samples is not None or annotation_set is not None:
            if graph_samples is None or annotation_set is None:
                raise ValueError(
                    "graph_samples and annotation_set must be supplied together."
                )

            self.prepare_data(graph_samples, annotation_set)

        data = self.prepared_data

        if self.model is None:
            self.build_model()
        elif self.model_config is None:
            # A supplied model without explicit metadata must still be
            # validated and described before the manifest is created.
            self.build_model(model=self.model)

        writer = ExperimentOutputWriter(root=str(output_root))
        experiment_dir = Path(
            writer.create_experiment_directory(
                experiment_name=experiment_name,
            )
        )

        started_at = _utc_now()
        started_clock = time.monotonic()

        status_path = experiment_dir / "run_status.json"
        error_path = experiment_dir / "error.json"

        def write_json(path: Path, payload: Mapping[str, Any]) -> None:
            """Write JSON atomically to avoid leaving partially written files."""
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = path.with_suffix(path.suffix + ".tmp")

            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(
                    _to_serializable(payload),
                    file,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                file.flush()

            temporary_path.replace(path)

        def write_status(
            status: str,
            error: Mapping[str, Any] | None = None,
        ) -> None:
            now = _utc_now()
            payload: dict[str, Any] = {
                "experiment_name": experiment_name,
                "status": status,
                "started_at_utc": started_at,
                "updated_at_utc": now,
                "elapsed_seconds": round(
                    time.monotonic() - started_clock,
                    3,
                ),
            }

            if status in {"completed", "failed"}:
                payload["finished_at_utc"] = now

            if error is not None:
                payload["error"] = _to_serializable(error)

            write_json(status_path, payload)

        try:
            # Keep checkpoints inside this experiment's output directory.
            checkpoint_dir = experiment_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.config.checkpoint_dir = str(checkpoint_dir)

            write_status("running")

            experiment_config = _config_to_dict(self.config)
            model_config = self.model_config or {
                "model_class": (
                    f"{self.model.__class__.__module__}."
                    f"{self.model.__class__.__qualname__}"
                ),
                "num_classes": len(data.label_mapping),
            }

            # These calls rely on the ExperimentOutputWriter API in the repo.
            writer.save_experiment_config(
                experiment_dir,
                experiment_config,
            )
            writer.save_model_config(
                experiment_dir,
                model_config,
            )

            split_sample_ids = {
                "train": _sample_ids(data.train_samples),
                "validation": _sample_ids(data.validation_samples),
                "test": _sample_ids(data.test_samples),
            }
            split_group_ids = {
                "train": _group_ids(
                    data.train_samples,
                    self.config.group_key,
                ),
                "validation": _group_ids(
                    data.validation_samples,
                    self.config.group_key,
                ),
                "test": _group_ids(
                    data.test_samples,
                    self.config.group_key,
                ),
            }

            writer.save_split_manifest(
                experiment_dir,
                split_sample_ids,
                split_group_ids=split_group_ids,
                diagnostics=_to_serializable(
                    data.split_diagnostics.to_dict()
                ),
            )

            manifest = self.create_manifest(
                experiment_name=experiment_name,
                extra_metadata=extra_metadata,
            )
            self.save_manifest(manifest, experiment_dir)

            result = self.train_and_evaluate(data)

            writer.save_training_history(
                experiment_dir,
                result.history,
            )
            writer.save_test_metrics(
                experiment_dir,
                _metrics_to_dict(result.test_metrics),
            )
            # Keep the error artifact semantically precise: all predictions
            # belong in the complete prediction file, while this file contains
            # only records whose predicted class differs from the true class.
            prediction_errors = [
                record
                for record in result.prediction_records
                if isinstance(record, Mapping)
                and "true_index" in record
                and "predicted_index" in record
                and record["true_index"] != record["predicted_index"]
            ]
            writer.save_prediction_errors(
                experiment_dir,
                prediction_errors,
            )

            if hasattr(writer, "save_all_predictions"):
                writer.save_all_predictions(
                    experiment_dir,
                    result.prediction_records,
                )

            summary = {
                "experiment_name": experiment_name,
                "model_type": result.model_type,
                "label_mapping": result.label_mapping,
                "num_train_samples": result.num_train_samples,
                "num_validation_samples": result.num_validation_samples,
                "num_test_samples": result.num_test_samples,
                "num_train_sequences": result.num_train_sequences,
                "num_validation_sequences": result.num_validation_sequences,
                "num_test_sequences": result.num_test_sequences,
                "num_test_predictions": len(result.prediction_records),
                "num_prediction_errors": len(prediction_errors),
                "checkpoint_path": result.checkpoint_path,
                "model_dimensions": result.model_dimensions,
                "split_groups": result.split_groups,
                "split_diagnostics": result.split_diagnostics,
                "test_metrics": _metrics_to_dict(result.test_metrics),
                "manifest_path": str(
                    experiment_dir / ExperimentManifestIO.FILENAME
                ),
            }

            writer.save_summary(experiment_dir, summary)

            write_status("completed")
            return result, experiment_dir

        except Exception as exc:
            error_payload = {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }

            # Logging must not hide the original training or evaluation error.
            try:
                write_json(error_path, error_payload)
                write_status("failed", error=error_payload)
            except Exception:
                pass

            raise
