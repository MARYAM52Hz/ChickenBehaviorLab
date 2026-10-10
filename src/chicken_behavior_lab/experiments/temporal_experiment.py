
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from chicken_behavior_lab.alignment import AnnotationGraphAligner
from chicken_behavior_lab.annotations.schema import AnnotationSet
from chicken_behavior_lab.dataset import (
    GroupAwareSplitter,
    TemporalPyGDataset,
    TemporalSequenceBuilder,
)
from chicken_behavior_lab.dataset.preflight import (
    validate_class_coverage,
)
from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.dataset.split_diagnostics import (
    SplitDiagnosticsReport,
    inspect_split_quality,
)
from chicken_behavior_lab.models.factory import (
    infer_temporal_model_dimensions,
)
from chicken_behavior_lab.training import (
    TemporalEvaluator,
    build_trainer,
)


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

    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 20

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    device: str = "cpu"
    checkpoint_dir: str = "checkpoints/temporal"

    def validate(self) -> None:
        """Validate experiment configuration before preparing data."""
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
            raise ValueError(
                "train_ratio, validation_ratio, and test_ratio "
                "must each be between 0 and 1."
            )

        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0.")

        if self.group_key not in {"video_id", "track_id"}:
            raise ValueError(
                "group_key must be either 'video_id' or 'track_id'."
            )

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

        if not self.device.strip():
            raise ValueError("device must not be empty.")

        if not self.checkpoint_dir.strip():
            raise ValueError("checkpoint_dir must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the config."""
        return asdict(self)


@dataclass(slots=True)
class TemporalExperimentData:
    """Prepared graph samples, temporal sequences, and split diagnostics."""

    train_samples: list[GraphSample]
    validation_samples: list[GraphSample]
    test_samples: list[GraphSample]

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
    """Summary of model training and held-out test evaluation."""

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


class TemporalExperiment:
    """Prepare, train, and evaluate a temporal skeleton-graph model.

    The experiment follows these methodological constraints:

    1. Align annotations to graph samples before splitting.
    2. Split by groups before creating temporal windows.
    3. Check split integrity and class coverage before training.
    4. Build the label mapping from training samples only.
    5. Reuse the same label mapping for every split.
    6. Evaluate the held-out test set after training.
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
        """Return prepared data or raise if preparation has not run."""
        if self._prepared_data is None:
            raise RuntimeError(
                "Experiment data has not been prepared. "
                "Call prepare_data() first."
            )

        return self._prepared_data

    @property
    def model_dimensions(self) -> dict[str, int | None]:
        """Infer feature dimensions from the prepared training dataset."""
        dimensions = infer_temporal_model_dimensions(
            self.prepared_data.train_dataset
        )

        # Return a copy so callers cannot mutate the inferred dictionary.
        return dict(dimensions)

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the training label mapping."""
        return len(self.prepared_data.label_mapping)

    def prepare_data(
        self,
        graph_samples: Sequence[GraphSample],
        annotation_set: AnnotationSet,
    ) -> TemporalExperimentData:
        """Align annotations, split groups, validate, and build sequences."""
        if not graph_samples:
            raise ValueError("graph_samples must not be empty.")

        if not isinstance(annotation_set, AnnotationSet):
            raise TypeError(
                "annotation_set must be an AnnotationSet instance."
            )

        annotation_set.validate()

        # Align graph samples with behavior annotations.
        aligner = AnnotationGraphAligner(strict_conflicts=True)
        alignment_result = aligner.align(
            graph_samples,
            annotation_set,
        )

        labeled_samples = list(alignment_result.labeled_samples)

        if not labeled_samples:
            raise ValueError(
                "Annotation alignment produced no labeled graph samples."
            )

        # Split original graph samples before generating temporal windows.
        # This prevents overlapping windows from crossing split boundaries.
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

        if not train_samples:
            raise ValueError("The training split is empty.")

        if not validation_samples:
            raise ValueError("The validation split is empty.")

        if not test_samples:
            raise ValueError("The test split is empty.")

        # Inspect group isolation, duplicate sample IDs, and class coverage.
        # Run this before creating temporal windows.
        split_diagnostics = inspect_split_quality(
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            group_key=self.config.group_key,
        )

        if split_diagnostics.has_group_leakage:
            raise ValueError(
                "Dataset split contains group leakage. "
                f"Overlaps: {split_diagnostics.group_overlaps}"
            )

        if split_diagnostics.duplicate_sample_ids:
            raise ValueError(
                "Dataset split contains duplicate sample IDs: "
                f"{split_diagnostics.duplicate_sample_ids}"
            )

        # Reject validation/test classes absent from the training split.
        # Do not use validation or test labels to construct the mapping.
        validate_class_coverage(
            train_samples,
            validation_samples,
            test_samples,
        )

        # Construct the canonical label mapping from training data only.
        label_mapping = aligner.build_label_mapping(train_samples)

        if not label_mapping:
            raise ValueError(
                "Could not build a label mapping from training samples."
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

        # Build temporal windows independently within each split.
        sequence_builder = TemporalSequenceBuilder(
            sequence_length=self.config.sequence_length,
            stride=self.config.sequence_stride,
        )

        train_temporal = list(
            sequence_builder.build(train_samples)
        )
        validation_temporal = list(
            sequence_builder.build(validation_samples)
        )
        test_temporal = list(
            sequence_builder.build(test_samples)
        )

        if not train_temporal:
            raise ValueError(
                "No training sequences were generated. "
                "Check sequence_length, sequence_stride, and data coverage."
            )

        if not validation_temporal:
            raise ValueError(
                "No validation sequences were generated. "
                "Check sequence_length, sequence_stride, and group sizes."
            )

        if not test_temporal:
            raise ValueError(
                "No test sequences were generated. "
                "Check sequence_length, sequence_stride, and group sizes."
            )

        # Apply the exact same label mapping to every dataset.
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

        prepared_data = TemporalExperimentData(
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            label_mapping=dict(label_mapping),
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

        self._prepared_data = prepared_data

        # A new dataset invalidates any previously built model.
        self.model = None

        return prepared_data

    def build_model(self, model: nn.Module) -> nn.Module:
        """Register a model and check its classifier against class count."""
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module instance.")

        num_classes = self.num_classes

        # Validate the final classifier layer when the model exposes one.
        classifier = getattr(model, "classifier", None)

        if isinstance(classifier, nn.Sequential) and len(classifier) > 0:
            final_layer = classifier[-1]

            if isinstance(final_layer, nn.Linear):
                if final_layer.out_features != num_classes:
                    raise ValueError(
                        "Model classifier output size does not match the "
                        f"number of classes: {final_layer.out_features} "
                        f"!= {num_classes}."
                    )

        self.model = model
        return model

    def _build_loaders(self) -> tuple[Any, Any, Any]:
        """Build train, validation, and test temporal data loaders."""
        from chicken_behavior_lab.dataset.temporal_collate import (
            make_temporal_dataloader,
        )

        data = self.prepared_data

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

    def train_and_evaluate(
        self,
        data: TemporalExperimentData | None = None,
    ) -> TemporalExperimentResult:
        """Train the registered model and evaluate the held-out test set."""
        if self.model is None:
            raise RuntimeError(
                "No model has been registered. "
                "Call build_model() before train_and_evaluate()."
            )

        if self.model_config is None:
            raise RuntimeError(
                "model_config is required by this experiment."
            )

        prepared_data = self.prepared_data

        if data is not None and data is not prepared_data:
            raise ValueError(
                "The supplied data is not this experiment's prepared data. "
                "Call prepare_data() and use its returned object."
            )

        train_loader, validation_loader, test_loader = (
            self._build_loaders()
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
            label_mapping=prepared_data.label_mapping,
            checkpoint_dir=self.config.checkpoint_dir,
        )

        history = trainer.fit(
            train_loader,
            validation_loader,
            epochs=self.config.epochs,
        )

        # Restore the best checkpoint before the final test evaluation.
        checkpoint_manager = getattr(
            trainer,
            "checkpoint_manager",
            None,
        )

        checkpoint_path: Path | None = None

        if checkpoint_manager is not None:
            candidate_path = getattr(
                checkpoint_manager,
                "checkpoint_path",
                None,
            )

            if candidate_path is not None:
                checkpoint_path = Path(candidate_path)

        if checkpoint_path is None:
            checkpoint_path = (
                Path(self.config.checkpoint_dir) / "checkpoint.pt"
            )

        if checkpoint_path.exists():
            trainer.load_checkpoint(checkpoint_path)
        else:
            checkpoint_path = None

        # Evaluate the test set only after training and checkpoint selection.
        index_to_label = {
            index: behavior_id
            for behavior_id, index in prepared_data.label_mapping.items()
        }

        evaluator = TemporalEvaluator(
            model=self.model,
            device=device,
            index_to_label=index_to_label,
        )

        evaluation_result = evaluator.evaluate(test_loader)

        metrics = evaluation_result.metrics

        split_groups = {
            "train": sorted(
                str(group)
                for group in getattr(
                    prepared_data.split_result,
                    "train_groups",
                    [],
                )
            ),
            "validation": sorted(
                str(group)
                for group in getattr(
                    prepared_data.split_result,
                    "validation_groups",
                    [],
                )
            ),
            "test": sorted(
                str(group)
                for group in getattr(
                    prepared_data.split_result,
                    "test_groups",
                    [],
                )
            ),
        }

        model_type = getattr(
            self.model,
            "model_type",
            type(self.model).__name__,
        )

        return TemporalExperimentResult(
            history=history,
            test_metrics=metrics,
            label_mapping=dict(prepared_data.label_mapping),
            train_size=len(prepared_data.train_samples),
            validation_size=len(prepared_data.validation_samples),
            test_size=len(prepared_data.test_samples),
            train_temporal_size=len(prepared_data.train_temporal),
            validation_temporal_size=len(
                prepared_data.validation_temporal
            ),
            test_temporal_size=len(prepared_data.test_temporal),
            checkpoint_path=(
                str(checkpoint_path)
                if checkpoint_path is not None
                else None
            ),
            model_type=str(model_type),
            model_dimensions=dict(self.model_dimensions),
            split_groups=split_groups,
            split_diagnostics=(
                prepared_data.split_diagnostics.to_dict()
            ),
            prediction_records=list(
                evaluation_result.prediction_records
            ),
        )
