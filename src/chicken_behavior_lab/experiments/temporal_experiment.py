from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from chicken_behavior_lab.alignment import AnnotationGraphAligner
from chicken_behavior_lab.annotations.schema import AnnotationSet
from chicken_behavior_lab.dataset import (
    GroupAwareSplitter,
    TemporalPyGDataset,
    TemporalSequenceBuilder,
)
from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.models.factory import infer_temporal_model_dimensions
from chicken_behavior_lab.training import TemporalEvaluator, build_trainer


@dataclass(slots=True)
class TemporalExperimentConfig:
    """Configuration for an end-to-end temporal behavior experiment."""

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
        """Validate experiment configuration."""

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")

        if self.sequence_stride <= 0:
            raise ValueError("sequence_stride must be positive.")

        if self.sequence_stride > self.sequence_length:
            raise ValueError(
                "sequence_stride cannot be greater than sequence_length."
            )

        if self.train_ratio <= 0:
            raise ValueError("train_ratio must be positive.")

        if self.validation_ratio <= 0:
            raise ValueError("validation_ratio must be positive.")

        if self.test_ratio <= 0:
            raise ValueError("test_ratio must be positive.")

        ratio_sum = (
            self.train_ratio
            + self.validation_ratio
            + self.test_ratio
        )

        if abs(ratio_sum - 1.0) > 1e-8:
            raise ValueError(
                "train_ratio + validation_ratio + test_ratio must equal 1.0."
            )

        if self.group_key not in {"video_id", "track_id"}:
            raise ValueError(
                "group_key must be either 'video_id' or 'track_id'."
            )

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")

        if not self.device:
            raise ValueError("device cannot be empty.")

        if not self.checkpoint_dir:
            raise ValueError("checkpoint_dir cannot be empty.")


@dataclass(slots=True)
class TemporalExperimentData:
    """Prepared datasets and metadata for a temporal experiment."""

    train_samples: list[GraphSample]
    validation_samples: list[GraphSample]
    test_samples: list[GraphSample]

    label_mapping: dict[str, int]

    train_temporal: Any
    validation_temporal: Any
    test_temporal: Any

    train_dataset: TemporalPyGDataset
    validation_dataset: TemporalPyGDataset
    test_dataset: TemporalPyGDataset

    alignment_result: Any
    split_result: Any


@dataclass(slots=True)
class TemporalExperimentResult:
    """Final result of a temporal experiment."""

    history: dict[str, list[float]]
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
    model_dimensions: dict[str, int | None] | None = None

    split_groups: dict[str, list[str]] | None = None

    # Preserve evaluator output so prediction errors can be saved
    # without evaluating the test set a second time.
    prediction_records: list[dict[str, Any]] | None = None


class TemporalExperiment:
    """End-to-end experiment orchestration for temporal behavior recognition."""

    def __init__(
        self,
        model_config: Any | None,
        config: TemporalExperimentConfig,
    ) -> None:
        config.validate()

        self.config = config
        self.model_config = model_config

        self.model: nn.Module | None = None
        self._prepared_data: TemporalExperimentData | None = None

    @property
    def prepared_data(self) -> TemporalExperimentData:
        """Return prepared experiment data."""

        if self._prepared_data is None:
            raise RuntimeError(
                "Experiment data has not been prepared. "
                "Call prepare_data() first."
            )

        return self._prepared_data

    @property
    def model_dimensions(self) -> dict[str, int | None]:
        """Infer temporal model dimensions from the training dataset."""

        return infer_temporal_model_dimensions(
            self.prepared_data.train_dataset
        )

    @property
    def num_classes(self) -> int:
        """Return the number of behavior classes."""

        return len(self.prepared_data.label_mapping)

    def prepare_data(
        self,
        graph_samples: list[GraphSample],
        annotation_set: AnnotationSet,
    ) -> TemporalExperimentData:
        """
        Align annotations, split groups, build temporal windows, and create
        PyG datasets.

        The order is intentionally:

            alignment
                ↓
            group split
                ↓
            label mapping from train only
                ↓
            temporal window construction
                ↓
            PyG dataset creation

        This prevents temporal-window leakage between train/validation/test.
        """

        if not graph_samples:
            raise ValueError("graph_samples cannot be empty.")

        if not isinstance(annotation_set, AnnotationSet):
            raise TypeError(
                "annotation_set must be an AnnotationSet."
            )

        annotation_set.validate()

        # ---------------------------------------------------------------
        # 1. Align frame-level graph samples with behavior annotations.
        # ---------------------------------------------------------------
        aligner = AnnotationGraphAligner(
            strict_conflicts=True,
        )

        alignment_result = aligner.align(
            graph_samples,
            annotation_set,
        )

        labeled_samples = alignment_result.labeled_samples

        if not labeled_samples:
            raise ValueError(
                "Annotation alignment produced no labeled graph samples."
            )

        # ---------------------------------------------------------------
        # 2. Split by group BEFORE temporal windows are created.
        # ---------------------------------------------------------------
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
            raise ValueError("Training split is empty.")

        if not validation_samples:
            raise ValueError("Validation split is empty.")

        if not test_samples:
            raise ValueError("Test split is empty.")

        # ---------------------------------------------------------------
        # 3. Build label mapping from TRAINING DATA ONLY.
        # ---------------------------------------------------------------
        label_mapping = aligner.build_label_mapping(
            train_samples
        )

        if not label_mapping:
            raise ValueError(
                "Training split produced an empty label mapping."
            )

        # ---------------------------------------------------------------
        # 4. Apply the train-derived mapping to every split.
        #
        # This deliberately rejects validation/test-only classes.
        # Such a class would indicate that the split is incompatible
        # with the supervised classification setup.
        # ---------------------------------------------------------------
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

        # ---------------------------------------------------------------
        # 5. Create temporal sequences AFTER group splitting.
        # ---------------------------------------------------------------
        sequence_builder = TemporalSequenceBuilder(
            sequence_length=self.config.sequence_length,
            stride=self.config.sequence_stride,
        )

        train_temporal = sequence_builder.build(train_samples)
        validation_temporal = sequence_builder.build(
            validation_samples
        )
        test_temporal = sequence_builder.build(test_samples)

        if len(train_temporal) == 0:
            raise ValueError(
                "Temporal sequence construction produced no training samples."
            )

        if len(validation_temporal) == 0:
            raise ValueError(
                "Temporal sequence construction produced no validation samples."
            )

        if len(test_temporal) == 0:
            raise ValueError(
                "Temporal sequence construction produced no test samples."
            )

        # ---------------------------------------------------------------
        # 6. Create PyG datasets with the SAME canonical mapping.
        # ---------------------------------------------------------------
        train_dataset = TemporalPyGDataset(
            train_temporal.samples,
            label_to_index=label_mapping,
        )

        validation_dataset = TemporalPyGDataset(
            validation_temporal.samples,
            label_to_index=label_mapping,
        )

        test_dataset = TemporalPyGDataset(
            test_temporal.samples,
            label_to_index=label_mapping,
        )

        data = TemporalExperimentData(
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
        )

        self._prepared_data = data

        # Rebuilding the data invalidates any previously attached model.
        self.model = None

        return data

    def build_model(
        self,
        model: nn.Module,
    ) -> nn.Module:
        """Attach and validate a model for the prepared experiment."""

        if not isinstance(model, nn.Module):
            raise TypeError(
                "model must be an instance of torch.nn.Module."
            )

        expected_num_classes = self.num_classes

        # TemporalBehaviorGNN currently exposes its classifier as a
        # Sequential module whose final layer is Linear.
        classifier = getattr(model, "classifier", None)

        if isinstance(classifier, nn.Sequential) and len(classifier) > 0:
            final_layer = classifier[-1]

            if isinstance(final_layer, nn.Linear):
                actual_num_classes = final_layer.out_features

                if actual_num_classes != expected_num_classes:
                    raise ValueError(
                        "Model output dimension does not match the "
                        f"experiment label mapping: expected "
                        f"{expected_num_classes}, got "
                        f"{actual_num_classes}."
                    )

        self.model = model

        return model

    def _build_loaders(self) -> tuple[Any, Any, Any]:
        """Build train, validation, and test data loaders."""

        if self.model is None:
            raise RuntimeError(
                "Model has not been built. Call build_model() first."
            )

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

        return (
            train_loader,
            validation_loader,
            test_loader,
        )

    def train_and_evaluate(
        self,
        data: TemporalExperimentData | None = None,
    ) -> TemporalExperimentResult:
        """
        Train the model and evaluate the BEST checkpoint on the test set.
        """

        if self.model is None:
            raise RuntimeError(
                "Model has not been built. Call build_model() first."
            )

        if data is None:
            data = self.prepared_data

        if not isinstance(data, TemporalExperimentData):
            raise TypeError(
                "data must be a TemporalExperimentData instance."
            )

        if self._prepared_data is not data:
            raise ValueError(
                "The supplied data object is not the experiment's "
                "currently prepared data."
            )

        if self.model_config is None:
            raise RuntimeError(
                "model_config is required before training."
            )

        (
            train_loader,
            validation_loader,
            test_loader,
        ) = self._build_loaders()

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
            label_mapping=data.label_mapping,
            model_config=self.model_config,
            experiment_config=asdict(self.config),
            checkpoint_dir=self.config.checkpoint_dir,
            device=self.config.device,
        )

        history = trainer.fit(
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=self.config.epochs,
        )

        # ---------------------------------------------------------------
        # IMPORTANT:
        # The trainer may have observed a better validation checkpoint
        # during training than the final epoch.
        #
        # Therefore we explicitly restore the best checkpoint BEFORE
        # evaluating on the test set.
        # ---------------------------------------------------------------
        checkpoint_path: str | None = None

        checkpoint_manager = getattr(
            trainer,
            "checkpoint_manager",
            None,
        )

        if checkpoint_manager is not None:
            checkpoint_file = getattr(
                checkpoint_manager,
                "checkpoint_path",
                None,
            )

            if checkpoint_file is not None:
                checkpoint_path = str(checkpoint_file)

        # Fall back to the conventional checkpoint path if the trainer
        # does not expose checkpoint_path directly.
        if checkpoint_path is None:
            candidate = (
                Path(self.config.checkpoint_dir)
                / "checkpoint.pt"
            )

            if candidate.exists():
                checkpoint_path = str(candidate)

        if checkpoint_path is not None and Path(
            checkpoint_path
        ).exists():
            trainer.load_checkpoint(checkpoint_path)

        # ---------------------------------------------------------------
        # Evaluate the restored best model.
        # ---------------------------------------------------------------
        evaluator = TemporalEvaluator(
            model=self.model,
            device=self.config.device,
            label_to_index=data.label_mapping,
        )

        evaluation = evaluator.evaluate(test_loader)

        test_metrics = evaluation.metrics

        prediction_records = list(
            getattr(
                evaluation,
                "prediction_records",
                [],
            )
        )

        split_groups = {
            "train": list(data.split_result.train_groups),
            "validation": list(data.split_result.validation_groups),
            "test": list(data.split_result.test_groups),
        }

        model_type = getattr(
            self.model,
            "model_type",
            self.model.__class__.__name__,
        )

        return TemporalExperimentResult(
            history=history,
            test_metrics=test_metrics,
            label_mapping=dict(data.label_mapping),
            train_size=len(data.train_samples),
            validation_size=len(data.validation_samples),
            test_size=len(data.test_samples),
            train_temporal_size=len(data.train_temporal),
            validation_temporal_size=len(
                data.validation_temporal
            ),
            test_temporal_size=len(data.test_temporal),
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            model_dimensions=self.model_dimensions,
            split_groups=split_groups,
            prediction_records=prediction_records,
        )
