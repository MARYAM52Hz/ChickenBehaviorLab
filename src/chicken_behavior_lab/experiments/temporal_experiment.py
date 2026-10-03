from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from chicken_behavior_lab.alignment import (
    AnnotationGraphAligner,
)
from chicken_behavior_lab.dataset import (
    GroupAwareSplitter,
    TemporalPyGDataset,
    TemporalSequenceBuilder,
)
from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)
from chicken_behavior_lab.annotations.schema import (
    AnnotationSet,
)
from chicken_behavior_lab.training import (
    ClassificationLoss,
    TemporalEvaluator,
    build_trainer,
)


@dataclass(slots=True)
class TemporalExperimentConfig:

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

        if self.sequence_length < 1:
            raise ValueError(
                "sequence_length must be >= 1."
            )

        if self.sequence_stride < 1:
            raise ValueError(
                "sequence_stride must be >= 1."
            )

        if self.batch_size < 1:
            raise ValueError(
                "batch_size must be >= 1."
            )

        if self.epochs < 1:
            raise ValueError(
                "epochs must be >= 1."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be > 0."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative."
            )


@dataclass(slots=True)
class TemporalExperimentData:

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


class TemporalExperiment:

    def __init__(
        self,
        *,
        model,
        model_config,
        config: TemporalExperimentConfig,
    ) -> None:

        config.validate()

        self.model = model
        self.model_config = model_config
        self.config = config

    def prepare_data(
        self,
        graph_samples: list[GraphSample],
        annotation_set: AnnotationSet,
    ) -> TemporalExperimentData:

        # ---------------------------------------------------------
        # 1. Annotation ↔ graph alignment
        # ---------------------------------------------------------

        aligner = AnnotationGraphAligner(
            strict_conflicts=True,
        )

        alignment_result = aligner.align(
            graph_samples,
            annotation_set,
        )

        if not alignment_result.labeled_samples:
            raise ValueError(
                "Annotation alignment produced no labeled samples."
            )

        labeled_samples = (
            alignment_result.labeled_samples
        )

        # ---------------------------------------------------------
        # 2. Group-level split
        #
        # IMPORTANT:
        # Temporal windows are NOT created yet.
        # ---------------------------------------------------------

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

        train_samples = split_result.train
        validation_samples = (
            split_result.validation
        )
        test_samples = split_result.test

        # ---------------------------------------------------------
        # 3. Build label mapping from TRAIN ONLY
        # ---------------------------------------------------------

        label_mapping = (
            AnnotationGraphAligner.build_label_mapping(
                train_samples
            )
        )

        # ---------------------------------------------------------
        # 4. Apply identical mapping everywhere
        # ---------------------------------------------------------

        train_samples = (
            AnnotationGraphAligner.apply_label_mapping(
                train_samples,
                label_mapping,
            )
        )

        validation_samples = (
            AnnotationGraphAligner.apply_label_mapping(
                validation_samples,
                label_mapping,
            )
        )

        test_samples = (
            AnnotationGraphAligner.apply_label_mapping(
                test_samples,
                label_mapping,
            )
        )

        # ---------------------------------------------------------
        # 5. Temporal windowing
        # ---------------------------------------------------------

        builder = TemporalSequenceBuilder(
            sequence_length=(
                self.config.sequence_length
            ),
            sequence_stride=(
                self.config.sequence_stride
            ),
        )

        train_temporal = builder.build(
            train_samples
        )

        validation_temporal = builder.build(
            validation_samples
        )

        test_temporal = builder.build(
            test_samples
        )

        if not train_temporal.samples:
            raise ValueError(
                "Temporal windowing produced no training samples."
            )

        if not validation_temporal.samples:
            raise ValueError(
                "Temporal windowing produced no validation samples."
            )

        if not test_temporal.samples:
            raise ValueError(
                "Temporal windowing produced no test samples."
            )

        # ---------------------------------------------------------
        # 6. PyG temporal datasets
        # ---------------------------------------------------------

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

        return TemporalExperimentData(
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

    def train_and_evaluate(
        self,
        data: TemporalExperimentData,
    ) -> TemporalExperimentResult:

        from chicken_behavior_lab.dataset.temporal_collate import (
            make_temporal_dataloader,
        )

        # ---------------------------------------------------------
        # 1. DataLoaders
        # ---------------------------------------------------------

        train_loader = (
            make_temporal_dataloader(
                data.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
            )
        )

        validation_loader = (
            make_temporal_dataloader(
                data.validation_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )
        )

        test_loader = (
            make_temporal_dataloader(
                data.test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )
        )

        # ---------------------------------------------------------
        # 2. Optimizer
        # ---------------------------------------------------------

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # ---------------------------------------------------------
        # 3. Loss
        # ---------------------------------------------------------

        criterion = ClassificationLoss()

        # ---------------------------------------------------------
        # 4. Trainer
        # ---------------------------------------------------------

        trainer = build_trainer(
            model=self.model,
            model_config=self.model_config,
            optimizer=optimizer,
            criterion=criterion,
            device=self.config.device,
            checkpoint_dir=(
                self.config.checkpoint_dir
            ),
            label_mapping=data.label_mapping,
            training_config=asdict(
                self.config
            ),
        )

        # ---------------------------------------------------------
        # 5. Training
        # ---------------------------------------------------------

        history = trainer.fit(
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=self.config.epochs,
        )

        # ---------------------------------------------------------
        # 6. Test evaluation
        # ---------------------------------------------------------

        evaluator = TemporalEvaluator(
            model=self.model,
            device=self.config.device,
            label_to_index=data.label_mapping,
        )

        evaluation = evaluator.evaluate(
            test_loader
        )

        checkpoint_path = None

        if trainer.checkpoint_manager is not None:
            checkpoint_path = str(
                trainer.checkpoint_manager.path
            )

        return TemporalExperimentResult(
            history=history,
            test_metrics=evaluation.metrics,
            label_mapping=data.label_mapping,
            train_size=len(
                data.train_samples
            ),
            validation_size=len(
                data.validation_samples
            ),
            test_size=len(
                data.test_samples
            ),
            train_temporal_size=len(
                data.train_temporal
            ),
            validation_temporal_size=len(
                data.validation_temporal
            ),
            test_temporal_size=len(
                data.test_temporal
            ),
            checkpoint_path=checkpoint_path,
        )
