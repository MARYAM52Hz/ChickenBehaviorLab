from __future__ import annotations

import json
from pathlib import Path

import pytest

from chicken_behavior_lab.annotations.schema import (
    AnnotationSet,
    BehaviorAnnotation,
)
from chicken_behavior_lab.experiments.temporal_experiment import (
    TemporalExperiment,
    TemporalExperimentConfig,
)
from chicken_behavior_lab.models.factory import (
    TemporalModelConfig,
    build_temporal_model,
)


def _make_graph_sample(
    sample_id: str,
    video_id: str,
    track_id: int,
    frame_index: int,
    behavior_id: str,
    node_feature_dim: int = 4,
):
    """
    Create a minimal synthetic frame-level graph sample.

    The graph contains three nodes arranged as a simple chain:

        0 ---- 1 ---- 2
    """

    from chicken_behavior_lab.dataset.sample import GraphSample
    from chicken_behavior_lab.graph.graph import TemporalSkeletonGraph

    node_features = [
        [
            float(frame_index),
            0.0,
            1.0,
            0.0,
        ]
        for _ in range(3)
    ]

    if node_feature_dim != 4:
        node_features = [
            [
                float(frame_index)
                if feature_index == 0
                else 0.0
                for feature_index in range(node_feature_dim)
            ]
            for _ in range(3)
        ]

    edge_index = [
        [0, 1],
        [1, 2],
    ]

    edge_features = [
        [1.0],
        [1.0],
    ]

    graph = TemporalSkeletonGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
    )

    return GraphSample(
        graph=graph,
        label=0,
        behavior_id=behavior_id,
        sample_id=sample_id,
        metadata={
            "video_id": video_id,
            "track_id": track_id,
            "start_frame": frame_index,
            "end_frame": frame_index,
        },
    )


def _make_synthetic_samples(
    frames_per_group: int = 16,
):
    """
    Create three video groups.

    Each video contains one chicken track and a single behavior.

    Video 1 -> feeding
    Video 2 -> feeding
    Video 3 -> walking

    This gives the group-aware splitter enough independent groups for
    train/validation/test.
    """

    samples = []

    group_specs = [
        ("video_001", 1, "feeding"),
        ("video_002", 1, "feeding"),
        ("video_003", 1, "walking"),
    ]

    for video_id, track_id, behavior_id in group_specs:
        for frame_index in range(frames_per_group):
            samples.append(
                _make_graph_sample(
                    sample_id=(
                        f"{video_id}_"
                        f"{track_id}_"
                        f"{frame_index:04d}"
                    ),
                    video_id=video_id,
                    track_id=track_id,
                    frame_index=frame_index,
                    behavior_id=behavior_id,
                )
            )

    return samples


def _make_annotations() -> AnnotationSet:
    """
    Create annotations covering the complete synthetic videos.
    """

    annotations = [
        BehaviorAnnotation(
            annotation_id="ann_001",
            video_id="video_001",
            track_id=1,
            start_frame=0,
            end_frame=15,
            behavior_id="feeding",
        ),
        BehaviorAnnotation(
            annotation_id="ann_002",
            video_id="video_002",
            track_id=1,
            start_frame=0,
            end_frame=15,
            behavior_id="feeding",
        ),
        BehaviorAnnotation(
            annotation_id="ann_003",
            video_id="video_003",
            track_id=1,
            start_frame=0,
            end_frame=15,
            behavior_id="walking",
        ),
    ]

    annotation_set = AnnotationSet(
        annotations=annotations,
    )

    annotation_set.validate()

    return annotation_set


def test_temporal_experiment_end_to_end(
    tmp_path: Path,
) -> None:
    """
    End-to-end smoke test for temporal behavior recognition.

    The test verifies:

    1. Annotation alignment.
    2. Group-aware splitting.
    3. Train-only label mapping.
    4. Temporal sequence construction.
    5. PyG dataset creation.
    6. Model construction.
    7. Training.
    8. Best-checkpoint creation/restoration.
    9. Test evaluation.
    10. Prediction records.
    11. Experiment result metadata.
    """

    graph_samples = _make_synthetic_samples(
        frames_per_group=16,
    )

    annotation_set = _make_annotations()

    config = TemporalExperimentConfig(
        sequence_length=4,
        sequence_stride=4,
        train_ratio=1 / 3,
        validation_ratio=1 / 3,
        test_ratio=1 / 3,
        group_key="video_id",
        split_seed=42,
        batch_size=2,
        num_workers=0,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        checkpoint_dir=str(
            tmp_path / "checkpoints"
        ),
    )

    experiment = TemporalExperiment(
        model_config=None,
        config=config,
    )

    # ---------------------------------------------------------------
    # Prepare data.
    # ---------------------------------------------------------------
    data = experiment.prepare_data(
        graph_samples=graph_samples,
        annotation_set=annotation_set,
    )

    assert data.train_samples
    assert data.validation_samples
    assert data.test_samples

    assert data.label_mapping

    assert len(data.train_temporal) > 0
    assert len(data.validation_temporal) > 0
    assert len(data.test_temporal) > 0

    # ---------------------------------------------------------------
    # Verify group isolation.
    # ---------------------------------------------------------------
    train_groups = set(
        data.split_result.train_groups
    )

    validation_groups = set(
        data.split_result.validation_groups
    )

    test_groups = set(
        data.split_result.test_groups
    )

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)

    # ---------------------------------------------------------------
    # Infer dimensions from actual dataset.
    # ---------------------------------------------------------------
    dimensions = experiment.model_dimensions

    assert dimensions["node_feature_dim"] == 4
    assert dimensions["edge_feature_dim"] == 1

    # ---------------------------------------------------------------
    # Build model from inferred dimensions.
    # ---------------------------------------------------------------
    model_config = TemporalModelConfig(
        node_feature_dim=dimensions["node_feature_dim"],
        edge_feature_dim=dimensions["edge_feature_dim"],
        spatial_hidden_dim=8,
        temporal_hidden_dim=8,
        num_gnn_layers=1,
        num_gru_layers=1,
        dropout=0.0,
        bidirectional_gru=False,
    )

    model = build_temporal_model(
        model_config=model_config,
        num_classes=experiment.num_classes,
    )

    experiment.model_config = model_config

    experiment.build_model(
        model=model,
    )

    assert experiment.model is model

    # ---------------------------------------------------------------
    # Train and evaluate.
    # ---------------------------------------------------------------
    result = experiment.train_and_evaluate(
        data=data,
    )

    assert result.history
    assert result.test_metrics
    assert result.label_mapping == data.label_mapping

    assert result.train_size == len(
        data.train_samples
    )

    assert result.validation_size == len(
        data.validation_samples
    )

    assert result.test_size == len(
        data.test_samples
    )

    assert result.train_temporal_size == len(
        data.train_temporal
    )

    assert result.validation_temporal_size == len(
        data.validation_temporal
    )

    assert result.test_temporal_size == len(
        data.test_temporal
    )

    # ---------------------------------------------------------------
    # Verify prediction records are retained.
    # ---------------------------------------------------------------
    assert result.prediction_records is not None
    assert len(result.prediction_records) > 0

    for record in result.prediction_records:
        assert "sample_id" in record
        assert "true_index" in record
        assert "predicted_index" in record
        assert "confidence" in record

    # ---------------------------------------------------------------
    # Verify checkpoint exists.
    # ---------------------------------------------------------------
    assert result.checkpoint_path is not None

    checkpoint_path = Path(
        result.checkpoint_path
    )

    assert checkpoint_path.exists()
    assert checkpoint_path.is_file()

    # ---------------------------------------------------------------
    # Verify model metadata.
    # ---------------------------------------------------------------
    assert result.model_type is not None

    assert result.model_dimensions is not None
    assert (
        result.model_dimensions["node_feature_dim"]
        == 4
    )

    # ---------------------------------------------------------------
    # Verify split groups are persisted in the result.
    # ---------------------------------------------------------------
    assert result.split_groups is not None

    assert set(
        result.split_groups["train"]
    ).isdisjoint(
        result.split_groups["validation"]
    )

    assert set(
        result.split_groups["train"]
    ).isdisjoint(
        result.split_groups["test"]
    )

    assert set(
        result.split_groups["validation"]
    ).isdisjoint(
        result.split_groups["test"]
    )


def test_synthetic_samples_have_expected_structure() -> None:
    """Verify the synthetic fixture before running the full pipeline."""

    samples = _make_synthetic_samples(
        frames_per_group=16,
    )

    assert len(samples) == 48

    video_ids = {
        sample.metadata["video_id"]
        for sample in samples
    }

    assert video_ids == {
        "video_001",
        "video_002",
        "video_003",
    }

    behavior_ids = {
        sample.behavior_id
        for sample in samples
    }

    assert behavior_ids == {
        "feeding",
        "walking",
    }
