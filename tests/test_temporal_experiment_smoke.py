from __future__ import annotations

from pathlib import Path

import numpy as np

from chicken_behavior_lab.annotations.schema import AnnotationSet, BehaviorAnnotation
from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.experiments.temporal_experiment import (
    TemporalExperiment,
    TemporalExperimentConfig,
)
from chicken_behavior_lab.graph.graph import TemporalSkeletonGraph
from chicken_behavior_lab.models.factory import TemporalModelConfig, build_temporal_model


NODE_FEATURE_DIM = 4
EDGE_FEATURE_DIM = 1
FRAMES_PER_ANNOTATION = 16


def _make_graph_sample(video_id: str, behavior_id: str) -> GraphSample:
    rng = np.random.default_rng(sum(ord(ch) for ch in video_id))
    features = rng.normal(size=(FRAMES_PER_ANNOTATION, 3, NODE_FEATURE_DIM)).astype(np.float32)
    edge_index = np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    edge_features = np.ones((FRAMES_PER_ANNOTATION, 4, EDGE_FEATURE_DIM), dtype=np.float32)
    graph = TemporalSkeletonGraph(
        node_features=features,
        edge_index=edge_index,
        frame_mask=np.ones(FRAMES_PER_ANNOTATION, dtype=bool),
        node_mask=np.ones((FRAMES_PER_ANNOTATION, 3), dtype=bool),
        edge_features=edge_features,
    )
    return GraphSample(
        graph=graph,
        label=0,
        behavior_id=behavior_id,
        sample_id=f"{video_id}_track_1_annotation",
        metadata={
            "video_id": video_id,
            "track_id": 1,
            "start_frame": 0,
            "end_frame": FRAMES_PER_ANNOTATION - 1,
        },
    )


def _make_data():
    group_specs = [
        ("video_001", "feeding"), ("video_002", "feeding"), ("video_003", "feeding"),
        ("video_004", "walking"), ("video_005", "walking"), ("video_006", "walking"),
        ("video_007", "resting"), ("video_008", "resting"), ("video_009", "resting"),
    ]
    samples = [_make_graph_sample(video_id, behavior) for video_id, behavior in group_specs]
    annotations = tuple(
        AnnotationSet(
            video_id=video_id,
            annotations=(
                BehaviorAnnotation(
                    annotation_id=f"annotation_{index:03d}",
                    track_id=1,
                    start_frame=0,
                    end_frame=FRAMES_PER_ANNOTATION - 1,
                    behavior_id=behavior,
                ),
            ),
        )
        for index, (video_id, behavior) in enumerate(group_specs, start=1)
    )
    return samples, annotations, {video_id for video_id, _ in group_specs}


def test_temporal_experiment_end_to_end(tmp_path: Path) -> None:
    graph_samples, annotation_sets, expected_videos = _make_data()
    config = TemporalExperimentConfig(
        train_ratio=0.67,
        validation_ratio=0.22,
        test_ratio=0.11,
        group_key="video_id",
        split_seed=42,
        batch_size=2,
        num_workers=0,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    experiment = TemporalExperiment(model_config=None, config=config)
    data = experiment.prepare_data(graph_samples, annotation_sets)

    assert len(data.train_samples) + len(data.validation_samples) + len(data.test_samples) == 9
    assert len(data.train_temporal_samples) == len(data.train_samples)
    assert len(data.validation_temporal_samples) == len(data.validation_samples)
    assert len(data.test_temporal_samples) == len(data.test_samples)

    train_groups = set(data.split_result.train_groups)
    validation_groups = set(data.split_result.validation_groups)
    test_groups = set(data.split_result.test_groups)
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)
    assert train_groups | validation_groups | test_groups == expected_videos
    assert set(data.label_mapping) == {"feeding", "walking", "resting"}

    dimensions = experiment.model_dimensions
    assert dimensions["node_feature_dim"] == NODE_FEATURE_DIM
    assert dimensions["edge_feature_dim"] == EDGE_FEATURE_DIM
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
    model = build_temporal_model(model_config=model_config, num_classes=experiment.num_classes)
    experiment.model_config = model_config
    experiment.build_model(model=model)
    result = experiment.train_and_evaluate(data=data)

    assert result.history
    assert result.test_metrics
    assert result.label_mapping == data.label_mapping
    assert result.train_size == len(data.train_samples)
    assert result.validation_size == len(data.validation_samples)
    assert result.test_size == len(data.test_samples)
    assert result.train_temporal_size == len(data.train_samples)
    assert result.validation_temporal_size == len(data.validation_samples)
    assert result.test_temporal_size == len(data.test_samples)
    assert result.prediction_records
    assert result.checkpoint_path is not None
    assert Path(result.checkpoint_path).is_file()


def test_temporal_experiment_requires_prepared_data(tmp_path: Path) -> None:
    config = TemporalExperimentConfig(
        epochs=1,
        device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    experiment = TemporalExperiment(model_config=None, config=config)
    assert experiment.model is None
    try:
        experiment.train_and_evaluate()
    except RuntimeError as exc:
        assert "model" in str(exc).lower()
    else:
        raise AssertionError("Expected RuntimeError before model construction.")
