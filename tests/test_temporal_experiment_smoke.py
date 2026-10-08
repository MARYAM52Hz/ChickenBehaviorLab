from __future__ import annotations

from pathlib import Path

from chicken_behavior_lab.annotations.schema import (
    AnnotationSet,
    BehaviorAnnotation,
)
from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.experiments.temporal_experiment import (
    TemporalExperiment,
    TemporalExperimentConfig,
)
from chicken_behavior_lab.graph.graph import TemporalSkeletonGraph
from chicken_behavior_lab.models.factory import (
    TemporalModelConfig,
    build_temporal_model,
)


NODE_FEATURE_DIM = 4
EDGE_FEATURE_DIM = 1
FRAMES_PER_VIDEO = 16


def _make_graph_sample(
    sample_id: str,
    video_id: str,
    track_id: int,
    frame_index: int,
    behavior_id: str,
) -> GraphSample:
    """Create one synthetic frame-level skeleton graph."""

    node_features = [
        [
            float(frame_index),
            float(node_index),
            1.0,
            0.0,
        ]
        for node_index in range(3)
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


def _make_synthetic_samples() -> list[GraphSample]:
    """
    Create six independent video groups.

    Two groups are provided for each behavior so that a group-aware
    train/validation/test split can preserve all classes in training.
    """

    group_specs = [
        ("video_001", "feeding"),
        ("video_002", "feeding"),
        ("video_003", "walking"),
        ("video_004", "walking"),
        ("video_005", "resting"),
        ("video_006", "resting"),
    ]

    samples: list[GraphSample] = []

    for video_id, behavior_id in group_specs:
        for frame_index in range(FRAMES_PER_VIDEO):
            samples.append(
                _make_graph_sample(
                    sample_id=(
                        f"{video_id}_"
                        f"track_1_"
                        f"frame_{frame_index:04d}"
                    ),
                    video_id=video_id,
                    track_id=1,
                    frame_index=frame_index,
                    behavior_id=behavior_id,
                )
            )

    return samples


def _make_annotations() -> AnnotationSet:
    """Create annotations covering all synthetic videos."""

    group_specs = [
        ("video_001", "feeding"),
        ("video_002", "feeding"),
        ("video_003", "walking"),
        ("video_004", "walking"),
        ("video_005", "resting"),
        ("video_006", "resting"),
    ]

    annotations = [
        BehaviorAnnotation(
            annotation_id=f"annotation_{index:03d}",
            video_id=video_id,
            track_id=1,
            start_frame=0,
            end_frame=FRAMES_PER_VIDEO - 1,
            behavior_id=behavior_id,
        )
        for index, (video_id, behavior_id)
        in enumerate(group_specs, start=1)
    ]

    annotation_set = AnnotationSet(
        annotations=annotations,
    )

    annotation_set.validate()

    return annotation_set


def _assert_group_disjoint(
    first: set[str],
    second: set[str],
) -> None:
    """Assert that two split group sets do not overlap."""

    assert first.isdisjoint(second)


def test_synthetic_fixture_structure() -> None:
    """Verify the synthetic dataset before running the full pipeline."""

    samples = _make_synthetic_samples()

    assert len(samples) == 6 * FRAMES_PER_VIDEO

    video_ids = {
        sample.metadata["video_id"]
        for sample in samples
    }

    assert video_ids == {
        "video_001",
        "video_002",
        "video_003",
        "video_004",
        "video_005",
        "video_006",
    }

    behavior_ids = {
        sample.behavior_id
        for sample in samples
    }

    assert behavior_ids == {
        "feeding",
        "walking",
        "resting",
    }

    for behavior_id in {
        "feeding",
        "walking",
        "resting",
    }:
        behavior_videos = {
            sample.metadata["video_id"]
            for sample in samples
            if sample.behavior_id == behavior_id
        }

        assert len(behavior_videos) == 2


def test_annotation_fixture_structure() -> None:
    """Verify that every synthetic video has one valid annotation."""

    annotation_set = _make_annotations()

    assert len(annotation_set.annotations) == 6

    video_ids = {
        annotation.video_id
        for annotation in annotation_set.annotations
    }

    assert len(video_ids) == 6

    for annotation in annotation_set.annotations:
        assert annotation.start_frame == 0
        assert annotation.end_frame == FRAMES_PER_VIDEO - 1
        assert annotation.track_id == 1


def test_temporal_experiment_end_to_end(
    tmp_path: Path,
) -> None:
    """
    End-to-end smoke test.

    This test validates pipeline integrity rather than scientific model
    performance.
    """

    graph_samples = _make_synthetic_samples()
    annotation_set = _make_annotations()

    config = TemporalExperimentConfig(
        sequence_length=4,
        sequence_stride=4,
        train_ratio=0.50,
        validation_ratio=0.25,
        test_ratio=0.25,
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
    # 1. Alignment + group split + temporal dataset creation.
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
    # 2. Verify group isolation.
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

    _assert_group_disjoint(
        train_groups,
        validation_groups,
    )

    _assert_group_disjoint(
        train_groups,
        test_groups,
    )

    _assert_group_disjoint(
        validation_groups,
        test_groups,
    )

    # Every original video must belong to exactly one split.
    all_groups = (
        train_groups
        | validation_groups
        | test_groups
    )

    assert all_groups == {
        "video_001",
        "video_002",
        "video_003",
        "video_004",
        "video_005",
        "video_006",
    }

    # ---------------------------------------------------------------
    # 3. The training label mapping must contain every class used
    #    by validation and test.
    #
    #    This is a smoke-test property. The actual research dataset
    #    should be checked explicitly before training.
    # ---------------------------------------------------------------
    expected_behaviors = {
        "feeding",
        "walking",
        "resting",
    }

    assert set(data.label_mapping) == expected_behaviors

    # ---------------------------------------------------------------
    # 4. Verify graph dimensions.
    # ---------------------------------------------------------------
    dimensions = experiment.model_dimensions

    assert (
        dimensions["node_feature_dim"]
        == NODE_FEATURE_DIM
    )

    assert (
        dimensions["edge_feature_dim"]
        == EDGE_FEATURE_DIM
    )

    # ---------------------------------------------------------------
    # 5. Build the temporal model.
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
    # 6. Train and evaluate.
    # ---------------------------------------------------------------
    result = experiment.train_and_evaluate(
        data=data,
    )

    # ---------------------------------------------------------------
    # 7. Validate experiment result.
    # ---------------------------------------------------------------
    assert result.history
    assert result.test_metrics

    assert (
        result.label_mapping
        == data.label_mapping
    )

    assert (
        result.train_size
        == len(data.train_samples)
    )

    assert (
        result.validation_size
        == len(data.validation_samples)
    )

    assert (
        result.test_size
        == len(data.test_samples)
    )

    assert (
        result.train_temporal_size
        == len(data.train_temporal)
    )

    assert (
        result.validation_temporal_size
        == len(data.validation_temporal)
    )

    assert (
        result.test_temporal_size
        == len(data.test_temporal)
    )

    # ---------------------------------------------------------------
    # 8. Verify prediction records.
    # ---------------------------------------------------------------
    assert result.prediction_records is not None
    assert len(result.prediction_records) > 0

    required_prediction_fields = {
        "sample_id",
        "true_index",
        "predicted_index",
        "confidence",
    }

    for record in result.prediction_records:
        assert required_prediction_fields.issubset(
            record.keys()
        )

    # ---------------------------------------------------------------
    # 9. Verify checkpoint.
    # ---------------------------------------------------------------
    assert result.checkpoint_path is not None

    checkpoint_path = Path(
        result.checkpoint_path
    )

    assert checkpoint_path.exists()
    assert checkpoint_path.is_file()

    # ---------------------------------------------------------------
    # 10. Verify model metadata.
    # ---------------------------------------------------------------
    assert result.model_type is not None

    assert result.model_dimensions is not None

    assert (
        result.model_dimensions["node_feature_dim"]
        == NODE_FEATURE_DIM
    )

    assert (
        result.model_dimensions["edge_feature_dim"]
        == EDGE_FEATURE_DIM
    )

    # ---------------------------------------------------------------
    # 11. Verify split metadata is returned.
    # ---------------------------------------------------------------
    assert result.split_groups is not None

    result_train = set(
        result.split_groups["train"]
    )

    result_validation = set(
        result.split_groups["validation"]
    )

    result_test = set(
        result.split_groups["test"]
    )

    _assert_group_disjoint(
        result_train,
        result_validation,
    )

    _assert_group_disjoint(
        result_train,
        result_test,
    )

    _assert_group_disjoint(
        result_validation,
        result_test,
    )


def test_temporal_experiment_requires_prepared_data(
    tmp_path: Path,
) -> None:
    """Verify that training cannot start before data preparation."""

    config = TemporalExperimentConfig(
        sequence_length=4,
        sequence_stride=4,
        epochs=1,
        device="cpu",
        checkpoint_dir=str(
            tmp_path / "checkpoints"
        ),
    )

    experiment = TemporalExperiment(
        model_config=None,
        config=config,
    )

    assert experiment.model is None

    try:
        experiment.train_and_evaluate()
    except RuntimeError as exc:
        assert "model" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected RuntimeError before model construction."
        )
