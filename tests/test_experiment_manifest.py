
import json

import pytest

from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.graph.graph import TemporalSkeletonGraph
from chicken_behavior_lab.experiments.manifest import (
    ExperimentManifest,
    fingerprint_samples,
)
from chicken_behavior_lab.io.manifest_io import ExperimentManifestIO


def make_sample(
    sample_id: str,
    behavior_id: str = "feeding",
    video_id: str | None = None,
) -> GraphSample:
    graph = TemporalSkeletonGraph(
        node_features=[
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 1.0, 0.0],
        ],
        edge_index=[[0, 1], [1, 2]],
        edge_features=[[1.0], [1.0]],
    )

    return GraphSample(
        graph=graph,
        label=0,
        behavior_id=behavior_id,
        sample_id=sample_id,
        metadata={
            "video_id": video_id or f"video_{sample_id}",
            "track_id": 1,
            "start_frame": 0,
            "end_frame": 10,
        },
    )


def make_manifest() -> ExperimentManifest:
    train = [
        make_sample("train_1", "feeding"),
        make_sample("train_2", "walking"),
    ]
    validation = [make_sample("validation_1", "feeding")]
    test = [make_sample("test_1", "walking")]

    return ExperimentManifest.create(
        experiment_name="temporal_baseline",
        experiment_config={
            "sequence_length": 16,
            "sequence_stride": 4,
        },
        model_config={
            "spatial_hidden_dim": 64,
            "temporal_hidden_dim": 64,
        },
        label_mapping={"feeding": 0, "walking": 1},
        train_samples=train,
        validation_samples=validation,
        test_samples=test,
        split_diagnostics={
            "is_valid": True,
            "has_group_leakage": False,
        },
        split_group_ids={
            "train": ["video_train_1", "video_train_2"],
            "validation": ["video_validation_1"],
            "test": ["video_test_1"],
        },
        seed=42,
    )


def test_fingerprint_is_stable():
    samples = [
        make_sample("sample_1", "feeding"),
        make_sample("sample_2", "walking"),
    ]

    assert fingerprint_samples(samples) == fingerprint_samples(samples)


def test_fingerprint_changes_when_graph_features_change():
    first = make_sample("sample_1", "feeding")
    second = make_sample("sample_1", "feeding")

    second.graph.node_features[0][0] = 99.0

    assert fingerprint_samples([first]) != fingerprint_samples([second])


def test_manifest_round_trip(tmp_path):
    manifest = make_manifest()

    saved_path = ExperimentManifestIO.save(manifest, tmp_path)
    loaded = ExperimentManifestIO.load(saved_path)

    assert saved_path.name == "experiment_manifest.json"
    assert loaded.experiment_name == manifest.experiment_name
    assert loaded.label_mapping == manifest.label_mapping
    assert loaded.dataset_fingerprint == manifest.dataset_fingerprint
    assert loaded.split_sample_ids == manifest.split_sample_ids


def test_manifest_json_is_valid(tmp_path):
    manifest = make_manifest()
    path = ExperimentManifestIO.save(manifest, tmp_path)

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    assert payload["schema_version"] == 1
    assert payload["reproducibility"]["seed"] == 42


def test_manifest_rejects_overlapping_sample_ids():
    train = [make_sample("duplicate", "feeding", "video_a")]
    validation = [make_sample("duplicate", "feeding", "video_b")]
    test = [make_sample("test_1", "feeding", "video_c")]

    with pytest.raises(ValueError, match="overlap"):
        ExperimentManifest.create(
            experiment_name="invalid_split",
            experiment_config={"sequence_length": 16},
            model_config={"hidden_dim": 64},
            label_mapping={"feeding": 0},
            train_samples=train,
            validation_samples=validation,
            test_samples=test,
            split_diagnostics={"is_valid": False},
            seed=42,
        )


def test_manifest_rejects_duplicate_label_indices():
    manifest = make_manifest()
    manifest.label_mapping = {"feeding": 0, "walking": 0}

    with pytest.raises(ValueError, match="unique"):
        manifest.validate()
