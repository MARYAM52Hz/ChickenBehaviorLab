from pathlib import Path

import numpy as np
import torch

from chicken_behavior_lab.dataset import DatasetFactory, PyGGraphDataset


def create_feature_file(path: Path, offset: float = 0.0) -> None:
    node_features = (np.arange(42, dtype=np.float32).reshape(6, 7) + offset)
    edge_index = np.array(
        [[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]],
        dtype=np.int64,
    )
    edge_attr = np.arange(48, dtype=np.float32).reshape(6, 8)
    np.savez(path, node_features=node_features, edge_index=edge_index, edge_attr=edge_attr)


def write_manifest(path: Path, samples: list[dict]) -> None:
    import json
    path.write_text(json.dumps({"dataset_version": "0.1.0", "samples": samples}), encoding="utf-8")


def test_dataset_factory_builds_canonical_graph_samples(tmp_path):
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    create_feature_file(feature_root / "sample_001.npz")
    manifest = tmp_path / "annotations.json"
    write_manifest(manifest, [{
        "sample_id": "sample_001",
        "track_id": 1,
        "video_id": "video_001",
        "feature_file": "sample_001.npz",
        "label": "walking",
    }])

    factory = DatasetFactory(annotation_file=manifest, feature_root=feature_root)
    samples = factory.build()

    assert len(samples) == 1
    sample = samples[0]
    assert sample.sample_id == "sample_001"
    assert sample.label == 0
    assert sample.behavior_id == "walking"
    assert sample.metadata["track_id"] == "track_1"
    assert sample.metadata["video_id"] == "video_001"
    assert sample.graph.node_features.shape == (1, 6, 7)
    assert sample.graph.edge_index.shape == (2, 6)
    assert sample.graph.edge_features.shape == (1, 6, 8)
    assert sample.graph.frame_mask.tolist() == [True]
    assert sample.graph.node_mask.shape == (1, 6)
    assert factory.label_to_index == {"walking": 0}


def test_dataset_factory_uses_stable_sorted_label_mapping(tmp_path):
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    create_feature_file(feature_root / "a.npz")
    create_feature_file(feature_root / "b.npz", offset=100.0)
    manifest = tmp_path / "annotations.json"
    write_manifest(manifest, [
        {"sample_id": "s1", "track_id": "track_1", "video_id": "v1", "feature_file": "a.npz", "label": "walking"},
        {"sample_id": "s2", "track_id": "track_2", "video_id": "v1", "feature_file": "b.npz", "label": "feeding"},
    ])

    factory = DatasetFactory(manifest, feature_root)
    samples = factory.build()

    assert factory.label_to_index == {"feeding": 0, "walking": 1}
    assert {sample.behavior_id: sample.label for sample in samples} == {"walking": 1, "feeding": 0}


def test_dataset_factory_accepts_explicit_label_mapping(tmp_path):
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    create_feature_file(feature_root / "sample.npz")
    manifest = tmp_path / "annotations.json"
    write_manifest(manifest, [{
        "sample_id": "s1", "track_id": "bird-A", "video_id": "v1",
        "feature_file": "sample.npz", "label": "walking",
    }])

    factory = DatasetFactory(manifest, feature_root, label_to_index={"walking": 4})
    sample = factory.build()[0]
    assert sample.label == 4
    assert sample.metadata["track_id"] == "bird-A"


def test_pyg_dataset_adapts_factory_graph_samples(tmp_path):
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    create_feature_file(feature_root / "sample_001.npz")
    manifest = tmp_path / "annotations.json"
    write_manifest(manifest, [{
        "sample_id": "sample_001", "track_id": 1, "video_id": "video_001",
        "feature_file": "sample_001.npz", "label": "walking",
    }])

    samples = DatasetFactory(manifest, feature_root).build()
    dataset = PyGGraphDataset(samples)
    assert len(dataset) == 1
    data = dataset[0]
    assert data.x.shape == (1, 6, 7)
    assert data.edge_index.shape == (2, 6)
    assert data.edge_attr.shape == (1, 6, 8)
    assert data.y.dtype == torch.long
    assert data.y.item() == 0
    assert data.track_id == "track_1"
    assert dataset.label_to_index == {"walking": 0}


def test_dataset_factory_rejects_manifest_with_unknown_label(tmp_path):
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    create_feature_file(feature_root / "sample.npz")
    manifest = tmp_path / "annotations.json"
    write_manifest(manifest, [{
        "sample_id": "s1", "track_id": 1, "video_id": "v1",
        "feature_file": "sample.npz", "label": "unknown",
    }])

    factory = DatasetFactory(manifest, feature_root, label_to_index={"walking": 0})
    try:
        factory.build()
        assert False, "Expected unknown manifest label to be rejected."
    except ValueError as error:
        assert "absent from label_to_index" in str(error)

