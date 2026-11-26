from __future__ import annotations

import json

import numpy as np
import pytest

from chicken_behavior_lab.dataset import DatasetFactory
from chicken_behavior_lab.features.edge_contract import (
    EDGE_FEATURE_NAMES,
    build_combined_edge_features,
)
from chicken_behavior_lab.io.graph_json import GraphJsonLoader


def _arrays():
    nodes = np.asarray(
        [[[0, 0, 0, 0, 0, 0, 0.9],
          [1, 1, 0, 0, 0, 0, 0.8],
          [2, 0, 0, 0, 0, 0, 0.7]]],
        dtype=np.float32,
    )
    edges = np.asarray(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        dtype=np.int64,
    )
    attrs = build_combined_edge_features(nodes, edges)
    return nodes, edges, attrs


def _manifest(tmp_path, feature_file):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({
            "dataset_version": "0.1.0",
            "samples": [{
                "sample_id": "s1",
                "track_id": "bird-1",
                "video_id": "video-1",
                "feature_file": feature_file,
                "label": "walking",
            }],
        }),
        encoding="utf-8",
    )
    return path


def test_dataset_factory_accepts_declared_canonical_nine_channel_contract(tmp_path):
    nodes, edges, attrs = _arrays()
    np.savez(
        tmp_path / "sample.npz",
        node_features=nodes,
        edge_index=edges,
        edge_attr=attrs,
        edge_feature_names=np.asarray(EDGE_FEATURE_NAMES),
    )

    samples = DatasetFactory(_manifest(tmp_path, "sample.npz")).build()

    assert len(samples) == 1
    assert samples[0].graph.edge_features.shape == (1, 4, 9)


def test_dataset_factory_rejects_declared_edge_feature_dimension_mismatch(tmp_path):
    nodes, edges, _ = _arrays()
    np.savez(
        tmp_path / "sample.npz",
        node_features=nodes,
        edge_index=edges,
        edge_attr=np.zeros((1, 4, 8), dtype=np.float32),
        edge_feature_names=np.asarray(EDGE_FEATURE_NAMES),
    )

    with pytest.raises(ValueError, match="requires edge_features shape"):
        DatasetFactory(_manifest(tmp_path, "sample.npz")).build()


def test_dataset_factory_rejects_noncanonical_declared_topology(tmp_path):
    nodes, _, _ = _arrays()
    directed_edges = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    attrs = build_combined_edge_features(nodes, directed_edges)
    np.savez(
        tmp_path / "sample.npz",
        node_features=nodes,
        edge_index=directed_edges,
        edge_attr=attrs,
        edge_feature_names=np.asarray(EDGE_FEATURE_NAMES),
    )

    with pytest.raises(ValueError, match="canonical, deduplicated"):
        DatasetFactory(_manifest(tmp_path, "sample.npz")).build()


def _graph_json_payload(edge_index, edge_features, edge_feature_names):
    nodes, _, _ = _arrays()
    return {
        "sample_id": "json-s1",
        "label": 0,
        "behavior_id": "walking",
        "graph": {
            "node_features": nodes.tolist(),
            "edge_index": edge_index.tolist(),
            "edge_features": edge_features.tolist(),
            "edge_feature_names": list(edge_feature_names),
        },
    }


def test_graph_json_loader_validates_declared_combined_edge_contract(tmp_path):
    nodes, edges, attrs = _arrays()
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(_graph_json_payload(edges, attrs, EDGE_FEATURE_NAMES)),
        encoding="utf-8",
    )

    samples = GraphJsonLoader().load(path)

    assert len(samples) == 1
    assert samples[0].graph.edge_feature_dimension == 9


def test_graph_json_loader_rejects_noncanonical_declared_topology(tmp_path):
    nodes, _, _ = _arrays()
    directed_edges = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    attrs = build_combined_edge_features(nodes, directed_edges)
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(_graph_json_payload(directed_edges, attrs, EDGE_FEATURE_NAMES)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical, deduplicated"):
        GraphJsonLoader().load(path)


def test_dataset_factory_rejects_edge_values_that_disagree_with_node_features(tmp_path):
    nodes, edges, attrs = _arrays()
    attrs = attrs.copy()
    attrs[0, 0, 0] += 0.25
    np.savez(
        tmp_path / "sample.npz",
        node_features=nodes,
        edge_index=edges,
        edge_attr=attrs,
        edge_feature_names=np.asarray(EDGE_FEATURE_NAMES),
    )

    with pytest.raises(ValueError, match="do not match the canonical combined"):
        DatasetFactory(_manifest(tmp_path, "sample.npz")).build()
