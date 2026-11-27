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


def _graph_json_payload(
    edge_index, edge_features, edge_feature_names, *, frame_mask=None, node_mask=None
):
    nodes, _, _ = _arrays()
    graph = {
        "node_features": nodes.tolist(),
        "edge_index": edge_index.tolist(),
        "edge_features": edge_features.tolist(),
        "edge_feature_names": list(edge_feature_names),
    }
    if frame_mask is not None:
        graph["frame_mask"] = np.asarray(frame_mask, dtype=bool).tolist()
    if node_mask is not None:
        graph["node_mask"] = np.asarray(node_mask, dtype=bool).tolist()
    return {
        "sample_id": "json-s1",
        "label": 0,
        "behavior_id": "walking",
        "graph": graph,
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



def test_dataset_factory_accepts_masked_edges_zeroed_by_graph_builder(tmp_path):
    nodes, edges, attrs = _arrays()
    frame_mask = np.asarray([True])
    node_mask = np.asarray([[True, False, True]])
    masked_attrs = attrs.copy()
    edge_valid = frame_mask[:, None] & node_mask[:, edges[0]] & node_mask[:, edges[1]]
    masked_attrs[~edge_valid] = 0.0
    np.savez(
        tmp_path / "sample.npz",
        node_features=nodes,
        edge_index=edges,
        edge_attr=masked_attrs,
        edge_feature_names=np.asarray(EDGE_FEATURE_NAMES),
        frame_mask=frame_mask,
        node_mask=node_mask,
    )

    samples = DatasetFactory(_manifest(tmp_path, "sample.npz")).build()

    assert len(samples) == 1
    assert np.array_equal(samples[0].graph.node_mask, node_mask)
    assert np.array_equal(samples[0].graph.edge_features, masked_attrs)


def test_graph_json_loader_accepts_masked_edges_zeroed_by_graph_builder(tmp_path):
    nodes, edges, attrs = _arrays()
    frame_mask = np.asarray([True])
    node_mask = np.asarray([[True, False, True]])
    masked_attrs = attrs.copy()
    edge_valid = frame_mask[:, None] & node_mask[:, edges[0]] & node_mask[:, edges[1]]
    masked_attrs[~edge_valid] = 0.0
    path = tmp_path / "masked.json"
    path.write_text(
        json.dumps(_graph_json_payload(
            edges, masked_attrs, EDGE_FEATURE_NAMES,
            frame_mask=frame_mask, node_mask=node_mask,
        )),
        encoding="utf-8",
    )

    samples = GraphJsonLoader().load(path)

    assert len(samples) == 1
    assert np.array_equal(samples[0].graph.node_mask, node_mask)
    assert np.array_equal(samples[0].graph.edge_features, masked_attrs)


def test_declared_edge_contract_rejects_malformed_mask_shape(tmp_path):
    nodes, edges, attrs = _arrays()
    np.savez(
        tmp_path / "sample.npz",
        node_features=nodes,
        edge_index=edges,
        edge_attr=attrs,
        edge_feature_names=np.asarray(EDGE_FEATURE_NAMES),
        node_mask=np.ones((1, 2), dtype=bool),
    )

    with pytest.raises(ValueError, match="node_mask must have shape"):
        DatasetFactory(_manifest(tmp_path, "sample.npz")).build()



def test_dataset_factory_rejects_fractional_edge_indices_before_integer_cast(tmp_path):
    nodes, edges, attrs = _arrays()
    fractional_edges = edges.astype(np.float64)
    fractional_edges[0, 0] = 0.5
    np.savez(
        tmp_path / "sample.npz",
        node_features=nodes,
        edge_index=fractional_edges,
        edge_attr=attrs,
        edge_feature_names=np.asarray(EDGE_FEATURE_NAMES),
    )

    with pytest.raises(ValueError, match="edge_index must contain integer-valued"):
        DatasetFactory(_manifest(tmp_path, "sample.npz")).build()


def test_dataset_factory_rejects_non_binary_masks_before_bool_cast(tmp_path):
    nodes, edges, attrs = _arrays()
    np.savez(
        tmp_path / "sample.npz",
        node_features=nodes,
        edge_index=edges,
        edge_attr=attrs,
        edge_feature_names=np.asarray(EDGE_FEATURE_NAMES),
        node_mask=np.asarray([[1, 2, 1]]),
    )

    with pytest.raises(ValueError, match="node_mask must contain only boolean or 0/1"):
        DatasetFactory(_manifest(tmp_path, "sample.npz")).build()


def test_dataset_factory_rejects_non_finite_node_features(tmp_path):
    nodes, edges, attrs = _arrays()
    nodes[0, 1, 0] = np.nan
    np.savez(
        tmp_path / "sample.npz",
        node_features=nodes,
        edge_index=edges,
        edge_attr=attrs,
        edge_feature_names=np.asarray(EDGE_FEATURE_NAMES),
    )

    with pytest.raises(ValueError, match="node_features must contain only finite"):
        DatasetFactory(_manifest(tmp_path, "sample.npz")).build()


def test_graph_json_loader_rejects_fractional_edge_indices(tmp_path):
    nodes, edges, attrs = _arrays()
    fractional_edges = edges.astype(np.float64)
    fractional_edges[0, 0] = 0.5
    payload = _graph_json_payload(fractional_edges, attrs, EDGE_FEATURE_NAMES)
    path = tmp_path / "fractional.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="edge_index must contain integer-valued"):
        GraphJsonLoader().load(path)


def test_graph_json_loader_rejects_non_binary_masks(tmp_path):
    nodes, edges, attrs = _arrays()
    payload = _graph_json_payload(
        edges, attrs, EDGE_FEATURE_NAMES, node_mask=np.asarray([[1, 2, 1]])
    )
    path = tmp_path / "bad-mask.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="node_mask must contain only boolean or 0/1"):
        GraphJsonLoader().load(path)


def test_graph_json_loader_rejects_non_finite_node_features(tmp_path):
    nodes, edges, attrs = _arrays()
    nodes[0, 1, 0] = np.inf
    payload = _graph_json_payload(edges, attrs, EDGE_FEATURE_NAMES)
    payload["graph"]["node_features"] = nodes.tolist()
    path = tmp_path / "non-finite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="node_features must contain only finite"):
        GraphJsonLoader().load(path)
