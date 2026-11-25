"""Tests for temporal graph construction and the shared edge contract."""

import numpy as np

from chicken_behavior_lab.features.temporal_features import TemporalFeatureSequence
from chicken_behavior_lab.graph.graph import SkeletonGraph, TemporalSkeletonGraph
from chicken_behavior_lab.graph.construction import SkeletonGraphBuilder


def create_sequence():
    return TemporalFeatureSequence(
        features=np.zeros((4, 3, 7), dtype=np.float32),
        frame_mask=np.ones(4, dtype=bool),
        keypoint_mask=np.ones((4, 3), dtype=bool),
        frame_ids=("0", "1", "2", "3"),
    )


def test_edge_index_is_undirected():
    edge_index = SkeletonGraphBuilder([(0, 1), (1, 2)]).build_edge_index()
    expected = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    assert np.array_equal(edge_index, expected)


def test_temporal_graph_shape():
    graph = SkeletonGraphBuilder([(0, 1), (1, 2)]).build(create_sequence())
    assert graph.node_features.shape == (4, 3, 7)
    assert graph.edge_index.shape == (2, 4)


def test_combined_edge_features_are_attached():
    sequence = create_sequence()
    sequence.features[0, 0, 0:2] = [0.0, 0.0]
    sequence.features[0, 1, 0:2] = [3.0, 4.0]
    graph = SkeletonGraphBuilder([(0, 1)]).build(sequence)
    assert graph.edge_features.shape == (4, 2, 9)
    np.testing.assert_allclose(
        graph.edge_features[0, 0],
        [3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        graph.edge_features[0, 1],
        [-3.0, -4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )


def test_invalid_edges_are_zeroed_and_masks_preserved():
    sequence = create_sequence()
    sequence.frame_mask[0] = False
    sequence.keypoint_mask[1, 2] = False
    graph = SkeletonGraphBuilder([(0, 1), (1, 2)]).build(sequence)
    assert not graph.frame_mask[0]
    assert not graph.node_mask[1, 2]
    invalid = ~(
        graph.frame_mask[:, None]
        & graph.node_mask[:, graph.edge_index[0]]
        & graph.node_mask[:, graph.edge_index[1]]
    )
    assert np.allclose(graph.edge_features[invalid], 0.0)


def test_graph_validation_and_combined_edge_dimension():
    graph = SkeletonGraph(
        node_features=np.zeros((3, 7), dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
    )
    graph.validate()
    assert graph.num_nodes == 3
    assert graph.num_edges == 2

    temporal = SkeletonGraphBuilder([(0, 1), (1, 2)]).build(create_sequence())
    temporal.validate()
    assert isinstance(temporal, TemporalSkeletonGraph)
    assert temporal.temporal_length == 4
    assert temporal.num_nodes == 3
    assert temporal.num_edges == 4
    assert temporal.edge_feature_dimension == 9
