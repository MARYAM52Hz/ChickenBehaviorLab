import numpy as np

from chicken_behavior_lab.features import TemporalFeatureSequence
from chicken_behavior_lab.graphs import TemporalSkeletonGraphBuilder


def test_temporal_graph_builder_uses_combined_edge_contract():
    features = np.array(
        [
            [[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]],
            [[0.5, 0.0], [1.5, 1.0], [2.5, 3.0]],
        ],
        dtype=np.float32,
    )
    sequence = TemporalFeatureSequence(
        features=features,
        frame_mask=np.array([True, True]),
        keypoint_mask=np.ones((2, 3), dtype=bool),
        frame_ids=("100", "101"),
    )
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    graph = TemporalSkeletonGraphBuilder(edge_index).build(sequence)

    assert graph.num_frames == 2
    assert graph.num_nodes == 3
    assert graph.num_edges == 4
    assert graph.edge_features.shape == (2, 4, 9)
    np.testing.assert_array_equal(
        graph.edge_index,
        np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
    )
    np.testing.assert_allclose(
        graph.edge_features[0],
        [
            [1.0, 1.0, np.sqrt(2.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, np.sqrt(2.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, np.sqrt(5.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -2.0, np.sqrt(5.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    )


def test_temporal_graph_builder_preserves_masks_and_zeros_invalid_edges():
    features = np.ones((2, 3, 2), dtype=np.float32)
    frame_mask = np.array([True, False], dtype=bool)
    node_mask = np.ones((2, 3), dtype=bool)
    node_mask[0, 1] = False
    sequence = TemporalFeatureSequence(
        features=features,
        frame_mask=frame_mask,
        keypoint_mask=node_mask,
        frame_ids=("200", "201"),
    )
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    graph = TemporalSkeletonGraphBuilder(edge_index).build(sequence)

    assert np.array_equal(graph.frame_mask, frame_mask)
    assert np.array_equal(graph.node_mask, node_mask)
    assert graph.edge_index.shape == (2, 4)
    assert np.allclose(graph.edge_features[0], 0.0)
    assert np.allclose(graph.edge_features[1], 0.0)
