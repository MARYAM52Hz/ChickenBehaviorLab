import numpy as np

from chicken_behavior_lab.features import TemporalFeatureSequence
from chicken_behavior_lab.graphs import SpatioTemporalGraphBuilder
from chicken_behavior_lab.graph.construction import SkeletonGraphBuilder


def create_sequence():
    features = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.1, 0.0], [1.1, 0.0], [2.1, 0.0]],
        ],
        dtype=np.float32,
    )
    return TemporalFeatureSequence(
        features=features,
        frame_mask=np.array([True, True]),
        keypoint_mask=np.ones((2, 3), dtype=bool),
        frame_ids=("100", "101"),
    )


def test_spatiotemporal_graph_has_bidirectional_spatial_and_temporal_edges():
    graph = SpatioTemporalGraphBuilder(
        np.array([[0, 1], [1, 2]], dtype=np.int64),
        bidirectional_temporal_edges=True,
    ).build(create_sequence())

    assert graph.num_frames == 2
    assert graph.num_keypoints == 3
    assert graph.num_nodes == 6
    assert graph.node_features.shape == (6, 2)
    assert graph.num_spatial_edges == 8
    assert graph.num_temporal_edges == 6
    assert graph.num_edges == 14
    assert graph.edge_features.shape == (14, 9)


def test_temporal_edge_type_is_preserved():
    graph = SpatioTemporalGraphBuilder(
        np.array([[0], [1]], dtype=np.int64)
    ).build(create_sequence())
    assert set(np.unique(graph.edge_type).tolist()) == {0, 1}


def test_frame_ids_are_preserved():
    graph = SpatioTemporalGraphBuilder(
        np.array([[0], [1]], dtype=np.int64)
    ).build(create_sequence())
    assert graph.frame_ids == ("100", "101")


def test_spatiotemporal_graph_preserves_masks_and_zeros_invalid_edges():
    sequence = create_sequence()
    sequence.frame_mask[1] = False
    sequence.keypoint_mask[0, 1] = False
    graph = SpatioTemporalGraphBuilder(
        np.array([[0, 1], [1, 2]], dtype=np.int64)
    ).build(sequence)

    expected_node_mask = (
        sequence.keypoint_mask & sequence.frame_mask[:, None]
    ).reshape(-1)
    assert np.array_equal(graph.frame_mask, sequence.frame_mask)
    assert np.array_equal(graph.node_mask, expected_node_mask)
    source, target = graph.edge_index
    invalid = ~(expected_node_mask[source] & expected_node_mask[target])
    assert np.allclose(graph.edge_features[invalid], 0.0)



def test_both_graph_representations_share_spatial_topology_and_edge_features():
    sequence = create_sequence()
    connections = ((0, 1), (1, 2))
    temporal_graph = SkeletonGraphBuilder(connections).build(sequence)
    flattened_graph = SpatioTemporalGraphBuilder(
        np.array(connections, dtype=np.int64).T
    ).build(sequence)

    np.testing.assert_array_equal(
        flattened_graph.edge_index[:, :temporal_graph.num_edges],
        temporal_graph.edge_index,
    )
    np.testing.assert_allclose(
        flattened_graph.edge_features[:temporal_graph.num_edges],
        temporal_graph.edge_features[0],
    )
