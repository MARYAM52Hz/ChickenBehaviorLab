"""Tests for combined geometric and motion edge features."""

import numpy as np

from chicken_behavior_lab.graph.edge_features import EdgeFeatureExtractor


def test_edge_feature_shape():
    nodes = np.zeros((5, 3, 7), dtype=np.float32)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    result = EdgeFeatureExtractor().extract(nodes, edges)
    assert result.shape == (5, 2, 9)


def test_relative_position():
    nodes = np.zeros((1, 2, 7), dtype=np.float32)
    nodes[0, 0, :2] = [10.0, 20.0]
    nodes[0, 1, :2] = [13.0, 24.0]
    edges = np.array([[0], [1]], dtype=np.int64)
    result = EdgeFeatureExtractor().extract(nodes, edges)
    np.testing.assert_allclose(result[0, 0], [3.0, 4.0, 5.0, 0, 0, 0, 0, 0, 0])


def test_reverse_edge():
    nodes = np.zeros((1, 2, 7), dtype=np.float32)
    nodes[0, 1, :2] = [3.0, 4.0]
    edges = np.array([[0, 1], [1, 0]], dtype=np.int64)
    result = EdgeFeatureExtractor().extract(nodes, edges)
    np.testing.assert_allclose(result[0, 0], [3.0, 4.0, 5.0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(result[0, 1], [-3.0, -4.0, 5.0, 0, 0, 0, 0, 0, 0])


def test_multiple_frames():
    nodes = np.zeros((2, 2, 7), dtype=np.float32)
    nodes[0, 1, 0] = 3.0
    nodes[1, 1, 0] = 4.0
    edges = np.array([[0], [1]], dtype=np.int64)
    result = EdgeFeatureExtractor().extract(nodes, edges)
    assert np.isclose(result[0, 0, 2], 3.0)
    assert np.isclose(result[1, 0, 2], 4.0)
