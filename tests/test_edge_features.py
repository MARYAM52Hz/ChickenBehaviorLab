import numpy as np
import pytest

from chicken_behavior_lab.graphs import EdgeFeatureBuilder


def test_edge_features_use_combined_contract():
    nodes = np.array(
        [[[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]]],
        dtype=np.float32,
    )
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    result = EdgeFeatureBuilder().build(nodes, edges)

    assert result.shape == (1, 2, 9)
    np.testing.assert_allclose(
        result[0, 0],
        [1.0, 1.0, np.sqrt(2.0), 0, 0, 0, 0, 0, 0],
    )
    np.testing.assert_allclose(
        result[0, 1],
        [1.0, 2.0, np.sqrt(5.0), 0, 0, 0, 0, 0, 0],
    )


def test_invalid_edge_index():
    nodes = np.zeros((2, 3, 2), dtype=np.float32)
    edges = np.array([[0, 3], [1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="invalid node"):
        EdgeFeatureBuilder().build(nodes, edges)
