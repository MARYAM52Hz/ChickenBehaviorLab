import numpy as np

from chicken_behavior_lab.graphs import (
    EdgeFeatureBuilder,
)


def test_edge_features():

    node_features = np.array(
        [
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 3.0],
            ]
        ],
        dtype=np.float32,
    )

    edge_index = np.array(
        [
            [0, 1],
            [1, 2],
        ],
        dtype=np.int64,
    )

    builder = EdgeFeatureBuilder()

    edge_features = builder.build(
        node_features,
        edge_index,
    )

    assert edge_features.shape == (
        1,
        2,
        2,
    )

    np.testing.assert_allclose(
        edge_features[0, 0],
        [1.0, 1.0],
    )

    np.testing.assert_allclose(
        edge_features[0, 1],
        [1.0, 2.0],
    )


def test_invalid_edge_index():

    node_features = np.zeros(
        (2, 3, 2),
        dtype=np.float32,
    )

    edge_index = np.array(
        [
            [0, 3],
            [1, 2],
        ],
        dtype=np.int64,
    )

    builder = EdgeFeatureBuilder()

    try:
        builder.build(
            node_features,
            edge_index,
        )

        assert False

    except ValueError:
        pass
