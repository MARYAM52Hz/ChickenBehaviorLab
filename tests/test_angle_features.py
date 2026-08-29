"""
Tests for ChickenBehaviorLab edge features.
"""

import numpy as np

from chicken_behavior_lab.graph.edge_features import (
    EdgeFeatureExtractor,
)


def test_edge_feature_shape():

    node_features = np.zeros(
        (
            5,
            3,
            7,
        ),
        dtype=np.float32,
    )

    edge_index = np.array(
        [
            [0, 1],
            [1, 2],
        ],
        dtype=np.int64,
    )

    extractor = EdgeFeatureExtractor()

    edge_features = extractor.extract(
        node_features,
        edge_index,
    )

    assert (
        edge_features.shape
        == (5, 2, 3)
    )


def test_relative_position():

    node_features = np.zeros(
        (
            1,
            2,
            7,
        ),
        dtype=np.float32,
    )

    node_features[
        0,
        0,
        0,
    ] = 10.0

    node_features[
        0,
        0,
        1,
    ] = 20.0

    node_features[
        0,
        1,
        0,
    ] = 13.0

    node_features[
        0,
        1,
        1,
    ] = 24.0

    edge_index = np.array(
        [
            [0],
            [1],
        ],
        dtype=np.int64,
    )

    extractor = EdgeFeatureExtractor()

    edge_features = extractor.extract(
        node_features,
        edge_index,
    )

    assert np.allclose(
        edge_features[
            0,
            0,
        ],
        [
            3.0,
            4.0,
            5.0,
        ],
    )


def test_reverse_edge():

    node_features = np.zeros(
        (
            1,
            2,
            7,
        ),
        dtype=np.float32,
    )

    node_features[
        0,
        0,
        0:2,
    ] = [
        0.0,
        0.0,
    ]

    node_features[
        0,
        1,
        0:2,
    ] = [
        3.0,
        4.0,
    ]

    edge_index = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=np.int64,
    )

    extractor = EdgeFeatureExtractor()

    edge_features = extractor.extract(
        node_features,
        edge_index,
    )

    assert np.allclose(
        edge_features[
            0,
            0,
        ],
        [
            3.0,
            4.0,
            5.0,
        ],
    )

    assert np.allclose(
        edge_features[
            0,
            1,
        ],
        [
            -3.0,
            -4.0,
            5.0,
        ],
    )


def test_multiple_frames():

    node_features = np.zeros(
        (
            2,
            2,
            7,
        ),
        dtype=np.float32,
    )

    node_features[
        0,
        1,
        0,
    ] = 3.0

    node_features[
        1,
        1,
        0,
    ] = 4.0

    edge_index = np.array(
        [
            [0],
            [1],
        ],
        dtype=np.int64,
    )

    extractor = EdgeFeatureExtractor()

    edge_features = extractor.extract(
        node_features,
        edge_index,
    )

    assert np.isclose(
        edge_features[
            0,
            0,
            2,
        ],
        3.0,
    )

    assert np.isclose(
        edge_features[
            1,
            0,
            2,
        ],
        4.0,
    )
