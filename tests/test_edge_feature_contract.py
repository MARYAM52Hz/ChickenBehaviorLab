import numpy as np
import pytest

from chicken_behavior_lab.features.edge_contract import (
    EDGE_FEATURE_NAMES,
    build_combined_edge_features,
)


def test_combined_edge_features_include_geometry_motion_and_endpoint_confidence():
    nodes = np.array(
        [[[0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.8],
          [3.0, 4.0, 4.0, 6.0, 8.0, 10.0, 0.6]]],
        dtype=np.float32,
    )
    edges = np.array([[0], [1]], dtype=np.int64)
    result = build_combined_edge_features(
        nodes, edges,
        ("x", "y", "vx", "vy", "ax", "ay", "confidence"),
    )
    assert result.shape == (1, 1, 9)
    assert EDGE_FEATURE_NAMES == (
        "relative_dx", "relative_dy", "distance",
        "delta_vx", "delta_vy", "delta_ax", "delta_ay",
        "source_confidence", "target_confidence",
    )
    np.testing.assert_allclose(
        result[0, 0],
        [3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0, 0.8, 0.6],
    )


def test_combined_edge_features_support_legacy_xy_inputs():
    nodes = np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    edges = np.array([[0], [1]], dtype=np.int64)
    result = build_combined_edge_features(nodes, edges)
    assert result.shape == (1, 9)
    np.testing.assert_allclose(result[0], [0.0, 2.0, 2.0, 0, 0, 0, 0, 0, 0])


def test_combined_edge_features_reject_ambiguous_unnamed_dimensions():
    nodes = np.zeros((3, 4), dtype=np.float32)
    edges = np.array([[0], [1]], dtype=np.int64)
    with pytest.raises(ValueError, match="feature_names are required"):
        build_combined_edge_features(nodes, edges)
