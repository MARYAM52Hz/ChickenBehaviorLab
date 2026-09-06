import numpy as np

from chicken_behavior_lab.graphs import (
    SpatialEdgeExpander,
)


def test_spatial_edge_expansion():

    skeleton_edges = np.array(
        [
            [0, 1],
            [1, 2],
        ],
        dtype=np.int64,
    )

    builder = SpatialEdgeExpander()

    edges = builder.build(
        skeleton_edge_index=(
            skeleton_edges
        ),
        num_frames=2,
        num_keypoints=3,
    )

    expected = np.array(
        [
            [
                0,
                1,
                3,
                4,
            ],
            [
                1,
                2,
                4,
                5,
            ],
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(
        edges,
        expected,
    )
