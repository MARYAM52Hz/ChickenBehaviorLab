import numpy as np

from chicken_behavior_lab.graphs import (
    TemporalEdgeBuilder,
)


def test_temporal_edges_bidirectional():

    builder = TemporalEdgeBuilder(
        bidirectional=True
    )

    edges = builder.build(
        num_frames=3,
        num_keypoints=2,
    )

    expected = np.array(
        [
            [
                0, 2,
                1, 3,
                2, 4,
                3, 5,
            ],
            [
                2, 0,
                3, 1,
                4, 2,
                5, 3,
            ],
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(
        edges,
        expected,
    )


def test_temporal_edges_unidirectional():

    builder = TemporalEdgeBuilder(
        bidirectional=False
    )

    edges = builder.build(
        num_frames=3,
        num_keypoints=2,
    )

    expected = np.array(
        [
            [
                0,
                1,
                2,
                3,
            ],
            [
                2,
                3,
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


def test_single_frame_has_no_temporal_edges():

    builder = TemporalEdgeBuilder()

    edges = builder.build(
        num_frames=1,
        num_keypoints=5,
    )

    assert edges.shape == (
        2,
        0,
    )


def test_invalid_num_frames():

    builder = TemporalEdgeBuilder()

    try:

        builder.build(
            num_frames=0,
            num_keypoints=5,
        )

        assert False

    except ValueError:
        pass
