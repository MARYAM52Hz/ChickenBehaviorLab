"""
Tests for temporal feature sequence construction.
"""

import numpy as np

from chicken_behavior_lab.features.motion import (
    MotionFeatureFrame,
)

from chicken_behavior_lab.features.temporal_features import (
    TemporalFeatureBuilder,
)


def create_frame(
    value: float,
    num_keypoints: int = 2,
    feature_dimension: int = 7,
) -> MotionFeatureFrame:

    features = np.full(
        (
            num_keypoints,
            feature_dimension,
        ),
        value,
        dtype=np.float32,
    )

    return MotionFeatureFrame(
        features=features,
        angles=np.empty(
            (0,),
            dtype=np.float32,
        ),
        angle_valid_mask=np.empty(
            (0,),
            dtype=bool,
        ),
        feature_names=(
            "x",
            "y",
            "vx",
            "vy",
            "ax",
            "ay",
            "confidence",
        ),
    )


def test_temporal_sequence_shape():

    frames = [
        create_frame(1.0),
        create_frame(2.0),
        create_frame(3.0),
    ]

    builder = TemporalFeatureBuilder(
        sequence_length=5
    )

    sequence = builder.build(
        frames,
        frame_ids=[
            "frame_1",
            "frame_2",
            "frame_3",
        ],
    )

    assert sequence.shape == (
        5,
        2,
        7,
    )


def test_temporal_padding():

    frames = [
        create_frame(1.0),
        create_frame(2.0),
        create_frame(3.0),
    ]

    builder = TemporalFeatureBuilder(
        sequence_length=5
    )

    sequence = builder.build(
        frames
    )

    assert np.all(
        sequence.frame_mask
        == np.array(
            [
                False,
                False,
                True,
                True,
                True,
            ]
        )
    )

    assert np.allclose(
        sequence.features[2],
        1.0,
    )

    assert np.allclose(
        sequence.features[3],
        2.0,
    )

    assert np.allclose(
        sequence.features[4],
        3.0,
    )


def test_sequence_keeps_latest_frames():

    frames = [
        create_frame(1.0),
        create_frame(2.0),
        create_frame(3.0),
        create_frame(4.0),
        create_frame(5.0),
    ]

    builder = TemporalFeatureBuilder(
        sequence_length=3
    )

    sequence = builder.build(
        frames
    )

    assert np.allclose(
        sequence.features[0],
        3.0,
    )

    assert np.allclose(
        sequence.features[1],
        4.0,
    )

    assert np.allclose(
        sequence.features[2],
        5.0,
    )


def test_frame_ids_are_preserved():

    frames = [
        create_frame(1.0),
        create_frame(2.0),
    ]

    builder = TemporalFeatureBuilder(
        sequence_length=4
    )

    sequence = builder.build(
        frames,
        frame_ids=[
            "frame_100",
            "frame_101",
        ],
    )

    assert sequence.frame_ids == (
        "",
        "",
        "frame_100",
        "frame_101",
    )
