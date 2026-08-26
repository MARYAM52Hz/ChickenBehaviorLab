"""
Tests for acceleration feature extraction.
"""

import numpy as np

from chicken_behavior_lab.features.acceleration import (
    AccelerationFeatureExtractor,
)


def test_acceleration_extraction():

    previous_velocity = np.array(
        [
            [100.0, 50.0],
        ],
        dtype=np.float32,
    )

    current_velocity = np.array(
        [
            [160.0, 80.0],
        ],
        dtype=np.float32,
    )

    extractor = (
        AccelerationFeatureExtractor()
    )

    features = extractor.extract(
        previous_velocity=previous_velocity,
        current_velocity=current_velocity,
        delta_time=0.1,
    )

    expected_acceleration = np.array(
        [
            [600.0, 300.0],
        ],
        dtype=np.float32,
    )

    assert np.allclose(
        features.acceleration,
        expected_acceleration,
    )

    expected_magnitude = np.sqrt(
        600.0**2 + 300.0**2
    )

    assert np.isclose(
        features.magnitude[0],
        expected_magnitude,
    )

    assert features.valid_mask[0]


def test_invalid_velocity_is_masked():

    previous_velocity = np.array(
        [
            [100.0, 50.0],
        ],
        dtype=np.float32,
    )

    current_velocity = np.array(
        [
            [160.0, 80.0],
        ],
        dtype=np.float32,
    )

    previous_valid_mask = np.array(
        [False],
        dtype=bool,
    )

    current_valid_mask = np.array(
        [True],
        dtype=bool,
    )

    extractor = (
        AccelerationFeatureExtractor()
    )

    features = extractor.extract(
        previous_velocity=previous_velocity,
        current_velocity=current_velocity,
        delta_time=0.1,
        previous_valid_mask=previous_valid_mask,
        current_valid_mask=current_valid_mask,
    )

    assert not features.valid_mask[0]

    assert np.allclose(
        features.acceleration[0],
        [0.0, 0.0],
    )

    assert np.isclose(
        features.magnitude[0],
        0.0,
    )


def test_invalid_delta_time_raises_error():

    velocity = np.array(
        [
            [100.0, 50.0],
        ],
        dtype=np.float32,
    )

    extractor = (
        AccelerationFeatureExtractor()
    )

    try:

        extractor.extract(
            previous_velocity=velocity,
            current_velocity=velocity,
            delta_time=0.0,
        )

        assert False

    except ValueError as error:

        assert "delta_time" in str(
            error
        )


def test_velocity_shape_mismatch_raises_error():

    previous_velocity = np.array(
        [
            [100.0, 50.0],
            [20.0, 10.0],
        ],
        dtype=np.float32,
    )

    current_velocity = np.array(
        [
            [160.0, 80.0],
        ],
        dtype=np.float32,
    )

    extractor = (
        AccelerationFeatureExtractor()
    )

    try:

        extractor.extract(
            previous_velocity=previous_velocity,
            current_velocity=current_velocity,
            delta_time=0.1,
        )

        assert False

    except ValueError as error:

        assert "Velocity shape mismatch" in str(
            error
        )
