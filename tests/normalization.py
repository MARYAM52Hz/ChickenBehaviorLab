"""
Tests for ChickenBehaviorLab feature normalization.
"""

import numpy as np
import pytest

from chicken_behavior_lab.features.motion import (
    MotionFeatureFrame,
)

from chicken_behavior_lab.features.temporal_features import (
    TemporalFeatureSequence,
)

from chicken_behavior_lab.features.normalization import (
    MotionFeatureNormalizer,
)


# =========================================================
# Helpers
# =========================================================

def create_motion_frame(
    value: float,
    num_keypoints: int = 3,
) -> MotionFeatureFrame:
    """
    Create a synthetic MotionFeatureFrame for testing.
    """

    features = np.zeros(
        (
            num_keypoints,
            7,
        ),
        dtype=np.float32,
    )

    features[:, 0] = value
    features[:, 1] = value

    features[:, 2] = value
    features[:, 3] = value

    features[:, 4] = value
    features[:, 5] = value

    features[:, 6] = 1.0

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


def create_sequence(
    value: float,
    temporal_length: int = 5,
    num_keypoints: int = 3,
) -> TemporalFeatureSequence:
    """
    Create a synthetic temporal sequence.
    """

    features = np.full(
        (
            temporal_length,
            num_keypoints,
            7,
        ),
        value,
        dtype=np.float32,
    )

    frame_mask = np.ones(
        temporal_length,
        dtype=bool,
    )

    keypoint_mask = np.ones(
        (
            temporal_length,
            num_keypoints,
        ),
        dtype=bool,
    )

    return TemporalFeatureSequence(
        features=features,
        frame_mask=frame_mask,
        keypoint_mask=keypoint_mask,
        frame_ids=tuple(
            f"frame_{i}"
            for i in range(
                temporal_length
            )
        ),
    )


# =========================================================
# Fit tests
# =========================================================

def test_normalizer_can_be_fitted():

    sequences = [
        create_sequence(1.0),
        create_sequence(2.0),
        create_sequence(3.0),
    ]

    normalizer = MotionFeatureNormalizer(
        reference_keypoint=0,
        scale_keypoints=(0, 1),
    )

    normalizer.fit(
        sequences
    )

    assert normalizer.is_fitted()


def test_fit_creates_statistics():

    sequences = [
        create_sequence(1.0),
        create_sequence(2.0),
        create_sequence(3.0),
    ]

    normalizer = MotionFeatureNormalizer()

    normalizer.fit(
        sequences
    )

    assert (
        normalizer.statistics
        is not None
    )

    assert (
        normalizer.statistics
        .velocity_mean.shape
        == (2,)
    )

    assert (
        normalizer.statistics
        .velocity_std.shape
        == (2,)
    )

    assert (
        normalizer.statistics
        .acceleration_mean.shape
        == (2,)
    )

    assert (
        normalizer.statistics
        .acceleration_std.shape
        == (2,)
    )


# =========================================================
# Transform tests
# =========================================================

def test_transform_preserves_shape():

    sequences = [
        create_sequence(1.0),
        create_sequence(2.0),
        create_sequence(3.0),
    ]

    normalizer = MotionFeatureNormalizer()

    normalizer.fit(
        sequences
    )

    sequence = create_sequence(
        2.0
    )

    normalized = normalizer.transform(
        sequence
    )

    assert (
        normalized.features.shape
        == sequence.features.shape
    )


def test_transform_preserves_frame_mask():

    sequences = [
        create_sequence(1.0),
        create_sequence(2.0),
    ]

    normalizer = MotionFeatureNormalizer()

    normalizer.fit(
        sequences
    )

    sequence = create_sequence(
        2.0
    )

    sequence.frame_mask[:] = [
        True,
        True,
        True,
        False,
        False,
    ]

    normalized = normalizer.transform(
        sequence
    )

    assert np.array_equal(
        normalized.frame_mask,
        sequence.frame_mask,
    )


def test_transform_preserves_keypoint_mask():

    sequences = [
        create_sequence(1.0),
        create_sequence(2.0),
    ]

    normalizer = MotionFeatureNormalizer()

    normalizer.fit(
        sequences
    )

    sequence = create_sequence(
        2.0
    )

    sequence.keypoint_mask[
        2,
        1,
    ] = False

    normalized = normalizer.transform(
        sequence
    )

    assert np.array_equal(
        normalized.keypoint_mask,
        sequence.keypoint_mask,
    )


# =========================================================
# Padding tests
# =========================================================

def test_invalid_keypoints_are_zeroed():

    sequences = [
        create_sequence(1.0),
        create_sequence(2.0),
    ]

    normalizer = MotionFeatureNormalizer()

    normalizer.fit(
        sequences
    )

    sequence = create_sequence(
        2.0
    )

    sequence.keypoint_mask[
        2,
        1,
    ] = False

    normalized = normalizer.transform(
        sequence
    )

    assert np.all(
        normalized.features[
            2,
            1,
        ]
        == 0.0
    )


def test_padding_frames_are_zeroed():

    sequences = [
        create_sequence(1.0),
        create_sequence(2.0),
    ]

    normalizer = MotionFeatureNormalizer()

    normalizer.fit(
        sequences
    )

    sequence = create_sequence(
        2.0
    )

    sequence.frame_mask[:] = [
        False,
        False,
        True,
        True,
        True,
    ]

    normalized = normalizer.transform(
        sequence
    )

    assert np.all(
        normalized.features[
            0
        ]
        == 0.0
    )

    assert np.all(
        normalized.features[
            1
        ]
        == 0.0
    )


# =========================================================
# Confidence tests
# =========================================================

def test_confidence_is_clipped():

    sequences = [
        create_sequence(1.0),
        create_sequence(2.0),
    ]

    normalizer = MotionFeatureNormalizer()

    normalizer.fit(
        sequences
    )

    sequence = create_sequence(
        2.0
    )

    sequence.features[
        :, :, 6
    ] = 2.5

    normalized = normalizer.transform(
        sequence
    )

    assert np.all(
        normalized.features[
            :, :, 6
        ]
        <= 1.0
    )


def test_negative_confidence_is_clipped():

    sequences = [
        create_sequence(1.0),
        create_sequence(2.0),
    ]

    normalizer = MotionFeatureNormalizer()

    normalizer.fit(
        sequences
    )

    sequence = create_sequence(
        2.0
    )

    sequence.features[
        :, :, 6
    ] = -1.0

    normalized = normalizer.transform(
        sequence
    )

    assert np.all(
        normalized.features[
            :, :, 6
        ]
        >= 0.0
    )


# =========================================================
# Error handling
# =========================================================

def test_transform_before_fit_raises_error():

    normalizer = MotionFeatureNormalizer()

    sequence = create_sequence(
        1.0
    )

    with pytest.raises(
        RuntimeError
    ):

        normalizer.transform(
            sequence
        )


def test_fit_with_empty_sequences_raises_error():

    normalizer = MotionFeatureNormalizer()

    with pytest.raises(
        ValueError
    ):

        normalizer.fit([])


# =========================================================
# Numerical stability
# =========================================================

def test_zero_variance_features_are_safe():

    sequences = [
        create_sequence(1.0),
        create_sequence(1.0),
        create_sequence(1.0),
    ]

    normalizer = MotionFeatureNormalizer()

    normalizer.fit(
        sequences
    )

    assert np.all(
        normalizer.statistics
        .velocity_std
        >= normalizer.epsilon
    )

    assert np.all(
        normalizer.statistics
        .acceleration_std
        >= normalizer.epsilon
    )
