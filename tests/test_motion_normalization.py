"""Tests for anatomy-aware, mask-preserving motion normalization."""

import numpy as np
import pytest

from chicken_behavior_lab.core.enums.keypoints import KeypointType
from chicken_behavior_lab.features.normalization import MotionFeatureNormalizer
from chicken_behavior_lab.features.temporal_features import TemporalFeatureSequence


def make_sequence(*, missing_left_hip=False):
    features = np.zeros((2, len(KeypointType), 7), dtype=np.float32)
    mask = np.ones((2, len(KeypointType)), dtype=bool)
    center = tuple(KeypointType).index(KeypointType.BODY_CENTER)
    left = tuple(KeypointType).index(KeypointType.LEFT_HIP)
    right = tuple(KeypointType).index(KeypointType.RIGHT_HIP)
    for frame in range(2):
        features[frame, :, 0] = np.arange(len(KeypointType), dtype=np.float32)
        features[frame, :, 1] = 1.0
        features[frame, center, :2] = (5.0, 5.0)
        features[frame, left, :2] = (0.0, 5.0)
        features[frame, right, :2] = (10.0, 5.0)
        features[frame, :, 2:6] = (1.0, 2.0, 3.0, 4.0)
        features[frame, :, 6] = 0.9
    if missing_left_hip:
        mask[:, left] = False
    return TemporalFeatureSequence(
        features=features,
        frame_mask=np.array([True, True]),
        keypoint_mask=mask,
        frame_ids=("10", "11"),
    )


def test_normalizer_uses_body_center_and_hip_distance():
    training = make_sequence()
    normalizer = MotionFeatureNormalizer()
    normalizer.fit([training])
    transformed = normalizer.transform(training)

    center = tuple(KeypointType).index(KeypointType.BODY_CENTER)
    left = tuple(KeypointType).index(KeypointType.LEFT_HIP)
    right = tuple(KeypointType).index(KeypointType.RIGHT_HIP)
    assert np.allclose(transformed.features[:, center, :2], 0.0)
    assert np.allclose(transformed.features[:, left, 0], -0.5)
    assert np.allclose(transformed.features[:, right, 0], 0.5)
    assert transformed.keypoint_mask.all()


def test_missing_required_anatomical_landmark_masks_whole_frame():
    training = make_sequence()
    normalizer = MotionFeatureNormalizer()
    normalizer.fit([training])
    transformed = normalizer.transform(make_sequence(missing_left_hip=True))
    assert not transformed.keypoint_mask.any()
    assert np.allclose(transformed.features, 0.0)


def test_fit_rejects_sequences_without_valid_motion_rows():
    sequence = make_sequence(missing_left_hip=True)
    sequence.keypoint_mask[:] = False
    with pytest.raises(ValueError, match="No valid motion feature rows"):
        MotionFeatureNormalizer().fit([sequence])


def test_normalizer_requires_fit_before_transform():
    with pytest.raises(RuntimeError, match="must be fitted"):
        MotionFeatureNormalizer().transform(make_sequence())


def test_fit_excludes_frames_without_required_body_landmarks():
    sequence = make_sequence()
    left = tuple(KeypointType).index(KeypointType.LEFT_HIP)
    sequence.keypoint_mask[0, left] = False
    sequence.features[0, :, 2:4] = 1000.0
    normalizer = MotionFeatureNormalizer()
    normalizer.fit([sequence])
    np.testing.assert_allclose(normalizer.statistics.velocity_mean, [1.0, 2.0])
