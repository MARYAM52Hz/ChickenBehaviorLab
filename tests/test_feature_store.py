import numpy as np

from chicken_behavior_lab.features import (
    FeatureStore,
    TemporalFeatureSequence,
)


def create_sequence():
    """
    Create a deterministic feature sequence
    for testing.
    """

    features = np.arange(
        4 * 3 * 7,
        dtype=np.float32,
    ).reshape(
        4,
        3,
        7,
    )

    frame_mask = np.array(
        [
            True,
            True,
            True,
            True,
        ],
        dtype=bool,
    )

    keypoint_mask = np.ones(
        (
            4,
            3,
        ),
        dtype=bool,
    )

    frame_ids = (
        "100",
        "101",
        "102",
        "103",
    )

    return TemporalFeatureSequence(
        features=features,
        frame_mask=frame_mask,
        keypoint_mask=keypoint_mask,
        frame_ids=frame_ids,
    )


def test_feature_store_save_and_load(
    tmp_path,
):

    store = FeatureStore(
        tmp_path
    )

    sequence = create_sequence()

    path = store.save(
        video_id="video_001",
        track_id=1,
        sequence=sequence,
    )

    assert path.exists()

    loaded = store.load(
        video_id="video_001",
        track_id=1,
    )

    assert np.array_equal(
        loaded.features,
        sequence.features,
    )

    assert np.array_equal(
        loaded.frame_mask,
        sequence.frame_mask,
    )

    assert np.array_equal(
        loaded.keypoint_mask,
        sequence.keypoint_mask,
    )

    assert (
        loaded.frame_ids
        == sequence.frame_ids
    )


def test_feature_store_exists(
    tmp_path,
):

    store = FeatureStore(
        tmp_path
    )

    sequence = create_sequence()

    assert not store.exists(
        "video_001",
        1,
    )

    store.save(
        "video_001",
        1,
        sequence,
    )

    assert store.exists(
        "video_001",
        1,
    )


def test_feature_store_delete(
    tmp_path,
):

    store = FeatureStore(
        tmp_path
    )

    sequence = create_sequence()

    store.save(
        "video_001",
        1,
        sequence,
    )

    assert store.exists(
        "video_001",
        1,
    )

    store.delete(
        "video_001",
        1,
    )

    assert not store.exists(
        "video_001",
        1,
    )


def test_feature_store_missing_file(
    tmp_path,
):

    store = FeatureStore(
        tmp_path
    )

    try:
        store.load(
            "video_001",
            1,
        )

        assert False, (
            "Expected FileNotFoundError"
        )

    except FileNotFoundError:
        pass


def test_feature_store_frame_slice(
    tmp_path,
):

    store = FeatureStore(
        tmp_path
    )

    sequence = create_sequence()

    store.save(
        "video_001",
        1,
        sequence,
    )

    sliced = store.load(
        video_id="video_001",
        track_id=1,
        start_frame=101,
        end_frame=102,
    )

    assert (
        sliced.features.shape
        == (2, 3, 7)
    )

    assert sliced.frame_ids == (
        "101",
        "102",
    )

    assert np.array_equal(
        sliced.features,
        sequence.features[1:3],
    )
