"""Integration test for the canonical tracked-skeleton feature path."""

import numpy as np

from chicken_behavior_lab.core.enums.keypoints import KeypointType
from chicken_behavior_lab.features.feature_store import FeatureStore
from chicken_behavior_lab.features.motion import MotionFeatureExtractor
from chicken_behavior_lab.features.normalization import MotionFeatureNormalizer
from chicken_behavior_lab.features.temporal_features import TemporalFeatureBuilder
from chicken_behavior_lab.graph.construction import SkeletonGraphBuilder
from chicken_behavior_lab.models.keypoint import Keypoint
from chicken_behavior_lab.models.skeleton import Skeleton
from chicken_behavior_lab.models.tracked_skeleton import TrackedSkeleton


BASE_COORDINATES = {
    KeypointType.BEAK: (0.0, 0.0),
    KeypointType.HEAD: (1.0, 1.0),
    KeypointType.NECK: (2.0, 2.0),
    KeypointType.BODY_CENTER: (3.0, 3.0),
    KeypointType.LEFT_WING: (2.0, 4.0),
    KeypointType.RIGHT_WING: (4.0, 4.0),
    KeypointType.TAIL: (3.0, 5.0),
    KeypointType.LEFT_HIP: (2.0, 6.0),
    KeypointType.RIGHT_HIP: (4.0, 6.0),
    KeypointType.LEFT_KNEE: (2.0, 7.0),
    KeypointType.RIGHT_KNEE: (4.0, 7.0),
    KeypointType.LEFT_FOOT: (2.0, 8.0),
    KeypointType.RIGHT_FOOT: (4.0, 8.0),
}


def make_observation(frame_index):
    points = [
        Keypoint(
            keypoint_type=kind,
            x=xy[0] + frame_index * 0.1,
            y=xy[1] + frame_index * 0.05,
            confidence=0.95,
        )
        for kind, xy in BASE_COORDINATES.items()
    ]
    skeleton = Skeleton(
        skeleton_id=f"bird_7_{frame_index}",
        frame_id=str(frame_index),
        keypoints=points,
        edges=[],
        track_id="bird_7",
        confidence=0.95,
    )
    return TrackedSkeleton(track_id="bird_7", frame_id=str(frame_index), skeleton=skeleton)


def test_tracked_skeleton_to_normalized_graph_and_feature_store(tmp_path):
    observations = [make_observation(index) for index in range(4)]
    extractor = MotionFeatureExtractor()
    motion_frames = [
        extractor.extract(observations[index - 2], observations[index - 1], observations[index], 1.0)
        for index in (2, 3)
    ]

    sequence = TemporalFeatureBuilder(sequence_length=3).build(
        motion_frames,
        frame_ids=["2", "3"],
    )
    assert sequence.features.shape == (3, len(KeypointType), 7)
    assert sequence.frame_mask.tolist() == [False, True, True]
    assert sequence.keypoint_mask[1:].all()

    normalizer = MotionFeatureNormalizer()
    normalizer.fit([sequence])
    normalized = normalizer.transform(sequence)
    assert not normalized.keypoint_mask[0].any()
    assert normalized.keypoint_mask[1:].all()
    assert np.isfinite(normalized.features).all()

    store = FeatureStore(tmp_path)
    store.save("video_01", "bird_7", normalized)
    restored = store.load("video_01", "bird_7")
    np.testing.assert_array_equal(restored.features, normalized.features)
    np.testing.assert_array_equal(restored.keypoint_mask, normalized.keypoint_mask)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (3, 5),
        (3, 6), (6, 7), (6, 8), (7, 9), (8, 10),
        (9, 11), (10, 12),
    ]
    graph = SkeletonGraphBuilder(edges).build(restored)
    assert graph.node_features.shape == (3, len(KeypointType), 7)
    assert np.array_equal(graph.frame_mask, restored.frame_mask)
    assert np.array_equal(graph.node_mask, restored.keypoint_mask)
    assert graph.edge_features.shape == (3, 24, 3)
