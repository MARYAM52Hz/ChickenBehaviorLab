"""Regression tests for canonical track-ID handling across the pipeline."""

import numpy as np
import pytest

from chicken_behavior_lab.core.identifiers import normalize_track_id
from chicken_behavior_lab.annotations.schema import AnnotationSet, BehaviorAnnotation
from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.dataset.temporal_builder import TemporalSequenceBuilder
from chicken_behavior_lab.features.feature_store import FeatureStore
from chicken_behavior_lab.features.temporal_features import TemporalFeatureSequence
from chicken_behavior_lab.models.detection import Detection as ModelDetection
from chicken_behavior_lab.models.geometry import BoundingBox
from chicken_behavior_lab.models.skeleton import Skeleton as ModelSkeleton
from chicken_behavior_lab.models.tracked_skeleton import TrackedSkeleton
from chicken_behavior_lab.pipeline.contracts import (
    ContractValidationError,
    Detection,
    Keypoint,
    PoseEstimate,
    SkeletonFrame,
    TrackObservation,
    validate_pose_matches_track,
    validate_skeleton_matches_pose,
)


class DummyGraph:
    def validate(self):
        pass


def make_graph_sample(frame, track_id):
    return GraphSample(
        graph=DummyGraph(),
        label=0,
        behavior_id="feeding",
        sample_id=f"sample_{track_id}_{frame}",
        metadata={
            "video_id": "video_01",
            "track_id": track_id,
            "start_frame": frame,
            "end_frame": frame,
        },
    )


def test_normalize_track_id_migrates_numeric_legacy_ids():
    assert normalize_track_id(1) == "track_1"
    assert normalize_track_id("1") == "track_1"
    assert normalize_track_id("track_1") == "track_1"
    assert normalize_track_id("bird-A") == "bird-A"


@pytest.mark.parametrize("value", [True, -1, "", "../escape", "folder/name", "bad:id", "name.", "CON"])
def test_normalize_track_id_rejects_invalid_values(value):
    with pytest.raises((TypeError, ValueError)):
        normalize_track_id(value)


def test_pipeline_contracts_normalize_numeric_and_named_ids_consistently():
    detection = Detection(
        frame_index=2,
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        confidence=0.9,
        class_id=0,
    )
    track = TrackObservation(track_id=1, detection=detection)
    pose = PoseEstimate(
        track_id="track_1",
        frame_index=2,
        keypoints={"head": Keypoint(1.0, 2.0, 0.9)},
    )
    skeleton = SkeletonFrame(
        track_id=1,
        frame_index=2,
        nodes={"head": Keypoint(1.0, 2.0, 0.9)},
        edges=(),
    )

    assert track.track_id == "track_1"
    assert pose.track_id == "track_1"
    assert skeleton.track_id == "track_1"
    validate_pose_matches_track(track, pose)
    validate_skeleton_matches_pose(pose, skeleton)

    wrong_pose = PoseEstimate(
        track_id="track_2",
        frame_index=2,
        keypoints={"head": Keypoint(1.0, 2.0, 0.9)},
    )
    with pytest.raises(ContractValidationError, match="Track ID mismatch"):
        validate_pose_matches_track(track, wrong_pose)


def test_model_track_and_skeleton_models_share_the_same_id_format():
    model_detection = ModelDetection(
        detection_id="det_1",
        frame_id="frame_1",
        bbox=BoundingBox(0.0, 0.0, 10.0, 10.0),
        confidence=0.9,
    )
    skeleton = ModelSkeleton(
        skeleton_id="skel_1",
        frame_id="frame_1",
        keypoints=[],
        edges=[],
        track_id=1,
    )
    observation = TrackedSkeleton(
        track_id="1",
        frame_id="frame_1",
        skeleton=skeleton,
    )
    assert observation.track_id == "track_1"
    assert observation.skeleton.track_id == "track_1"


def test_graph_samples_and_temporal_windows_canonicalize_legacy_ids():
    numeric = make_graph_sample(0, 1)
    named = make_graph_sample(1, "track_1")
    assert numeric.metadata["track_id"] == "track_1"
    assert named.metadata["track_id"] == "track_1"

    dataset = TemporalSequenceBuilder(sequence_length=2).build([numeric, named])
    assert len(dataset) == 1
    assert dataset[0].track_id == "track_1"
    assert dataset[0].metadata["track_id"] == "track_1"
    assert "__track_1__" in dataset[0].sample_id


def test_feature_store_numeric_and_named_ids_address_the_same_file(tmp_path):
    store = FeatureStore(tmp_path)
    sequence = TemporalFeatureSequence(
        features=np.ones((2, 1, 1), dtype=np.float32),
        frame_mask=np.array([True, True]),
        keypoint_mask=np.array([[True], [True]]),
        frame_ids=("10", "11"),
    )
    saved_path = store.save("video_01", 1, sequence)
    assert saved_path.name == "track_1.npz"
    assert store.exists("video_01", "track_1")
    loaded = store.load("video_01", "track_1")
    np.testing.assert_array_equal(loaded.features, sequence.features)



def test_annotation_queries_compare_canonical_track_identity():
    legacy = BehaviorAnnotation(
        track_id=1,
        behavior_id="feeding",
        start_frame=10,
        end_frame=20,
        annotation_id="legacy",
    )
    canonical = BehaviorAnnotation(
        track_id="track_1",
        behavior_id="walking",
        start_frame=15,
        end_frame=25,
        annotation_id="canonical",
    )
    other = BehaviorAnnotation(
        track_id="track_2",
        behavior_id="standing",
        start_frame=15,
        end_frame=25,
        annotation_id="other",
    )

    assert legacy.overlaps(canonical)
    assert not legacy.overlaps(other)

    annotations = AnnotationSet(
        video_id="video_01",
        annotations=(legacy, canonical, other),
    )
    assert {item.annotation_id for item in annotations.for_track("track_1")} == {
        "legacy", "canonical"
    }
    assert {item.annotation_id for item in annotations.for_track(1)} == {
        "legacy", "canonical"
    }
    assert {item.annotation_id for item in annotations.at_frame(16, track_id=1)} == {
        "legacy", "canonical"
    }
