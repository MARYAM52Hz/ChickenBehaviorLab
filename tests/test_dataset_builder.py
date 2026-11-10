from __future__ import annotations

import pytest

from chicken_behavior_lab.annotations.labels import BehaviorLabel, BehaviorLabelEncoder
from chicken_behavior_lab.annotations.schema import AnnotationSet, BehaviorAnnotation
from chicken_behavior_lab.dataset.builder import DatasetBuilder
from chicken_behavior_lab.dataset.graph_dataset import GraphDataset


class FakeFeatureSequence:
    def __init__(self, video_id, track_id, start_frame, end_frame):
        self.video_id = video_id
        self.track_id = track_id
        self.start_frame = start_frame
        self.end_frame = end_frame


class FakeGraph:
    def __init__(self, sequence):
        self.sequence = sequence
        self.validation_called = False

    def validate(self):
        self.validation_called = True


class FakeGraphBuilder:
    def build(self, sequence):
        return FakeGraph(sequence)


def fake_feature_provider(video_id, track_id, start_frame, end_frame):
    return FakeFeatureSequence(video_id, track_id, start_frame, end_frame)


@pytest.fixture
def encoder():
    return BehaviorLabelEncoder([
        BehaviorLabel(behavior_id="feeding", name="Feeding"),
        BehaviorLabel(behavior_id="walking", name="Walking"),
        BehaviorLabel(behavior_id="standing", name="Standing"),
    ])


@pytest.fixture
def dataset_builder(encoder):
    return DatasetBuilder(
        graph_builder=FakeGraphBuilder(),
        label_encoder=encoder,
        feature_sequence_provider=fake_feature_provider,
    )


@pytest.fixture
def feeding_annotation():
    return BehaviorAnnotation(
        annotation_id="ann_000001",
        track_id=1,
        behavior_id="feeding",
        start_frame=100,
        end_frame=180,
        annotator_id="annotator_01",
        confidence=0.95,
        attributes={"location": "feeder", "note": "near feeder"},
    )


def test_build_single_sample_uses_video_scope_and_canonical_track_id(dataset_builder, feeding_annotation):
    sample = dataset_builder.build_sample(feeding_annotation, video_id="video_001")
    assert sample.behavior_id == "feeding"
    assert sample.label == 0
    assert sample.sample_id == "video_001_track_1_ann_000001"
    assert sample.metadata["track_id"] == "track_1"
    assert sample.graph.sequence.track_id == "track_1"


def test_sample_contains_feature_sequence(dataset_builder, feeding_annotation):
    sample = dataset_builder.build_sample(feeding_annotation, video_id="video_001")
    sequence = sample.graph.sequence
    assert sequence.video_id == "video_001"
    assert sequence.track_id == "track_1"
    assert sequence.start_frame == 100
    assert sequence.end_frame == 180


def test_sample_metadata_matches_annotation_schema(dataset_builder, feeding_annotation):
    sample = dataset_builder.build_sample(feeding_annotation, video_id="video_001")
    assert sample.metadata == {
        "annotation_id": "ann_000001",
        "video_id": "video_001",
        "track_id": "track_1",
        "start_frame": 100,
        "end_frame": 180,
        "num_frames": 81,
        "annotator_id": "annotator_01",
        "confidence": 0.95,
        "attributes": {"location": "feeder", "note": "near feeder"},
    }


def test_build_dataset_from_annotation_set(dataset_builder):
    annotations = AnnotationSet(
        video_id="video_001",
        annotations=(
            BehaviorAnnotation(
                annotation_id="ann_000001", track_id=1, behavior_id="feeding",
                start_frame=100, end_frame=180,
            ),
            BehaviorAnnotation(
                annotation_id="ann_000002", track_id="track_1", behavior_id="walking",
                start_frame=181, end_frame=230,
            ),
        ),
    )
    dataset = dataset_builder.build(annotations)
    assert isinstance(dataset, GraphDataset)
    assert len(dataset) == 2
    assert dataset.labels == [0, 1]
    assert dataset.sample_ids == [
        "video_001_track_1_ann_000001",
        "video_001_track_1_ann_000002",
    ]


def test_build_from_iterable_requires_video_id(dataset_builder):
    annotations = [
        BehaviorAnnotation(track_id=1, behavior_id="feeding", start_frame=0, end_frame=20),
        BehaviorAnnotation(track_id=2, behavior_id="standing", start_frame=21, end_frame=40),
    ]
    with pytest.raises(ValueError, match="video_id is required"):
        dataset_builder.build(annotations)
    dataset = dataset_builder.build(annotations, video_id="video_001")
    assert len(dataset) == 2
    assert dataset[0].metadata["track_id"] == "track_1"
    assert dataset[1].metadata["track_id"] == "track_2"


def test_explicit_video_id_must_match_annotation_set(dataset_builder):
    annotation_set = AnnotationSet(
        video_id="video_001",
        annotations=(BehaviorAnnotation(track_id=1, behavior_id="feeding", start_frame=0, end_frame=1),),
    )
    with pytest.raises(ValueError, match="does not match"):
        dataset_builder.build(annotation_set, video_id="video_002")


def test_annotation_without_id_gets_a_deterministic_sample_id(dataset_builder):
    annotation = BehaviorAnnotation(
        track_id="bird-A", behavior_id="feeding", start_frame=10, end_frame=20,
    )
    first = dataset_builder.build_sample(annotation, video_id="video_001")
    second = dataset_builder.build_sample(annotation, video_id="video_001")
    assert first.sample_id == second.sample_id
    assert first.sample_id.startswith("video_001_bird-A_ann_")


def test_unknown_behavior_raises_error(dataset_builder):
    annotation = BehaviorAnnotation(
        annotation_id="ann_unknown", track_id=1, behavior_id="flying", start_frame=0, end_frame=20,
    )
    with pytest.raises(KeyError):
        dataset_builder.build_sample(annotation, video_id="video_001")


def test_invalid_annotation_type_raises_error(dataset_builder):
    with pytest.raises(TypeError):
        dataset_builder.build_sample("not-an-annotation", video_id="video_001")


def test_none_feature_sequence_raises_error(encoder, feeding_annotation):
    builder = DatasetBuilder(
        graph_builder=FakeGraphBuilder(),
        label_encoder=encoder,
        feature_sequence_provider=lambda *args: None,
    )
    with pytest.raises(ValueError, match="feature_sequence_provider returned None"):
        builder.build_sample(feeding_annotation, video_id="video_001")


def test_none_graph_raises_error(encoder, feeding_annotation):
    class BrokenGraphBuilder:
        def build(self, sequence):
            return None

    builder = DatasetBuilder(
        graph_builder=BrokenGraphBuilder(),
        label_encoder=encoder,
        feature_sequence_provider=fake_feature_provider,
    )
    with pytest.raises(ValueError, match="graph_builder returned None"):
        builder.build_sample(feeding_annotation, video_id="video_001")


def test_graph_is_validated(dataset_builder, feeding_annotation):
    sample = dataset_builder.build_sample(feeding_annotation, video_id="video_001")
    assert sample.graph.validation_called is True
