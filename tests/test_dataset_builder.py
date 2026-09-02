import pytest

from chicken_behavior_lab.annotations.labels import (
    BehaviorLabel,
    BehaviorLabelEncoder,
)
from chicken_behavior_lab.annotations.schema import (
    AnnotationSet,
    BehaviorAnnotation,
)
from chicken_behavior_lab.dataset.builder import DatasetBuilder
from chicken_behavior_lab.dataset.graph_dataset import GraphDataset


class FakeFeatureSequence:
    def __init__(
        self,
        video_id: str,
        track_id: int,
        start_frame: int,
        end_frame: int,
    ) -> None:
        self.video_id = video_id
        self.track_id = track_id
        self.start_frame = start_frame
        self.end_frame = end_frame


class FakeGraph:
    def __init__(self, sequence: FakeFeatureSequence) -> None:
        self.sequence = sequence
        self.validation_called = False

    def validate(self) -> None:
        self.validation_called = True


class FakeGraphBuilder:
    def build(
        self,
        sequence: FakeFeatureSequence,
    ) -> FakeGraph:
        return FakeGraph(sequence)


def fake_feature_provider(
    video_id: str,
    track_id: int,
    start_frame: int,
    end_frame: int,
) -> FakeFeatureSequence:
    return FakeFeatureSequence(
        video_id=video_id,
        track_id=track_id,
        start_frame=start_frame,
        end_frame=end_frame,
    )


@pytest.fixture
def encoder() -> BehaviorLabelEncoder:
    return BehaviorLabelEncoder(
        [
            BehaviorLabel(
                behavior_id="feeding",
                name="Feeding",
            ),
            BehaviorLabel(
                behavior_id="walking",
                name="Walking",
            ),
            BehaviorLabel(
                behavior_id="standing",
                name="Standing",
            ),
        ]
    )


@pytest.fixture
def dataset_builder(
    encoder: BehaviorLabelEncoder,
) -> DatasetBuilder:
    return DatasetBuilder(
        graph_builder=FakeGraphBuilder(),
        label_encoder=encoder,
        feature_sequence_provider=fake_feature_provider,
    )


@pytest.fixture
def feeding_annotation() -> BehaviorAnnotation:
    return BehaviorAnnotation(
        annotation_id="ann_000001",
        video_id="video_001",
        track_id=1,
        behavior_id="feeding",
        start_frame=100,
        end_frame=180,
        annotator="annotator_01",
        confidence=0.95,
        notes="Chicken feeding near feeder.",
    )


def test_build_single_sample(
    dataset_builder: DatasetBuilder,
    feeding_annotation: BehaviorAnnotation,
) -> None:
    sample = dataset_builder.build_sample(
        feeding_annotation
    )

    assert sample.behavior_id == "feeding"
    assert sample.label == 0

    assert (
        sample.sample_id
        == "video_001_track_1_ann_000001"
    )


def test_sample_contains_graph(
    dataset_builder: DatasetBuilder,
    feeding_annotation: BehaviorAnnotation,
) -> None:
    sample = dataset_builder.build_sample(
        feeding_annotation
    )

    assert isinstance(sample.graph, FakeGraph)

    sequence = sample.graph.sequence

    assert sequence.video_id == "video_001"
    assert sequence.track_id == 1
    assert sequence.start_frame == 100
    assert sequence.end_frame == 180


def test_sample_metadata(
    dataset_builder: DatasetBuilder,
    feeding_annotation: BehaviorAnnotation,
) -> None:
    sample = dataset_builder.build_sample(
        feeding_annotation
    )

    assert sample.metadata is not None

    assert sample.metadata["annotation_id"] == "ann_000001"
    assert sample.metadata["video_id"] == "video_001"
    assert sample.metadata["track_id"] == 1

    assert sample.metadata["start_frame"] == 100
    assert sample.metadata["end_frame"] == 180

    assert sample.metadata["num_frames"] == 81

    assert sample.metadata["annotator"] == "annotator_01"
    assert sample.metadata["confidence"] == 0.95
    assert (
        sample.metadata["notes"]
        == "Chicken feeding near feeder."
    )


def test_build_dataset(
    dataset_builder: DatasetBuilder,
) -> None:
    annotations = AnnotationSet(
        annotations=[
            BehaviorAnnotation(
                annotation_id="ann_000001",
                video_id="video_001",
                track_id=1,
                behavior_id="feeding",
                start_frame=100,
                end_frame=180,
            ),
            BehaviorAnnotation(
                annotation_id="ann_000002",
                video_id="video_001",
                track_id=1,
                behavior_id="walking",
                start_frame=181,
                end_frame=230,
            ),
        ]
    )

    dataset = dataset_builder.build(annotations)

    assert isinstance(dataset, GraphDataset)

    assert len(dataset) == 2

    assert dataset.labels == [0, 1]

    assert dataset.sample_ids == [
        "video_001_track_1_ann_000001",
        "video_001_track_1_ann_000002",
    ]


def test_build_from_annotation_iterable(
    dataset_builder: DatasetBuilder,
) -> None:
    annotations = [
        BehaviorAnnotation(
            annotation_id="ann_001",
            video_id="video_001",
            track_id=1,
            behavior_id="feeding",
            start_frame=0,
            end_frame=20,
        ),
        BehaviorAnnotation(
            annotation_id="ann_002",
            video_id="video_001",
            track_id=2,
            behavior_id="standing",
            start_frame=0,
            end_frame=20,
        ),
    ]

    dataset = dataset_builder.build(annotations)

    assert len(dataset) == 2

    assert dataset[0].behavior_id == "feeding"
    assert dataset[1].behavior_id == "standing"

    assert dataset.labels == [0, 2]


def test_unknown_behavior_raises_error(
    dataset_builder: DatasetBuilder,
) -> None:
    annotation = BehaviorAnnotation(
        annotation_id="ann_unknown",
        video_id="video_001",
        track_id=1,
        behavior_id="flying",
        start_frame=0,
        end_frame=20,
    )

    with pytest.raises(KeyError):
        dataset_builder.build_sample(annotation)


def test_invalid_annotation_type_raises_error(
    dataset_builder: DatasetBuilder,
) -> None:
    with pytest.raises(TypeError):
        dataset_builder.build_sample(
            "not-an-annotation"  # type: ignore[arg-type]
        )


def test_none_feature_sequence_raises_error(
    encoder: BehaviorLabelEncoder,
    feeding_annotation: BehaviorAnnotation,
) -> None:
    def provider(
        video_id: str,
        track_id: int,
        start_frame: int,
        end_frame: int,
    ):
        return None

    builder = DatasetBuilder(
        graph_builder=FakeGraphBuilder(),
        label_encoder=encoder,
        feature_sequence_provider=provider,
    )

    with pytest.raises(
        ValueError,
        match="feature_sequence_provider returned None",
    ):
        builder.build_sample(feeding_annotation)


def test_none_graph_raises_error(
    encoder: BehaviorLabelEncoder,
    feeding_annotation: BehaviorAnnotation,
) -> None:
    class BrokenGraphBuilder:
        def build(self, sequence):
            return None

    builder = DatasetBuilder(
        graph_builder=BrokenGraphBuilder(),
        label_encoder=encoder,
        feature_sequence_provider=fake_feature_provider,
    )

    with pytest.raises(
        ValueError,
        match="graph_builder returned None",
    ):
        builder.build_sample(feeding_annotation)


def test_graph_is_validated(
    dataset_builder: DatasetBuilder,
    feeding_annotation: BehaviorAnnotation,
) -> None:
    sample = dataset_builder.build_sample(
        feeding_annotation
    )

    assert sample.graph.validation_called is True
