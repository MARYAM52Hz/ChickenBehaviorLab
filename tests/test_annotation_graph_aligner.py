from __future__ import annotations

import numpy as np
import pytest

from chicken_behavior_lab.alignment import (
    AnnotationGraphAligner,
)

from chicken_behavior_lab.annotations.schema import (
    AnnotationSet,
    BehaviorAnnotation,
)

from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)


class DummyGraph:
    def __init__(self) -> None:

        self.node_features = np.zeros(
            (3, 4),
            dtype=np.float32,
        )

        self.edge_index = np.array(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=np.int64,
        )

        self.edge_features = None

    def validate(self) -> None:
        pass


def make_graph_sample(
    sample_id: str,
    frame: int,
    *,
    video_id: str = "video_001",
    track_id: int = 1,
) -> GraphSample:

    return GraphSample(
        graph=DummyGraph(),
        label=0,
        behavior_id="unknown",
        sample_id=sample_id,
        metadata={
            "video_id": video_id,
            "track_id": track_id,
            "start_frame": frame,
            "end_frame": frame,
        },
    )


def make_annotation(
    annotation_id: str,
    start_frame: int,
    end_frame: int,
    behavior_id: str,
    *,
    video_id: str = "video_001",
    track_id: int = 1,
) -> BehaviorAnnotation:

    return BehaviorAnnotation(
        annotation_id=annotation_id,
        video_id=video_id,
        track_id=track_id,
        behavior_id=behavior_id,
        start_frame=start_frame,
        end_frame=end_frame,
        annotator="annotator_001",
        confidence=0.95,
    )


def test_annotation_graph_alignment() -> None:

    samples = [
        make_graph_sample(
            "sample_001",
            10,
        ),
        make_graph_sample(
            "sample_002",
            11,
        ),
        make_graph_sample(
            "sample_003",
            12,
        ),
    ]

    annotations = AnnotationSet(
        annotations=[
            make_annotation(
                "ann_001",
                10,
                11,
                "feeding",
            ),
            make_annotation(
                "ann_002",
                12,
                12,
                "walking",
            ),
        ]
    )

    aligner = AnnotationGraphAligner()

    result = aligner.align(
        samples,
        annotations,
    )

    assert result.num_labeled == 3
    assert result.num_unlabeled == 0
    assert result.num_conflicts == 0

    assert (
        result.labeled_samples[0].behavior_id
        == "feeding"
    )

    assert (
        result.labeled_samples[1].behavior_id
        == "feeding"
    )

    assert (
        result.labeled_samples[2].behavior_id
        == "walking"
    )


def test_unlabeled_sample_is_preserved() -> None:

    samples = [
        make_graph_sample(
            "sample_001",
            10,
        ),
        make_graph_sample(
            "sample_002",
            20,
        ),
    ]

    annotations = AnnotationSet(
        annotations=[
            make_annotation(
                "ann_001",
                10,
                10,
                "feeding",
            ),
        ]
    )

    aligner = AnnotationGraphAligner()

    result = aligner.align(
        samples,
        annotations,
    )

    assert result.num_labeled == 1
    assert result.num_unlabeled == 1

    assert (
        result.unlabeled_samples[0].sample_id
        == "sample_002"
    )


def test_track_id_prevents_wrong_alignment() -> None:

    samples = [
        make_graph_sample(
            "sample_001",
            10,
            track_id=1,
        ),
        make_graph_sample(
            "sample_002",
            10,
            track_id=2,
        ),
    ]

    annotations = AnnotationSet(
        annotations=[
            make_annotation(
                "ann_001",
                10,
                10,
                "feeding",
                track_id=1,
            ),
        ]
    )

    aligner = AnnotationGraphAligner()

    result = aligner.align(
        samples,
        annotations,
    )

    assert result.num_labeled == 1
    assert result.num_unlabeled == 1

    assert (
        result.labeled_samples[0].track_id
        == 1
    )

    assert (
        result.unlabeled_samples[0].track_id
        == 2
    )


def test_video_id_prevents_wrong_alignment() -> None:

    samples = [
        make_graph_sample(
            "sample_001",
            10,
            video_id="video_001",
        ),
        make_graph_sample(
            "sample_002",
            10,
            video_id="video_002",
        ),
    ]

    annotations = AnnotationSet(
        annotations=[
            make_annotation(
                "ann_001",
                10,
                10,
                "feeding",
                video_id="video_001",
            ),
        ]
    )

    aligner = AnnotationGraphAligner()

    result = aligner.align(
        samples,
        annotations,
    )

    assert result.num_labeled == 1
    assert result.num_unlabeled == 1

    assert (
        result.labeled_samples[0].video_id
        == "video_001"
    )


def test_conflicting_annotations_are_detected() -> None:

    samples = [
        make_graph_sample(
            "sample_001",
            10,
        ),
    ]

    annotations = AnnotationSet(
        annotations=[
            make_annotation(
                "ann_001",
                10,
                10,
                "feeding",
            ),
            make_annotation(
                "ann_002",
                10,
                10,
                "walking",
            ),
        ]
    )

    aligner = AnnotationGraphAligner(
        strict_conflicts=False,
    )

    result = aligner.align(
        samples,
        annotations,
    )

    assert result.num_labeled == 0
    assert result.num_unlabeled == 0
    assert result.num_conflicts == 1

    conflict = result.conflicts[0]

    assert conflict.sample_id == (
        "sample_001"
    )

    assert conflict.behavior_ids == (
        "feeding",
        "walking",
    )


def test_strict_conflicts_raise_error() -> None:

    samples = [
        make_graph_sample(
            "sample_001",
            10,
        ),
    ]

    annotations = AnnotationSet(
        annotations=[
            make_annotation(
                "ann_001",
                10,
                10,
                "feeding",
            ),
            make_annotation(
                "ann_002",
                10,
                10,
                "walking",
            ),
        ]
    )

    aligner = AnnotationGraphAligner(
        strict_conflicts=True,
    )

    with pytest.raises(ValueError):
        aligner.align(
            samples,
            annotations,
        )


def test_require_full_coverage() -> None:

    samples = [
        make_graph_sample(
            "sample_001",
            10,
        ),
        make_graph_sample(
            "sample_002",
            20,
        ),
    ]

    annotations = AnnotationSet(
        annotations=[
            make_annotation(
                "ann_001",
                10,
                10,
                "feeding",
            ),
        ]
    )

    aligner = AnnotationGraphAligner(
        require_full_coverage=True,
    )

    with pytest.raises(ValueError):
        aligner.align(
            samples,
            annotations,
        )


def test_build_label_mapping() -> None:

    samples = [
        make_graph_sample(
            "sample_001",
            10,
        ),
        make_graph_sample(
            "sample_002",
            11,
        ),
    ]

    samples[0] = GraphSample(
        graph=samples[0].graph,
        label=0,
        behavior_id="walking",
        sample_id=samples[0].sample_id,
        metadata=samples[0].metadata,
    )

    samples[1] = GraphSample(
        graph=samples[1].graph,
        label=0,
        behavior_id="feeding",
        sample_id=samples[1].sample_id,
        metadata=samples[1].metadata,
    )

    mapping = (
        AnnotationGraphAligner.build_label_mapping(
            samples
        )
    )

    assert mapping == {
        "feeding": 0,
        "walking": 1,
    }


def test_apply_label_mapping() -> None:

    samples = [
        make_graph_sample(
            "sample_001",
            10,
        ),
        make_graph_sample(
            "sample_002",
            11,
        ),
    ]

    samples[0] = GraphSample(
        graph=samples[0].graph,
        label=0,
        behavior_id="feeding",
        sample_id=samples[0].sample_id,
        metadata=samples[0].metadata,
    )

    samples[1] = GraphSample(
        graph=samples[1].graph,
        label=0,
        behavior_id="walking",
        sample_id=samples[1].sample_id,
        metadata=samples[1].metadata,
    )

    mapping = {
        "feeding": 3,
        "walking": 7,
    }

    labeled = (
        AnnotationGraphAligner.apply_label_mapping(
            samples,
            mapping,
        )
    )

    assert labeled[0].label == 3
    assert labeled[1].label == 7

    assert (
        labeled[0].behavior_id
        == "feeding"
    )

    assert (
        labeled[1].behavior_id
        == "walking"
    )


def test_apply_label_mapping_rejects_unknown_behavior() -> None:

    sample = make_graph_sample(
        "sample_001",
        10,
    )

    sample = GraphSample(
        graph=sample.graph,
        label=0,
        behavior_id="drinking",
        sample_id=sample.sample_id,
        metadata=sample.metadata,
    )

    with pytest.raises(ValueError):
        AnnotationGraphAligner.apply_label_mapping(
            [sample],
            {
                "feeding": 0,
                "walking": 1,
            },
        )
