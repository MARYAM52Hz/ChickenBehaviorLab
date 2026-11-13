from __future__ import annotations

import pytest

from chicken_behavior_lab.alignment.annotation_graph_aligner import AnnotationGraphAligner
from chicken_behavior_lab.annotations.schema import AnnotationSet, BehaviorAnnotation
from chicken_behavior_lab.dataset.sample import GraphSample


class FakeGraph:
    def validate(self):
        pass


def make_sample(start_frame: int = 10, end_frame: int = 19) -> GraphSample:
    return GraphSample(
        graph=FakeGraph(),
        label=0,
        behavior_id="unassigned",
        sample_id="sample_001",
        metadata={
            "video_id": "video_001",
            "track_id": "track_1",
            "start_frame": start_frame,
            "end_frame": end_frame,
        },
    )


def annotation(start_frame: int, end_frame: int, behavior_id: str = "feeding") -> BehaviorAnnotation:
    return BehaviorAnnotation(
        track_id=1,
        behavior_id=behavior_id,
        start_frame=start_frame,
        end_frame=end_frame,
    )


@pytest.mark.parametrize(
    ("annotations", "missing_region"),
    [
        ((annotation(11, 19),), "start"),
        ((annotation(10, 18),), "end"),
        ((annotation(10, 13), annotation(15, 19)), "middle"),
    ],
)
def test_require_full_coverage_rejects_partial_interval(annotations, missing_region):
    aligner = AnnotationGraphAligner(require_full_coverage=True)
    annotation_set = AnnotationSet(video_id="video_001", annotations=annotations)

    with pytest.raises(ValueError, match="do not fully cover"):
        aligner.align([make_sample()], annotation_set)


def test_require_full_coverage_accepts_union_of_adjacent_intervals():
    aligner = AnnotationGraphAligner(require_full_coverage=True)
    annotation_set = AnnotationSet(
        video_id="video_001",
        annotations=(
            annotation(10, 14),
            annotation(15, 19),
        ),
    )

    result = aligner.align([make_sample()], annotation_set)

    assert result.num_labeled == 1
    assert result.labeled_samples[0].behavior_id == "feeding"


def test_partial_overlap_remains_allowed_when_full_coverage_is_disabled():
    aligner = AnnotationGraphAligner(require_full_coverage=False)
    annotation_set = AnnotationSet(
        video_id="video_001",
        annotations=(annotation(12, 15),),
    )

    result = aligner.align([make_sample()], annotation_set)

    assert result.num_labeled == 1
