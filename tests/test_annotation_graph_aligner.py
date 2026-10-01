from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from chicken_behavior_lab.annotations.schema import (
    AnnotationSet,
    BehaviorAnnotation,
)

from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)


@dataclass(frozen=True, slots=True)
class AlignmentConflict:
    """
    Description of a conflicting annotation for one graph sample.
    """

    sample_id: str
    video_id: str
    track_id: int
    frame: int
    behavior_ids: tuple[str, ...]


@dataclass(slots=True)
class AlignmentResult:
    """
    Result of aligning annotations with graph samples.
    """

    labeled_samples: list[GraphSample]
    unlabeled_samples: list[GraphSample]
    conflicts: list[AlignmentConflict]

    @property
    def num_labeled(self) -> int:
        return len(
            self.labeled_samples
        )

    @property
    def num_unlabeled(self) -> int:
        return len(
            self.unlabeled_samples
        )

    @property
    def num_conflicts(self) -> int:
        return len(
            self.conflicts
        )

    @property
    def total_samples(self) -> int:
        return (
            self.num_labeled
            + self.num_unlabeled
            + self.num_conflicts
        )

    def validate(self) -> None:
        sample_ids: set[str] = set()

        for sample in self.labeled_samples:
            sample.validate()

            if sample.sample_id in sample_ids:
                raise ValueError(
                    "Duplicate sample_id in alignment result: "
                    f"{sample.sample_id}"
                )

            sample_ids.add(
                sample.sample_id
            )

        for sample in self.unlabeled_samples:
            sample.validate()

            if sample.sample_id in sample_ids:
                raise ValueError(
                    "A sample cannot appear in both labeled and "
                    f"unlabeled collections: {sample.sample_id}"
                )

            sample_ids.add(
                sample.sample_id
            )

        conflict_ids = {
            conflict.sample_id
            for conflict in self.conflicts
        }

        if sample_ids & conflict_ids:
            raise ValueError(
                "A sample cannot appear in labeled/unlabeled "
                "collections and conflicts simultaneously."
            )


class AnnotationGraphAligner:
    """
    Align BehaviorAnnotation objects with GraphSample objects.

    Matching key:

        video_id + track_id + frame interval

    The aligner does not perform train/validation/test splitting.
    """

    def __init__(
        self,
        *,
        strict_conflicts: bool = True,
        require_full_coverage: bool = False,
    ) -> None:

        self.strict_conflicts = (
            strict_conflicts
        )

        self.require_full_coverage = (
            require_full_coverage
        )

    def align(
        self,
        graph_samples: Sequence[GraphSample],
        annotations: AnnotationSet,
    ) -> AlignmentResult:

        if not graph_samples:
            raise ValueError(
                "graph_samples cannot be empty."
            )

        annotations.validate()

        annotation_index = (
            self._build_annotation_index(
                annotations
            )
        )

        labeled_samples: list[
            GraphSample
        ] = []

        unlabeled_samples: list[
            GraphSample
        ] = []

        conflicts: list[
            AlignmentConflict
        ] = []

        for graph_sample in graph_samples:

            graph_sample.validate()

            (
                video_id,
                track_id,
                start_frame,
                end_frame,
            ) = self._extract_graph_identity(
                graph_sample
            )

            matched_annotations = (
                self._find_overlapping_annotations(
                    annotation_index=annotation_index,
                    video_id=video_id,
                    track_id=track_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )

            if not matched_annotations:

                if self.require_full_coverage:
                    raise ValueError(
                        "GraphSample has no matching annotation: "
                        f"{graph_sample.sample_id}"
                    )

                unlabeled_samples.append(
                    graph_sample
                )

                continue

            behavior_ids = tuple(
                sorted(
                    {
                        annotation.behavior_id
                        for annotation
                        in matched_annotations
                    }
                )
            )

            if len(behavior_ids) > 1:

                conflict = (
                    AlignmentConflict(
                        sample_id=graph_sample.sample_id,
                        video_id=video_id,
                        track_id=track_id,
                        frame=start_frame,
                        behavior_ids=behavior_ids,
                    )
                )

                conflicts.append(
                    conflict
                )

                if self.strict_conflicts:
                    raise ValueError(
                        self._format_conflict_message(
                            conflict
                        )
                    )

                continue

            annotation = (
                matched_annotations[0]
            )

            labeled_sample = (
                self._apply_annotation(
                    graph_sample,
                    annotation,
                )
            )

            labeled_samples.append(
                labeled_sample
            )

        result = AlignmentResult(
            labeled_samples=labeled_samples,
            unlabeled_samples=unlabeled_samples,
            conflicts=conflicts,
        )

        result.validate()

        return result

    @staticmethod
    def apply_label_mapping(
        samples: Sequence[GraphSample],
        label_to_index: dict[str, int],
    ) -> list[GraphSample]:
        """
        Apply a canonical behavior-to-index mapping to graph samples.

        The mapping should normally be created once from the training
        split and then reused for validation and test data.
        """

        if not label_to_index:
            raise ValueError(
                "label_to_index cannot be empty."
            )

        values = list(
            label_to_index.values()
        )

        if len(values) != len(
            set(values)
        ):
            raise ValueError(
                "label_to_index values must be unique."
            )

        if any(
            not isinstance(
                value,
                int,
            )
            for value in values
        ):
            raise TypeError(
                "label_to_index values must be integers."
            )

        if any(
            value < 0
            for value in values
        ):
            raise ValueError(
                "label_to_index values cannot be negative."
            )

        result: list[
            GraphSample
        ] = []

        for sample in samples:

            sample.validate()

            behavior_id = (
                sample.behavior_id
            )

            if behavior_id not in (
                label_to_index
            ):
                raise ValueError(
                    "Behavior "
                    f"'{behavior_id}' "
                    "is missing from label_to_index."
                )

            labeled_sample = (
                GraphSample(
                    graph=sample.graph,
                    label=int(
                        label_to_index[
                            behavior_id
                        ]
                    ),
                    behavior_id=behavior_id,
                    sample_id=sample.sample_id,
                    metadata=dict(
                        sample.metadata or {}
                    ),
                )
            )

            labeled_sample.validate()

            result.append(
                labeled_sample
            )

        return result

    @staticmethod
    def build_label_mapping(
        samples: Sequence[GraphSample],
    ) -> dict[str, int]:
        """
        Build a deterministic behavior-to-index mapping.

        The mapping is sorted alphabetically by behavior_id.
        """

        behavior_ids = sorted(
            {
                sample.behavior_id
                for sample in samples
            }
        )

        if not behavior_ids:
            raise ValueError(
                "Cannot build label mapping from empty samples."
            )

        return {
            behavior_id: index
            for index, behavior_id
            in enumerate(behavior_ids)
        }

    @staticmethod
    def _build_annotation_index(
        annotations: AnnotationSet,
    ) -> dict[
        tuple[str, int],
        list[BehaviorAnnotation],
    ]:

        index: dict[
            tuple[str, int],
            list[BehaviorAnnotation],
        ] = {}

        for annotation in annotations:

            key = (
                annotation.video_id,
                annotation.track_id,
            )

            index.setdefault(
                key,
                [],
            ).append(
                annotation
            )

        for values in index.values():

            values.sort(
                key=lambda annotation: (
                    annotation.start_frame,
                    annotation.end_frame,
                    annotation.annotation_id,
                )
            )

        return index

    @staticmethod
    def _extract_graph_identity(
        graph_sample: GraphSample,
    ) -> tuple[str, int, int, int]:

        video_id = graph_sample.get_metadata(
            "video_id"
        )

        track_id = graph_sample.get_metadata(
            "track_id"
        )

        start_frame = graph_sample.get_metadata(
            "start_frame"
        )

        end_frame = graph_sample.get_metadata(
            "end_frame"
        )

        if video_id is None:
            raise ValueError(
                "GraphSample is missing metadata.video_id: "
                f"{graph_sample.sample_id}"
            )

        if track_id is None:
            raise ValueError(
                "GraphSample is missing metadata.track_id: "
                f"{graph_sample.sample_id}"
            )

        if start_frame is None:
            raise ValueError(
                "GraphSample is missing metadata.start_frame: "
                f"{graph_sample.sample_id}"
            )

        if end_frame is None:
            raise ValueError(
                "GraphSample is missing metadata.end_frame: "
                f"{graph_sample.sample_id}"
            )

        video_id = str(
            video_id
        )

        track_id = int(
            track_id
        )

        start_frame = int(
            start_frame
        )

        end_frame = int(
            end_frame
        )

        if track_id < 0:
            raise ValueError(
                "track_id cannot be negative."
            )

        if start_frame < 0:
            raise ValueError(
                "start_frame cannot be negative."
            )

        if end_frame < start_frame:
            raise ValueError(
                "end_frame must be >= start_frame: "
                f"{graph_sample.sample_id}"
            )

        return (
            video_id,
            track_id,
            start_frame,
            end_frame,
        )

    @staticmethod
    def _find_overlapping_annotations(
        *,
        annotation_index: dict[
            tuple[str, int],
            list[BehaviorAnnotation],
        ],
        video_id: str,
        track_id: int,
        start_frame: int,
        end_frame: int,
    ) -> list[BehaviorAnnotation]:

        candidates = annotation_index.get(
            (
                video_id,
                track_id,
            ),
            [],
        )

        matched: list[
            BehaviorAnnotation
        ] = []

        for annotation in candidates:

            overlaps = (
                annotation.start_frame
                <= end_frame
                and annotation.end_frame
                >= start_frame
            )

            if overlaps:
                matched.append(
                    annotation
                )

        return matched

    @staticmethod
    def _apply_annotation(
        graph_sample: GraphSample,
        annotation: BehaviorAnnotation,
    ) -> GraphSample:

        metadata = dict(
            graph_sample.metadata or {}
        )

        metadata.update(
            {
                "annotation_id": (
                    annotation.annotation_id
                ),
                "annotator": (
                    annotation.annotator
                ),
                "annotation_confidence": (
                    annotation.confidence
                ),
            }
        )

        return GraphSample(
            graph=graph_sample.graph,
            label=graph_sample.label,
            behavior_id=annotation.behavior_id,
            sample_id=graph_sample.sample_id,
            metadata=metadata,
        )

    @staticmethod
    def _format_conflict_message(
        conflict: AlignmentConflict,
    ) -> str:

        behaviors = ", ".join(
            conflict.behavior_ids
        )

        return (
            "Conflicting annotations found for "
            f"sample '{conflict.sample_id}' "
            f"(video_id='{conflict.video_id}', "
            f"track_id={conflict.track_id}, "
            f"frame={conflict.frame}): "
            f"{behaviors}"
        )
