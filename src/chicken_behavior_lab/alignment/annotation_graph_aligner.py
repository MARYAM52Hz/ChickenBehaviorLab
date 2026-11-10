from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

from chicken_behavior_lab.annotations.schema import AnnotationSet, BehaviorAnnotation
from chicken_behavior_lab.core.identifiers import normalize_track_id
from chicken_behavior_lab.dataset.sample import GraphSample


@dataclass(frozen=True, slots=True)
class AlignmentConflict:
    """Conflicting behavior labels for one sample interval."""

    sample_id: str
    video_id: str
    track_id: str
    frame: int
    behavior_ids: tuple[str, ...]


@dataclass(slots=True)
class AlignmentResult:
    """Labeled, unlabeled, and conflicted graph samples."""

    labeled_samples: list[GraphSample]
    unlabeled_samples: list[GraphSample]
    conflicts: list[AlignmentConflict]

    @property
    def num_labeled(self) -> int:
        return len(self.labeled_samples)

    @property
    def num_unlabeled(self) -> int:
        return len(self.unlabeled_samples)

    @property
    def num_conflicts(self) -> int:
        return len(self.conflicts)

    @property
    def total_samples(self) -> int:
        return self.num_labeled + self.num_unlabeled + self.num_conflicts

    def validate(self) -> None:
        sample_ids: set[str] = set()
        for sample in self.labeled_samples:
            sample.validate()
            if sample.sample_id in sample_ids:
                raise ValueError(f"Duplicate sample_id in alignment result: {sample.sample_id}")
            sample_ids.add(sample.sample_id)

        for sample in self.unlabeled_samples:
            sample.validate()
            if sample.sample_id in sample_ids:
                raise ValueError(
                    "A sample cannot appear in both labeled and unlabeled collections: "
                    f"{sample.sample_id}"
                )
            sample_ids.add(sample.sample_id)

        conflict_ids = {conflict.sample_id for conflict in self.conflicts}
        if sample_ids & conflict_ids:
            raise ValueError(
                "A sample cannot appear in labeled/unlabeled collections and conflicts simultaneously."
            )
        if len(conflict_ids) != len(self.conflicts):
            raise ValueError("A sample cannot have duplicate conflict records.")


class AnnotationGraphAligner:
    """Align frame/interval graph samples with one or more video-scoped AnnotationSets.

    AnnotationSet owns the video_id; each BehaviorAnnotation owns a track ID,
    behavior ID, and inclusive frame interval. Track IDs are canonicalized at
    the boundary so legacy integer IDs and named IDs match consistently.
    """

    def __init__(
        self,
        *,
        strict_conflicts: bool = True,
        require_full_coverage: bool = False,
    ) -> None:
        self.strict_conflicts = bool(strict_conflicts)
        self.require_full_coverage = bool(require_full_coverage)

    def align(
        self,
        graph_samples: Sequence[GraphSample],
        annotations: AnnotationSet | Sequence[AnnotationSet],
    ) -> AlignmentResult:
        """Align graph samples against one or more video-scoped annotation sets."""
        if not graph_samples:
            raise ValueError("graph_samples cannot be empty.")

        annotation_sets = self._coerce_annotation_sets(annotations)
        annotation_index = self._build_annotation_index(annotation_sets)

        labeled_samples: list[GraphSample] = []
        unlabeled_samples: list[GraphSample] = []
        conflicts: list[AlignmentConflict] = []

        seen_sample_ids: set[str] = set()
        for graph_sample in graph_samples:
            graph_sample.validate()
            if graph_sample.sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate graph sample_id: {graph_sample.sample_id}")
            seen_sample_ids.add(graph_sample.sample_id)

            video_id, track_id, start_frame, end_frame = self._extract_graph_identity(graph_sample)
            matched = self._find_overlapping_annotations(
                annotation_index=annotation_index,
                video_id=video_id,
                track_id=track_id,
                start_frame=start_frame,
                end_frame=end_frame,
            )

            if not matched:
                if self.require_full_coverage:
                    raise ValueError(
                        f"GraphSample has no matching annotation: {graph_sample.sample_id}"
                    )
                unlabeled_samples.append(graph_sample)
                continue

            behavior_ids = tuple(sorted({item.behavior_id for item in matched}))
            if len(behavior_ids) > 1:
                conflict = AlignmentConflict(
                    sample_id=graph_sample.sample_id,
                    video_id=video_id,
                    track_id=track_id,
                    frame=start_frame,
                    behavior_ids=behavior_ids,
                )
                conflicts.append(conflict)
                if self.strict_conflicts:
                    raise ValueError(self._format_conflict_message(conflict))
                continue

            # All matches share one behavior label. If several same-label
            # intervals overlap, prefer the most specific interval and then
            # stable annotation ID ordering for deterministic provenance.
            annotation = min(
                matched,
                key=lambda item: (
                    item.duration_frames,
                    item.start_frame,
                    item.end_frame,
                    item.annotation_id or "",
                ),
            )
            labeled_samples.append(self._apply_annotation(graph_sample, annotation))

        result = AlignmentResult(
            labeled_samples=labeled_samples,
            unlabeled_samples=unlabeled_samples,
            conflicts=conflicts,
        )
        result.validate()
        return result

    @staticmethod
    def _coerce_annotation_sets(
        annotations: AnnotationSet | Sequence[AnnotationSet],
    ) -> tuple[AnnotationSet, ...]:
        if isinstance(annotations, AnnotationSet):
            items = (annotations,)
        else:
            if isinstance(annotations, (str, bytes)):
                raise TypeError("annotations must be an AnnotationSet or a sequence of AnnotationSets.")
            items = tuple(annotations)

        if not items:
            raise ValueError("At least one AnnotationSet is required.")
        if any(not isinstance(item, AnnotationSet) for item in items):
            raise TypeError("Every annotation collection must be an AnnotationSet.")

        video_ids = [item.video_id for item in items]
        if len(video_ids) != len(set(video_ids)):
            raise ValueError("Only one AnnotationSet is allowed per video_id.")
        return items

    @staticmethod
    def _build_annotation_index(
        annotation_sets: Sequence[AnnotationSet],
    ) -> dict[tuple[str, str], list[BehaviorAnnotation]]:
        index: dict[tuple[str, str], list[BehaviorAnnotation]] = {}
        for annotation_set in annotation_sets:
            for annotation in annotation_set.annotations:
                key = (annotation_set.video_id, normalize_track_id(annotation.track_id))
                index.setdefault(key, []).append(annotation)

        for values in index.values():
            values.sort(
                key=lambda item: (
                    item.start_frame,
                    item.end_frame,
                    item.behavior_id,
                    item.annotation_id or "",
                )
            )
        return index

    @staticmethod
    def _extract_graph_identity(
        graph_sample: GraphSample,
    ) -> tuple[str, str, int, int]:
        video_id = graph_sample.get_metadata("video_id")
        raw_track_id = graph_sample.get_metadata("track_id")
        start_frame = graph_sample.get_metadata("start_frame")
        end_frame = graph_sample.get_metadata("end_frame")

        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError(f"GraphSample is missing a valid metadata.video_id: {graph_sample.sample_id}")
        if raw_track_id is None:
            raise ValueError(f"GraphSample is missing metadata.track_id: {graph_sample.sample_id}")
        if isinstance(start_frame, bool) or not isinstance(start_frame, Integral) or start_frame < 0:
            raise ValueError(f"GraphSample has invalid metadata.start_frame: {graph_sample.sample_id}")
        if isinstance(end_frame, bool) or not isinstance(end_frame, Integral) or end_frame < start_frame:
            raise ValueError(f"GraphSample has invalid metadata.end_frame: {graph_sample.sample_id}")
        try:
            track_id = normalize_track_id(raw_track_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"GraphSample has invalid metadata.track_id: {graph_sample.sample_id}") from exc

        return video_id.strip(), track_id, int(start_frame), int(end_frame)

    @staticmethod
    def _find_overlapping_annotations(
        *,
        annotation_index: dict[tuple[str, str], list[BehaviorAnnotation]],
        video_id: str,
        track_id: str,
        start_frame: int,
        end_frame: int,
    ) -> list[BehaviorAnnotation]:
        candidates = annotation_index.get((video_id, track_id), ())
        return [
            annotation
            for annotation in candidates
            if annotation.start_frame <= end_frame and annotation.end_frame >= start_frame
        ]

    @staticmethod
    def _apply_annotation(
        graph_sample: GraphSample,
        annotation: BehaviorAnnotation,
    ) -> GraphSample:
        """Return a new graph sample with annotation provenance metadata."""
        metadata = dict(graph_sample.metadata or {})
        metadata.update(
            {
                "annotation_id": annotation.annotation_id,
                "annotator_id": annotation.annotator_id,
                "annotation_confidence": annotation.confidence,
                "annotation_attributes": dict(annotation.attributes),
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
    def _format_conflict_message(conflict: AlignmentConflict) -> str:
        behaviors = ", ".join(conflict.behavior_ids)
        return (
            "Conflicting annotations found for "
            f"sample '{conflict.sample_id}' "
            f"(video_id='{conflict.video_id}', track_id='{conflict.track_id}', "
            f"frame={conflict.frame}): {behaviors}"
        )


    @staticmethod
    def build_label_mapping(samples: Sequence[GraphSample]) -> dict[str, int]:
        """Create a deterministic zero-based mapping from training behavior IDs."""
        if not samples:
            raise ValueError("Cannot build a label mapping from an empty sample sequence.")
        behavior_ids = sorted({sample.behavior_id for sample in samples})
        if any(not isinstance(value, str) or not value.strip() for value in behavior_ids):
            raise ValueError("Every training sample must have a non-empty behavior_id.")
        return {behavior_id: index for index, behavior_id in enumerate(behavior_ids)}

    @staticmethod
    def apply_label_mapping(
        samples: Sequence[GraphSample],
        label_mapping: dict[str, int],
    ) -> list[GraphSample]:
        """Return graph samples with their behavior IDs mapped to class indices."""
        if not label_mapping:
            raise ValueError("label_mapping cannot be empty.")
        values = list(label_mapping.values())
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("label_mapping values must be integers.")
        if len(values) != len(set(values)) or sorted(values) != list(range(len(values))):
            raise ValueError("label_mapping values must be unique, contiguous, and start at 0.")

        result: list[GraphSample] = []
        for sample in samples:
            if sample.behavior_id not in label_mapping:
                raise ValueError(
                    f"Behavior '{sample.behavior_id}' is missing from label_mapping."
                )
            result.append(
                GraphSample(
                    graph=sample.graph,
                    label=label_mapping[sample.behavior_id],
                    behavior_id=sample.behavior_id,
                    sample_id=sample.sample_id,
                    metadata=dict(sample.metadata or {}),
                )
            )
        return result
