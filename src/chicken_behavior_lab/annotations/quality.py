"""Quality diagnostics for interval-based chicken behavior annotations.

The analyzer reports suspicious patterns but does not modify or silently reject
annotations. Frame intervals are inclusive, matching the annotation schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .schema import AnnotationSet, BehaviorAnnotation


@dataclass(frozen=True, slots=True)
class AnnotationIssue:
    """One machine-readable annotation-quality finding."""
    code: str
    message: str
    track_id: str | int | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    annotation_ids: tuple[str, ...] = ()
    behavior_ids: tuple[str, ...] = ()
    severity: str = "warning"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "annotation_ids": list(self.annotation_ids),
            "behavior_ids": list(self.behavior_ids),
            "severity": self.severity,
        }


@dataclass(frozen=True, slots=True)
class AnnotationQualityReport:
    """Summary and detailed findings for one annotation set."""
    video_id: str
    annotation_count: int
    track_count: int
    behavior_counts: Mapping[str, int] = field(default_factory=dict)
    issues: tuple[AnnotationIssue, ...] = ()
    uncovered_frames_by_track: Mapping[str, int] = field(default_factory=dict)

    @property
    def issue_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.code] = counts.get(issue.code, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def has_conflicting_overlaps(self) -> bool:
        return any(issue.code == "conflicting_overlap" for issue in self.issues)

    @property
    def is_clean(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "annotation_count": self.annotation_count,
            "track_count": self.track_count,
            "behavior_counts": dict(sorted(self.behavior_counts.items())),
            "issue_counts": self.issue_counts,
            "uncovered_frames_by_track": dict(sorted(self.uncovered_frames_by_track.items())),
            "issues": [issue.to_dict() for issue in self.issues],
            "has_conflicting_overlaps": self.has_conflicting_overlaps,
            "is_clean": self.is_clean,
        }


def analyze_annotation_quality(
    annotation_set: AnnotationSet,
    *,
    video_length_frames: int | None = None,
    expected_track_ids: Iterable[str | int] | None = None,
) -> AnnotationQualityReport:
    """Find duplicate intervals, conflicting overlaps, gaps and annotator disagreement.

    Args:
        annotation_set: Validated annotations for a single video.
        video_length_frames: Optional total number of frames. When supplied,
            it must be positive and uncovered frames are calculated over the
            range ``[0, video_length_frames - 1]`` for each expected/observed track.
        expected_track_ids: Optional track IDs expected to be present. This is
            useful for reporting tracks with no annotations; it does not infer
            that every track must have a label at every frame.

    Notes:
        - Different behavior labels overlapping for the same track are reported
          as conflicts, not automatically resolved.
        - Gaps are counted only when ``video_length_frames`` is supplied.
        - The analyzer does not decide whether overlapping behaviors are
          semantically forbidden; findings require review against the label guide.
    """
    if not isinstance(annotation_set, AnnotationSet):
        raise TypeError("annotation_set must be an AnnotationSet.")
    if video_length_frames is not None:
        if isinstance(video_length_frames, bool) or not isinstance(video_length_frames, int):
            raise TypeError("video_length_frames must be an integer or None.")
        if video_length_frames <= 0:
            raise ValueError("video_length_frames must be greater than zero.")

    annotations = list(annotation_set.annotations)
    by_track: dict[str | int, list[BehaviorAnnotation]] = {}
    behavior_counts: dict[str, int] = {}
    for ann in annotations:
        by_track.setdefault(ann.track_id, []).append(ann)
        behavior_counts[ann.behavior_id] = behavior_counts.get(ann.behavior_id, 0) + 1

    issues: list[AnnotationIssue] = []
    # Stable sorting makes reports deterministic across runs.
    for track_id, items in by_track.items():
        items.sort(key=lambda a: (a.start_frame, a.end_frame, a.behavior_id, a.annotation_id or ""))
        for index, current in enumerate(items):
            for other in items[index + 1:]:
                if other.start_frame > current.end_frame:
                    break
                overlap_start = max(current.start_frame, other.start_frame)
                overlap_end = min(current.end_frame, other.end_frame)
                if overlap_start > overlap_end:
                    continue
                ids = tuple(sorted(x for x in (current.annotation_id, other.annotation_id) if x))
                labels = tuple(sorted({current.behavior_id, other.behavior_id}))
                if (current.start_frame, current.end_frame, current.behavior_id) == (
                    other.start_frame, other.end_frame, other.behavior_id
                ):
                    issues.append(AnnotationIssue(
                        code="duplicate_interval",
                        message="Two annotations have the same track, behavior and frame interval.",
                        track_id=track_id, start_frame=overlap_start, end_frame=overlap_end,
                        annotation_ids=ids, behavior_ids=labels,
                    ))
                elif current.behavior_id != other.behavior_id:
                    issues.append(AnnotationIssue(
                        code="conflicting_overlap",
                        message="Different behavior labels overlap for the same track.",
                        track_id=track_id, start_frame=overlap_start, end_frame=overlap_end,
                        annotation_ids=ids, behavior_ids=labels, severity="error",
                    ))
                elif current.annotator_id and other.annotator_id and current.annotator_id != other.annotator_id:
                    issues.append(AnnotationIssue(
                        code="same_behavior_overlap_across_annotators",
                        message="Different annotators supplied overlapping intervals for the same behavior.",
                        track_id=track_id, start_frame=overlap_start, end_frame=overlap_end,
                        annotation_ids=ids, behavior_ids=labels,
                    ))

    # Exact same track and interval but different labels, explicitly marked as
    # annotator disagreement only when at least two annotators are known.
    interval_groups: dict[tuple[str | int, int, int], list[BehaviorAnnotation]] = {}
    for ann in annotations:
        interval_groups.setdefault((ann.track_id, ann.start_frame, ann.end_frame), []).append(ann)
    for (track_id, start, end), items in interval_groups.items():
        annotators = {a.annotator_id for a in items if a.annotator_id}
        labels = {a.behavior_id for a in items}
        if len(annotators) >= 2 and len(labels) >= 2:
            issues.append(AnnotationIssue(
                code="annotator_disagreement",
                message="Multiple annotators assigned different behaviors to the same track and interval.",
                track_id=track_id, start_frame=start, end_frame=end,
                annotation_ids=tuple(sorted(a.annotation_id for a in items if a.annotation_id)),
                behavior_ids=tuple(sorted(labels)), severity="error",
            ))

    uncovered: dict[str, int] = {}
    if video_length_frames is not None:
        tracks = set(by_track)
        if expected_track_ids is not None:
            for track_id in expected_track_ids:
                if isinstance(track_id, bool) or not isinstance(track_id, (str, int)):
                    raise TypeError("expected_track_ids must contain only string or integer IDs.")
                tracks.add(track_id)
        for track_id in sorted(tracks, key=lambda value: (type(value).__name__, str(value))):
            intervals = sorted(
                (max(0, a.start_frame), min(video_length_frames - 1, a.end_frame))
                for a in by_track.get(track_id, ())
                if a.end_frame >= 0 and a.start_frame < video_length_frames
            )
            covered = 0
            cursor = 0
            for start, end in intervals:
                if end < start:
                    continue
                start = max(start, cursor)
                if end >= start:
                    covered += end - start + 1
                    cursor = end + 1
            uncovered[str(track_id)] = max(0, video_length_frames - covered)
            if not by_track.get(track_id):
                issues.append(AnnotationIssue(
                    code="track_without_annotations",
                    message="Expected or observed track has no annotations in this video.",
                    track_id=track_id, start_frame=0, end_frame=video_length_frames - 1,
                ))

    issues.sort(key=lambda issue: (
        issue.start_frame if issue.start_frame is not None else -1,
        str(issue.track_id), issue.code, issue.end_frame if issue.end_frame is not None else -1,
        issue.annotation_ids,
    ))
    return AnnotationQualityReport(
        video_id=annotation_set.video_id,
        annotation_count=len(annotations),
        track_count=len(by_track),
        behavior_counts=behavior_counts,
        issues=tuple(issues),
        uncovered_frames_by_track=uncovered,
    )
