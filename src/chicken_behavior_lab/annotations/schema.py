"""Typed schemas for interval-based chicken behavior annotations.

The public annotation contract deliberately separates behavior labels from
higher-level events and mortality outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class AnnotationValidationError(ValueError):
    """Raised when an annotation or annotation set violates the data contract."""


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnnotationValidationError(f"'{field_name}' must be a non-empty string.")
    return value.strip()


def _validate_track_id(value: Any) -> str | int:
    # Booleans are integers in Python, but are not valid track identifiers.
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise AnnotationValidationError("'track_id' must be a non-empty string or an integer.")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise AnnotationValidationError("'track_id' must not be an empty string.")
    return value


def _validate_frame(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnnotationValidationError(f"'{field_name}' must be an integer frame index.")
    if value < 0:
        raise AnnotationValidationError(f"'{field_name}' must be greater than or equal to zero.")
    return value


@dataclass(frozen=True, slots=True)
class BehaviorAnnotation:
    """A behavior label assigned to one tracked individual over an inclusive frame interval."""

    track_id: str | int
    behavior_id: str
    start_frame: int
    end_frame: int
    annotation_id: str | None = None
    annotator_id: str | None = None
    confidence: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _validate_track_id(self.track_id))
        object.__setattr__(self, "behavior_id", _require_non_empty_string(self.behavior_id, "behavior_id"))
        start = _validate_frame(self.start_frame, "start_frame")
        end = _validate_frame(self.end_frame, "end_frame")
        if end < start:
            raise AnnotationValidationError("'end_frame' must be greater than or equal to 'start_frame'.")
        object.__setattr__(self, "start_frame", start)
        object.__setattr__(self, "end_frame", end)

        if self.annotation_id is not None:
            object.__setattr__(self, "annotation_id", _require_non_empty_string(self.annotation_id, "annotation_id"))
        if self.annotator_id is not None:
            object.__setattr__(self, "annotator_id", _require_non_empty_string(self.annotator_id, "annotator_id"))
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
                raise AnnotationValidationError("'confidence' must be a number between 0 and 1.")
            if not 0.0 <= float(self.confidence) <= 1.0:
                raise AnnotationValidationError("'confidence' must be between 0 and 1, inclusive.")
            object.__setattr__(self, "confidence", float(self.confidence))
        if not isinstance(self.attributes, Mapping):
            raise AnnotationValidationError("'attributes' must be a JSON object / mapping.")
        object.__setattr__(self, "attributes", dict(self.attributes))

    @property
    def duration_frames(self) -> int:
        """Return interval length, counting both start and end frames."""
        return self.end_frame - self.start_frame + 1

    def contains_frame(self, frame_index: int) -> bool:
        """Return whether a frame belongs to this inclusive interval."""
        frame_index = _validate_frame(frame_index, "frame_index")
        return self.start_frame <= frame_index <= self.end_frame

    def overlaps(self, other: "BehaviorAnnotation") -> bool:
        """Return whether two annotations overlap in time for the same track."""
        if not isinstance(other, BehaviorAnnotation):
            raise TypeError("'other' must be a BehaviorAnnotation.")
        return self.track_id == other.track_id and self.start_frame <= other.end_frame and other.start_frame <= self.end_frame

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BehaviorAnnotation":
        if not isinstance(data, Mapping):
            raise AnnotationValidationError("Each item in 'annotations' must be a JSON object.")
        required = ("track_id", "behavior_id", "start_frame", "end_frame")
        missing = [key for key in required if key not in data]
        if missing:
            raise AnnotationValidationError(f"Annotation is missing required field(s): {', '.join(missing)}.")
        allowed = set(required) | {"annotation_id", "annotator_id", "confidence", "attributes"}
        unexpected = sorted(set(data) - allowed)
        if unexpected:
            raise AnnotationValidationError(f"Unexpected annotation field(s): {', '.join(map(str, unexpected))}.")
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "track_id": self.track_id,
            "behavior_id": self.behavior_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }
        optional = {
            "annotation_id": self.annotation_id,
            "annotator_id": self.annotator_id,
            "confidence": self.confidence,
        }
        result.update({key: value for key, value in optional.items() if value is not None})
        if self.attributes:
            result["attributes"] = dict(self.attributes)
        return result


@dataclass(frozen=True, slots=True)
class AnnotationSet:
    """All behavior annotations associated with one video."""

    video_id: str
    annotations: tuple[BehaviorAnnotation, ...]
    schema_version: str = "1.0"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_id", _require_non_empty_string(self.video_id, "video_id"))
        object.__setattr__(self, "schema_version", _require_non_empty_string(self.schema_version, "schema_version"))
        normalized = tuple(self.annotations)
        if any(not isinstance(item, BehaviorAnnotation) for item in normalized):
            raise AnnotationValidationError("'annotations' must contain only BehaviorAnnotation instances.")
        ids = [item.annotation_id for item in normalized if item.annotation_id is not None]
        if len(ids) != len(set(ids)):
            raise AnnotationValidationError("'annotation_id' values must be unique within an AnnotationSet.")
        if not isinstance(self.metadata, Mapping):
            raise AnnotationValidationError("'metadata' must be a JSON object / mapping.")
        object.__setattr__(self, "annotations", normalized)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def for_track(self, track_id: str | int) -> tuple[BehaviorAnnotation, ...]:
        return tuple(item for item in self.annotations if item.track_id == track_id)

    def for_behavior(self, behavior_id: str) -> tuple[BehaviorAnnotation, ...]:
        behavior_id = _require_non_empty_string(behavior_id, "behavior_id")
        return tuple(item for item in self.annotations if item.behavior_id == behavior_id)

    def at_frame(self, frame_index: int, *, track_id: str | int | None = None) -> tuple[BehaviorAnnotation, ...]:
        frame_index = _validate_frame(frame_index, "frame_index")
        return tuple(
            item for item in self.annotations
            if item.contains_frame(frame_index) and (track_id is None or item.track_id == track_id)
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnnotationSet":
        if not isinstance(data, Mapping):
            raise AnnotationValidationError("The annotation document root must be a JSON object.")
        required = ("video_id", "annotations")
        missing = [key for key in required if key not in data]
        if missing:
            raise AnnotationValidationError(f"Annotation document is missing required field(s): {', '.join(missing)}.")
        allowed = {"schema_version", "video_id", "annotations", "metadata"}
        unexpected = sorted(set(data) - allowed)
        if unexpected:
            raise AnnotationValidationError(f"Unexpected document field(s): {', '.join(map(str, unexpected))}.")
        raw_annotations = data["annotations"]
        if not isinstance(raw_annotations, list):
            raise AnnotationValidationError("'annotations' must be a JSON array.")
        annotations = tuple(BehaviorAnnotation.from_dict(item) for item in raw_annotations)
        return cls(
            video_id=data["video_id"],
            annotations=annotations,
            schema_version=data.get("schema_version", "1.0"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "annotations": [item.to_dict() for item in self.annotations],
            "metadata": dict(self.metadata),
        }
