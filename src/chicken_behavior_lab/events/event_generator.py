"""Convert temporal behavior predictions into interval-based behavior events.

This module is deliberately independent of mortality prediction and clinical
interpretation. It converts model outputs into event intervals only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence


class EventGenerationError(ValueError):
    """Raised when prediction records do not satisfy the event-generation contract."""


@dataclass(frozen=True, slots=True)
class BehaviorEvent:
    """A predicted behavior interval; start_frame and end_frame are inclusive."""

    video_id: str
    track_id: str | int
    behavior_id: str
    start_frame: int
    end_frame: int
    confidence: float | None = None
    source_sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise EventGenerationError("video_id must be a non-empty string.")
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, (str, int)):
            raise EventGenerationError("track_id must be a non-empty string or integer.")
        if isinstance(self.track_id, str) and not self.track_id.strip():
            raise EventGenerationError("track_id must not be empty.")
        if not isinstance(self.behavior_id, str) or not self.behavior_id.strip():
            raise EventGenerationError("behavior_id must be a non-empty string.")
        for name, value in (("start_frame", self.start_frame), ("end_frame", self.end_frame)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise EventGenerationError(f"{name} must be a non-negative integer.")
        if self.end_frame < self.start_frame:
            raise EventGenerationError("end_frame must be greater than or equal to start_frame.")
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
                raise EventGenerationError("confidence must be numeric or None.")
            if not isfinite(float(self.confidence)) or not 0.0 <= float(self.confidence) <= 1.0:
                raise EventGenerationError("confidence must be finite and in [0, 1].")
            object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "video_id", self.video_id.strip())
        object.__setattr__(self, "behavior_id", self.behavior_id.strip())
        object.__setattr__(self, "source_sample_ids", tuple(str(x) for x in self.source_sample_ids))

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_sample_ids"] = list(self.source_sample_ids)
        result["duration_frames"] = self.duration_frames
        return result


def _read(record: Any, key: str, *aliases: str, default: Any = None) -> Any:
    for name in (key, *aliases):
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    metadata = record.get("metadata") if isinstance(record, Mapping) else getattr(record, "metadata", None)
    if isinstance(metadata, Mapping):
        for name in (key, *aliases):
            if name in metadata:
                return metadata[name]
    return default


class EventGenerator:
    """Merge adjacent/overlapping same-label prediction windows into events.

    Prediction records may be mappings or objects. Required fields are
    ``video_id``, ``track_id``, inclusive frame bounds, and a predicted label
    (``predicted_label`` / ``behavior_id`` / ``predicted_index``). If an index
    is supplied, ``index_to_label`` must map it to a behavior name.
    """

    def __init__(self, *, max_gap_frames: int = 0, minimum_confidence: float = 0.0,
                 index_to_label: Mapping[int, str] | Sequence[str] | None = None) -> None:
        if isinstance(max_gap_frames, bool) or not isinstance(max_gap_frames, int) or max_gap_frames < 0:
            raise ValueError("max_gap_frames must be a non-negative integer.")
        if isinstance(minimum_confidence, bool) or not isinstance(minimum_confidence, (int, float)):
            raise ValueError("minimum_confidence must be numeric.")
        if not isfinite(float(minimum_confidence)) or not 0.0 <= float(minimum_confidence) <= 1.0:
            raise ValueError("minimum_confidence must be finite and in [0, 1].")
        self.max_gap_frames = max_gap_frames
        self.minimum_confidence = float(minimum_confidence)
        self.index_to_label = index_to_label

    def _label(self, record: Any) -> str:
        label = _read(record, "predicted_label", "behavior_id", "label")
        if label is not None:
            if not isinstance(label, str) or not label.strip():
                raise EventGenerationError("Predicted behavior label must be a non-empty string.")
            return label.strip()
        index = _read(record, "predicted_index", "predicted_class", "y_pred")
        if index is None:
            raise EventGenerationError("Prediction record needs predicted_label/behavior_id or predicted_index.")
        if self.index_to_label is None:
            raise EventGenerationError("index_to_label is required when records contain only predicted indices.")
        try:
            label = self.index_to_label[index]  # supports mappings and sequences
        except (KeyError, IndexError, TypeError) as exc:
            raise EventGenerationError(f"No behavior label configured for predicted index {index!r}.") from exc
        if not isinstance(label, str) or not label.strip():
            raise EventGenerationError(f"Mapped label for predicted index {index!r} is invalid.")
        return label.strip()

    def _normalize(self, record: Any) -> dict[str, Any] | None:
        video_id = _read(record, "video_id")
        track_id = _read(record, "track_id")
        start = _read(record, "start_frame", "frame_start")
        end = _read(record, "end_frame", "frame_end")
        confidence = _read(record, "confidence", "probability", default=None)
        sample_id = _read(record, "sample_id", "id", default=None)
        if not isinstance(video_id, str) or not video_id.strip():
            raise EventGenerationError("Each prediction record needs a non-empty video_id.")
        if isinstance(track_id, bool) or not isinstance(track_id, (str, int)) or (isinstance(track_id, str) and not track_id.strip()):
            raise EventGenerationError("Each prediction record needs a non-empty string or integer track_id.")
        for name, value in (("start_frame", start), ("end_frame", end)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise EventGenerationError(f"Each prediction record needs a non-negative integer {name}.")
        if end < start:
            raise EventGenerationError("Prediction end_frame must be greater than or equal to start_frame.")
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
                raise EventGenerationError("Prediction confidence must be finite and in [0, 1].")
            confidence = float(confidence)
            if confidence < self.minimum_confidence:
                return None
        return {"video_id": video_id.strip(), "track_id": track_id, "behavior_id": self._label(record),
                "start_frame": start, "end_frame": end, "confidence": confidence,
                "sample_id": None if sample_id is None else str(sample_id)}

    def generate(self, predictions: Iterable[Any]) -> list[BehaviorEvent]:
        """Return deterministic event intervals from model prediction records."""
        if predictions is None:
            raise TypeError("predictions must be an iterable, not None.")
        normalized = [item for raw in predictions if (item := self._normalize(raw)) is not None]
        normalized.sort(key=lambda r: (r["video_id"], str(r["track_id"]), r["behavior_id"], r["start_frame"], r["end_frame"], r["sample_id"] or ""))
        events: list[BehaviorEvent] = []
        current: dict[str, Any] | None = None
        source_ids: list[str] = []
        confidences: list[float] = []

        def flush() -> None:
            nonlocal current, source_ids, confidences
            if current is None:
                return
            confidence = sum(confidences) / len(confidences) if confidences else None
            events.append(BehaviorEvent(
                video_id=current["video_id"], track_id=current["track_id"], behavior_id=current["behavior_id"],
                start_frame=current["start_frame"], end_frame=current["end_frame"], confidence=confidence,
                source_sample_ids=tuple(source_ids),
            ))
            current, source_ids, confidences = None, [], []

        for row in normalized:
            same_group = current is not None and all(row[k] == current[k] for k in ("video_id", "track_id", "behavior_id"))
            within_gap = same_group and row["start_frame"] <= current["end_frame"] + self.max_gap_frames + 1
            if not within_gap:
                flush()
                current = dict(row)
                source_ids = [row["sample_id"]] if row["sample_id"] is not None else []
                confidences = [row["confidence"]] if row["confidence"] is not None else []
            else:
                current["end_frame"] = max(current["end_frame"], row["end_frame"])
                if row["sample_id"] is not None:
                    source_ids.append(row["sample_id"])
                if row["confidence"] is not None:
                    confidences.append(row["confidence"])
        flush()
        return events
