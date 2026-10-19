"""Descriptive analytics for generated behavior events.

These statistics describe model outputs; they are not clinical conclusions or
mortality predictions.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable, Mapping

from chicken_behavior_lab.events.event_generator import BehaviorEvent


class AnalyticsError(ValueError):
    """Raised when analytics inputs or parameters are invalid."""


@dataclass(frozen=True, slots=True)
class BehaviorAnalyticsReport:
    video_id: str | None
    event_count: int
    total_duration_frames: int
    total_duration_seconds: float | None
    by_behavior: dict[str, dict[str, int | float | None]]
    by_track: dict[str, dict[str, int | float | None]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "event_count": self.event_count,
            "total_duration_frames": self.total_duration_frames,
            "total_duration_seconds": self.total_duration_seconds,
            "by_behavior": self.by_behavior,
            "by_track": self.by_track,
        }


class Analytics:
    """Compute descriptive event counts and durations by behavior and track."""

    @staticmethod
    def _event_value(event: Any, name: str) -> Any:
        if isinstance(event, Mapping):
            return event.get(name)
        return getattr(event, name, None)

    def compute(self, events: Iterable[BehaviorEvent | Mapping[str, Any]], *,
                frame_rate: float | None = None, video_id: str | None = None) -> BehaviorAnalyticsReport:
        if events is None:
            raise TypeError("events must be an iterable, not None.")
        if frame_rate is not None:
            if isinstance(frame_rate, bool) or not isinstance(frame_rate, (int, float)) or not isfinite(float(frame_rate)) or frame_rate <= 0:
                raise AnalyticsError("frame_rate must be a finite positive number or None.")
            frame_rate = float(frame_rate)
        if video_id is not None and (not isinstance(video_id, str) or not video_id.strip()):
            raise AnalyticsError("video_id must be a non-empty string or None.")
        normalized: list[dict[str, Any]] = []
        for index, event in enumerate(events):
            vid = self._event_value(event, "video_id")
            track = self._event_value(event, "track_id")
            behavior = self._event_value(event, "behavior_id")
            start = self._event_value(event, "start_frame")
            end = self._event_value(event, "end_frame")
            if not isinstance(vid, str) or not vid.strip():
                raise AnalyticsError(f"Event at index {index} has invalid video_id.")
            if isinstance(track, bool) or not isinstance(track, (str, int)) or (isinstance(track, str) and not track.strip()):
                raise AnalyticsError(f"Event at index {index} has invalid track_id.")
            if not isinstance(behavior, str) or not behavior.strip():
                raise AnalyticsError(f"Event at index {index} has invalid behavior_id.")
            for name, value in (("start_frame", start), ("end_frame", end)):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise AnalyticsError(f"Event at index {index} has invalid {name}.")
            if end < start:
                raise AnalyticsError(f"Event at index {index} has end_frame before start_frame.")
            if video_id is not None and vid != video_id:
                continue
            normalized.append({"video_id": vid, "track_id": track, "behavior_id": behavior.strip(),
                               "duration_frames": end - start + 1})

        behavior_stats: dict[str, dict[str, int | float | None]] = {}
        track_stats: dict[str, dict[str, int | float | None]] = {}
        behavior_acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        track_acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        total_frames = 0
        for row in normalized:
            duration = row["duration_frames"]
            total_frames += duration
            behavior_acc[row["behavior_id"]][0] += 1
            behavior_acc[row["behavior_id"]][1] += duration
            track_acc[str(row["track_id"])][0] += 1
            track_acc[str(row["track_id"])][1] += duration

        def stats_map(acc: dict[str, list[int]]) -> dict[str, dict[str, int | float | None]]:
            result: dict[str, dict[str, int | float | None]] = {}
            for key in sorted(acc):
                count, frames = acc[key]
                result[key] = {
                    "event_count": count,
                    "total_duration_frames": frames,
                    "total_duration_seconds": frames / frame_rate if frame_rate is not None else None,
                }
            return result

        videos = {row["video_id"] for row in normalized}
        report_video_id = video_id.strip() if video_id is not None else (next(iter(videos)) if len(videos) == 1 else None)
        return BehaviorAnalyticsReport(
            video_id=report_video_id,
            event_count=len(normalized),
            total_duration_frames=total_frames,
            total_duration_seconds=total_frames / frame_rate if frame_rate is not None else None,
            by_behavior=stats_map(behavior_acc),
            by_track=stats_map(track_acc),
        )
