"""
ChickenBehaviorLab Simple IoU Tracker
======================================

A lightweight baseline tracker based on
Bounding Box IoU.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

from chicken_behavior_lab.models.detection import Detection
from chicken_behavior_lab.models.track import Track
from chicken_behavior_lab.trackers.base import BaseTracker


class SimpleIoUTracker(BaseTracker):
    """Simple multi-object tracker based on greedy IoU matching."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed_frames: int = 15,
    ) -> None:
        if (
            isinstance(iou_threshold, bool)
            or not isinstance(iou_threshold, Real)
            or not math.isfinite(float(iou_threshold))
            or not 0.0 <= float(iou_threshold) <= 1.0
        ):
            raise ValueError("iou_threshold must be a finite number in [0, 1].")

        if (
            isinstance(max_missed_frames, bool)
            or not isinstance(max_missed_frames, Integral)
            or max_missed_frames < 0
        ):
            raise ValueError("max_missed_frames must be a non-negative integer.")

        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self.tracks: list[Track] = []
        self.next_track_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        matched_detection_ids: set[str] = set()

        # Greedy matching is intentionally retained as a lightweight baseline.
        for track in self.tracks:
            best_detection = None
            best_iou = 0.0

            for detection in detections:
                if detection.detection_id in matched_detection_ids:
                    continue

                iou = track.detection.bbox.iou(detection.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_detection = detection

            if best_detection is not None and best_iou >= self.iou_threshold:
                track.update(best_detection)
                matched_detection_ids.add(best_detection.detection_id)
            else:
                track.mark_missed()

        for detection in detections:
            if detection.detection_id in matched_detection_ids:
                continue

            new_track = Track(
                track_id=f"track_{self.next_track_id}",
                detection=detection,
            )
            self.tracks.append(new_track)
            self.next_track_id += 1

        self.tracks = [
            track
            for track in self.tracks
            if track.missed_frames <= self.max_missed_frames
        ]

        return list(self.tracks)

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1
