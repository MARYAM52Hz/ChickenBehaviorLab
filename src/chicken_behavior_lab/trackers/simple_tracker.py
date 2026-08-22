"""
ChickenBehaviorLab Simple IoU Tracker
======================================

A lightweight baseline tracker based on
Bounding Box IoU.
"""

from __future__ import annotations

from chicken_behavior_lab.models.detection import (
    Detection,
)

from chicken_behavior_lab.models.track import (
    Track,
)

from chicken_behavior_lab.trackers.base import (
    BaseTracker,
)


class SimpleIoUTracker(BaseTracker):
    """
    Simple multi-object tracker based on IoU matching.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed_frames: int = 15,
    ) -> None:

        self.iou_threshold = (
            iou_threshold
        )

        self.max_missed_frames = (
            max_missed_frames
        )

        self.tracks: list[Track] = []

        self.next_track_id: int = 1

    # =====================================================
    # Update
    # =====================================================

    def update(
        self,
        detections: list[Detection],
    ) -> list[Track]:

        matched_track_ids: set[int] = set()

        matched_detection_ids: set[str] = set()

        # -------------------------------------------------
        # Match existing tracks to detections
        # -------------------------------------------------

        for track in self.tracks:

            best_detection = None

            best_iou = 0.0

            for detection in detections:

                if (
                    detection.detection_id
                    in matched_detection_ids
                ):
                    continue

                iou = track.detection.bbox.iou(
                    detection.bbox
                )

                if iou > best_iou:

                    best_iou = iou

                    best_detection = detection

            # -------------------------------------------------
            # Match found
            # -------------------------------------------------

            if (
                best_detection is not None
                and best_iou
                >= self.iou_threshold
            ):

                track.update(
                    best_detection
                )

                matched_track_ids.add(
                    track.track_id
                )

                matched_detection_ids.add(
                    best_detection.detection_id
                )

            # -------------------------------------------------
            # No match
            # -------------------------------------------------

            else:

                track.mark_missed()

        # -------------------------------------------------
        # Create new tracks
        # -------------------------------------------------

        for detection in detections:

            if (
                detection.detection_id
                in matched_detection_ids
            ):
                continue

            new_track = Track(
                track_id=self.next_track_id,
                detection=detection,
            )

            self.tracks.append(
                new_track
            )

            self.next_track_id += 1

        # -------------------------------------------------
        # Remove dead tracks
        # -------------------------------------------------

        self.tracks = [
            track
            for track in self.tracks
            if track.missed_frames
            <= self.max_missed_frames
        ]

        return list(self.tracks)

    # =====================================================
    # Reset
    # =====================================================

    def reset(self) -> None:

        self.tracks.clear()

        self.next_track_id = 1
