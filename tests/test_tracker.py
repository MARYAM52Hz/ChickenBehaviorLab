"""
Tests for ChickenBehaviorLab Tracker.
"""

import pytest

from chicken_behavior_lab.models.detection import Detection
from chicken_behavior_lab.models.geometry import BoundingBox
from chicken_behavior_lab.trackers.simple_tracker import SimpleIoUTracker


def create_detection(
    detection_id: str,
    x: float,
    y: float,
) -> Detection:
    return Detection(
        detection_id=detection_id,
        frame_id="frame",
        bbox=BoundingBox(
            x_min=x,
            y_min=y,
            x_max=x + 100,
            y_max=y + 100,
        ),
        confidence=0.95,
    )


def test_tracker_preserves_identity_and_uses_string_ids():
    tracker = SimpleIoUTracker(iou_threshold=0.3)

    tracks = tracker.update([create_detection("f1_d1", 100, 100)])

    assert len(tracks) == 1
    first_track_id = tracks[0].track_id
    assert first_track_id == "track_1"
    assert isinstance(first_track_id, str)

    tracks = tracker.update([create_detection("f2_d1", 105, 105)])

    assert len(tracks) == 1
    assert tracks[0].track_id == first_track_id


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_tracker_rejects_invalid_iou_threshold(threshold):
    with pytest.raises(ValueError, match="iou_threshold"):
        SimpleIoUTracker(iou_threshold=threshold)


@pytest.mark.parametrize("max_missed", [-1, 1.5, True])
def test_tracker_rejects_invalid_max_missed_frames(max_missed):
    with pytest.raises(ValueError, match="max_missed_frames"):
        SimpleIoUTracker(max_missed_frames=max_missed)


def test_tracker_allows_boundary_parameters():
    tracker = SimpleIoUTracker(iou_threshold=0.0, max_missed_frames=0)
    assert tracker.iou_threshold == 0.0
    assert tracker.max_missed_frames == 0
