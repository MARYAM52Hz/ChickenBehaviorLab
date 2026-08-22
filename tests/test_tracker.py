"""
Tests for ChickenBehaviorLab Tracker.
"""

from chicken_behavior_lab.models.detection import (
    Detection,
)

from chicken_behavior_lab.models.geometry import (
    BoundingBox,
)

from chicken_behavior_lab.trackers.simple_tracker import (
    SimpleIoUTracker,
)


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


def test_tracker_preserves_identity():

    tracker = SimpleIoUTracker(
        iou_threshold=0.3
    )

    # Frame 1
    detections_frame_1 = [
        create_detection(
            "f1_d1",
            100,
            100,
        )
    ]

    tracks = tracker.update(
        detections_frame_1
    )

    assert len(tracks) == 1

    first_track_id = (
        tracks[0].track_id
    )

    # Frame 2
    detections_frame_2 = [
        create_detection(
            "f2_d1",
            105,
            105,
        )
    ]

    tracks = tracker.update(
        detections_frame_2
    )

    assert len(tracks) == 1

    assert (
        tracks[0].track_id
        == first_track_id
    )
