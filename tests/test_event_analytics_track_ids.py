""""Track identity remains canonical in generated events and analytics."""

import pytest

from chicken_behavior_lab.analytics.behavior_analytics import Analytics
from chicken_behavior_lab.core.enums.behavior import BehaviorType
from chicken_behavior_lab.core.enums.event import EventType
from chicken_behavior_lab.events.event_generator import BehaviorEvent, EventGenerator
from chicken_behavior_lab.models.behavior_event import BehaviorEvent as TransitionBehaviorEvent


def test_event_generator_merges_numeric_and_canonical_track_ids():
    generator = EventGenerator(max_gap_frames=0)
    events = generator.generate([
        {
            "video_id": "video_001", "track_id": 1,
            "behavior_id": "feeding", "start_frame": 10, "end_frame": 12,
            "confidence": 0.8, "sample_id": "sample_1",
        },
        {
            "video_id": "video_001", "track_id": "track_1",
            "behavior_id": "feeding", "start_frame": 13, "end_frame": 15,
            "confidence": 0.9, "sample_id": "sample_2",
        },
    ])

    assert len(events) == 1
    assert events[0].track_id == "track_1"
    assert events[0].start_frame == 10
    assert events[0].end_frame == 15
    assert events[0].source_sample_ids == ("sample_1", "sample_2")


def test_behavior_event_canonicalizes_legacy_integer_track_id():
    event = BehaviorEvent(
        video_id="video_001",
        track_id=7,
        behavior_id="walking",
        start_frame=1,
        end_frame=3,
    )
    assert event.track_id == "track_7"


def test_transition_event_model_canonicalizes_legacy_track_ids():
    event = TransitionBehaviorEvent(
        event_id="event_001",
        track_id=7,
        event_type=EventType.BEHAVIOR_STARTED,
        previous_behavior=BehaviorType.STANDING,
        current_behavior=BehaviorType.FEEDING,
        frame_number=10,
        confidence=0.9,
    )

    assert event.track_id == "track_7"


def test_transition_event_model_rejects_unsafe_track_ids():
    with pytest.raises(ValueError, match="track_id contains unsupported"):
        TransitionBehaviorEvent(
            event_id="event_002",
            track_id="../bird",
            event_type=EventType.BEHAVIOR_STARTED,
            previous_behavior=BehaviorType.STANDING,
            current_behavior=BehaviorType.FEEDING,
            frame_number=11,
            confidence=0.9,
        )


def test_analytics_normalizes_ids_and_does_not_merge_tracks_across_videos():
    report = Analytics().compute([
        {
            "video_id": "video_001", "track_id": 1,
            "behavior_id": "feeding", "start_frame": 0, "end_frame": 9,
        },
        {
            "video_id": "video_001", "track_id": "track_1",
            "behavior_id": "walking", "start_frame": 10, "end_frame": 19,
        },
        {
            "video_id": "video_002", "track_id": 1,
            "behavior_id": "feeding", "start_frame": 0, "end_frame": 4,
        },
    ])

    assert report.by_track == {
        "video_001::track_1": {
            "event_count": 2,
            "total_duration_frames": 20,
            "total_duration_seconds": None,
        },
        "video_002::track_1": {
            "event_count": 1,
            "total_duration_frames": 5,
            "total_duration_seconds": None,
        },
    }
