"""
Tests for the temporal skeleton buffer.
"""

from chicken_behavior_lab.models.skeleton import (
    Skeleton,
)

from chicken_behavior_lab.models.tracked_skeleton import (
    TrackedSkeleton,
)

from chicken_behavior_lab.temporal.buffer import (
    TemporalSkeletonBuffer,
)


def create_observation(
    track_id: str,
    frame_id: str,
) -> TrackedSkeleton:

    skeleton = Skeleton(
        skeleton_id=(
            f"{track_id}_{frame_id}"
        ),
        frame_id=frame_id,
        keypoints=[],
        edges=[],
        track_id=track_id,
        confidence=0.9,
    )

    return TrackedSkeleton(
        track_id=track_id,
        frame_id=frame_id,
        skeleton=skeleton,
    )


def test_buffer_separates_tracks():

    buffer = TemporalSkeletonBuffer(
        max_length=10
    )

    buffer.add(
        create_observation(
            "track_1",
            "frame_1",
        )
    )

    buffer.add(
        create_observation(
            "track_2",
            "frame_1",
        )
    )

    buffer.add(
        create_observation(
            "track_1",
            "frame_2",
        )
    )

    sequence_1 = buffer.get(
        "track_1"
    )

    sequence_2 = buffer.get(
        "track_2"
    )

    assert sequence_1 is not None

    assert sequence_2 is not None

    assert sequence_1.length == 2

    assert sequence_2.length == 1
