"""
Tests for temporal skeleton sequences.
"""

from chicken_behavior_lab.models.tracked_skeleton import (
    TrackedSkeleton,
)

from chicken_behavior_lab.models.skeleton import (
    Skeleton,
)

from chicken_behavior_lab.temporal.sequence import (
    TemporalSkeletonSequence,
)


def create_observation(
    frame_id: str,
) -> TrackedSkeleton:

    skeleton = Skeleton(
        skeleton_id=(
            f"track_1_{frame_id}"
        ),
        frame_id=frame_id,
        keypoints=[],
        edges=[],
        track_id="track_1",
        confidence=0.9,
    )

    return TrackedSkeleton(
        track_id="track_1",
        frame_id=frame_id,
        skeleton=skeleton,
    )


def test_sequence_keeps_temporal_order():

    sequence = (
        TemporalSkeletonSequence(
            track_id="track_1",
            max_length=3,
        )
    )

    sequence.add(
        create_observation("frame_1")
    )

    sequence.add(
        create_observation("frame_2")
    )

    sequence.add(
        create_observation("frame_3")
    )

    assert sequence.length == 3

    assert (
        sequence.observations[0].frame_id
        == "frame_1"
    )

    assert (
        sequence.latest.frame_id
        == "frame_3"
    )


def test_sequence_sliding_window():

    sequence = (
        TemporalSkeletonSequence(
            track_id="track_1",
            max_length=3,
        )
    )

    for i in range(1, 6):

        sequence.add(
            create_observation(
                f"frame_{i}"
            )
        )

    assert sequence.length == 3

    assert (
        sequence.observations[0].frame_id
        == "frame_3"
    )

    assert (
        sequence.latest.frame_id
        == "frame_5"
    )
