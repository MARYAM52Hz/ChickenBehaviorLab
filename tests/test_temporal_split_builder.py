from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.dataset.group_splitter import (
    split_dataset_by_group,
)

from chicken_behavior_lab.dataset.temporal_split_builder import (
    build_temporal_splits,
)


@dataclass
class DummySample:
    metadata: dict


class DummyDataset:

    def __init__(
        self,
        samples,
    ):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(
        self,
        index,
    ):
        return self.samples[index]


def make_dataset():

    samples = []

    for video_number in range(
        1,
        7,
    ):

        video_id = (
            f"video_{video_number:03d}"
        )

        for frame in range(
            40
        ):

            samples.append(
                DummySample(
                    metadata={
                        "video_id": video_id,
                        "track_id": 1,
                        "frame_id": frame,
                        "behavior_id": (
                            "feeding"
                            if frame < 20
                            else "walking"
                        ),
                    }
                )
            )

    return DummyDataset(
        samples
    )


def test_temporal_windows_do_not_leak():

    dataset = make_dataset()

    dataset_splits = (
        split_dataset_by_group(
            dataset,
            group_by="video",
            train_ratio=0.50,
            validation_ratio=0.25,
            test_ratio=0.25,
            seed=42,
        )
    )

    temporal_splits = (
        build_temporal_splits(
            dataset_splits,
            sequence_length=8,
            sequence_stride=4,
        )
    )

    train_groups = {
        (
            window.video_id,
            window.track_id,
        )
        for window
        in temporal_splits.train
    }

    validation_groups = {
        (
            window.video_id,
            window.track_id,
        )
        for window
        in temporal_splits.validation
    }

    test_groups = {
        (
            window.video_id,
            window.track_id,
        )
        for window
        in temporal_splits.test
    }

    assert not (
        train_groups
        & validation_groups
    )

    assert not (
        train_groups
        & test_groups
    )

    assert not (
        validation_groups
        & test_groups
    )


def test_windows_are_temporal():

    dataset = make_dataset()

    dataset_splits = (
        split_dataset_by_group(
            dataset,
            group_by="video",
            train_ratio=0.50,
            validation_ratio=0.25,
            test_ratio=0.25,
            seed=42,
        )
    )

    temporal_splits = (
        build_temporal_splits(
            dataset_splits,
            sequence_length=8,
            sequence_stride=4,
        )
    )

    for window in (
        temporal_splits.train
    ):

        assert len(
            window.samples
        ) == 8

        assert (
            window.end_frame
            >=
            window.start_frame
        )

        assert (
            window.end_frame
            - window.start_frame
            == 7
        )
