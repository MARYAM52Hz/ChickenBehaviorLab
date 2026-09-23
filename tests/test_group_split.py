from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.dataset.group_split import (
    split_group_ids,
)

from chicken_behavior_lab.dataset.group_splitter import (
    split_dataset_by_group,
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

    for video_id in [
        "video_001",
        "video_002",
        "video_003",
        "video_004",
        "video_005",
        "video_006",
        "video_007",
        "video_008",
        "video_009",
        "video_010",
    ]:
        for frame in range(3):
            samples.append(
                DummySample(
                    metadata={
                        "video_id": video_id,
                        "track_id": 1,
                        "frame_id": frame,
                    }
                )
            )

    return DummyDataset(
        samples
    )


def test_group_ids_do_not_overlap():
    split = split_group_ids(
        [
            "video_001",
            "video_002",
            "video_003",
            "video_004",
            "video_005",
            "video_006",
            "video_007",
            "video_008",
            "video_009",
            "video_010",
        ],
        seed=42,
    )

    train = set(
        split.train
    )

    validation = set(
        split.validation
    )

    test = set(
        split.test
    )

    assert not train.intersection(
        validation
    )

    assert not train.intersection(
        test
    )

    assert not validation.intersection(
        test
    )


def test_dataset_split_has_no_video_leakage():
    dataset = make_dataset()

    splits = split_dataset_by_group(
        dataset,
        group_by="video",
        seed=42,
    )

    def get_videos(subset):
        videos = set()

        for index in range(
            len(subset)
        ):
            sample = subset[index]

            videos.add(
                sample.metadata[
                    "video_id"
                ]
            )

        return videos

    train_videos = get_videos(
        splits.train
    )

    validation_videos = get_videos(
        splits.validation
    )

    test_videos = get_videos(
        splits.test
    )

    assert not train_videos.intersection(
        validation_videos
    )

    assert not train_videos.intersection(
        test_videos
    )

    assert not validation_videos.intersection(
        test_videos
    )


def test_all_samples_are_preserved():
    dataset = make_dataset()

    splits = split_dataset_by_group(
        dataset,
        group_by="video",
        seed=42,
    )

    total_split_samples = (
        len(splits.train)
        + len(splits.validation)
        + len(splits.test)
    )

    assert total_split_samples == len(
        dataset
    )
