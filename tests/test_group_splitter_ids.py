"""Tests for group-level splits using canonical track identities."""

from dataclasses import dataclass

from chicken_behavior_lab.dataset.group_split import build_group_id
from chicken_behavior_lab.dataset.group_splitter import split_dataset_by_group


@dataclass
class Sample:
    sample_id: str
    metadata: dict


class Dataset:
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_group_identity_normalizes_legacy_and_canonical_track_ids():
    numeric = Sample("numeric", {"video_id": "video_01", "track_id": 1})
    named = Sample("named", {"video_id": "video_01", "track_id": "track_1"})
    other_video = Sample("other", {"video_id": "video_02", "track_id": 1})

    assert build_group_id(numeric, group_by="track") == "video_01::track::track_1"
    assert build_group_id(named, group_by="track") == "video_01::track::track_1"
    assert build_group_id(other_video, group_by="track") == "video_02::track::track_1"


def test_group_split_keeps_the_same_track_identity_together():
    samples = []
    for video_index in range(3):
        video_id = f"video_{video_index:02d}"
        samples.extend([
            Sample(
                f"{video_id}_numeric",
                {"video_id": video_id, "track_id": 1},
            ),
            Sample(
                f"{video_id}_named",
                {"video_id": video_id, "track_id": "track_1"},
            ),
        ])

    splits = split_dataset_by_group(
        Dataset(samples),
        group_by="track",
        train_ratio=0.34,
        validation_ratio=0.33,
        test_ratio=0.33,
        seed=17,
    )

    group_sets = [
        set(splits.groups.train),
        set(splits.groups.validation),
        set(splits.groups.test),
    ]
    assert all(group_sets[i].isdisjoint(group_sets[j]) for i in range(3) for j in range(i + 1, 3))

    for subset in (splits.train, splits.validation, splits.test):
        items = [subset[index] for index in range(len(subset))]
        assert len(items) % 2 == 0
        for index in range(0, len(items), 2):
            assert items[index].metadata["video_id"] == items[index + 1].metadata["video_id"]
