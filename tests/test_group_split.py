from __future__ import annotations

import numpy as np
import pytest

from chicken_behavior_lab.dataset.group_split import (
    GroupAwareSplitter,
)
from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)


class DummyGraph:

    def __init__(self) -> None:
        self.node_features = np.zeros(
            (3, 4),
            dtype=np.float32,
        )

        self.edge_index = np.array(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=np.int64,
        )

        self.edge_features = None

    def validate(self) -> None:
        pass


def make_sample(
    sample_id: str,
    video_id: str,
    track_id: int,
) -> GraphSample:

    return GraphSample(
        graph=DummyGraph(),
        label=0,
        behavior_id="feeding",
        sample_id=sample_id,
        metadata={
            "video_id": video_id,
            "track_id": track_id,
            "start_frame": 0,
            "end_frame": 10,
        },
    )


def test_group_split_has_no_video_overlap() -> None:

    samples = []

    for video_index in range(10):

        video_id = (
            f"video_{video_index:03d}"
        )

        for track_id in range(2):

            samples.append(
                make_sample(
                    sample_id=(
                        f"{video_id}_"
                        f"track_{track_id}"
                    ),
                    video_id=video_id,
                    track_id=track_id,
                )
            )

    splitter = GroupAwareSplitter(
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
        group_key="video_id",
    )

    result = splitter.split(
        samples,
        seed=42,
    )

    train_groups = set(
        result.train_groups
    )

    validation_groups = set(
        result.validation_groups
    )

    test_groups = set(
        result.test_groups
    )

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


def test_split_is_reproducible() -> None:

    samples = [
        make_sample(
            sample_id=f"sample_{index}",
            video_id=f"video_{index}",
            track_id=0,
        )
        for index in range(10)
    ]

    splitter = GroupAwareSplitter(
        group_key="video_id"
    )

    first = splitter.split(
        samples,
        seed=123,
    )

    second = splitter.split(
        samples,
        seed=123,
    )

    assert first.train_groups == (
        second.train_groups
    )

    assert first.validation_groups == (
        second.validation_groups
    )

    assert first.test_groups == (
        second.test_groups
    )


def test_track_split_uses_video_and_track() -> None:

    samples = [
        make_sample(
            sample_id="a",
            video_id="video_a",
            track_id=1,
        ),
        make_sample(
            sample_id="b",
            video_id="video_a",
            track_id=2,
        ),
        make_sample(
            sample_id="c",
            video_id="video_b",
            track_id=1,
        ),
        make_sample(
            sample_id="d",
            video_id="video_b",
            track_id=2,
        ),
        make_sample(
            sample_id="e",
            video_id="video_c",
            track_id=1,
        ),
        make_sample(
            sample_id="f",
            video_id="video_c",
            track_id=2,
        ),
    ]

    splitter = GroupAwareSplitter(
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
        group_key="track_id",
    )

    result = splitter.split(
        samples,
        seed=42,
    )

    all_groups = (
        result.train_groups
        + result.validation_groups
        + result.test_groups
    )

    assert len(all_groups) == 6

    assert len(all_groups) == len(
        set(all_groups)
    )


@pytest.mark.parametrize("video_id", [None, "", "   "])
def test_track_split_rejects_missing_or_blank_video_id(video_id) -> None:
    samples = [
        make_sample(
            sample_id=f"sample_{index}",
            video_id=video_id,
            track_id=index,
        )
        for index in range(3)
    ]
    splitter = GroupAwareSplitter(
        train_ratio=0.34,
        validation_ratio=0.33,
        test_ratio=0.33,
        group_key="track_id",
    )

    import pytest
    with pytest.raises(ValueError, match="metadata.video_id"):
        splitter.split(samples, seed=42)



def test_train_validation_split_keeps_video_groups_together() -> None:
    from chicken_behavior_lab.dataset.split import (
        group_train_validation_split,
    )

    class SampleDataset:
        def __init__(self, samples):
            self.samples = list(samples)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    samples = [
        make_sample(
            sample_id=f"video_{video_index}_track_{track_id}",
            video_id=f"video_{video_index}",
            track_id=track_id,
        )
        for video_index in range(8)
        for track_id in range(2)
    ]
    dataset = SampleDataset(samples)

    train, validation = group_train_validation_split(
        dataset,
        validation_fraction=0.25,
        random_seed=19,
        group_by="video",
    )

    train_groups = {
        dataset.samples[index].metadata["video_id"]
        for index in train.indices
    }
    validation_groups = {
        dataset.samples[index].metadata["video_id"]
        for index in validation.indices
    }

    assert train_groups.isdisjoint(validation_groups)
    assert len(train) + len(validation) == len(dataset)
    assert len(train_groups) == 6
    assert len(validation_groups) == 2


def test_train_validation_group_split_rejects_single_group() -> None:
    from chicken_behavior_lab.dataset.split import (
        group_train_validation_split,
    )

    class SampleDataset:
        def __init__(self, samples):
            self.samples = list(samples)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    dataset = SampleDataset([
        make_sample("sample_a", "same_video", 1),
        make_sample("sample_b", "same_video", 2),
    ])

    with pytest.raises(ValueError, match="At least two distinct groups"):
        group_train_validation_split(dataset, group_by="video")
