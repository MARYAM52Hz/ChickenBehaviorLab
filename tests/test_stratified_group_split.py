from __future__ import annotations

import numpy as np

from chicken_behavior_lab.dataset.group_split import GroupAwareSplitter
from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.dataset.split import group_train_validation_split


class DummyGraph:
    def validate(self) -> None:
        pass


def make_sample(sample_id: str, video_id: str, behavior_id: str, label: int) -> GraphSample:
    return GraphSample(
        graph=DummyGraph(),
        label=label,
        behavior_id=behavior_id,
        sample_id=sample_id,
        metadata={
            "video_id": video_id,
            "track_id": "track_1",
            "start_frame": 0,
            "end_frame": 0,
        },
    )


def make_stratified_samples() -> list[GraphSample]:
    samples = []
    for video_index in range(12):
        video_id = f"video_{video_index:02d}"
        behaviors = []
        if video_index < 8:
            behaviors.append(("feeding", 0))
        if video_index >= 4:
            behaviors.append(("walking", 1))
        if video_index == 0:
            behaviors.append(("dust_bathing", 2))
        for behavior_id, label in behaviors:
            for annotation_index in range(3):
                samples.append(
                    make_sample(
                        f"{video_id}_{behavior_id}_{annotation_index}",
                        video_id,
                        behavior_id,
                        label,
                    )
                )
    return samples


def test_three_way_split_stratifies_classes_without_group_leakage() -> None:
    samples = make_stratified_samples()
    splitter = GroupAwareSplitter(
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        group_key="video_id",
    )

    result = splitter.split(samples, seed=17)

    assert result.train and result.validation and result.test
    split_groups = [
        set(result.train_groups),
        set(result.validation_groups),
        set(result.test_groups),
    ]
    assert split_groups[0].isdisjoint(split_groups[1])
    assert split_groups[0].isdisjoint(split_groups[2])
    assert split_groups[1].isdisjoint(split_groups[2])

    for split_samples in (result.train, result.validation, result.test):
        behaviors = {sample.behavior_id for sample in split_samples}
        assert {"feeding", "walking"} <= behaviors

    # The class represented by only one video is retained in training.
    assert "dust_bathing" in {sample.behavior_id for sample in result.train}
    assert "dust_bathing" not in {
        sample.behavior_id
        for sample in result.validation + result.test
    }


def test_two_way_split_stratifies_behavior_distribution() -> None:
    samples = make_stratified_samples()

    class SampleDataset:
        def __init__(self, items):
            self.samples = list(items)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    dataset = SampleDataset(samples)
    train, validation = group_train_validation_split(
        dataset,
        validation_fraction=0.25,
        random_seed=29,
        group_by="video",
    )
    train_samples = [dataset.samples[index] for index in train.indices]
    validation_samples = [dataset.samples[index] for index in validation.indices]

    assert {sample.behavior_id for sample in train_samples} >= {"feeding", "walking"}
    assert {sample.behavior_id for sample in validation_samples} >= {"feeding", "walking"}

    train_groups = {sample.metadata["video_id"] for sample in train_samples}
    validation_groups = {sample.metadata["video_id"] for sample in validation_samples}
    assert train_groups.isdisjoint(validation_groups)


def test_stratified_split_is_reproducible_for_same_seed() -> None:
    samples = make_stratified_samples()
    splitter = GroupAwareSplitter(
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        group_key="video_id",
    )

    first = splitter.split(samples, seed=123)
    second = splitter.split(samples, seed=123)

    assert first.train_groups == second.train_groups
    assert first.validation_groups == second.validation_groups
    assert first.test_groups == second.test_groups
