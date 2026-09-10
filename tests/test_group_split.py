import torch

from chicken_behavior_lab.dataset import (
    GraphSample,
    group_train_validation_test_split,
)


def make_sample(
    sample_id: str,
    video_id: str,
    track_id: str,
    label: str,
) -> GraphSample:

    return GraphSample(
        sample_id=sample_id,
        node_features=torch.randn(
            6,
            7,
        ),
        edge_index=torch.tensor(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=torch.long,
        ),
        edge_features=torch.randn(
            2,
            8,
        ),
        label=label,
        metadata={
            "video_id": video_id,
            "track_id": track_id,
        },
    )


def test_video_group_split():

    samples = []

    for video_index in range(10):

        video_id = (
            f"video_{video_index:02d}"
        )

        for track_index in range(2):

            track_id = (
                f"track_{track_index:02d}"
            )

            for sample_index in range(5):

                samples.append(
                    make_sample(
                        sample_id=(
                            f"{video_id}_"
                            f"{track_id}_"
                            f"{sample_index}"
                        ),
                        video_id=video_id,
                        track_id=track_id,
                        label="walking",
                    )
                )

    split = (
        group_train_validation_test_split(
            samples,
            validation_fraction=0.2,
            test_fraction=0.2,
            group_by="video",
            random_seed=42,
        )
    )

    train_videos = {
        sample.video_id
        for sample in split.train
    }

    validation_videos = {
        sample.video_id
        for sample in split.validation
    }

    test_videos = {
        sample.video_id
        for sample in split.test
    }

    assert (
        train_videos
        .isdisjoint(
            validation_videos
        )
    )

    assert (
        train_videos
        .isdisjoint(
            test_videos
        )
    )

    assert (
        validation_videos
        .isdisjoint(
            test_videos
        )
    )


def test_all_samples_are_preserved():

    samples = []

    for video_index in range(6):

        video_id = (
            f"video_{video_index}"
        )

        for sample_index in range(4):

            samples.append(
                make_sample(
                    sample_id=(
                        f"{video_id}_"
                        f"sample_{sample_index}"
                    ),
                    video_id=video_id,
                    track_id="track_01",
                    label="walking",
                )
            )

    split = (
        group_train_validation_test_split(
            samples,
            validation_fraction=0.2,
            test_fraction=0.2,
            group_by="video",
            random_seed=42,
        )
    )

    total = (
        len(split.train)
        + len(split.validation)
        + len(split.test)
    )

    assert total == len(samples)
