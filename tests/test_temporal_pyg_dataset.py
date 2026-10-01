def test_validation_dataset_uses_training_mapping() -> None:
    samples = [
        make_sample(frame)
        for frame in range(4)
    ]

    builder = TemporalSequenceBuilder(
        sequence_length=2,
        sequence_stride=2,
    )

    temporal_dataset = builder.build(
        samples
    )

    train_dataset = TemporalPyGDataset(
        temporal_dataset.samples
    )

    validation_dataset = TemporalPyGDataset(
        temporal_dataset.samples,
        label_to_index=train_dataset.label_to_index,
    )

    assert (
        validation_dataset.label_to_index
        == train_dataset.label_to_index
    )


def test_dataset_rejects_inconsistent_external_mapping() -> None:
    samples = [
        make_sample(frame)
        for frame in range(4)
    ]

    builder = TemporalSequenceBuilder(
        sequence_length=4,
    )

    temporal_dataset = builder.build(
        samples
    )

    with pytest.raises(ValueError):
        TemporalPyGDataset(
            temporal_dataset.samples,
            label_to_index={
                "feeding": 5,
            },
        )
