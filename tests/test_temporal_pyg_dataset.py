def test_temporal_topology_must_be_constant():
    first = make_graph_sample(
        "graph_001",
        100,
        109,
        0.0,
    )

    second = make_graph_sample(
        "graph_002",
        110,
        119,
        0.5,
    )

    second.graph.edge_index = np.array(
        [
            [0, 1],
            [1, 2],
        ],
        dtype=np.int64,
    )

    temporal_sample = TemporalGraphSample(
        graphs=[
            first,
            second,
        ],
        label=0,
        behavior_id="feeding",
        sample_id="sequence_001",
        metadata={
            "video_id": "video_001",
            "track_id": 1,
        },
    )

    dataset = TemporalPyGDataset(
        [temporal_sample]
    )

    try:
        dataset[0]
    except ValueError as error:
        assert (
            "different edge_index structures"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected topology validation to fail."
        )
