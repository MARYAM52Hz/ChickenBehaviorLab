import torch
from torch_geometric.data import Batch, Data


def test_pyg_batch_preserves_sample_metadata():
    data_1 = Data(
        x=torch.tensor(
            [[1.0, 2.0]]
        ),
        edge_index=torch.empty(
            (2, 0),
            dtype=torch.long,
        ),
        y=torch.tensor([0]),
    )

    data_1.sample_id = "sample_001"
    data_1.video_id = "video_001"
    data_1.track_id = 1
    data_1.start_frame = 100
    data_1.end_frame = 150

    data_2 = Data(
        x=torch.tensor(
            [[3.0, 4.0]]
        ),
        edge_index=torch.empty(
            (2, 0),
            dtype=torch.long,
        ),
        y=torch.tensor([1]),
    )

    data_2.sample_id = "sample_002"
    data_2.video_id = "video_002"
    data_2.track_id = 2
    data_2.start_frame = 200
    data_2.end_frame = 250

    batch = Batch.from_data_list(
        [data_1, data_2]
    )

    assert batch.sample_id == [
        "sample_001",
        "sample_002",
    ]

    assert batch.video_id == [
        "video_001",
        "video_002",
    ]

    assert batch.track_id.tolist() == [
        1,
        2,
    ]

    assert batch.start_frame.tolist() == [
        100,
        200,
    ]

    assert batch.end_frame.tolist() == [
        150,
        250,
    ]
