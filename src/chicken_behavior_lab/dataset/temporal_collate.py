from __future__ import annotations

from collections.abc import Sequence

import torch
from torch_geometric.data import Batch, Data

from chicken_behavior_lab.core.identifiers import normalize_track_id
from chicken_behavior_lab.dataset.temporal_batch import TemporalBatch


class TemporalCollator:
    """Collate temporal PyG samples and create one batched graph per time step."""

    def __call__(self, data_list: Sequence) -> TemporalBatch:
        if not data_list:
            raise ValueError("Cannot collate an empty batch.")

        x_list = [data.x for data in data_list]
        edge_indices = [data.edge_index for data in data_list]
        reference_x = x_list[0]
        reference_edge_index = edge_indices[0]

        if reference_x.ndim != 3:
            raise ValueError("Each sample x must have shape (T, N, F).")
        if reference_edge_index.ndim != 2 or reference_edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E).")

        for index, x in enumerate(x_list):
            if x.shape != reference_x.shape:
                raise ValueError(
                    "All temporal samples in a batch must have identical x shapes. "
                    f"Sample 0: {tuple(reference_x.shape)}, sample {index}: {tuple(x.shape)}."
                )
        for index, edge_index in enumerate(edge_indices):
            if not torch.equal(edge_index, reference_edge_index):
                raise ValueError(
                    "All temporal samples in a batch must have identical graph topology. "
                    f"Mismatch at sample {index}."
                )

        edge_attr_list = [getattr(data, "edge_attr", None) for data in data_list]
        has_edge_attr = [value is not None for value in edge_attr_list]
        if any(has_edge_attr) and not all(has_edge_attr):
            raise ValueError("Either all temporal samples must contain edge_attr or none may contain it.")

        if all(has_edge_attr):
            expected_shape = edge_attr_list[0].shape
            if len(expected_shape) != 3 or expected_shape[:2] != (
                reference_x.shape[0], reference_edge_index.shape[1]
            ):
                raise ValueError("Each sample edge_attr must have shape (T, E, D).")
            for index, edge_attr in enumerate(edge_attr_list):
                if edge_attr.shape != expected_shape:
                    raise ValueError(f"edge_attr shape mismatch at sample {index}.")
            edge_attr = torch.stack(edge_attr_list, dim=0)
        else:
            edge_attr = None

        x = torch.stack(x_list, dim=0)
        y = torch.stack([data.y.view(-1)[0] for data in data_list], dim=0).long()

        frame_batches: list[Batch] = []
        for time_index in range(reference_x.shape[0]):
            frame_graphs = []
            for sample_index, data in enumerate(data_list):
                graph = Data(
                    x=data.x[time_index],
                    edge_index=data.edge_index,
                )
                if edge_attr is not None:
                    graph.edge_attr = edge_attr[sample_index, time_index]
                frame_graphs.append(graph)
            frame_batches.append(Batch.from_data_list(frame_graphs))

        track_ids = [
            None if getattr(data, "track_id", None) is None
            else normalize_track_id(data.track_id)
            for data in data_list
        ]
        batch = TemporalBatch(
            x=x,
            edge_index=reference_edge_index.clone().long(),
            edge_attr=edge_attr,
            y=y,
            sample_id=[str(data.sample_id) for data in data_list],
            video_id=[str(data.video_id) for data in data_list],
            track_id=track_ids,
            start_frame=torch.tensor([int(data.start_frame) for data in data_list], dtype=torch.long),
            end_frame=torch.tensor([int(data.end_frame) for data in data_list], dtype=torch.long),
            behavior_id=[str(data.behavior_id) for data in data_list],
            frame_batches=frame_batches,
        )
        batch.validate()
        return batch


def make_temporal_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
):
    if isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError("num_workers must be >= 0.")

    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=TemporalCollator(),
    )
