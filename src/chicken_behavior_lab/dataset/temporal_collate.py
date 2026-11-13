from __future__ import annotations

from collections.abc import Sequence

import torch
from torch_geometric.data import Batch, Data

from chicken_behavior_lab.core.identifiers import normalize_track_id
from chicken_behavior_lab.dataset.temporal_batch import TemporalBatch


class TemporalCollator:
    """Pad variable-length annotation graphs and preserve all validity masks."""

    def __call__(self, data_list: Sequence) -> TemporalBatch:
        if not data_list:
            raise ValueError("Cannot collate an empty batch.")

        first = data_list[0]
        if first.x.ndim != 3:
            raise ValueError("Each sample x must have shape (T, N, F).")
        n, f = first.x.shape[1:]
        edge_index = first.edge_index
        e = edge_index.shape[1]
        has_edge_attr = getattr(first, "edge_attr", None) is not None
        edge_dim = first.edge_attr.shape[-1] if has_edge_attr else 0

        max_t = max(int(data.x.shape[0]) for data in data_list)
        batch_size = len(data_list)
        x = first.x.new_zeros((batch_size, max_t, n, f))
        frame_mask = torch.zeros((batch_size, max_t), dtype=torch.bool)
        node_mask = torch.zeros((batch_size, max_t, n), dtype=torch.bool)
        edge_mask = torch.zeros((batch_size, max_t, e), dtype=torch.bool)
        edge_attr = (
            first.x.new_zeros((batch_size, max_t, e, edge_dim))
            if has_edge_attr else None
        )

        for index, data in enumerate(data_list):
            if data.x.ndim != 3 or data.x.shape[1:] != (n, f):
                raise ValueError(f"Incompatible node feature shape at sample {index}.")
            if not torch.equal(data.edge_index, edge_index):
                raise ValueError(f"Graph topology mismatch at sample {index}.")
            if (getattr(data, "edge_attr", None) is not None) != has_edge_attr:
                raise ValueError("Either all samples must contain edge_attr or none may contain it.")
            if data.edge_mask.shape != (data.x.shape[0], e):
                raise ValueError(f"edge_mask shape mismatch at sample {index}.")
            length = data.x.shape[0]
            x[index, :length] = data.x
            frame_mask[index, :length] = data.frame_mask
            node_mask[index, :length] = data.node_mask
            edge_mask[index, :length] = data.edge_mask
            if edge_attr is not None:
                if data.edge_attr.shape != (length, e, edge_dim):
                    raise ValueError(f"edge_attr shape mismatch at sample {index}.")
                edge_attr[index, :length] = data.edge_attr

        y = torch.stack([data.y.view(-1)[0] for data in data_list]).long()
        frame_batches: list[Batch] = []
        for t in range(max_t):
            graphs = []
            for i, data in enumerate(data_list):
                length = data.x.shape[0]
                valid_nodes = node_mask[i, t] if t < length else node_mask.new_zeros(n)
                valid_edges = edge_mask[i, t] if t < length else edge_mask.new_zeros(e)
                local_edges = edge_index[:, valid_edges]
                graph = Data(x=x[i, t], edge_index=local_edges)
                if edge_attr is not None:
                    graph.edge_attr = edge_attr[i, t, valid_edges]
                graphs.append(graph)
            frame_batches.append(Batch.from_data_list(graphs))

        track_ids = [
            None if getattr(data, "track_id", None) is None
            else normalize_track_id(data.track_id)
            for data in data_list
        ]
        batch = TemporalBatch(
            x=x,
            edge_index=edge_index.clone().long(),
            edge_attr=edge_attr,
            y=y,
            sample_id=[str(data.sample_id) for data in data_list],
            video_id=[str(data.video_id) for data in data_list],
            track_id=track_ids,
            start_frame=torch.tensor([int(data.start_frame) for data in data_list], dtype=torch.long),
            end_frame=torch.tensor([int(data.end_frame) for data in data_list], dtype=torch.long),
            behavior_id=[str(data.behavior_id) for data in data_list],
            frame_batches=frame_batches,
            frame_mask=frame_mask,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )
        batch.validate()
        return batch


def make_temporal_dataloader(dataset, batch_size: int, shuffle: bool = False, num_workers: int = 0):
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
