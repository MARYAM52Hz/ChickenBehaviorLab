from __future__ import annotations

from collections.abc import Sequence

import torch

from chicken_behavior_lab.dataset.temporal_batch import (
    TemporalBatch,
)

from chicken_behavior_lab.dataset.temporal_pyg_dataset import (
    TemporalPyGDataset,
)


class TemporalCollator:
    """
    Collate temporal graph samples into an explicit
    TemporalBatch.

    All sequences in one batch must have the same:
        - sequence length
        - number of nodes
        - node feature dimension
        - graph topology
        - edge feature dimension
    """

    def __call__(
        self,
        data_list: Sequence,
    ) -> TemporalBatch:
        if not data_list:
            raise ValueError(
                "Cannot collate an empty batch."
            )

        x_list = [
            data.x
            for data in data_list
        ]

        edge_indices = [
            data.edge_index
            for data in data_list
        ]

        y_list = [
            data.y.view(-1)[0]
            for data in data_list
        ]

        reference_x = x_list[0]
        reference_edge_index = (
            edge_indices[0]
        )

        for index, x in enumerate(
            x_list
        ):
            if x.shape != reference_x.shape:
                raise ValueError(
                    "All temporal samples in a batch "
                    "must have identical x shapes. "
                    f"Sample 0: {tuple(reference_x.shape)}, "
                    f"sample {index}: {tuple(x.shape)}."
                )

        for index, edge_index in enumerate(
            edge_indices
        ):
            if not torch.equal(
                edge_index,
                reference_edge_index,
            ):
                raise ValueError(
                    "All temporal samples in a batch "
                    "must have identical graph topology. "
                    f"Mismatch at sample {index}."
                )

        edge_attr_list = [
            getattr(
                data,
                "edge_attr",
                None,
            )
            for data in data_list
        ]

        has_edge_attr = [
            value is not None
            for value in edge_attr_list
        ]

        if any(has_edge_attr) and not all(
            has_edge_attr
        ):
            raise ValueError(
                "Either all temporal samples must "
                "contain edge_attr or none may contain it."
            )

        x = torch.stack(
            x_list,
            dim=0,
        )

        edge_index = reference_edge_index

        y = torch.stack(
            y_list,
            dim=0,
        ).long()

        edge_attr = None

        if all(has_edge_attr):
            edge_attr = torch.stack(
                edge_attr_list,
                dim=0,
            )

        batch = TemporalBatch(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            sample_id=[
                str(data.sample_id)
                for data in data_list
            ],
            video_id=[
                str(data.video_id)
                for data in data_list
            ],
            track_id=torch.tensor(
                [
                    int(data.track_id)
                    for data in data_list
                ],
                dtype=torch.long,
            ),
            start_frame=torch.tensor(
                [
                    int(data.start_frame)
                    for data in data_list
                ],
                dtype=torch.long,
            ),
            end_frame=torch.tensor(
                [
                    int(data.end_frame)
                    for data in data_list
                ],
                dtype=torch.long,
            ),
            behavior_id=[
                str(data.behavior_id)
                for data in data_list
            ],
        )

        batch.validate()

        return batch


def make_temporal_dataloader(
    dataset: TemporalPyGDataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
):
    """
    Create a DataLoader using the explicit temporal collator.
    """

    if batch_size < 1:
        raise ValueError(
            "batch_size must be >= 1."
        )

    from torch.utils.data import DataLoader

    collator = TemporalCollator()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )
