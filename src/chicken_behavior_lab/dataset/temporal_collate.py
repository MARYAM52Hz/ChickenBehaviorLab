from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from chicken_behavior_lab.dataset.temporal_batch import (
    TemporalBatch,
)


class TemporalCollator:
    """
    Collate temporal graph samples into TemporalBatch.

    Important:
        Graph topology can differ between frames, so edge_index
        and edge_attr are kept frame-aware rather than flattened
        into a single static graph.
    """

    def __call__(
        self,
        items: Sequence[dict],
    ) -> TemporalBatch:

        if not items:
            raise ValueError(
                "TemporalCollator received an empty batch."
            )

        sequence_lengths = {
            len(item["graphs"])
            for item in items
        }

        if len(sequence_lengths) != 1:
            raise ValueError(
                "All temporal sequences in a batch must "
                "have the same sequence length."
            )

        sequence_length = (
            sequence_lengths.pop()
        )

        batch_size = len(items)

        frame_batches: list[
            list[Batch]
        ] = []

        edge_indices: list[
            torch.Tensor
        ] = []

        edge_attrs: list[
            torch.Tensor | None
        ] = []

        x_frames: list[
            torch.Tensor
        ] = []

        for t in range(
            sequence_length
        ):

            graphs_at_t = [
                item["graphs"][t]
                for item in items
            ]

            pyg_batch = Batch.from_data_list(
                graphs_at_t
            )

            frame_batches.append(
                [
                    pyg_batch
                ]
            )

            x_frames.append(
                self._stack_node_features(
                    graphs_at_t
                )
            )

            edge_indices.append(
                self._extract_common_edge_index(
                    graphs_at_t
                )
            )

            edge_attrs.append(
                self._extract_common_edge_attr(
                    graphs_at_t
                )
            )

        x = torch.stack(
            x_frames,
            dim=1,
        )

        y = torch.stack(
            [
                item["y"].reshape(())
                for item in items
            ]
        ).long()

        track_id = torch.tensor(
            [
                item["track_id"]
                for item in items
            ],
            dtype=torch.long,
        )

        start_frame = torch.tensor(
            [
                item["start_frame"]
                for item in items
            ],
            dtype=torch.long,
        )

        end_frame = torch.tensor(
            [
                item["end_frame"]
                for item in items
            ],
            dtype=torch.long,
        )

        return TemporalBatch(
            x=x,
            edge_index=edge_indices,
            edge_attr=edge_attrs,
            y=y,
            sample_id=[
                item["sample_id"]
                for item in items
            ],
            video_id=[
                item["video_id"]
                for item in items
            ],
            track_id=track_id,
            start_frame=start_frame,
            end_frame=end_frame,
            behavior_id=[
                item["behavior_id"]
                for item in items
            ],
            frame_batches=frame_batches,
        )

    @staticmethod
    def _stack_node_features(
        graphs,
    ) -> torch.Tensor:
        """
        Stack node features from graphs belonging to the
        same temporal frame.

        Expected:
            each graph has [N, F]

        Result:
            [B, N, F]
        """

        node_counts = {
            graph.x.shape[0]
            for graph in graphs
        }

        if len(node_counts) != 1:
            raise ValueError(
                "All graphs in a temporal batch must "
                "have the same number of nodes."
            )

        return torch.stack(
            [
                graph.x
                for graph in graphs
            ],
            dim=0,
        )

    @staticmethod
    def _extract_common_edge_index(
        graphs,
    ) -> torch.Tensor:
        """
        Extract edge_index when graph topology is shared
        across the batch.
        """

        first = graphs[0].edge_index

        for graph in graphs[1:]:
            if not torch.equal(
                first,
                graph.edge_index,
            ):
                raise ValueError(
                    "TemporalCollator currently requires "
                    "the same edge_index for all samples "
                    "within a batch."
                )

        return first

    @staticmethod
    def _extract_common_edge_attr(
        graphs,
    ) -> torch.Tensor | None:
        """
        Extract edge attributes.

        If no graph has edge attributes, return None.

        If some graphs have edge attributes and others do not,
        raise an error rather than silently dropping features.
        """

        attrs = [
            graph.edge_attr
            for graph in graphs
        ]

        has_attr = [
            attr is not None
            for attr in attrs
        ]

        if not any(has_attr):
            return None

        if not all(has_attr):
            raise ValueError(
                "Either all graphs in a temporal frame "
                "must contain edge_attr or none of them."
            )

        first = attrs[0]

        for attr in attrs[1:]:
            if attr.shape != first.shape:
                raise ValueError(
                    "Edge attribute dimensions differ "
                    "between samples in the batch."
                )

        return torch.stack(
            attrs,
            dim=0,
        )


def make_temporal_dataloader(
    dataset,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:

    collator = TemporalCollator()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collator,
    )
