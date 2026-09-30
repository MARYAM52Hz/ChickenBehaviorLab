from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch_geometric.data import Data

from chicken_behavior_lab.dataset.temporal_sample import (
    TemporalGraphSample,
)


class TemporalPyGDataset:
    """
    Convert TemporalGraphSample objects into PyTorch-compatible
    temporal graph Data objects.

    Output tensor shapes:

        x:
            [T, N, F_node]

        edge_index:
            [2, E]

        edge_attr:
            [T, E, F_edge]

        y:
            [1]
    """

    def __init__(
        self,
        samples: Sequence[TemporalGraphSample],
    ) -> None:

        self.samples = list(
            samples
        )

        self._validate()

        self.label_to_index = (
            self._build_label_mapping()
        )

    def _validate(self) -> None:
        sample_ids: set[str] = set()

        for sample in self.samples:
            if not isinstance(
                sample,
                TemporalGraphSample,
            ):
                raise TypeError(
                    "Every item must be a "
                    "TemporalGraphSample."
                )

            sample.validate()

            if sample.sample_id in sample_ids:
                raise ValueError(
                    "Duplicate sample_id: "
                    f"{sample.sample_id}"
                )

            sample_ids.add(
                sample.sample_id
            )

    def _build_label_mapping(
        self,
    ) -> dict[str, int]:

        mapping: dict[str, int] = {}

        for sample in self.samples:

            if sample.behavior_id in mapping:
                expected = mapping[
                    sample.behavior_id
                ]

                if expected != sample.label:
                    raise ValueError(
                        "Inconsistent label mapping for "
                        f"behavior '{sample.behavior_id}': "
                        f"expected {expected}, "
                        f"found {sample.label}."
                    )

            else:
                mapping[
                    sample.behavior_id
                ] = sample.label

        return dict(
            sorted(
                mapping.items(),
                key=lambda item: item[1],
            )
        )

    def __len__(self) -> int:
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ) -> Data:

        sample = self.samples[
            index
        ]

        sample.validate()

        graphs = sample.graphs

        first_graph = graphs[0].graph

        reference_node_shape = (
            first_graph.node_features.shape
        )

        reference_edge_index = (
            first_graph.edge_index
        )

        reference_edge_shape = (
            first_graph.edge_features.shape
            if first_graph.edge_features is not None
            else None
        )

        node_features: list[torch.Tensor] = []
        edge_features: list[torch.Tensor] = []

        has_edge_features = (
            first_graph.edge_features
            is not None
        )

        for time_index, graph_sample in enumerate(
            graphs
        ):

            graph = graph_sample.graph

            if (
                graph.node_features.shape
                != reference_node_shape
            ):
                raise ValueError(
                    "All graphs in a temporal sample must have "
                    "the same node feature shape. "
                    f"Mismatch at time index {time_index}."
                )

            if not torch.equal(
                torch.as_tensor(
                    graph.edge_index,
                    dtype=torch.long,
                ),
                torch.as_tensor(
                    reference_edge_index,
                    dtype=torch.long,
                ),
            ):
                raise ValueError(
                    "All graphs in a temporal sample must have "
                    "identical edge topology. "
                    f"Mismatch at time index {time_index}."
                )

            current_has_edge_features = (
                graph.edge_features is not None
            )

            if (
                current_has_edge_features
                != has_edge_features
            ):
                raise ValueError(
                    "Either all graphs in a temporal sample "
                    "must have edge features or none may have them."
                )

            if has_edge_features:
                if (
                    graph.edge_features.shape
                    != reference_edge_shape
                ):
                    raise ValueError(
                        "All graphs in a temporal sample must have "
                        "the same edge feature shape."
                    )

            node_features.append(
                torch.as_tensor(
                    graph.node_features,
                    dtype=torch.float32,
                )
            )

            if has_edge_features:
                edge_features.append(
                    torch.as_tensor(
                        graph.edge_features,
                        dtype=torch.float32,
                    )
                )

        x = torch.stack(
            node_features,
            dim=0,
        )

        edge_index = torch.as_tensor(
            reference_edge_index,
            dtype=torch.long,
        )

        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(
                [sample.label],
                dtype=torch.long,
            ),
        )

        if has_edge_features:
            data.edge_attr = torch.stack(
                edge_features,
                dim=0,
            )

        data.sample_id = (
            sample.sample_id
        )

        data.video_id = (
            sample.video_id or ""
        )

        data.track_id = (
            -1
            if sample.track_id is None
            else int(sample.track_id)
        )

        data.start_frame = int(
            sample.first_frame
        )

        data.end_frame = int(
            sample.last_frame
        )

        data.behavior_id = (
            sample.behavior_id
        )

        data.sequence_length = (
            len(sample)
        )

        return data

    @property
    def samples_metadata(
        self,
    ) -> list[dict[str, Any]]:

        metadata_list: list[
            dict[str, Any]
        ] = []

        for sample in self.samples:

            metadata = dict(
                sample.metadata or {}
            )

            metadata["sample_id"] = (
                sample.sample_id
            )

            metadata["behavior_id"] = (
                sample.behavior_id
            )

            metadata["label"] = int(
                sample.label
            )

            metadata_list.append(
                metadata
            )

        return metadata_list
