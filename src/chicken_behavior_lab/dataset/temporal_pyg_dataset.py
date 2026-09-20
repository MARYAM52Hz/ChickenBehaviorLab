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
    Adapter between TemporalGraphSample objects and PyTorch
    Geometric Data objects.

    Expected temporal tensor contract:

        node features:
            [T, N, F_node]

        edge index:
            [2, E]

        edge features:
            [T, E, F_edge] or None

        target:
            [1]

    T = number of temporal graph windows
    N = number of skeleton nodes
    E = number of graph edges
    """

    def __init__(
        self,
        samples: Sequence[TemporalGraphSample],
    ) -> None:
        self.samples = list(samples)
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
                    f"Duplicate sample_id: "
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
            behavior_id = sample.behavior_id
            label = int(sample.label)

            if behavior_id in mapping:
                if mapping[behavior_id] != label:
                    raise ValueError(
                        "Inconsistent label mapping for "
                        f"behavior '{behavior_id}': "
                        f"expected {mapping[behavior_id]}, "
                        f"found {label}."
                    )
            else:
                mapping[behavior_id] = label

        return dict(
            sorted(
                mapping.items(),
                key=lambda item: item[1],
            )
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> Data:
        sample = self.samples[index]

        sample.validate()

        if not sample.graphs:
            raise ValueError(
                f"Temporal sample '{sample.sample_id}' "
                "contains no graphs."
            )

        node_features = []
        edge_features = []

        edge_index = None
        has_edge_features = True

        for time_index, graph_sample in enumerate(
            sample.graphs
        ):
            graph = graph_sample.graph

            graph.validate()

            current_node_features = (
                self._to_node_tensor(
                    graph.node_features,
                    sample.sample_id,
                    time_index,
                )
            )

            node_features.append(
                current_node_features
            )

            current_edge_index = (
                self._to_edge_index_tensor(
                    graph.edge_index,
                    sample.sample_id,
                    time_index,
                )
            )

            if edge_index is None:
                edge_index = current_edge_index
            elif not torch.equal(
                edge_index,
                current_edge_index,
            ):
                raise ValueError(
                    f"Temporal sample "
                    f"'{sample.sample_id}' has "
                    "different edge_index structures "
                    "across time."
                )

            current_edge_features = getattr(
                graph,
                "edge_features",
                None,
            )

            if current_edge_features is None:
                has_edge_features = False
            else:
                edge_features.append(
                    self._to_edge_feature_tensor(
                        current_edge_features,
                        sample.sample_id,
                        time_index,
                    )
                )

        x = torch.stack(
            node_features,
            dim=0,
        )

        if edge_index is None:
            raise ValueError(
                f"Temporal sample "
                f"'{sample.sample_id}' has no edge_index."
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
            if len(edge_features) != len(
                node_features
            ):
                raise ValueError(
                    f"Temporal sample "
                    f"'{sample.sample_id}' has "
                    "inconsistent edge feature availability."
                )

            data.edge_attr = torch.stack(
                edge_features,
                dim=0,
            )

        self._attach_metadata(
            data,
            sample,
        )

        self._validate_data_contract(
            data,
            sample,
        )

        return data

    @staticmethod
    def _to_node_tensor(
        values: Any,
        sample_id: str,
        time_index: int,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(
            values,
            dtype=torch.float32,
        )

        if tensor.ndim != 2:
            raise ValueError(
                f"Sample '{sample_id}', time index "
                f"{time_index}: node_features must "
                f"have shape [N, F], got "
                f"{tuple(tensor.shape)}."
            )

        return tensor

    @staticmethod
    def _to_edge_index_tensor(
        values: Any,
        sample_id: str,
        time_index: int,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(
            values,
            dtype=torch.long,
        )

        if tensor.ndim != 2:
            raise ValueError(
                f"Sample '{sample_id}', time index "
                f"{time_index}: edge_index must "
                f"have shape [2, E], got "
                f"{tuple(tensor.shape)}."
            )

        if tensor.shape[0] != 2:
            raise ValueError(
                f"Sample '{sample_id}', time index "
                f"{time_index}: edge_index must "
                f"have shape [2, E], got "
                f"{tuple(tensor.shape)}."
            )

        return tensor

    @staticmethod
    def _to_edge_feature_tensor(
        values: Any,
        sample_id: str,
        time_index: int,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(
            values,
            dtype=torch.float32,
        )

        if tensor.ndim != 2:
            raise ValueError(
                f"Sample '{sample_id}', time index "
                f"{time_index}: edge_features must "
                f"have shape [E, F], got "
                f"{tuple(tensor.shape)}."
            )

        return tensor

    @staticmethod
    def _attach_metadata(
        data: Data,
        sample: TemporalGraphSample,
    ) -> None:
        data.sample_id = sample.sample_id
        data.behavior_id = sample.behavior_id

        video_id = sample.video_id
        track_id = sample.track_id

        data.video_id = (
            str(video_id)
            if video_id is not None
            else ""
        )

        data.track_id = (
            int(track_id)
            if track_id is not None
            else -1
        )

        data.start_frame = (
            int(sample.first_frame)
            if sample.first_frame is not None
            else 0
        )

        data.end_frame = (
            int(sample.last_frame)
            if sample.last_frame is not None
            else 0
        )

        data.sequence_length = len(
            sample
        )

    @staticmethod
    def _validate_data_contract(
        data: Data,
        sample: TemporalGraphSample,
    ) -> None:
        if data.x.ndim != 3:
            raise ValueError(
                "Temporal Data.x must have shape "
                "[T, N, F_node]."
            )

        if data.x.shape[0] != len(
            sample.graphs
        ):
            raise ValueError(
                "Temporal dimension of x does not "
                "match sequence length."
            )

        if data.edge_index.ndim != 2:
            raise ValueError(
                "edge_index must have shape [2, E]."
            )

        if data.edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, E]."
            )

        if hasattr(
            data,
            "edge_attr",
        ):
            if data.edge_attr.ndim != 3:
                raise ValueError(
                    "Temporal edge_attr must have shape "
                    "[T, E, F_edge]."
                )

            if (
                data.edge_attr.shape[0]
                != data.x.shape[0]
            ):
                raise ValueError(
                    "Temporal dimension of edge_attr "
                    "must match x."
                )

            if (
                data.edge_attr.shape[1]
                != data.edge_index.shape[1]
            ):
                raise ValueError(
                    "Number of edges in edge_attr "
                    "must match edge_index."
                )

    @property
    def samples_metadata(
        self,
    ) -> list[dict[str, Any]]:
        result = []

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
            metadata["sequence_length"] = len(
                sample
            )

            result.append(metadata)

        return result
