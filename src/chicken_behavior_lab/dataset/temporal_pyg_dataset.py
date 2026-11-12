from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from chicken_behavior_lab.dataset.sample import GraphSample


class TemporalPyGDataset:
    """Convert one annotation-level GraphSample into one temporal PyG Data.

    Each input sample owns a TemporalSkeletonGraph with x shaped [T, V, F].
    No second temporal windowing stage is performed here.
    """

    def __init__(
        self,
        samples: Sequence[GraphSample],
        label_to_index: dict[str, int] | None = None,
    ) -> None:
        self.samples = list(samples)
        self._validate()

        if label_to_index is None:
            self.label_to_index = self._build_label_mapping()
        else:
            self.label_to_index = dict(label_to_index)
            self._validate_label_mapping()

    def _validate(self) -> None:
        sample_ids: set[str] = set()
        for sample in self.samples:
            if not isinstance(sample, GraphSample):
                raise TypeError("Every item must be a GraphSample.")
            sample.validate()
            if sample.sample_id in sample_ids:
                raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
            sample_ids.add(sample.sample_id)

    def _build_label_mapping(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for sample in self.samples:
            previous = mapping.setdefault(sample.behavior_id, sample.label)
            if previous != sample.label:
                raise ValueError(
                    f"Inconsistent label mapping for behavior '{sample.behavior_id}'."
                )
        return dict(sorted(mapping.items(), key=lambda item: item[1]))

    def _validate_label_mapping(self) -> None:
        if not self.label_to_index:
            if self.samples:
                raise ValueError("label_to_index cannot be empty.")
            return
        values = list(self.label_to_index.values())
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("label_to_index values must be integers.")
        if any(value < 0 for value in values) or len(values) != len(set(values)):
            raise ValueError("label_to_index values must be unique non-negative integers.")
        for sample in self.samples:
            expected = self.label_to_index.get(sample.behavior_id)
            if expected is None:
                raise ValueError(f"Behavior '{sample.behavior_id}' is missing from label_to_index.")
            if expected != sample.label:
                raise ValueError(
                    f"Inconsistent label mapping for behavior '{sample.behavior_id}': "
                    f"expected {expected}, found {sample.label}."
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Data:
        sample = self.samples[index]
        sample.validate()
        graph = sample.graph

        if not hasattr(graph, "temporal_length"):
            raise TypeError(
                "GraphSample.graph must be a TemporalSkeletonGraph with shape (T, V, F)."
            )

        x_array = np.asarray(graph.node_features, dtype=np.float32)
        frame_mask = np.asarray(graph.frame_mask, dtype=bool)
        node_mask = np.asarray(graph.node_mask, dtype=bool)
        edge_index_array = np.asarray(graph.edge_index, dtype=np.int64)

        if x_array.ndim != 3:
            raise ValueError("Temporal graph node_features must have shape (T, V, F).")
        temporal_length, num_nodes, _ = x_array.shape
        if frame_mask.shape != (temporal_length,):
            raise ValueError("frame_mask must have shape (T,).")
        if node_mask.shape != (temporal_length, num_nodes):
            raise ValueError("node_mask must have shape (T, V).")
        if not frame_mask.any():
            raise ValueError(f"Sample '{sample.sample_id}' contains no valid frames.")

        finite_nodes = np.all(np.isfinite(x_array), axis=-1)
        node_mask = node_mask & finite_nodes & frame_mask[:, None]
        if not node_mask.any():
            raise ValueError(f"Sample '{sample.sample_id}' contains no valid nodes.")

        if edge_index_array.ndim != 2 or edge_index_array.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E).")
        if edge_index_array.size and (
            edge_index_array.min() < 0 or edge_index_array.max() >= num_nodes
        ):
            raise ValueError("edge_index contains a node index outside [0, V).")

        x_array = np.nan_to_num(x_array, nan=0.0, posinf=0.0, neginf=0.0)
        edge_features = graph.edge_features
        edge_attr_tensor = None
        edge_mask = np.zeros((temporal_length, edge_index_array.shape[1]), dtype=bool)

        if edge_features is not None:
            edge_features = np.asarray(edge_features, dtype=np.float32)
            if edge_features.ndim != 3 or edge_features.shape[:2] != (
                temporal_length, edge_index_array.shape[1]
            ):
                raise ValueError("edge_features must have shape (T, E, D).")
            if not np.isfinite(edge_features).all():
                edge_features = np.nan_to_num(
                    edge_features, nan=0.0, posinf=0.0, neginf=0.0
                )
            edge_attr_tensor = torch.as_tensor(edge_features, dtype=torch.float32)

        if edge_index_array.shape[1]:
            source, target = edge_index_array
            edge_mask = node_mask[:, source] & node_mask[:, target]

        metadata = dict(sample.metadata or {})
        video_id = metadata.get("video_id")
        track_id = metadata.get("track_id")
        start_frame = metadata.get("start_frame")
        end_frame = metadata.get("end_frame")
        if video_id is None or not str(video_id).strip():
            raise ValueError(f"Sample '{sample.sample_id}' requires metadata.video_id.")
        if start_frame is None or end_frame is None:
            raise ValueError(
                f"Sample '{sample.sample_id}' requires start_frame and end_frame metadata."
            )

        expected_length = int(end_frame) - int(start_frame) + 1
        if expected_length != temporal_length:
            raise ValueError(
                f"Sample '{sample.sample_id}' spans {expected_length} inclusive frames, "
                f"but its temporal graph contains {temporal_length} steps."
            )

        data = Data(
            x=torch.as_tensor(x_array, dtype=torch.float32),
            edge_index=torch.as_tensor(edge_index_array, dtype=torch.long),
            y=torch.tensor([sample.label], dtype=torch.long),
            frame_mask=torch.as_tensor(frame_mask, dtype=torch.bool),
            node_mask=torch.as_tensor(node_mask, dtype=torch.bool),
            edge_mask=torch.as_tensor(edge_mask, dtype=torch.bool),
        )
        if edge_attr_tensor is not None:
            data.edge_attr = edge_attr_tensor

        data.sample_id = sample.sample_id
        data.video_id = str(video_id).strip()
        data.track_id = str(track_id) if track_id is not None else None
        data.start_frame = int(start_frame)
        data.end_frame = int(end_frame)
        data.behavior_id = sample.behavior_id
        data.sequence_length = temporal_length
        return data

    @property
    def samples_metadata(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for sample in self.samples:
            metadata = dict(sample.metadata or {})
            metadata.update(
                sample_id=sample.sample_id,
                behavior_id=sample.behavior_id,
                label=int(sample.label),
            )
            result.append(metadata)
        return result
