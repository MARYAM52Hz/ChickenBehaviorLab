from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.dataset.temporal_sample import TemporalGraphSample


class TemporalPyGDataset:
    """Convert one annotation-level GraphSample into one temporal PyG Data.

    Each input sample owns a TemporalSkeletonGraph with x shaped [T, V, F].
    No second temporal windowing stage is performed here.
    """

    def __init__(
        self,
        samples: Sequence[GraphSample | TemporalGraphSample],
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
        sample_type = type(self.samples[0]) if self.samples else None
        if sample_type not in {None, GraphSample, TemporalGraphSample}:
            raise TypeError(
                "Every item must be a GraphSample or TemporalGraphSample."
            )

        expected_keypoint_types = None
        if sample_type is GraphSample:
            expected_keypoint_types = getattr(
                self.samples[0].graph, "keypoint_types", None
            )

        for sample in self.samples:
            if not isinstance(sample, (GraphSample, TemporalGraphSample)):
                raise TypeError(
                    "Every item must be a GraphSample or TemporalGraphSample."
                )
            if type(sample) is not sample_type:
                raise TypeError(
                    "A TemporalPyGDataset cannot mix GraphSample and "
                    "TemporalGraphSample items."
                )
            sample.validate()

            if isinstance(sample, GraphSample):
                keypoint_types = getattr(sample.graph, "keypoint_types", None)
                if keypoint_types != expected_keypoint_types:
                    raise ValueError(
                        "All temporal graphs in a dataset must use the same keypoint order."
                    )

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
        if isinstance(sample, TemporalGraphSample):
            return self._convert_legacy_temporal_sample(sample)

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

    @staticmethod
    def _convert_legacy_temporal_sample(sample: TemporalGraphSample) -> Data:
        """Adapt the legacy frame-graph sequence API to the current temporal Data contract."""
        graphs = [item.graph for item in sample.graphs]
        first_graph = graphs[0]
        reference_node_shape = first_graph.node_features.shape
        reference_edge_index = np.asarray(first_graph.edge_index, dtype=np.int64)
        has_edge_features = first_graph.edge_features is not None
        reference_edge_shape = (
            first_graph.edge_features.shape if has_edge_features else None
        )

        node_frames: list[np.ndarray] = []
        node_masks: list[np.ndarray] = []
        edge_frames: list[np.ndarray] = []

        for frame_index, graph in enumerate(graphs):
            graph.validate()
            if graph.node_features.shape != reference_node_shape:
                raise ValueError(
                    "All graphs in a legacy temporal sample must have "
                    f"the same node feature shape (frame {frame_index})."
                )
            current_edges = np.asarray(graph.edge_index, dtype=np.int64)
            if not np.array_equal(current_edges, reference_edge_index):
                raise ValueError(
                    "All graphs in a legacy temporal sample must have "
                    f"identical edge topology (frame {frame_index})."
                )

            current_has_edge_features = graph.edge_features is not None
            if current_has_edge_features != has_edge_features:
                raise ValueError(
                    "Either all legacy frame graphs must contain edge features or none may."
                )
            if has_edge_features and graph.edge_features.shape != reference_edge_shape:
                raise ValueError(
                    "All legacy frame graphs must have the same edge feature shape."
                )

            features = np.asarray(graph.node_features, dtype=np.float32)
            if features.ndim != 2:
                raise ValueError(
                    "Legacy TemporalGraphSample must contain one SkeletonGraph "
                    f"with node_features shaped (V, F) per frame; got {features.shape} "
                    f"at frame {frame_index}."
                )
            finite_nodes = np.all(np.isfinite(features), axis=-1)
            valid_nodes = (
                np.asarray(graph.node_mask, dtype=bool)
                if graph.node_mask is not None
                else np.ones(features.shape[0], dtype=bool)
            )
            valid_nodes = valid_nodes & finite_nodes
            node_frames.append(np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0))
            node_masks.append(valid_nodes)

            if has_edge_features:
                edge_frames.append(
                    np.nan_to_num(
                        np.asarray(graph.edge_features, dtype=np.float32),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                )

        x_array = np.stack(node_frames, axis=0)
        node_mask = np.stack(node_masks, axis=0)
        frame_mask = node_mask.any(axis=1)
        if not frame_mask.any():
            raise ValueError(
                f"Legacy temporal sample '{sample.sample_id}' contains no valid nodes."
            )

        if reference_edge_index.ndim != 2 or reference_edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E).")
        if reference_edge_index.size and (
            reference_edge_index.min() < 0
            or reference_edge_index.max() >= x_array.shape[1]
        ):
            raise ValueError("edge_index contains a node index outside [0, V).")

        if reference_edge_index.shape[1]:
            source, target = reference_edge_index
            edge_mask = node_mask[:, source] & node_mask[:, target]
        else:
            edge_mask = np.zeros((len(graphs), 0), dtype=bool)

        edge_attr = (
            torch.as_tensor(np.stack(edge_frames, axis=0), dtype=torch.float32)
            if has_edge_features
            else None
        )
        video_id = sample.video_id
        if video_id is None or not str(video_id).strip():
            raise ValueError(
                f"Legacy temporal sample '{sample.sample_id}' requires video_id."
            )
        track_id = sample.track_id
        start_frame = sample.first_frame
        end_frame = sample.last_frame
        if end_frame - start_frame + 1 != len(graphs):
            raise ValueError(
                "Legacy TemporalGraphSample metadata interval must match "
                "the number of frame graphs."
            )

        data = Data(
            x=torch.as_tensor(x_array, dtype=torch.float32),
            edge_index=torch.as_tensor(reference_edge_index, dtype=torch.long),
            y=torch.tensor([sample.label], dtype=torch.long),
            frame_mask=torch.as_tensor(frame_mask, dtype=torch.bool),
            node_mask=torch.as_tensor(node_mask, dtype=torch.bool),
            edge_mask=torch.as_tensor(edge_mask, dtype=torch.bool),
        )
        if edge_attr is not None:
            data.edge_attr = edge_attr

        data.sample_id = sample.sample_id
        data.video_id = str(video_id).strip()
        data.track_id = track_id
        data.start_frame = int(start_frame)
        data.end_frame = int(end_frame)
        data.behavior_id = sample.behavior_id
        data.sequence_length = len(graphs)
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
