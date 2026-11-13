from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chicken_behavior_lab.core.identifiers import normalize_track_id
from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.graph.graph import TemporalSkeletonGraph


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    """One sample entry in a feature-manifest JSON file."""

    sample_id: str
    track_id: str
    video_id: str
    feature_file: Path
    label: str

    def __post_init__(self) -> None:
        if not self.sample_id or not self.video_id or not self.label:
            raise ValueError("sample_id, video_id, and label must be non-empty.")
        object.__setattr__(self, "track_id", normalize_track_id(self.track_id))


class DatasetFactory:
    """Build canonical GraphSamples from a manifest and NumPy feature files.

    Feature files use node_features[N,F] (or [T,N,F]), edge_index[2,E],
    and edge_attr[E,D] (or [T,E,D]). Single-frame arrays are promoted to a
    temporal dimension of length one.
    """

    def __init__(
        self,
        annotation_file: str | Path,
        feature_root: str | Path | None = None,
        *,
        label_to_index: dict[str, int] | None = None,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.feature_root = (
            self.annotation_file.parent
            if feature_root is None
            else Path(feature_root)
        )
        self._provided_label_to_index = (
            dict(label_to_index) if label_to_index is not None else None
        )
        self.label_to_index: dict[str, int] = {}

    def build(self) -> list[GraphSample]:
        """Load a manifest and create validated graph samples."""
        records = self._load_annotations()
        if not records:
            raise ValueError("Annotation manifest contains no samples.")

        if self._provided_label_to_index is None:
            labels = sorted({record.label for record in records})
            self.label_to_index = {label: index for index, label in enumerate(labels)}
        else:
            self.label_to_index = dict(self._provided_label_to_index)
            if len(set(self.label_to_index.values())) != len(self.label_to_index):
                raise ValueError("label_to_index values must be unique.")
            missing = sorted({record.label for record in records} - self.label_to_index.keys())
            if missing:
                raise ValueError(f"Manifest contains labels absent from label_to_index: {missing}.")

        sample_ids = [record.sample_id for record in records]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Manifest sample_id values must be unique.")

        samples = [self._build_sample(record) for record in records]
        return samples

    def _load_annotations(self) -> list[AnnotationRecord]:
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        if self.annotation_file.suffix.lower() != ".json":
            raise ValueError("DatasetFactory expects a JSON manifest.")
        try:
            payload: Any = json.loads(self.annotation_file.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid dataset manifest JSON: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
            raise ValueError("Annotation manifest must contain a 'samples' list.")

        records: list[AnnotationRecord] = []
        required = {"sample_id", "track_id", "video_id", "feature_file", "label"}
        for index, item in enumerate(payload["samples"]):
            if not isinstance(item, dict):
                raise ValueError(f"Sample {index} must be a JSON object.")
            missing = sorted(required - item.keys())
            if missing:
                raise ValueError(f"Sample {index} is missing required field(s): {missing}.")
            feature_file = Path(item["feature_file"])
            if not feature_file.is_absolute():
                feature_file = self.feature_root / feature_file
            records.append(
                AnnotationRecord(
                    sample_id=str(item["sample_id"]),
                    track_id=item["track_id"],
                    video_id=str(item["video_id"]),
                    feature_file=feature_file,
                    label=str(item["label"]),
                )
            )
        return records

    @staticmethod
    def _load_feature_file(feature_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_file}")
        with np.load(feature_file, allow_pickle=False) as data:
            required = {"node_features", "edge_index", "edge_attr"}
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"Feature file {feature_file} is missing fields: {missing}.")
            node_features = np.asarray(data["node_features"], dtype=np.float32)
            edge_index = np.asarray(data["edge_index"], dtype=np.int64)
            edge_attr = np.asarray(data["edge_attr"], dtype=np.float32)
            frame_mask = np.asarray(data["frame_mask"], dtype=bool) if "frame_mask" in data else None
            node_mask = np.asarray(data["node_mask"], dtype=bool) if "node_mask" in data else None
        if node_features.ndim == 2:
            node_features = node_features[None, ...]
        if edge_attr.ndim == 2:
            edge_attr = edge_attr[None, ...]
        return node_features, edge_index, edge_attr, frame_mask, node_mask

    def _build_sample(self, record: AnnotationRecord) -> GraphSample:
        node_features, edge_index, edge_attr, frame_mask, node_mask = self._load_feature_file(record.feature_file)
        temporal_length, num_nodes, _ = node_features.shape
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"{record.sample_id}: edge_index must have shape (2, E).")
        if edge_attr.ndim != 3 or edge_attr.shape[:2] != (temporal_length, edge_index.shape[1]):
            raise ValueError(f"{record.sample_id}: edge_attr must have shape (T, E, D).")
        if frame_mask is None:
            frame_mask = np.ones((temporal_length,), dtype=bool)
        if node_mask is None:
            node_mask = np.ones((temporal_length, num_nodes), dtype=bool)

        graph = TemporalSkeletonGraph(
            node_features=node_features,
            edge_index=edge_index,
            frame_mask=frame_mask,
            node_mask=node_mask,
            edge_features=edge_attr,
        )
        graph.validate()
        sample = GraphSample(
            graph=graph,
            label=self.label_to_index[record.label],
            behavior_id=record.label,
            sample_id=record.sample_id,
            metadata={
                "video_id": record.video_id,
                "track_id": record.track_id,
                "start_frame": 0,
                "end_frame": temporal_length - 1,
            },
        )
        sample.validate()
        return sample
