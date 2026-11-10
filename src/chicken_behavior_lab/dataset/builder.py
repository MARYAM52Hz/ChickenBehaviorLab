from __future__ import annotations

from collections.abc import Callable, Iterable
import hashlib
import json
from typing import Any, Protocol

from chicken_behavior_lab.annotations.labels import BehaviorLabelEncoder
from chicken_behavior_lab.annotations.schema import AnnotationSet, BehaviorAnnotation
from chicken_behavior_lab.core.identifiers import normalize_track_id
from chicken_behavior_lab.dataset.graph_dataset import GraphDataset
from chicken_behavior_lab.dataset.sample import GraphSample


class GraphBuilderProtocol(Protocol):
    """Minimal interface required from a skeleton graph builder."""

    def build(self, sequence: Any) -> Any:
        ...


FeatureSequenceProvider = Callable[[str, str, int, int], Any]


class DatasetBuilder:
    """Build graph samples using a video-level annotation set and feature provider.

    The feature provider receives (video_id, canonical_track_id, start_frame,
    end_frame). The source annotation schema remains unchanged; legacy numeric
    track IDs are normalized only at this dataset-construction boundary.
    """

    def __init__(
        self,
        graph_builder: GraphBuilderProtocol,
        label_encoder: BehaviorLabelEncoder,
        feature_sequence_provider: FeatureSequenceProvider,
    ) -> None:
        if not callable(getattr(graph_builder, "build", None)):
            raise TypeError("graph_builder must provide a callable build() method.")
        if not isinstance(label_encoder, BehaviorLabelEncoder):
            raise TypeError("label_encoder must be a BehaviorLabelEncoder.")
        if not callable(feature_sequence_provider):
            raise TypeError("feature_sequence_provider must be callable.")
        self.graph_builder = graph_builder
        self.label_encoder = label_encoder
        self.feature_sequence_provider = feature_sequence_provider

    def build(
        self,
        annotations: AnnotationSet | Iterable[BehaviorAnnotation],
        *,
        video_id: str | None = None,
    ) -> GraphDataset:
        """Build a dataset from one AnnotationSet or an iterable plus video_id."""
        if isinstance(annotations, AnnotationSet):
            annotations.validate()
            if video_id is not None and video_id != annotations.video_id:
                raise ValueError("Explicit video_id does not match AnnotationSet.video_id.")
            resolved_video_id = annotations.video_id
            annotation_items = annotations.annotations
        else:
            annotation_items = list(annotations)
            if video_id is None:
                raise ValueError(
                    "video_id is required when building from an annotation iterable."
                )
            if not isinstance(video_id, str) or not video_id.strip():
                raise ValueError("video_id must be a non-empty string.")
            resolved_video_id = video_id.strip()

        samples = [
            self.build_sample(annotation, video_id=resolved_video_id)
            for annotation in annotation_items
        ]
        return GraphDataset(samples)

    def build_sample(
        self,
        annotation: BehaviorAnnotation,
        *,
        video_id: str,
    ) -> GraphSample:
        """Build one graph sample from a video-scoped behavior annotation."""
        if not isinstance(annotation, BehaviorAnnotation):
            raise TypeError("annotation must be a BehaviorAnnotation.")
        annotation.validate()
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("video_id must be a non-empty string.")
        video_id = video_id.strip()

        track_id = normalize_track_id(annotation.track_id)
        label = self.label_encoder.encode(annotation.behavior_id)
        sequence = self.feature_sequence_provider(
            video_id,
            track_id,
            annotation.start_frame,
            annotation.end_frame,
        )
        if sequence is None:
            raise ValueError(
                "feature_sequence_provider returned None for "
                f"annotation '{annotation.annotation_id}'."
            )

        graph = self.graph_builder.build(sequence)
        if graph is None:
            raise ValueError(
                "graph_builder returned None for "
                f"annotation '{annotation.annotation_id}'."
            )

        sample = GraphSample(
            graph=graph,
            label=label,
            behavior_id=annotation.behavior_id,
            sample_id=self._make_sample_id(annotation, video_id),
            metadata=self._make_metadata(annotation, video_id),
        )
        sample.validate()
        return sample

    @staticmethod
    def _make_sample_id(annotation: BehaviorAnnotation, video_id: str) -> str:
        """Create a deterministic ID, including a digest if annotation_id is absent."""
        annotation_id = annotation.annotation_id
        if annotation_id is None:
            payload = json.dumps(
                annotation.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            digest = hashlib.sha256(
                f"{video_id}|{payload}".encode("utf-8")
            ).hexdigest()[:12]
            annotation_id = f"ann_{digest}"
        track_id = normalize_track_id(annotation.track_id)
        return f"{video_id}_{track_id}_{annotation_id}"

    @staticmethod
    def _make_metadata(annotation: BehaviorAnnotation, video_id: str) -> dict[str, Any]:
        """Preserve the fields actually defined by the public annotation schema."""
        return {
            "annotation_id": annotation.annotation_id,
            "video_id": video_id,
            "track_id": normalize_track_id(annotation.track_id),
            "start_frame": annotation.start_frame,
            "end_frame": annotation.end_frame,
            "num_frames": annotation.duration_frames,
            "annotator_id": annotation.annotator_id,
            "confidence": annotation.confidence,
            "attributes": dict(annotation.attributes),
        }
