from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

from chicken_behavior_lab.annotations.labels import BehaviorLabelEncoder
from chicken_behavior_lab.annotations.schema import (
    AnnotationSet,
    BehaviorAnnotation,
)
from chicken_behavior_lab.dataset.graph_dataset import GraphDataset
from chicken_behavior_lab.dataset.sample import GraphSample


class GraphBuilderProtocol(Protocol):
    """
    Minimal interface required from a skeleton graph builder.

    This protocol keeps DatasetBuilder independent from the concrete
    graph-construction implementation.
    """

    def build(self, sequence: Any) -> Any:
        ...


FeatureSequenceProvider = Callable[
    [str, int, int, int],
    Any,
]


class DatasetBuilder:
    """
    Build a graph-based behavior dataset from behavior annotations.

    DatasetBuilder connects:

        BehaviorAnnotation
                +
        temporal feature sequence
                +
        skeleton graph builder
                +
        behavior label encoder
                ↓
            GraphSample
                ↓
            GraphDataset

    The builder deliberately does not read videos, run tracking, perform
    pose estimation, or compute features itself.

    Instead, a feature_sequence_provider is supplied by the caller.

    The provider receives:

        video_id
        track_id
        start_frame
        end_frame

    and must return the temporal feature representation needed by the
    graph builder.
    """

    def __init__(
        self,
        graph_builder: GraphBuilderProtocol,
        label_encoder: BehaviorLabelEncoder,
        feature_sequence_provider: FeatureSequenceProvider,
    ) -> None:
        self.graph_builder = graph_builder
        self.label_encoder = label_encoder
        self.feature_sequence_provider = feature_sequence_provider

    def build(
        self,
        annotations: AnnotationSet | Iterable[BehaviorAnnotation],
    ) -> GraphDataset:
        """
        Convert annotations into a GraphDataset.

        Parameters
        ----------
        annotations:
            AnnotationSet or iterable of BehaviorAnnotation objects.

        Returns
        -------
        GraphDataset
            Dataset containing one GraphSample per annotation.
        """

        if isinstance(annotations, AnnotationSet):
            annotations.validate()
            annotation_items = annotations.annotations
        else:
            annotation_items = list(annotations)

        samples: list[GraphSample] = []

        for annotation in annotation_items:
            sample = self.build_sample(annotation)
            samples.append(sample)

        return GraphDataset(samples)

    def build_sample(
        self,
        annotation: BehaviorAnnotation,
    ) -> GraphSample:
        """
        Build one GraphSample from one behavior annotation.
        """

        if not isinstance(annotation, BehaviorAnnotation):
            raise TypeError(
                "annotation must be a BehaviorAnnotation."
            )

        annotation.validate()

        label = self.label_encoder.encode(
            annotation.behavior_id
        )

        sequence = self.feature_sequence_provider(
            annotation.video_id,
            annotation.track_id,
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

        sample_id = self._make_sample_id(annotation)

        metadata = self._make_metadata(annotation)

        sample = GraphSample(
            graph=graph,
            label=label,
            behavior_id=annotation.behavior_id,
            sample_id=sample_id,
            metadata=metadata,
        )

        sample.validate()

        return sample

    @staticmethod
    def _make_sample_id(
        annotation: BehaviorAnnotation,
    ) -> str:
        """
        Generate a deterministic sample identifier.

        Example:
            video_001_track_1_ann_000001
        """

        return (
            f"{annotation.video_id}"
            f"_track_{annotation.track_id}"
            f"_{annotation.annotation_id}"
        )

    @staticmethod
    def _make_metadata(
        annotation: BehaviorAnnotation,
    ) -> dict[str, Any]:
        """
        Preserve annotation provenance inside the dataset sample.
        """

        return {
            "annotation_id": annotation.annotation_id,
            "video_id": annotation.video_id,
            "track_id": annotation.track_id,
            "start_frame": annotation.start_frame,
            "end_frame": annotation.end_frame,
            "num_frames": annotation.num_frames,
            "annotator": annotation.annotator,
            "confidence": annotation.confidence,
            "notes": annotation.notes,
        }
