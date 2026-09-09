from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    """
    One annotation record from the dataset manifest.
    """

    sample_id: str
    track_id: str
    video_id: str
    feature_file: Path
    label: str


class DatasetFactory:
    """
    Converts annotation records and feature files
    into GraphSample objects.
    """

    def __init__(
        self,
        annotation_file: str | Path,
        feature_root: str | Path | None = None,
    ) -> None:

        self.annotation_file = Path(
            annotation_file
        )

        if feature_root is None:
            self.feature_root = (
                self.annotation_file.parent
            )
        else:
            self.feature_root = Path(
                feature_root
            )

    # =====================================================
    # Public API
    # =====================================================

    def build(self) -> list[GraphSample]:
        """
        Build the complete graph dataset.
        """

        records = (
            self._load_annotations()
        )

        samples: list[GraphSample] = []

        for record in records:

            sample = (
                self._build_sample(
                    record
                )
            )

            samples.append(
                sample
            )

        return samples

    # =====================================================
    # Annotation loading
    # =====================================================

    def _load_annotations(
        self,
    ) -> list[AnnotationRecord]:

        if not self.annotation_file.exists():
            raise FileNotFoundError(
                "Annotation file not found: "
                f"{self.annotation_file}"
            )

        with self.annotation_file.open(
            "r",
            encoding="utf-8",
        ) as file:

            payload: dict[str, Any] = (
                json.load(file)
            )

        if "samples" not in payload:
            raise ValueError(
                "Annotation file must contain "
                "a 'samples' field."
            )

        raw_samples = payload[
            "samples"
        ]

        if not isinstance(
            raw_samples,
            list,
        ):
            raise ValueError(
                "'samples' must be a list."
            )

        records: list[
            AnnotationRecord
        ] = []

        for index, item in enumerate(
            raw_samples
        ):

            if not isinstance(
                item,
                dict,
            ):
                raise ValueError(
                    f"Sample {index} must "
                    "be an object."
                )

            required_fields = [
                "sample_id",
                "track_id",
                "video_id",
                "feature_file",
                "label",
            ]

            for field in required_fields:

                if field not in item:
                    raise ValueError(
                        f"Sample {index} is "
                        f"missing '{field}'."
                    )

            feature_file = Path(
                item["feature_file"]
            )

            if not feature_file.is_absolute():
                feature_file = (
                    self.feature_root
                    / feature_file
                )

            records.append(
                AnnotationRecord(
                    sample_id=str(
                        item["sample_id"]
                    ),
                    track_id=str(
                        item["track_id"]
                    ),
                    video_id=str(
                        item["video_id"]
                    ),
                    feature_file=feature_file,
                    label=str(
                        item["label"]
                    ),
                )
            )

        return records

    # =====================================================
    # Feature loading
    # =====================================================

    @staticmethod
    def _load_feature_file(
        feature_file: Path,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
    ]:

        if not feature_file.exists():
            raise FileNotFoundError(
                "Feature file not found: "
                f"{feature_file}"
            )

        with np.load(
            feature_file,
            allow_pickle=False,
        ) as data:

            required_fields = [
                "node_features",
                "edge_index",
                "edge_attr",
            ]

            for field in required_fields:

                if field not in data:
                    raise ValueError(
                        f"Feature file "
                        f"{feature_file} "
                        f"is missing '{field}'."
                    )

            node_features = torch.as_tensor(
                data["node_features"],
                dtype=torch.float32,
            )

            edge_index = torch.as_tensor(
                data["edge_index"],
                dtype=torch.long,
            )

            edge_attr = torch.as_tensor(
                data["edge_attr"],
                dtype=torch.float32,
            )

        return (
            node_features,
            edge_index,
            edge_attr,
        )

    # =====================================================
    # GraphSample construction
    # =====================================================

    def _build_sample(
        self,
        record: AnnotationRecord,
    ) -> GraphSample:

        (
            node_features,
            edge_index,
            edge_attr,
        ) = self._load_feature_file(
            record.feature_file
        )

        self._validate_graph(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            sample_id=record.sample_id,
        )

        return GraphSample(
            sample_id=record.sample_id,
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_attr,
            label=record.label,
            metadata={
                "track_id": record.track_id,
                "video_id": record.video_id,
            },
        )

    # =====================================================
    # Validation
    # =====================================================

    @staticmethod
    def _validate_graph(
        node_features: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        sample_id: str,
    ) -> None:

        if node_features.ndim != 2:
            raise ValueError(
                f"{sample_id}: "
                "node_features must have "
                "shape (N, F)."
            )

        if edge_index.ndim != 2:
            raise ValueError(
                f"{sample_id}: "
                "edge_index must have "
                "shape (2, E)."
            )

        if edge_index.shape[0] != 2:
            raise ValueError(
                f"{sample_id}: "
                "edge_index must have "
                "shape (2, E)."
            )

        if edge_attr.ndim != 2:
            raise ValueError(
                f"{sample_id}: "
                "edge_attr must have "
                "shape (E, D)."
            )

        if (
            edge_index.shape[1]
            != edge_attr.shape[0]
        ):
            raise ValueError(
                f"{sample_id}: "
                "edge count in edge_index "
                "does not match edge_attr."
            )

        num_nodes = (
            node_features.shape[0]
        )

        if edge_index.numel() > 0:

            if torch.any(
                edge_index < 0
            ):
                raise ValueError(
                    f"{sample_id}: "
                    "edge_index contains "
                    "negative indices."
                )

            if torch.any(
                edge_index >= num_nodes
            ):
                raise ValueError(
                    f"{sample_id}: "
                    "edge_index references "
                    "non-existing nodes."
                )
