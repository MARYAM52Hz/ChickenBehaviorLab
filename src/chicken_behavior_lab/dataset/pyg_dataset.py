from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch_geometric.data import Data

from chicken_behavior_lab.dataset.sample import GraphSample


class PyGGraphDataset:
    """
    Adapter between ChickenBehaviorLab GraphSample objects
    and PyTorch Geometric Data objects.

    The class deliberately keeps sample metadata as explicit
    Data attributes instead of passing arbitrary dictionaries
    through PyG batching.
    """

    def __init__(
        self,
        samples: Sequence[GraphSample],
    ) -> None:
        self.samples = list(samples)
        self._validate()

        self.label_to_index = self._build_label_mapping()

    def _validate(self) -> None:
        sample_ids: set[str] = set()

        for sample in self.samples:
            if not isinstance(
                sample,
                GraphSample,
            ):
                raise TypeError(
                    "Every item must be a GraphSample."
                )

            sample.validate()

            if sample.sample_id in sample_ids:
                raise ValueError(
                    f"Duplicate sample_id: "
                    f"{sample.sample_id}"
                )

            sample_ids.add(sample.sample_id)

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
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> Data:
        sample = self.samples[index]

        graph = sample.graph

        graph.validate()

        metadata = sample.metadata or {}

        video_id = str(
            metadata.get(
                "video_id",
                "",
            )
        )

        track_id = int(
            metadata.get(
                "track_id",
                -1,
            )
        )

        start_frame = int(
            metadata.get(
                "start_frame",
                0,
            )
        )

        end_frame = int(
            metadata.get(
                "end_frame",
                start_frame,
            )
        )

        data = Data(
            x=torch.as_tensor(
                graph.node_features,
                dtype=torch.float32,
            ),
            edge_index=torch.as_tensor(
                graph.edge_index,
                dtype=torch.long,
            ),
            y=torch.tensor(
                [sample.label],
                dtype=torch.long,
            ),
        )

        if graph.edge_features is not None:
            data.edge_attr = torch.as_tensor(
                graph.edge_features,
                dtype=torch.float32,
            )

        # Explicit sample metadata.
        data.sample_id = sample.sample_id
        data.video_id = video_id
        data.track_id = track_id
        data.start_frame = start_frame
        data.end_frame = end_frame
        data.behavior_id = sample.behavior_id

        return data

    @property
    def samples_metadata(
        self,
    ) -> list[dict[str, Any]]:
        metadata_list = []

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

            metadata_list.append(metadata)

        return metadata_list
