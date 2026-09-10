from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch_geometric.data import (
    Data,
    Dataset,
)

from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)


class PyGGraphDataset(Dataset):
    """
    PyTorch Geometric dataset for
    ChickenBehaviorLab.
    """

    def __init__(
        self,
        samples: Sequence[GraphSample],
        label_to_index: Mapping[
            str,
            int,
        ] | None = None,
    ) -> None:

        super().__init__()

        self.samples = list(
            samples
        )

        if label_to_index is None:

            labels = sorted(
                {
                    sample.label
                    for sample in self.samples
                }
            )

            self.label_to_index = {
                label: index
                for index, label in enumerate(
                    labels
                )
            }

        else:

            self.label_to_index = dict(
                label_to_index
            )

        self.index_to_label = {
            index: label
            for label, index
            in self.label_to_index.items()
        }

        unknown_labels = {
            sample.label
            for sample in self.samples
            if sample.label
            not in self.label_to_index
        }

        if unknown_labels:

            raise ValueError(
                "Dataset contains labels that "
                "are not present in the shared "
                f"label mapping: {unknown_labels}"
            )

    @property
    def labels(self) -> list[str]:

        return list(
            self.label_to_index.keys()
        )

    def len(self) -> int:
        return len(
            self.samples
        )

    def get(
        self,
        index: int,
    ) -> Data:

        sample = self.samples[
            index
        ]

        label_index = (
            self.label_to_index[
                sample.label
            ]
        )

        data = Data(
            x=sample.node_features,
            edge_index=sample.edge_index,
            edge_attr=sample.edge_features,
            y=torch.tensor(
                [label_index],
                dtype=torch.long,
            ),
        )

        data.sample_id = (
            sample.sample_id
        )

        data.video_id = (
            sample.video_id
        )

        data.track_id = (
            sample.track_id
        )

        return data
