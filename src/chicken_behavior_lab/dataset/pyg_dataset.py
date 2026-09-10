from __future__ import annotations

from typing import Sequence

import torch
from torch_geometric.data import Data, Dataset

from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)


class PyGGraphDataset(Dataset):
    """
    PyTorch Geometric dataset for ChickenBehaviorLab.
    """

    def __init__(
        self,
        samples: Sequence[GraphSample],
    ) -> None:

        super().__init__()

        self.samples = list(samples)

        self.labels = sorted(
            {
                sample.label
                for sample in self.samples
            }
        )

        self.label_to_index = {
            label: index
            for index, label in enumerate(
                self.labels
            )
        }

        self.index_to_label = {
            index: label
            for label, index
            in self.label_to_index.items()
        }

    def len(self) -> int:
        return len(self.samples)

    def get(
        self,
        index: int,
    ) -> Data:

        sample = self.samples[index]

        if sample.label not in self.label_to_index:
            raise ValueError(
                f"Unknown label: {sample.label}"
            )

        label_index = self.label_to_index[
            sample.label
        ]

        data = Data(
            x=sample.node_features,
            edge_index=sample.edge_index,
            edge_attr=sample.edge_features,
            y=torch.tensor(
                [label_index],
                dtype=torch.long,
            ),
        )

        # Keep metadata attached to the graph.
        data.sample_id = sample.sample_id
        data.video_id = sample.video_id
        data.track_id = sample.track_id

        return data
