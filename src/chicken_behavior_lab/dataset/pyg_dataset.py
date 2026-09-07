from __future__ import annotations

from collections.abc import Sequence

import numpy as np
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
    Convert ChickenBehaviorLab graph samples into
    PyTorch Geometric Data objects.

    Expected graph representation:

        node_features:
            (N, F)

        edge_index:
            (2, E)

        edge_features:
            (E, D)

        edge_type:
            (E,)
    """

    def __init__(
        self,
        samples: Sequence[GraphSample],
    ) -> None:

        super().__init__()

        self.samples = list(
            samples
        )

        self._validate_samples()

    def _validate_samples(
        self,
    ) -> None:

        for index, sample in enumerate(
            self.samples
        ):

            if not isinstance(
                sample,
                GraphSample,
            ):
                raise TypeError(
                    f"Sample {index} must be "
                    "a GraphSample."
                )

            sample.validate()

    def len(
        self,
    ) -> int:

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

        graph = sample.graph

        graph.validate()

        # =============================================
        # Node features
        # =============================================

        x = torch.from_numpy(
            np.asarray(
                graph.node_features,
                dtype=np.float32,
            )
        )

        # =============================================
        # Edges
        # =============================================

        edge_index = torch.from_numpy(
            np.asarray(
                graph.edge_index,
                dtype=np.int64,
            )
        )

        # =============================================
        # Edge attributes
        # =============================================

        edge_attr = torch.from_numpy(
            np.asarray(
                graph.edge_features,
                dtype=np.float32,
            )
        )

        # =============================================
        # Edge type
        # =============================================

        edge_type = torch.from_numpy(
            np.asarray(
                graph.edge_type,
                dtype=np.int64,
            )
        )

        # =============================================
        # Label
        # =============================================

        y = torch.tensor(
            sample.label,
            dtype=torch.long,
        )

        # =============================================
        # PyG object
        # =============================================

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type,
            y=y,
        )

        # =============================================
        # Research metadata
        # =============================================

        data.sample_id = (
            sample.sample_id
        )

        data.behavior_id = (
            sample.behavior_id
        )

        data.frame_ids = list(
            graph.frame_ids
        )

        data.num_frames = (
            graph.num_frames
        )

        data.num_keypoints = (
            graph.num_keypoints
        )

        data.metadata = (
            sample.metadata
        )

        return data
