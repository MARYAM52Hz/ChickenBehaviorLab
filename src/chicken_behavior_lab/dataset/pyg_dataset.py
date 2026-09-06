from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from chicken_behavior_lab.dataset.sample import GraphSample


class PyGGraphDataset(Dataset):
    """
    PyTorch Geometric dataset for ChickenBehaviorLab.

    Each GraphSample is converted into a PyG Data object.

    The initial implementation keeps the temporal dimension.

    Input graph:

        x
        shape = (T, V, F)

        edge_index
        shape = (2, E)

        edge_attr
        shape = (T, E, D)

    where:

        T = number of frames
        V = number of skeleton nodes
        E = number of edges
        F = node feature dimension
        D = edge feature dimension
    """

    def __init__(
        self,
        samples: Sequence[GraphSample],
    ) -> None:

        super().__init__()

        self.samples = list(samples)

        self._validate_samples()

    # =====================================================
    # Validation
    # =====================================================

    def _validate_samples(self) -> None:
        """
        Validate all GraphSample objects.
        """

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

    # =====================================================
    # Length
    # =====================================================

    def len(self) -> int:
        """
        Return number of graph samples.
        """

        return len(self.samples)

    # =====================================================
    # Get Item
    # =====================================================

    def get(
        self,
        index: int,
    ) -> Data:
        """
        Convert one GraphSample into a PyG Data object.
        """

        sample = self.samples[index]

        graph = sample.graph

        node_features = np.asarray(
            graph.node_features,
            dtype=np.float32,
        )

        edge_index = np.asarray(
            graph.edge_index,
            dtype=np.int64,
        )

        edge_features = np.asarray(
            graph.edge_features,
            dtype=np.float32,
        )

        # -------------------------------------------------
        # Convert NumPy → PyTorch
        # -------------------------------------------------

        x = torch.from_numpy(
            node_features
        )

        edge_index_tensor = torch.from_numpy(
            edge_index
        )

        edge_attr = torch.from_numpy(
            edge_features
        )

        # -------------------------------------------------
        # Label
        # -------------------------------------------------

        y = torch.tensor(
            [sample.label],
            dtype=torch.long,
        )

        # -------------------------------------------------
        # Masks
        # -------------------------------------------------

        frame_ids = list(
            graph.frame_ids
        )

        data = Data(
            x=x,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr,
            y=y,
        )

        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        data.sample_id = sample.sample_id

        data.behavior_id = (
            sample.behavior_id
        )

        data.frame_ids = frame_ids

        data.metadata = sample.metadata

        return data
