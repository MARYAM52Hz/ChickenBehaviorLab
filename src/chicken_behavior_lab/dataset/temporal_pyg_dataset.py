from __future__ import annotations

from typing import Sequence

import torch
from torch_geometric.data import Data

from chicken_behavior_lab.dataset.temporal_builder import (
    TemporalWindow,
)


class TemporalPyGDataset:
    """
    Convert TemporalWindow objects into PyG-compatible
    temporal graph samples.

    Each item represents one temporal sequence.

    Expected structure of each frame sample:

        sample.x
        sample.edge_index
        sample.edge_attr
        sample.y

    The implementation intentionally keeps temporal frames
    separate. TemporalCollator is responsible for batching
    them.
    """

    def __init__(
        self,
        windows: Sequence[TemporalWindow],
    ) -> None:

        self.windows = list(windows)

        if not self.windows:
            raise ValueError(
                "TemporalPyGDataset cannot be empty."
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(
        self,
        index: int,
    ) -> dict:

        window = self.windows[index]

        frames = []

        for sample in window.samples:
            frames.append(
                self._sample_to_pyg(
                    sample
                )
            )

        return {
            "graphs": frames,
            "y": self._label_to_tensor(
                window.behavior_id
            ),
            "sample_id": window.sample_id,
            "video_id": window.video_id,
            "track_id": window.track_id,
            "start_frame": window.start_frame,
            "end_frame": window.end_frame,
            "behavior_id": window.behavior_id,
        }

    @staticmethod
    def _sample_to_pyg(
        sample,
    ) -> Data:
        """
        Convert one graph sample to PyG Data.
        """

        x = getattr(
            sample,
            "x",
            None,
        )

        edge_index = getattr(
            sample,
            "edge_index",
            None,
        )

        edge_attr = getattr(
            sample,
            "edge_attr",
            None,
        )

        if x is None:
            raise ValueError(
                "Graph sample does not contain x."
            )

        if edge_index is None:
            raise ValueError(
                "Graph sample does not contain "
                "edge_index."
            )

        if not isinstance(
            x,
            torch.Tensor,
        ):
            x = torch.as_tensor(
                x,
                dtype=torch.float32,
            )

        if not isinstance(
            edge_index,
            torch.Tensor,
        ):
            edge_index = torch.as_tensor(
                edge_index,
                dtype=torch.long,
            )

        if edge_attr is not None:
            if not isinstance(
                edge_attr,
                torch.Tensor,
            ):
                edge_attr = torch.as_tensor(
                    edge_attr,
                    dtype=torch.float32,
                )

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

    @staticmethod
    def _label_to_tensor(
        label,
    ) -> torch.Tensor:
        """
        Convert a numeric label directly.

        String labels are intentionally not converted here.
        The label encoder belongs to the experiment/config
        layer.
        """

        if isinstance(
            label,
            torch.Tensor,
        ):
            return label.long()

        if isinstance(
            label,
            int,
        ):
            return torch.tensor(
                label,
                dtype=torch.long,
            )

        raise TypeError(
            "TemporalPyGDataset expects an integer "
            "behavior label or a torch.Tensor. "
            f"Received: {type(label).__name__}"
        )
