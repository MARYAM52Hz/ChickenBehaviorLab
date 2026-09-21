from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class TemporalBatch:
    """
    Explicit batch representation for temporal skeleton graphs.

    Tensor contract
    --------------
    x:
        [B, T, N, F_node]

    edge_index:
        [2, E]

    edge_attr:
        [B, T, E, F_edge] or None

    y:
        [B]

    Metadata fields contain one value per sequence.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor | None
    y: torch.Tensor

    sample_id: list[str]
    video_id: list[str]
    track_id: torch.Tensor
    start_frame: torch.Tensor
    end_frame: torch.Tensor

    behavior_id: list[str]

    def validate(self) -> None:
        if self.x.ndim != 4:
            raise ValueError(
                "x must have shape [B, T, N, F_node]."
            )

        batch_size = self.x.shape[0]
        sequence_length = self.x.shape[1]

        if batch_size < 1:
            raise ValueError(
                "Batch cannot be empty."
            )

        if sequence_length < 1:
            raise ValueError(
                "Sequence length must be >= 1."
            )

        if self.edge_index.ndim != 2:
            raise ValueError(
                "edge_index must have shape [2, E]."
            )

        if self.edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, E]."
            )

        if self.y.ndim != 1:
            raise ValueError(
                "y must have shape [B]."
            )

        if self.y.shape[0] != batch_size:
            raise ValueError(
                "y and x must contain the same "
                "number of samples."
            )

        if self.edge_attr is not None:
            if self.edge_attr.ndim != 4:
                raise ValueError(
                    "edge_attr must have shape "
                    "[B, T, E, F_edge]."
                )

            if self.edge_attr.shape[0] != batch_size:
                raise ValueError(
                    "edge_attr batch dimension must "
                    "match x."
                )

            if self.edge_attr.shape[1] != sequence_length:
                raise ValueError(
                    "edge_attr temporal dimension must "
                    "match x."
                )

            if (
                self.edge_attr.shape[2]
                != self.edge_index.shape[1]
            ):
                raise ValueError(
                    "edge_attr edge dimension must "
                    "match edge_index."
                )

        if len(self.sample_id) != batch_size:
            raise ValueError(
                "sample_id length must match batch size."
            )

        if len(self.video_id) != batch_size:
            raise ValueError(
                "video_id length must match batch size."
            )

        if self.track_id.shape != (batch_size,):
            raise ValueError(
                "track_id must have shape [B]."
            )

        if self.start_frame.shape != (batch_size,):
            raise ValueError(
                "start_frame must have shape [B]."
            )

        if self.end_frame.shape != (batch_size,):
            raise ValueError(
                "end_frame must have shape [B]."
            )

        if len(self.behavior_id) != batch_size:
            raise ValueError(
                "behavior_id length must match batch size."
            )

    @property
    def batch_size(self) -> int:
        return self.x.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.x.shape[1]

    @property
    def num_nodes(self) -> int:
        return self.x.shape[2]

    @property
    def node_feature_dim(self) -> int:
        return self.x.shape[3]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def to(
        self,
        device: torch.device | str,
    ) -> "TemporalBatch":
        return TemporalBatch(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=(
                self.edge_attr.to(device)
                if self.edge_attr is not None
                else None
            ),
            y=self.y.to(device),
            sample_id=self.sample_id,
            video_id=self.video_id,
            track_id=self.track_id.to(device),
            start_frame=self.start_frame.to(device),
            end_frame=self.end_frame.to(device),
            behavior_id=self.behavior_id,
        )

    def metadata(
        self,
        index: int,
    ) -> dict[str, Any]:
        if not 0 <= index < self.batch_size:
            raise IndexError(
                f"Batch index out of range: {index}"
            )

        return {
            "sample_id": self.sample_id[index],
            "video_id": self.video_id[index],
            "track_id": int(
                self.track_id[index].item()
            ),
            "start_frame": int(
                self.start_frame[index].item()
            ),
            "end_frame": int(
                self.end_frame[index].item()
            ),
            "behavior_id": self.behavior_id[index],
        }
