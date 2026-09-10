from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(slots=True)
class GraphSample:
    """
    A single graph sample used by ChickenBehaviorLab.

    Each sample represents one temporal skeleton graph
    extracted from a chicken track.
    """

    sample_id: str

    node_features: Tensor

    edge_index: Tensor

    edge_features: Tensor

    label: str

    metadata: dict[str, Any]

    @property
    def video_id(self) -> str:
        """
        Return the source video identifier.
        """

        return str(
            self.metadata.get(
                "video_id",
                "unknown",
            )
        )

    @property
    def track_id(self) -> str:
        """
        Return the source track identifier.
        """

        return str(
            self.metadata.get(
                "track_id",
                "unknown",
            )
        )
