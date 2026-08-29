"""
ChickenBehaviorLab Dataset Sample
=================================

Data structures representing a single training sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chicken_behavior_lab.graph.graph import (
    TemporalSkeletonGraph,
)


@dataclass(slots=True)
class GraphSample:
    """
    A single graph-based behavior recognition sample.

    Parameters
    ----------
    graph:
        Temporal skeleton graph.

    label:
        Behavior class index.

    sample_id:
        Unique identifier for this sample.

    metadata:
        Optional metadata associated with the sample.
    """

    graph: TemporalSkeletonGraph

    label: int

    sample_id: str

    metadata: dict | None = None

    def validate(self) -> None:
        """
        Validate the graph sample.
        """

        self.graph.validate()

        if not isinstance(
            self.label,
            (int, np.integer),
        ):

            raise TypeError(
                "label must be an integer."
            )

        if not self.sample_id:

            raise ValueError(
                "sample_id cannot be empty."
            )
