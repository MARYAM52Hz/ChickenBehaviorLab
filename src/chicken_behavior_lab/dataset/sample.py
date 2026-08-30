"""
ChickenBehaviorLab Dataset Sample
=================================

Data structure representing a single graph-based
behavior recognition sample.
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
        Integer ML class index.

    behavior_id:
        Canonical CBO behavior identifier.

    sample_id:
        Unique sample identifier.

    metadata:
        Optional sample metadata.
    """

    graph: TemporalSkeletonGraph

    label: int

    behavior_id: str

    sample_id: str

    metadata: dict | None = None

    def validate(self) -> None:
        """
        Validate the graph sample.
        """

        self.graph.validate()

        # -------------------------------------------------
        # Label validation
        # -------------------------------------------------

        if not isinstance(
            self.label,
            (int, np.integer),
        ):

            raise TypeError(
                "label must be an integer."
            )

        if self.label < 0:

            raise ValueError(
                "label cannot be negative."
            )

        # -------------------------------------------------
        # Behavior ID
        # -------------------------------------------------

        if not self.behavior_id:

            raise ValueError(
                "behavior_id cannot be empty."
            )

        # -------------------------------------------------
        # Sample ID
        # -------------------------------------------------

        if not self.sample_id:

            raise ValueError(
                "sample_id cannot be empty."
            )
