from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from chicken_behavior_lab.graph.graph import TemporalSkeletonGraph


@dataclass(slots=True)
class GraphSample:
    """
    One supervised graph-learning sample.

    A GraphSample contains:
    - the temporal skeleton graph,
    - the integer training label,
    - the canonical behavior ID,
    - a unique sample ID,
    - optional metadata describing the source sample.
    """

    graph: TemporalSkeletonGraph
    label: int
    behavior_id: str
    sample_id: str
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        self.graph.validate()

        if not isinstance(self.label, (int, np.integer)):
            raise TypeError("label must be an integer.")

        if self.label < 0:
            raise ValueError("label cannot be negative.")

        if not self.behavior_id:
            raise ValueError("behavior_id cannot be empty.")

        if not self.sample_id:
            raise ValueError("sample_id cannot be empty.")

        if self.metadata is not None:
            if not isinstance(self.metadata, dict):
                raise TypeError(
                    "metadata must be a dictionary or None."
                )

            self._validate_metadata()

    def _validate_metadata(self) -> None:
        if "video_id" in self.metadata:
            if not self.metadata["video_id"]:
                raise ValueError(
                    "metadata.video_id cannot be empty."
                )

        if "track_id" in self.metadata:
            track_id = self.metadata["track_id"]

            if not isinstance(
                track_id,
                (int, np.integer),
            ):
                raise TypeError(
                    "metadata.track_id must be an integer."
                )

            if track_id < 0:
                raise ValueError(
                    "metadata.track_id cannot be negative."
                )

        if "start_frame" in self.metadata:
            start_frame = self.metadata["start_frame"]

            if not isinstance(
                start_frame,
                (int, np.integer),
            ):
                raise TypeError(
                    "metadata.start_frame must be an integer."
                )

            if start_frame < 0:
                raise ValueError(
                    "metadata.start_frame cannot be negative."
                )

        if "end_frame" in self.metadata:
            end_frame = self.metadata["end_frame"]

            if not isinstance(
                end_frame,
                (int, np.integer),
            ):
                raise TypeError(
                    "metadata.end_frame must be an integer."
                )

            if end_frame < 0:
                raise ValueError(
                    "metadata.end_frame cannot be negative."
                )

        if (
            self.metadata is not None
            and "start_frame" in self.metadata
            and "end_frame" in self.metadata
        ):
            if (
                self.metadata["end_frame"]
                < self.metadata["start_frame"]
            ):
                raise ValueError(
                    "metadata.end_frame must be greater "
                    "than or equal to metadata.start_frame."
                )

    def get_metadata(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        if self.metadata is None:
            return default

        return self.metadata.get(key, default)
