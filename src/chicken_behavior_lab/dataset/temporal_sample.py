from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chicken_behavior_lab.dataset.sample import GraphSample


@dataclass(slots=True)
class TemporalGraphSample:
    """
    A temporal sequence of graph samples belonging to one
    behavioral observation.

    Each GraphSample represents one temporal graph window.
    The TemporalGraphSample combines consecutive windows into
    a sequence for temporal behavior recognition.
    """

    graphs: list[GraphSample]
    label: int
    behavior_id: str
    sample_id: str
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        if not self.graphs:
            raise ValueError(
                "graphs cannot be empty."
            )

        if not isinstance(self.label, int):
            raise TypeError(
                "label must be an integer."
            )

        if self.label < 0:
            raise ValueError(
                "label cannot be negative."
            )

        if not self.behavior_id:
            raise ValueError(
                "behavior_id cannot be empty."
            )

        if not self.sample_id:
            raise ValueError(
                "sample_id cannot be empty."
            )

        for graph_sample in self.graphs:
            if not isinstance(
                graph_sample,
                GraphSample,
            ):
                raise TypeError(
                    "Every temporal element must be "
                    "a GraphSample."
                )

            graph_sample.validate()

            if graph_sample.label != self.label:
                raise ValueError(
                    "All graphs in a temporal sample must "
                    "have the same label."
                )

            if graph_sample.behavior_id != self.behavior_id:
                raise ValueError(
                    "All graphs in a temporal sample must "
                    "have the same behavior_id."
                )

        self._validate_temporal_order()
        self._validate_metadata()

    def _validate_temporal_order(self) -> None:
        previous_start = None

        for graph_sample in self.graphs:
            start_frame = graph_sample.get_metadata(
                "start_frame"
            )

            if start_frame is None:
                continue

            if previous_start is not None:
                if start_frame < previous_start:
                    raise ValueError(
                        "Temporal graphs must be ordered "
                        "by increasing start_frame."
                    )

            previous_start = start_frame

    def _validate_metadata(self) -> None:
        if self.metadata is None:
            return

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "metadata must be a dictionary or None."
            )

        if "video_id" in self.metadata:
            if not self.metadata["video_id"]:
                raise ValueError(
                    "metadata.video_id cannot be empty."
                )

        if "track_id" in self.metadata:
            track_id = self.metadata["track_id"]

            if not isinstance(track_id, int):
                raise TypeError(
                    "metadata.track_id must be an integer."
                )

            if track_id < 0:
                raise ValueError(
                    "metadata.track_id cannot be negative."
                )

    def __len__(self) -> int:
        return len(self.graphs)

    @property
    def first_frame(self) -> int | None:
        return self.graphs[0].get_metadata(
            "start_frame"
        )

    @property
    def last_frame(self) -> int | None:
        return self.graphs[-1].get_metadata(
            "end_frame"
        )

    @property
    def video_id(self) -> str | None:
        if self.metadata is not None:
            value = self.metadata.get(
                "video_id"
            )

            if value is not None:
                return str(value)

        return self.graphs[0].get_metadata(
            "video_id"
        )

    @property
    def track_id(self) -> int | None:
        if self.metadata is not None:
            value = self.metadata.get(
                "track_id"
            )

            if value is not None:
                return int(value)

        value = self.graphs[0].get_metadata(
            "track_id"
        )

        if value is None:
            return None

        return int(value)
