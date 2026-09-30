from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chicken_behavior_lab.dataset.sample import GraphSample


@dataclass(slots=True)
class TemporalGraphSample:
    """
    A temporal sequence of graph samples representing one behavior window.

    Each element in `graphs` corresponds to one temporal step.

    Expected structure:

        graphs[0] -> frame/window t
        graphs[1] -> frame/window t+1
        ...
        graphs[T-1] -> frame/window t+T-1
    """

    graphs: list[GraphSample]
    label: int
    behavior_id: str
    sample_id: str
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        if not self.graphs:
            raise ValueError(
                "TemporalGraphSample must contain at least one graph."
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

        for index, graph_sample in enumerate(self.graphs):
            if not isinstance(
                graph_sample,
                GraphSample,
            ):
                raise TypeError(
                    "Every temporal element must be a GraphSample. "
                    f"Invalid element at index {index}."
                )

            graph_sample.validate()

        self._validate_graph_consistency()
        self._validate_metadata()

    def _validate_graph_consistency(self) -> None:
        first_graph = self.graphs[0]

        first_video_id = first_graph.get_metadata(
            "video_id"
        )

        first_track_id = first_graph.get_metadata(
            "track_id"
        )

        for index, graph_sample in enumerate(
            self.graphs[1:],
            start=1,
        ):
            video_id = graph_sample.get_metadata(
                "video_id"
            )

            track_id = graph_sample.get_metadata(
                "track_id"
            )

            if (
                first_video_id is not None
                and video_id is not None
                and video_id != first_video_id
            ):
                raise ValueError(
                    "All graphs in a temporal sample must belong "
                    "to the same video_id. "
                    f"Mismatch at index {index}."
                )

            if (
                first_track_id is not None
                and track_id is not None
                and track_id != first_track_id
            ):
                raise ValueError(
                    "All graphs in a temporal sample must belong "
                    "to the same track_id. "
                    f"Mismatch at index {index}."
                )

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

            if not isinstance(
                track_id,
                int,
            ):
                raise TypeError(
                    "metadata.track_id must be an integer."
                )

            if track_id < 0:
                raise ValueError(
                    "metadata.track_id cannot be negative."
                )

        if "start_frame" in self.metadata:
            start_frame = self.metadata[
                "start_frame"
            ]

            if not isinstance(
                start_frame,
                int,
            ):
                raise TypeError(
                    "metadata.start_frame must be an integer."
                )

            if start_frame < 0:
                raise ValueError(
                    "metadata.start_frame cannot be negative."
                )

        if "end_frame" in self.metadata:
            end_frame = self.metadata[
                "end_frame"
            ]

            if not isinstance(
                end_frame,
                int,
            ):
                raise TypeError(
                    "metadata.end_frame must be an integer."
                )

            if end_frame < 0:
                raise ValueError(
                    "metadata.end_frame cannot be negative."
                )

        if (
            "start_frame" in self.metadata
            and "end_frame" in self.metadata
        ):
            if (
                self.metadata["end_frame"]
                < self.metadata["start_frame"]
            ):
                raise ValueError(
                    "metadata.end_frame must be greater than "
                    "or equal to metadata.start_frame."
                )

    def __len__(self) -> int:
        return len(self.graphs)

    @property
    def video_id(self) -> str | None:
        if self.metadata is not None:
            value = self.metadata.get(
                "video_id"
            )

            if value is not None:
                return str(value)

        value = self.graphs[0].get_metadata(
            "video_id"
        )

        if value is None:
            return None

        return str(value)

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

    @property
    def first_frame(self) -> int:
        if self.metadata is not None:
            value = self.metadata.get(
                "start_frame"
            )

            if value is not None:
                return int(value)

        value = self.graphs[0].get_metadata(
            "start_frame"
        )

        if value is None:
            return 0

        return int(value)

    @property
    def last_frame(self) -> int:
        if self.metadata is not None:
            value = self.metadata.get(
                "end_frame"
            )

            if value is not None:
                return int(value)

        value = self.graphs[-1].get_metadata(
            "end_frame"
        )

        if value is None:
            return self.first_frame

        return int(value)

    @property
    def sequence_length(self) -> int:
        return len(self.graphs)

    def get_metadata(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        if self.metadata is None:
            return default

        return self.metadata.get(
            key,
            default,
        )
