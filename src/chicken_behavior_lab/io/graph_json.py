from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.graph.graph import TemporalSkeletonGraph


class GraphJsonLoader:
    """
    Load GraphSample objects from JSON files.

    Supported input forms:

    1. A single GraphSample JSON object.
    2. An object containing a "samples" list.
    3. A directory containing multiple JSON files.
    """

    def load(
        self,
        path: str | Path,
    ) -> list[GraphSample]:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Graph input not found: {path}"
            )

        if path.is_dir():
            return self._load_directory(path)

        if path.suffix.lower() != ".json":
            raise ValueError(
                "Graph input must be a JSON file or "
                "a directory containing JSON files."
            )

        return self._load_file(path)

    def _load_directory(
        self,
        directory: Path,
    ) -> list[GraphSample]:
        files = sorted(
            directory.glob("*.json")
        )

        if not files:
            raise ValueError(
                f"No JSON graph files found in: {directory}"
            )

        samples: list[GraphSample] = []

        for path in files:
            samples.extend(
                self._load_file(path)
            )

        self._validate_unique_sample_ids(
            samples
        )

        return samples

    def _load_file(
        self,
        path: Path,
    ) -> list[GraphSample]:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise TypeError(
                f"Graph JSON must contain an object: {path}"
            )

        if "samples" in data:
            raw_samples = data["samples"]

            if not isinstance(
                raw_samples,
                list,
            ):
                raise TypeError(
                    "'samples' must be a list."
                )

            samples = [
                self._parse_sample(item)
                for item in raw_samples
            ]

        else:
            samples = [
                self._parse_sample(data)
            ]

        self._validate_unique_sample_ids(
            samples
        )

        return samples

    def _parse_sample(
        self,
        data: Any,
    ) -> GraphSample:
        if not isinstance(data, dict):
            raise TypeError(
                "Each graph sample must be a JSON object."
            )

        required_fields = {
            "sample_id",
            "label",
            "behavior_id",
            "graph",
        }

        missing = required_fields.difference(
            data.keys()
        )

        if missing:
            raise ValueError(
                "Graph sample is missing required fields: "
                f"{sorted(missing)}"
            )

        graph_data = data["graph"]

        if not isinstance(
            graph_data,
            dict,
        ):
            raise TypeError(
                "'graph' must be a JSON object."
            )

        required_graph_fields = {
            "node_features",
            "edge_index",
        }

        missing_graph = (
            required_graph_fields.difference(
                graph_data.keys()
            )
        )

        if missing_graph:
            raise ValueError(
                "Graph is missing required fields: "
                f"{sorted(missing_graph)}"
            )

        node_features = np.asarray(
            graph_data["node_features"],
            dtype=np.float32,
        )

        edge_index = np.asarray(
            graph_data["edge_index"],
            dtype=np.int64,
        )

        edge_features = None

        if graph_data.get(
            "edge_features"
        ) is not None:
            edge_features = np.asarray(
                graph_data["edge_features"],
                dtype=np.float32,
            )

        graph = TemporalSkeletonGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
        )

        sample = GraphSample(
            graph=graph,
            label=int(data["label"]),
            behavior_id=str(
                data["behavior_id"]
            ),
            sample_id=str(
                data["sample_id"]
            ),
            metadata=(
                dict(data["metadata"])
                if data.get("metadata") is not None
                else None
            ),
        )

        sample.validate()

        return sample

    @staticmethod
    def _validate_unique_sample_ids(
        samples: list[GraphSample],
    ) -> None:
        sample_ids: set[str] = set()

        for sample in samples:
            if sample.sample_id in sample_ids:
                raise ValueError(
                    "Duplicate sample_id encountered: "
                    f"{sample.sample_id}"
                )

            sample_ids.add(
                sample.sample_id
            )


def load_graph_samples(
    path: str | Path,
) -> list[GraphSample]:
    """
    Convenience function for loading graph samples.
    """

    loader = GraphJsonLoader()

    return loader.load(path)
