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

    Supported layouts:

    Single sample:
    {
        "sample_id": "...",
        "label": 0,
        "behavior_id": "feeding",
        "metadata": {...},
        "graph": {
            "node_features": [[...], [...]],
            "edge_index": [[0, 1], [1, 2]],
            "edge_features": [[...], [...]]
        }
    }

    Multiple samples:
    {
        "samples": [
            {...},
            {...}
        ]
    }

    A directory containing multiple JSON files is also supported.
    """

    def load(
        self,
        path: str | Path,
    ) -> list[GraphSample]:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Graph path not found: {path}"
            )

        if path.is_file():
            if path.suffix.lower() != ".json":
                raise ValueError(
                    "Graph file must have a .json extension."
                )

            return self._load_file(path)

        if path.is_dir():
            return self._load_directory(path)

        raise ValueError(
            f"Unsupported graph path: {path}"
        )

    def _load_directory(
        self,
        directory: Path,
    ) -> list[GraphSample]:

        files = sorted(
            directory.glob("*.json")
        )

        if not files:
            raise ValueError(
                f"No JSON graph files found in {directory}."
            )

        samples: list[GraphSample] = []

        for file_path in files:
            samples.extend(
                self._load_file(file_path)
            )

        self._validate_unique_sample_ids(samples)

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

        if isinstance(data, dict) and "samples" in data:
            raw_samples = data["samples"]

            if not isinstance(
                raw_samples,
                list,
            ):
                raise TypeError(
                    "'samples' must be a list."
                )

        elif isinstance(data, dict):
            raw_samples = [data]

        elif isinstance(data, list):
            raw_samples = data

        else:
            raise TypeError(
                "Graph JSON must contain either "
                "an object or a list of objects."
            )

        samples = []

        for index, raw_sample in enumerate(
            raw_samples
        ):
            if not isinstance(
                raw_sample,
                dict,
            ):
                raise TypeError(
                    f"Graph sample at index {index} "
                    "must be a JSON object."
                )

            samples.append(
                self._parse_sample(
                    raw_sample,
                    source_path=path,
                )
            )

        self._validate_unique_sample_ids(samples)

        return samples

    def _parse_sample(
        self,
        data: dict[str, Any],
        *,
        source_path: Path,
    ) -> GraphSample:

        required = {
            "sample_id",
            "label",
            "behavior_id",
            "graph",
        }

        missing = required.difference(
            data.keys()
        )

        if missing:
            raise ValueError(
                f"Graph sample in {source_path} "
                f"is missing fields: "
                f"{sorted(missing)}"
            )

        raw_graph = data["graph"]

        if not isinstance(
            raw_graph,
            dict,
        ):
            raise TypeError(
                "'graph' must be a JSON object."
            )

        graph = self._parse_graph(
            raw_graph
        )

        metadata = data.get(
            "metadata"
        )

        if metadata is not None:
            if not isinstance(
                metadata,
                dict,
            ):
                raise TypeError(
                    "'metadata' must be a dictionary."
                )

            metadata = dict(metadata)

        sample = GraphSample(
            graph=graph,
            label=int(data["label"]),
            behavior_id=str(
                data["behavior_id"]
            ),
            sample_id=str(
                data["sample_id"]
            ),
            metadata=metadata,
        )

        sample.validate()

        return sample

    def _parse_graph(
        self,
        data: dict[str, Any],
    ) -> TemporalSkeletonGraph:

        required = {
            "node_features",
            "edge_index",
        }

        missing = required.difference(
            data.keys()
        )

        if missing:
            raise ValueError(
                "Graph object is missing fields: "
                f"{sorted(missing)}"
            )

        node_features = np.asarray(
            data["node_features"],
            dtype=np.float32,
        )

        edge_index = np.asarray(
            data["edge_index"],
            dtype=np.int64,
        )

        edge_features_raw = data.get(
            "edge_features"
        )

        edge_features = None

        if edge_features_raw is not None:
            edge_features = np.asarray(
                edge_features_raw,
                dtype=np.float32,
            )

        graph = TemporalSkeletonGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
        )

        graph.validate()

        return graph

    @staticmethod
    def _validate_unique_sample_ids(
        samples: list[GraphSample],
    ) -> None:

        sample_ids = [
            sample.sample_id
            for sample in samples
        ]

        if len(sample_ids) != len(
            set(sample_ids)
        ):
            raise ValueError(
                "Duplicate sample_id values "
                "were found in graph data."
            )


def load_graph_samples(
    path: str | Path,
) -> list[GraphSample]:

    loader = GraphJsonLoader()

    return loader.load(path)
