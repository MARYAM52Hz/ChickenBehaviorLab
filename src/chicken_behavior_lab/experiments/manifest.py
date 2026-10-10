
from __future__ import annotations

import hashlib
import json
import platform
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import torch

from chicken_behavior_lab.dataset.sample import GraphSample


def _json_default(value: Any) -> Any:
    """Convert common scientific Python values to JSON-compatible values."""
    if hasattr(value, "tolist"):
        return value.tolist()

    if hasattr(value, "item"):
        return value.item()

    if hasattr(value, "__fspath__"):
        return str(value)

    if isinstance(value, set):
        return sorted(value)

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable."
    )


def _canonical_json(value: Any) -> str:
    """Serialize data deterministically for hashing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
        allow_nan=False,
    )


def _graph_payload(sample: GraphSample) -> dict[str, Any]:
    """Extract the graph fields used to identify a sample."""
    graph = sample.graph

    payload: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "behavior_id": sample.behavior_id,
        "label": int(sample.label),
        "metadata": sample.metadata or {},
        "node_features": graph.node_features,
        "edge_index": graph.edge_index,
    }

    edge_features = getattr(graph, "edge_features", None)
    payload["edge_features"] = edge_features

    return payload


def fingerprint_samples(
    samples: Sequence[GraphSample],
) -> str:
    """Return a stable SHA-256 fingerprint of an ordered sample collection.

    The fingerprint includes sample identifiers, labels, metadata, and graph
    structure/features. Reordering samples changes the fingerprint.
    """
    digest = hashlib.sha256()

    for sample in samples:
        payload = _canonical_json(_graph_payload(sample))
        encoded = payload.encode("utf-8")

        # Prefix each record with its length to avoid ambiguous concatenation.
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)

    return digest.hexdigest()


def fingerprint_sample_ids(
    samples: Sequence[GraphSample],
) -> str:
    """Fingerprint sample IDs independently of graph feature contents."""
    identifiers = sorted(sample.sample_id for sample in samples)
    encoded = _canonical_json(identifiers).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def collect_environment() -> dict[str, Any]:
    """Collect core runtime versions and hardware availability."""
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": (
            torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0
        ),
    }


def set_reproducibility_seed(seed: int) -> None:
    """Seed common random-number generators used by the experiment."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(slots=True)
class ExperimentManifest:
    """Metadata needed to identify and audit one experiment run."""

    experiment_name: str
    created_at_utc: str

    experiment_config: dict[str, Any]
    model_config: dict[str, Any]
    label_mapping: dict[str, int]

    environment: dict[str, Any]

    dataset_fingerprint: str
    sample_id_fingerprint: str

    split_diagnostics: dict[str, Any]
    split_sample_ids: dict[str, list[str]]
    split_group_ids: dict[str, list[str]]

    reproducibility: dict[str, Any] = field(default_factory=dict)
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    schema_version: int = 1

    def validate(self) -> None:
        """Validate required fields and basic manifest invariants."""
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty.")

        if not self.created_at_utc.strip():
            raise ValueError("created_at_utc must not be empty.")

        if self.schema_version < 1:
            raise ValueError("schema_version must be >= 1.")

        if not self.dataset_fingerprint:
            raise ValueError("dataset_fingerprint must not be empty.")

        if not self.sample_id_fingerprint:
            raise ValueError("sample_id_fingerprint must not be empty.")

        if not self.label_mapping:
            raise ValueError("label_mapping must not be empty.")

        values = list(self.label_mapping.values())

        if any(not isinstance(value, int) or value < 0 for value in values):
            raise ValueError(
                "Label mapping values must be non-negative integers."
            )

        if len(values) != len(set(values)):
            raise ValueError(
                "Label mapping indices must be unique."
            )

        expected_splits = {"train", "validation", "test"}

        if set(self.split_sample_ids) != expected_splits:
            raise ValueError(
                "split_sample_ids must contain train, validation, and test."
            )

        if set(self.split_group_ids) != expected_splits:
            raise ValueError(
                "split_group_ids must contain train, validation, and test."
            )

        # A sample ID should not be assigned to more than one split.
        split_sets = {
            name: set(ids)
            for name, ids in self.split_sample_ids.items()
        }

        names = ("train", "validation", "test")

        for index, left in enumerate(names):
            for right in names[index + 1:]:
                overlap = split_sets[left] & split_sets[right]

                if overlap:
                    raise ValueError(
                        f"Sample IDs overlap between {left} and {right}: "
                        f"{sorted(overlap)}"
                    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        self.validate()

        return json.loads(
            _canonical_json(asdict(self))
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ExperimentManifest":
        """Reconstruct a manifest from a parsed JSON object."""
        manifest = cls(**dict(payload))
        manifest.validate()
        return manifest

    @classmethod
    def create(
        cls,
        *,
        experiment_name: str,
        experiment_config: Any,
        model_config: Any,
        label_mapping: Mapping[str, int],
        train_samples: Sequence[GraphSample],
        validation_samples: Sequence[GraphSample],
        test_samples: Sequence[GraphSample],
        split_diagnostics: Mapping[str, Any],
        split_group_ids: Mapping[str, Sequence[str]] | None = None,
        seed: int = 42,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> "ExperimentManifest":
        """Build a manifest from the prepared experiment state."""
        splits = {
            "train": list(train_samples),
            "validation": list(validation_samples),
            "test": list(test_samples),
        }

        all_samples = (
            splits["train"]
            + splits["validation"]
            + splits["test"]
        )

        if not all_samples:
            raise ValueError(
                "Cannot create a manifest for an empty dataset."
            )

        # Dataclasses are supported; mappings can also be supplied directly.
        config_dict = (
            asdict(experiment_config)
            if hasattr(experiment_config, "__dataclass_fields__")
            else dict(experiment_config)
        )

        model_config_dict = (
            asdict(model_config)
            if hasattr(model_config, "__dataclass_fields__")
            else dict(model_config)
        )

        normalized_group_ids = {
            name: sorted(
                str(group_id)
                for group_id in (
                    (split_group_ids or {}).get(name, [])
                )
            )
            for name in ("train", "validation", "test")
        }

        manifest = cls(
            experiment_name=experiment_name,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            experiment_config=config_dict,
            model_config=model_config_dict,
            label_mapping={
                str(key): int(value)
                for key, value in label_mapping.items()
            },
            environment=collect_environment(),
            dataset_fingerprint=fingerprint_samples(all_samples),
            sample_id_fingerprint=fingerprint_sample_ids(all_samples),
            split_diagnostics=dict(split_diagnostics),
            split_sample_ids={
                name: sorted(
                    sample.sample_id for sample in samples
                )
                for name, samples in splits.items()
            },
            split_group_ids=normalized_group_ids,
            reproducibility={
                "seed": int(seed),
                "python_random_seeded": True,
                "torch_seeded": True,
                "cuda_seeded_if_available": True,
                "deterministic_algorithms_enabled": False,
            },
            extra_metadata=dict(extra_metadata or {}),
        )

        manifest.validate()
        return manifest
