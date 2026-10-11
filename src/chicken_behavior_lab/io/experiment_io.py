"""Persistence utilities for reproducible machine-learning experiments."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def _to_serializable(value: Any) -> Any:
    """Convert common scientific Python objects to JSON-compatible values."""

    if is_dataclass(value) and not isinstance(value, type):
        return _to_serializable(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): _to_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]

    if isinstance(value, set):
        return sorted(
            (_to_serializable(item) for item in value),
            key=str,
        )

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_serializable(value.to_dict())

    if hasattr(value, "__dict__"):
        return _to_serializable(vars(value))

    raise TypeError(
        f"Unsupported JSON value type: {type(value).__name__}"
    )


class ExperimentOutputWriter:
    """Write experiment artifacts into a single run directory."""

    def __init__(
        self,
        root: str | Path = "results/experiments",
    ) -> None:
        self.root = Path(root)

    def create_experiment_directory(
        self,
        timestamp: str | None = None,
        experiment_name: str = "temporal_experiment",
    ) -> Path:
        """Create a unique directory for one experiment run."""

        if not experiment_name.strip():
            raise ValueError("experiment_name cannot be empty.")

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S%fZ"
            )

        safe_name = "".join(
            char if char.isalnum() or char in "-_." else "_"
            for char in experiment_name.strip()
        )

        run_dir = self.root / f"{safe_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)

        return run_dir

    @staticmethod
    def _write_json(
        path: str | Path,
        payload: Any,
    ) -> Path:
        """Atomically write JSON using a temporary file in the same folder."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        serializable = _to_serializable(payload)

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )

        try:
            with os.fdopen(
                file_descriptor,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    serializable,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())

            os.replace(temporary_name, destination)

        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

        return destination

    @classmethod
    def save_json(
        cls,
        path: str | Path,
        payload: Any,
    ) -> Path:
        """Save an arbitrary JSON-compatible payload."""

        return cls._write_json(path, payload)

    @classmethod
    def save_experiment_config(
        cls,
        experiment_dir: str | Path,
        config: Any,
    ) -> Path:
        """Save experiment configuration."""

        return cls._write_json(
            Path(experiment_dir) / "experiment_config.json",
            config,
        )

    @classmethod
    def save_model_config(
        cls,
        experiment_dir: str | Path,
        model_config: Any,
    ) -> Path:
        """Save model architecture configuration."""

        return cls._write_json(
            Path(experiment_dir) / "model_config.json",
            model_config,
        )

    @classmethod
    def save_training_history(
        cls,
        experiment_dir: str | Path,
        history: Any,
    ) -> Path:
        """Save training and validation history."""

        return cls._write_json(
            Path(experiment_dir) / "training_history.json",
            history,
        )

    @classmethod
    def save_test_metrics(
        cls,
        experiment_dir: str | Path,
        metrics: Mapping[str, Any],
    ) -> Path:
        """Save final held-out test metrics."""

        return cls._write_json(
            Path(experiment_dir) / "test_metrics.json",
            metrics,
        )

    @classmethod
    def save_prediction_errors(
        cls,
        experiment_dir: str | Path,
        prediction_records: list[Mapping[str, Any]],
    ) -> Path:
        """Save incorrect predictions for targeted error analysis."""

        errors = []

        for record in prediction_records:
            true_index = record.get("true_index")
            predicted_index = record.get("predicted_index")

            if (
                true_index is not None
                and predicted_index is not None
                and true_index != predicted_index
            ):
                errors.append(dict(record))

        return cls._write_json(
            Path(experiment_dir) / "prediction_errors.json",
            {
                "num_prediction_records": len(prediction_records),
                "num_errors": len(errors),
                "errors": errors,
            },
        )

    @classmethod
    def save_all_predictions(
        cls,
        experiment_dir: str | Path,
        prediction_records: list[Mapping[str, Any]],
    ) -> Path:
        """Save correct and incorrect predictions for auditing."""

        return cls._write_json(
            Path(experiment_dir) / "predictions.json",
            {
                "num_predictions": len(prediction_records),
                "predictions": prediction_records,
            },
        )

    @classmethod
    def save_split_manifest(
        cls,
        experiment_dir: str | Path,
        split_sample_ids: Mapping[str, Any],
        split_group_ids: Mapping[str, Any] | None = None,
        diagnostics: Any = None,
    ) -> Path:
        """Save sample IDs, group IDs, and split diagnostics."""

        return cls._write_json(
            Path(experiment_dir) / "split_manifest.json",
            {
                "split_sample_ids": split_sample_ids,
                "split_group_ids": split_group_ids or {},
                "diagnostics": diagnostics,
            },
        )

    @classmethod
    def save_summary(
        cls,
        experiment_dir: str | Path,
        summary: Mapping[str, Any],
    ) -> Path:
        """Save the human-readable experiment summary."""

        return cls._write_json(
            Path(experiment_dir) / "summary.json",
            summary,
        )
