from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


class ExperimentOutputWriter:
    """
    Manage experiment output directories and JSON artifacts.
    """

    def __init__(
        self,
        root_directory: str | Path = "results/experiments",
    ) -> None:
        self.root_directory = Path(
            root_directory
        )

    def create_experiment_directory(
        self,
        timestamp: str | None = None,
    ) -> Path:
        self.root_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if timestamp is None:
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

        experiment_dir = (
            self.root_directory / timestamp
        )

        experiment_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        return experiment_dir

    def save_json(
        self,
        path: str | Path,
        data: Any,
    ) -> Path:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        serializable = self._to_serializable(
            data
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                serializable,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return path

    def save_experiment_config(
        self,
        experiment_dir: str | Path,
        config: Any,
    ) -> Path:
        return self.save_json(
            Path(experiment_dir)
            / "experiment_config.json",
            config,
        )

    def save_training_history(
        self,
        experiment_dir: str | Path,
        history: dict[str, Any],
    ) -> Path:
        return self.save_json(
            Path(experiment_dir)
            / "training_history.json",
            history,
        )

    def save_test_metrics(
        self,
        experiment_dir: str | Path,
        metrics: dict[str, Any],
    ) -> Path:
        return self.save_json(
            Path(experiment_dir)
            / "test_metrics.json",
            metrics,
        )

    def save_prediction_errors(
        self,
        experiment_dir: str | Path,
        prediction_errors: Any,
    ) -> Path:
        return self.save_json(
            Path(experiment_dir)
            / "prediction_errors.json",
            prediction_errors,
        )

    def save_split_manifest(
        self,
        experiment_dir: str | Path,
        manifest: Any,
    ) -> Path:
        return self.save_json(
            Path(experiment_dir)
            / "split_manifest.json",
            manifest,
        )

    def save_summary(
        self,
        experiment_dir: str | Path,
        summary: dict[str, Any],
    ) -> Path:
        return self.save_json(
            Path(experiment_dir)
            / "summary.json",
            summary,
        )

    @classmethod
    def _to_serializable(
        cls,
        value: Any,
    ) -> Any:
        if is_dataclass(value):
            return cls._to_serializable(
                asdict(value)
            )

        if isinstance(
            value,
            dict,
        ):
            return {
                str(key): cls._to_serializable(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple),
        ):
            return [
                cls._to_serializable(item)
                for item in value
            ]

        if isinstance(
            value,
            np.ndarray,
        ):
            return value.tolist()

        if isinstance(
            value,
            np.integer,
        ):
            return int(value)

        if isinstance(
            value,
            np.floating,
        ):
            return float(value)

        if isinstance(
            value,
            torch.Tensor,
        ):
            if value.ndim == 0:
                return value.item()

            return value.detach().cpu().tolist()

        if isinstance(
            value,
            Path,
        ):
            return str(value)

        if isinstance(
            value,
            (str, int, float, bool),
        ) or value is None:
            return value

        return str(value)
