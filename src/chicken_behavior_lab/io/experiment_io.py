from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from chicken_behavior_lab.dataset.sample import GraphSample


class ExperimentOutputWriter:
    """
    Persist reproducible experiment outputs.

    Output structure:

        experiment_dir/
            experiment_config.json
            training_history.json
            test_metrics.json
            split_manifest.json
            prediction_errors.json
            summary.json
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
        *,
        timestamp: str | None = None,
    ) -> Path:

        if timestamp is None:
            timestamp = datetime.now(
                timezone.utc
            ).strftime(
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

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self._make_serializable(data),
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
        errors: Any,
    ) -> Path:

        return self.save_json(
            Path(experiment_dir)
            / "prediction_errors.json",
            errors,
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

    def save_split_manifest(
        self,
        experiment_dir: str | Path,
        *,
        train_samples: Sequence[GraphSample],
        validation_samples: Sequence[GraphSample],
        test_samples: Sequence[GraphSample],
        group_key: str,
        train_groups: Sequence[str],
        validation_groups: Sequence[str],
        test_groups: Sequence[str],
    ) -> Path:

        manifest = {
            "group_key": group_key,
            "train_groups": list(
                train_groups
            ),
            "validation_groups": list(
                validation_groups
            ),
            "test_groups": list(
                test_groups
            ),
            "train": self._serialize_samples(
                train_samples
            ),
            "validation": self._serialize_samples(
                validation_samples
            ),
            "test": self._serialize_samples(
                test_samples
            ),
        }

        return self.save_json(
            Path(experiment_dir)
            / "split_manifest.json",
            manifest,
        )

    @staticmethod
    def _serialize_samples(
        samples: Sequence[GraphSample],
    ) -> list[dict[str, Any]]:

        records = []

        for sample in samples:

            metadata = dict(
                sample.metadata or {}
            )

            records.append(
                {
                    "sample_id": sample.sample_id,
                    "behavior_id": sample.behavior_id,
                    "label": int(sample.label),
                    "video_id": metadata.get(
                        "video_id"
                    ),
                    "track_id": metadata.get(
                        "track_id"
                    ),
                    "start_frame": metadata.get(
                        "start_frame"
                    ),
                    "end_frame": metadata.get(
                        "end_frame"
                    ),
                }
            )

        return records

    @staticmethod
    def _make_serializable(
        value: Any,
    ) -> Any:

        if is_dataclass(value):
            return ExperimentOutputWriter._make_serializable(
                asdict(value)
            )

        if isinstance(value, dict):
            return {
                str(key):
                    ExperimentOutputWriter._make_serializable(
                        item
                    )
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                ExperimentOutputWriter._make_serializable(
                    item
                )
                for item in value
            ]

        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, TypeError):
                pass

        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except (ValueError, TypeError):
                pass

        if isinstance(value, Path):
            return str(value)

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):
            return value

        if value is None:
            return None

        return str(value)
