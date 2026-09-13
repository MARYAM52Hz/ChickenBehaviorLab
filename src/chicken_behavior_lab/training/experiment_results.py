from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class ExperimentResultManager:
    """
    Manage experiment outputs and maintain a
    central CSV registry of all experiments.
    """

    CSV_COLUMNS = [
        "experiment_id",
        "created_at",
        "device",
        "seed",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "hidden_dim",
        "num_gnn_layers",
        "dropout",
        "validation_fraction",
        "test_fraction",
        "split_group",
        "best_epoch",
        "best_validation_f1",
        "test_accuracy",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "test_weighted_f1",
        "checkpoint",
    ]

    def __init__(
        self,
        results_root: str | Path = "results",
    ) -> None:
        """
        Initialize the experiment result manager.
        """

        self.results_root = Path(
            results_root
        )

        self.experiments_root = (
            self.results_root
            / "experiments"
        )

        self.registry_path = (
            self.results_root
            / "experiments.csv"
        )

        self.experiments_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------
    # Experiment directory
    # --------------------------------------------------

    def create_experiment_directory(
        self,
        experiment_id: str,
    ) -> Path:
        """
        Create a directory for one experiment.
        """

        experiment_dir = (
            self.experiments_root
            / experiment_id
        )

        experiment_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        return experiment_dir

    # --------------------------------------------------
    # JSON
    # --------------------------------------------------

    @staticmethod
    def save_json(
        path: Path,
        data: dict[str, Any],
    ) -> None:
        """
        Save a dictionary as formatted JSON.
        """

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
            )

    # --------------------------------------------------
    # Experiment configuration
    # --------------------------------------------------

    def save_experiment_config(
        self,
        experiment_dir: Path,
        config: dict[str, Any],
    ) -> None:
        """
        Save experiment configuration.
        """

        self.save_json(
            experiment_dir
            / "experiment_config.json",
            config,
        )

    # --------------------------------------------------
    # Training history
    # --------------------------------------------------

    def save_training_history(
        self,
        experiment_dir: Path,
        history: dict[str, Any],
    ) -> None:
        """
        Save training history.
        """

        self.save_json(
            experiment_dir
            / "training_history.json",
            history,
        )

    # --------------------------------------------------
    # Test metrics
    # --------------------------------------------------

    def save_test_metrics(
        self,
        experiment_dir: Path,
        metrics: dict[str, Any],
    ) -> None:
        """
        Save final test metrics.
        """

        self.save_json(
            experiment_dir
            / "test_metrics.json",
            metrics,
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    def save_summary(
        self,
        experiment_dir: Path,
        summary: dict[str, Any],
    ) -> None:
        """
        Save a compact experiment summary.
        """

        self.save_json(
            experiment_dir
            / "summary.json",
            summary,
        )

    # --------------------------------------------------
    # CSV registry
    # --------------------------------------------------

    def register_experiment(
        self,
        summary: dict[str, Any],
    ) -> None:
        """
        Add one experiment to the central CSV registry.
        """

        file_exists = (
            self.registry_path.exists()
        )

        row = {
            key: summary.get(
                key,
                "",
            )
            for key in self.CSV_COLUMNS
        }

        with self.registry_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=self.CSV_COLUMNS,
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

    # --------------------------------------------------
    # Load registry
    # --------------------------------------------------

    def load_registry(
        self,
    ) -> list[dict[str, str]]:
        """
        Load all registered experiments.
        """

        if not self.registry_path.exists():
            return []

        with self.registry_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(
                file
            )

            return list(reader)
