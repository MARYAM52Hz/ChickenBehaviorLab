
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from chicken_behavior_lab.experiments.manifest import (
    ExperimentManifest,
)


class ExperimentManifestIO:
    """Read and write experiment manifests as JSON."""

    FILENAME = "experiment_manifest.json"

    @classmethod
    def save(
        cls,
        manifest: ExperimentManifest,
        experiment_dir: str | Path,
    ) -> Path:
        """Atomically write a manifest into an experiment directory."""
        manifest.validate()

        directory = Path(experiment_dir)
        directory.mkdir(parents=True, exist_ok=True)

        destination = directory / cls.FILENAME
        temporary = directory / f".{cls.FILENAME}.tmp"

        payload = manifest.to_dict()

        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as file:
                json.dump(
                    payload,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())

            os.replace(temporary, destination)
        finally:
            # Clean up a temporary file if writing or replacement fails.
            if temporary.exists():
                temporary.unlink()

        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> ExperimentManifest:
        """Load and validate a previously saved manifest."""
        source = Path(path)

        if not source.is_file():
            raise FileNotFoundError(
                f"Experiment manifest not found: {source}"
            )

        with source.open("r", encoding="utf-8") as file:
            payload: Any = json.load(file)

        if not isinstance(payload, dict):
            raise ValueError(
                "Experiment manifest JSON must contain an object."
            )

        return ExperimentManifest.from_dict(payload)
