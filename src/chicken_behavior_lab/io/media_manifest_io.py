"""Atomic JSON persistence for media asset manifests."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from chicken_behavior_lab.media.manifest import MediaManifest


class MediaManifestIO:
    """Read and write versioned MediaManifest JSON files."""

    FILENAME = "media_manifest.json"
    SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})

    @classmethod
    def save(cls, manifest: MediaManifest, path: str | Path) -> Path:
        if not isinstance(manifest, MediaManifest):
            raise TypeError("manifest must be a MediaManifest instance.")
        if manifest.schema_version not in cls.SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"Unsupported media manifest schema version: {manifest.schema_version!r}.")
        destination = Path(path).expanduser()
        if destination.suffix.lower() != ".json":
            raise ValueError("Media manifest destination must have a .json extension.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(
                    manifest.to_dict(), stream, ensure_ascii=False,
                    indent=2, sort_keys=True, allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> MediaManifest:
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Media manifest not found: {source}")
        try:
            with source.open("r", encoding="utf-8-sig") as stream:
                payload: Any = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in media manifest at line {exc.lineno}, "
                f"column {exc.colno}: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("Media manifest JSON must contain an object.")
        manifest = MediaManifest.from_dict(payload)
        if manifest.schema_version not in cls.SUPPORTED_SCHEMA_VERSIONS:
            supported = ", ".join(sorted(cls.SUPPORTED_SCHEMA_VERSIONS))
            raise ValueError(
                f"Unsupported media manifest schema version {manifest.schema_version!r}. "
                f"Supported version(s): {supported}."
            )
        return manifest
