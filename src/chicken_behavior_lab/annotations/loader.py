"""JSON loader and writer for ChickenBehaviorLab behavior annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import AnnotationSet, AnnotationValidationError


class AnnotationLoadError(ValueError):
    """Raised when an annotation file cannot be read or parsed into the schema."""


class AnnotationLoader:
    """Load and save the project's versioned JSON annotation format.

    Expected document shape::

        {
          "schema_version": "1.0",
          "video_id": "video_001",
          "annotations": [
            {"track_id": 3, "behavior_id": "pecking",
             "start_frame": 120, "end_frame": 136}
          ],
          "metadata": {}
        }

    Frame intervals are inclusive at both ends.
    """

    SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})

    def __init__(self, *, allow_empty_annotations: bool = True) -> None:
        if not isinstance(allow_empty_annotations, bool):
            raise TypeError("allow_empty_annotations must be a bool.")
        self.allow_empty_annotations = allow_empty_annotations

    def load(self, path: str | Path) -> AnnotationSet:
        """Read a JSON annotation file and validate its structure and values."""
        source = Path(path).expanduser()
        if not source.exists():
            raise AnnotationLoadError(f"Annotation file does not exist: {source}")
        if not source.is_file():
            raise AnnotationLoadError(f"Annotation path is not a file: {source}")

        try:
            with source.open("r", encoding="utf-8-sig") as stream:
                document: Any = json.load(stream)
        except UnicodeDecodeError as exc:
            raise AnnotationLoadError(f"Annotation file is not valid UTF-8: {source}") from exc
        except json.JSONDecodeError as exc:
            raise AnnotationLoadError(
                f"Invalid JSON in {source} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc
        except OSError as exc:
            raise AnnotationLoadError(f"Could not read annotation file {source}: {exc}") from exc

        try:
            annotation_set = AnnotationSet.from_dict(document)
        except (AnnotationValidationError, TypeError) as exc:
            raise AnnotationLoadError(f"Invalid annotation document {source}: {exc}") from exc

        if annotation_set.schema_version not in self.SUPPORTED_SCHEMA_VERSIONS:
            supported = ", ".join(sorted(self.SUPPORTED_SCHEMA_VERSIONS))
            raise AnnotationLoadError(
                f"Unsupported schema_version {annotation_set.schema_version!r} in {source}. "
                f"Supported version(s): {supported}."
            )
        if not self.allow_empty_annotations and not annotation_set.annotations:
            raise AnnotationLoadError(f"Annotation document contains no annotations: {source}")
        return annotation_set

    def save(
        self,
        annotation_set: AnnotationSet,
        path: str | Path,
        *,
        create_parent: bool = True,
        indent: int = 2,
    ) -> Path:
        """Serialize an AnnotationSet as UTF-8 JSON and return the destination path."""
        if not isinstance(annotation_set, AnnotationSet):
            raise TypeError("annotation_set must be an AnnotationSet instance.")
        if annotation_set.schema_version not in self.SUPPORTED_SCHEMA_VERSIONS:
            raise AnnotationLoadError(
                f"Cannot save unsupported schema_version {annotation_set.schema_version!r}."
            )
        if isinstance(indent, bool) or not isinstance(indent, int) or indent < 0:
            raise ValueError("indent must be a non-negative integer.")

        destination = Path(path).expanduser()
        if create_parent:
            destination.parent.mkdir(parents=True, exist_ok=True)
        elif not destination.parent.exists():
            raise AnnotationLoadError(f"Parent directory does not exist: {destination.parent}")
        try:
            with destination.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(annotation_set.to_dict(), stream, ensure_ascii=False, indent=indent)
                stream.write("\n")
        except OSError as exc:
            raise AnnotationLoadError(f"Could not write annotation file {destination}: {exc}") from exc
        return destination
