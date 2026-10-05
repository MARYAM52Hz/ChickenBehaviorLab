from __future__ import annotations

from pathlib import Path

from chicken_behavior_lab.annotations.loader import (
    AnnotationLoader,
)
from chicken_behavior_lab.annotations.schema import (
    AnnotationSet,
)


def load_annotations(
    path: str | Path,
) -> AnnotationSet:
    """
    Load behavior annotations from a JSON file.
    """

    loader = AnnotationLoader()

    return loader.load(path)
