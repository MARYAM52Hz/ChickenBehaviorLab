"""
ChickenBehaviorLab Annotation Loader
====================================

Loads behavior annotations from JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chicken_behavior_lab.annotations.schema import (
    AnnotationSet,
    BehaviorAnnotation,
)


class AnnotationLoader:
    """
    Load ChickenBehaviorLab annotations from JSON.
    """

    def load(
        self,
        path: str | Path,
    ) -> AnnotationSet:
        """
        Load annotations from a JSON file.

        Parameters
        ----------
        path:
            Path to annotation JSON file.

        Returns
        -------
        AnnotationSet
            Parsed and validated annotations.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {path}"
            )

        if path.suffix.lower() != ".json":
            raise ValueError(
                "Only JSON annotation files "
                "are currently supported."
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        annotations = self._parse(
            data
        )

        annotation_set = AnnotationSet(
            annotations=annotations
        )

        annotation_set.validate()

        return annotation_set

    def _parse(
        self,
        data: dict[str, Any],
    ) -> list[BehaviorAnnotation]:
        """
        Parse raw JSON data into BehaviorAnnotation objects.
        """

        if "annotations" not in data:
            raise ValueError(
                "JSON must contain an "
                "'annotations' field."
            )

        raw_annotations = data[
            "annotations"
        ]

        if not isinstance(
            raw_annotations,
            list,
        ):
            raise TypeError(
                "'annotations' must be a list."
            )

        annotations = []

        for item in raw_annotations:

            if not isinstance(
                item,
                dict,
            ):
                raise TypeError(
                    "Each annotation must "
                    "be a JSON object."
                )

            annotation = (
                BehaviorAnnotation(
                    annotation_id=str(
                        item["annotation_id"]
                    ),
                    video_id=str(
                        item["video_id"]
                    ),
                    track_id=int(
                        item["track_id"]
                    ),
                    behavior_id=str(
                        item["behavior_id"]
                    ),
                    start_frame=int(
                        item["start_frame"]
                    ),
                    end_frame=int(
                        item["end_frame"]
                    ),
                    annotator=(
                        str(item["annotator"])
                        if item.get("annotator")
                        is not None
                        else None
                    ),
                    confidence=(
                        float(item["confidence"])
                        if item.get("confidence")
                        is not None
                        else None
                    ),
                    notes=(
                        str(item["notes"])
                        if item.get("notes")
                        is not None
                        else None
                    ),
                )
            )

            annotations.append(
                annotation
            )

        return annotations
