"""
YOLO Result Adapter
===================

Converts raw Ultralytics YOLO outputs into
ChickenBehaviorLab internal data structures.
"""

from __future__ import annotations

from typing import Any

from chicken_behavior_lab.models.geometry import (
    BoundingBox,
)

from chicken_behavior_lab.models.detection import (
    Detection,
)


class YOLOResultAdapter:
    """
    Convert Ultralytics results into framework models.
    """

    def adapt_detections(
        self,
        results: Any,
        frame_id: str,
    ) -> list[Detection]:
        """
        Convert YOLO detection results into
        ChickenBehaviorLab Detection objects.
        """

        detections: list[Detection] = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        boxes = result.boxes

        for index in range(len(boxes)):

            xyxy = boxes.xyxy[index].tolist()

            confidence = float(
                boxes.conf[index].item()
            )

            bbox = BoundingBox(
                x_min=float(xyxy[0]),
                y_min=float(xyxy[1]),
                x_max=float(xyxy[2]),
                y_max=float(xyxy[3]),
            )

            detection = Detection(
                detection_id=(
                    f"{frame_id}_det_{index}"
                ),
                frame_id=frame_id,
                bbox=bbox,
                confidence=confidence,
                class_name="chicken",
            )

            detections.append(detection)

        return detections
