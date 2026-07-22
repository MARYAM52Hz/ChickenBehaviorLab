# Detection Data Model

**Project:** ChickenBehaviorLab
**Document Type:** Canonical Data Model Specification
**Entity:** Detection
**Version:** 1.0.0

---

# 1. Overview

The **Detection** data model represents the output of an object detection algorithm.

A Detection identifies the presence and location of a chicken within a single video frame.

This model serves as the standardized interface between the Detection Module and all downstream components.

---

# 2. Purpose

The Detection model provides a unified representation of detected objects independent of the underlying detection algorithm.

Whether detections are produced by YOLO, RT-DETR, Faster R-CNN, or future models, they should all be converted into this canonical format.

---

# 3. Relationships

```text
Frame
   │
   ├── Detection 1
   ├── Detection 2
   ├── Detection 3
   └── ...
```

Each Detection belongs to exactly one Frame.

A single Frame may contain zero, one, or many detections.

---

# 4. Required Attributes

| Field        | Type        | Description                              |
| ------------ | ----------- | ---------------------------------------- |
| detection_id | String      | Globally unique detection identifier     |
| frame_id     | Integer     | Identifier of the source frame           |
| class_name   | String      | Detected object category (e.g., Chicken) |
| confidence   | Float       | Detection confidence score (0.0–1.0)     |
| bbox         | BoundingBox | Bounding box coordinates                 |

---

# 5. Bounding Box Specification

Bounding boxes are represented as:

| Field | Description       |
| ----- | ----------------- |
| x_min | Left coordinate   |
| y_min | Top coordinate    |
| x_max | Right coordinate  |
| y_max | Bottom coordinate |

Coordinate system:

* Origin at the upper-left corner.
* Pixel coordinates.
* Image coordinate convention.

---

# 6. Optional Attributes

| Field             | Type   | Description                              |
| ----------------- | ------ | ---------------------------------------- |
| detector_name     | String | Detection model name                     |
| detector_version  | String | Model version                            |
| inference_time    | Float  | Processing time (ms)                     |
| segmentation_mask | Mask   | Optional instance mask                   |
| feature_vector    | Vector | Optional embedding for re-identification |
| notes             | String | Additional information                   |

---

# 7. Validation Rules

A valid Detection shall satisfy:

* Confidence ∈ [0,1]
* Bounding box width > 0
* Bounding box height > 0
* Bounding box inside image boundaries
* Valid frame reference
* Valid class label

---

# 8. Produced By

* YOLO-based detectors
* RT-DETR
* Faster R-CNN
* DETR
* Future object detection models

---

# 9. Consumed By

* Multi-Object Tracking
* Visualization
* Annotation Validation
* Dataset Export
* Quality Assessment

---

# 10. Serialization Example

```json
{
  "detection_id": "DET_000145_02",
  "frame_id": 145,
  "class_name": "Chicken",
  "confidence": 0.982,
  "bbox": {
    "x_min": 421,
    "y_min": 208,
    "x_max": 683,
    "y_max": 547
  }
}
```

---

# 11. Design Decisions

This model intentionally stores only object localization information.

Tracking identity, anatomical pose, behavior labels, and temporal information are managed by separate data models.

This separation ensures loose coupling between software modules and simplifies algorithm replacement.

---

# 12. Future Extensions

Future versions may include:

* Rotated bounding boxes
* 3D bounding boxes
* Occlusion level
* Visibility score
* Detection uncertainty
* Multi-camera correspondence
* Temporal confidence estimation

---

# 13. Python Reference Model

Future software implementations should provide an equivalent Python representation, for example:

```python
Detection(
    detection_id: str,
    frame_id: int,
    class_name: str,
    confidence: float,
    bbox: BoundingBox
)
```

This specification serves as the canonical definition for all detection outputs within ChickenBehaviorLab.
