# Frame Data Model

**Project:** ChickenBehaviorLab
**Document Type:** Canonical Data Model Specification
**Entity:** Frame
**Version:** 1.0.0

---

# 1. Overview

The **Frame** is the fundamental visual entity in ChickenBehaviorLab.

It represents a single image extracted from a video stream together with its associated metadata.

Every downstream module—including detection, tracking, pose estimation, and behavior recognition—operates on or references Frame objects.

---

# 2. Purpose

The Frame model provides a standardized representation of an individual video frame.

It ensures that all modules receive consistent image data and contextual metadata regardless of the original video source.

---

# 3. Relationships

```text
Video
   │
   ├── Frame 000001
   ├── Frame 000002
   ├── Frame 000003
   └── ...
```

Each Frame belongs to exactly one Video.

---

# 4. Required Attributes

| Field     | Type    | Description                                     |
| --------- | ------- | ----------------------------------------------- |
| frame_id  | Integer | Unique frame identifier within a video          |
| video_id  | String  | Identifier of the source video                  |
| timestamp | Float   | Time in seconds from the beginning of the video |
| image     | Image   | RGB image data                                  |
| width     | Integer | Image width in pixels                           |
| height    | Integer | Image height in pixels                          |

---

# 5. Optional Attributes

| Field          | Type   | Description                        |
| -------------- | ------ | ---------------------------------- |
| camera_id      | String | Camera identifier                  |
| farm_id        | String | Poultry farm identifier            |
| fps            | Float  | Video frame rate                   |
| recording_date | Date   | Recording date                     |
| weather        | String | Optional environmental description |
| notes          | String | Additional comments                |

---

# 6. Validation Rules

A valid Frame shall satisfy the following conditions:

* `frame_id` must be unique within a video.
* `timestamp` must be non-negative.
* `width` and `height` must be greater than zero.
* The image shall use RGB color ordering.
* The image dimensions shall match the recorded metadata.

---

# 7. Produced By

* Video Import Module
* Video Decoding Module

---

# 8. Consumed By

* Detection Module
* Visualization Module
* Dataset Export Module

---

# 9. Serialization Example

```json
{
  "frame_id": 145,
  "video_id": "Farm01_Day03",
  "timestamp": 4.83,
  "width": 1920,
  "height": 1080,
  "camera_id": "CAM_A",
  "farm_id": "Farm01"
}
```

---

# 10. Future Extensions

The Frame model may later include:

* camera calibration parameters
* thermal image references
* depth image references
* synchronization information
* environmental sensor references

---

# 11. Design Notes

The Frame model intentionally contains only information describing a single image.

Objects such as detections, tracks, poses, and behaviors are represented by separate data models and reference the corresponding `frame_id`.

This separation improves modularity, reduces redundancy, and simplifies data management throughout the framework.
