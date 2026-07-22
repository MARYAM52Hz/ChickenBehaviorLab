# Pose Data Model

**Project:** ChickenBehaviorLab
**Document Type:** Canonical Data Model Specification
**Entity:** Pose
**Version:** 1.0.0

---

# 1. Overview

The **Pose** data model represents the anatomical configuration of an individual chicken in a single video frame.

A Pose consists of a predefined set of anatomical keypoints estimated from the tracked chicken and serves as the foundation for skeleton construction, motion analysis, and behavior recognition.

The Pose model is independent of the pose estimation algorithm and provides a standardized representation for downstream modules.

---

# 2. Purpose

The purpose of the Pose model is to describe the spatial arrangement of anatomical landmarks for one tracked chicken at a specific moment in time.

This standardized representation enables consistent behavior analysis across different datasets and pose estimation methods.

---

# 3. Relationships

```text
Track
   │
   ├── Pose (Frame 001)
   ├── Pose (Frame 002)
   ├── Pose (Frame 003)
   └── ...

Pose
   │
   └── Skeleton
```

Each Pose belongs to one Track and one Frame.

A Track contains a temporal sequence of Pose objects.

---

# 4. Required Attributes

| Field      | Type           | Description                 |
| ---------- | -------------- | --------------------------- |
| pose_id    | String         | Unique pose identifier      |
| frame_id   | Integer        | Source frame identifier     |
| track_id   | String         | Associated track identifier |
| keypoints  | List[Keypoint] | Anatomical keypoints        |
| confidence | Float          | Overall pose confidence     |

---

# 5. Keypoint Representation

Each keypoint shall contain:

| Field      | Type    |
| ---------- | ------- |
| name       | String  |
| x          | Float   |
| y          | Float   |
| confidence | Float   |
| visible    | Boolean |

Coordinates are expressed in image space.

---

# 6. Recommended Anatomical Landmarks

The exact set of landmarks follows the Chicken Behavior Annotation Standard (CBAS).

Typical landmarks include:

* Beak
* Head
* Neck
* Body Center
* Tail Base
* Tail Tip
* Left Wing
* Right Wing
* Left Hip
* Right Hip
* Left Knee
* Right Knee
* Left Foot
* Right Foot

Future revisions of CBAS define the canonical landmark list.

---

# 7. Optional Attributes

| Field             | Type   | Description           |
| ----------------- | ------ | --------------------- |
| estimator_name    | String | Pose estimation model |
| estimator_version | String | Model version         |
| inference_time    | Float  | Processing time       |
| notes             | String | Additional comments   |

---

# 8. Validation Rules

A valid Pose shall satisfy:

* Every keypoint has a unique anatomical name.
* Coordinates are inside image boundaries.
* Confidence values are within [0,1].
* Required landmarks are present or explicitly marked as missing.
* Pose references valid Frame and Track identifiers.

---

# 9. Produced By

Typical pose estimation models include:

* YOLO-Pose
* HRNet
* ViTPose
* RTMPose
* Future pose estimation algorithms

---

# 10. Consumed By

The Pose model serves as input for:

* Skeleton Builder
* Motion Analysis
* Graph Neural Networks
* Behavior Recognition
* Quality Assessment

---

# 11. Serialization Example

```json
{
  "pose_id": "POSE_000187",
  "frame_id": 187,
  "track_id": "TRK_00027",
  "confidence": 0.97,
  "keypoints": [
    {
      "name": "head",
      "x": 421.8,
      "y": 218.3,
      "confidence": 0.99,
      "visible": true
    }
  ]
}
```

---

# 12. Design Decisions

The Pose model describes anatomical landmarks only.

It does not encode skeletal connectivity, temporal relationships, or behavior labels.

These responsibilities belong to dedicated downstream data models.

---

# 13. Future Extensions

Future versions may support:

* 3D pose estimation
* Multi-view pose fusion
* Temporal pose smoothing
* Pose uncertainty estimation
* Occlusion reasoning
* Multi-camera pose association

---

# 14. Python Reference Model

Future implementations should provide an equivalent Python representation:

```python
Pose(
    pose_id: str,
    frame_id: int,
    track_id: str,
    keypoints: list[Keypoint],
    confidence: float
)
```

This specification defines the canonical pose representation used throughout ChickenBehaviorLab.
