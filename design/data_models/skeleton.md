# Skeleton Data Model

**Project:** ChickenBehaviorLab
**Document Type:** Canonical Data Model Specification
**Entity:** Skeleton
**Version:** 1.0.0

---

# 1. Overview

The **Skeleton** data model represents the graph-based anatomical structure of an individual chicken.

Unlike the Pose model, which stores independent keypoints, the Skeleton model explicitly defines the anatomical connectivity between keypoints and provides a graph representation suitable for downstream graph-based learning algorithms.

The Skeleton model serves as the primary structural representation for motion analysis and behavior recognition.

---

# 2. Purpose

The Skeleton model transforms anatomical keypoints into a structured graph.

This representation enables:

* Graph Neural Networks (GNNs)
* Spatio-Temporal Graph Convolutional Networks (ST-GCNs)
* Motion analysis
* Behavior recognition
* Explainable AI

---

# 3. Relationships

```text
Track
   │
   ├── Pose
   │      │
   │      └── Skeleton
   │
   └── Skeleton Sequence
```

Each Skeleton is generated from exactly one Pose.

A Track contains a temporal sequence of Skeletons.

---

# 4. Graph Representation

A Skeleton is represented as:

```
G = (V, E)
```

where:

* **V** = Anatomical keypoints (nodes)
* **E** = Anatomical connections (edges)

---

# 5. Node Attributes

Each node contains:

| Field      | Type    | Description              |
| ---------- | ------- | ------------------------ |
| node_id    | String  | Unique node identifier   |
| landmark   | String  | Anatomical landmark name |
| x          | Float   | X coordinate             |
| y          | Float   | Y coordinate             |
| confidence | Float   | Detection confidence     |
| visible    | Boolean | Visibility flag          |

---

# 6. Edge Attributes

Each edge represents a biological connection.

| Field     | Type   | Description           |
| --------- | ------ | --------------------- |
| source    | String | Source node           |
| target    | String | Target node           |
| edge_type | String | Anatomical connection |

Optional edge features:

* Euclidean distance
* Relative angle
* Connection confidence

---

# 7. Canonical Skeleton Topology

The exact topology follows the Chicken Behavior Annotation Standard (CBAS).

Example anatomical graph:

* Head ↔ Neck
* Neck ↔ Body Center
* Body Center ↔ Tail Base
* Tail Base ↔ Tail Tip
* Body Center ↔ Left Wing
* Body Center ↔ Right Wing
* Body Center ↔ Left Hip
* Body Center ↔ Right Hip
* Left Hip ↔ Left Knee
* Left Knee ↔ Left Foot
* Right Hip ↔ Right Knee
* Right Knee ↔ Right Foot

---

# 8. Required Attributes

| Field       | Type       |
| ----------- | ---------- |
| skeleton_id | String     |
| pose_id     | String     |
| frame_id    | Integer    |
| track_id    | String     |
| nodes       | List[Node] |
| edges       | List[Edge] |

---

# 9. Validation Rules

A valid Skeleton shall satisfy:

* All nodes are unique.
* Edge endpoints reference valid nodes.
* No duplicated edges.
* Graph is anatomically valid.
* Required landmarks are present or explicitly marked as missing.

---

# 10. Produced By

* Skeleton Builder
* Graph Construction Module

---

# 11. Consumed By

* Graph Neural Networks
* ST-GCN
* Temporal Graph Models
* Behavior Recognition
* Motion Analysis

---

# 12. Future Extensions

Future versions may include:

* 3D skeleton graphs
* Dynamic edge weights
* Learned graph topology
* Multi-view graph fusion
* Hypergraph representations
* Physics-informed skeletal models

---

# 13. Python Reference Model

Future implementations should provide an equivalent Python representation:

```python
Skeleton(
    skeleton_id: str,
    pose_id: str,
    frame_id: int,
    track_id: str,
    nodes: list[Node],
    edges: list[Edge]
)
```

This specification defines the canonical graph representation used throughout ChickenBehaviorLab.
