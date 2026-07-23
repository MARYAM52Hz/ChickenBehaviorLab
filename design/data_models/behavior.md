# Behavior Data Model

**Project:** ChickenBehaviorLab
**Document Type:** Canonical Data Model Specification
**Entity:** Behavior
**Version:** 1.0.0

---

# 1. Overview

The **Behavior** data model represents the semantic interpretation of a chicken's activity at a specific point in time or during a short temporal interval.

Unlike lower-level representations such as Detection, Track, Pose, or Skeleton, the Behavior model describes *what the chicken is doing* rather than *how it appears*.

The Behavior model bridges computer vision outputs with biologically meaningful concepts defined by the Chicken Behavior Ontology (CBO).

---

# 2. Purpose

The Behavior model provides a standardized representation of recognized behaviors independent of the underlying machine learning algorithm.

It serves as the primary semantic layer for welfare assessment, health monitoring, anomaly detection, and long-term behavioral analysis.

---

# 3. Relationships

```text
Skeleton Sequence
        │
        ▼
Behavior
        │
        ▼
Behavior Event
        │
        ▼
Flock Analytics
```

Each Behavior is inferred from one or more Skeleton observations and may contribute to a larger Behavior Event.

---

# 4. Required Attributes

| Field          | Type    | Description                |
| -------------- | ------- | -------------------------- |
| behavior_id    | String  | Unique behavior identifier |
| track_id       | String  | Associated chicken track   |
| frame_id       | Integer | Reference frame            |
| behavior_class | String  | Predicted behavior label   |
| confidence     | Float   | Prediction confidence      |
| timestamp      | Float   | Time within the video      |

---

# 5. Behavior Categories

Behavior classes should follow the Chicken Behavior Ontology (CBO).

Example categories include:

### Maintenance Behaviors

* Feeding
* Drinking
* Preening
* Dust Bathing

### Locomotion

* Standing
* Walking
* Running
* Jumping

### Resting

* Sitting
* Sleeping

### Social Behaviors

* Following
* Pecking
* Aggressive Interaction
* Social Avoidance

### Abnormal Behaviors

* Lameness
* Prolonged Inactivity
* Repetitive Motion
* Wing Drooping
* Isolation

The ontology may evolve as new behaviors are defined and validated.

---

# 6. Optional Attributes

| Field                    | Type       | Description                            |
| ------------------------ | ---------- | -------------------------------------- |
| model_name               | String     | Behavior recognition model             |
| model_version            | String     | Model version                          |
| probability_distribution | Dictionary | Confidence for all candidate behaviors |
| notes                    | String     | Additional information                 |

---

# 7. Validation Rules

A valid Behavior shall satisfy:

* Confidence ∈ [0,1]
* Valid Track reference
* Valid Frame reference
* Behavior class defined in CBO
* Timestamp within video duration

---

# 8. Produced By

Typical producers include:

* Graph Neural Networks (GNN)
* ST-GCN
* Graph Transformer Networks
* Temporal Transformer Models
* Hybrid Rule-Based Systems

---

# 9. Consumed By

The Behavior model serves as input for:

* Behavior Event Generation
* Welfare Assessment
* Health Monitoring
* Anomaly Detection
* Flock Analytics
* Mortality Prediction

---

# 10. Serialization Example

```json
{
  "behavior_id": "BEH_000451",
  "track_id": "TRK_00027",
  "frame_id": 451,
  "timestamp": 15.03,
  "behavior_class": "Feeding",
  "confidence": 0.96
}
```

---

# 11. Design Decisions

The Behavior model represents an instantaneous semantic prediction.

Long-duration activities are represented separately as Behavior Events.

This separation allows researchers to distinguish between frame-level predictions and temporally aggregated activities.

---

# 12. Future Extensions

Future versions may support:

* Multi-label behaviors
* Hierarchical behavior classification
* Behavioral uncertainty estimation
* Context-aware behavior prediction
* Environmental context integration
* Multi-animal interaction reasoning

---

# 13. Python Reference Model

Future implementations should provide an equivalent Python representation:

```python
Behavior(
    behavior_id: str,
    track_id: str,
    frame_id: int,
    behavior_class: str,
    confidence: float,
    timestamp: float
)
```

This specification defines the canonical semantic behavior representation used throughout ChickenBehaviorLab.
