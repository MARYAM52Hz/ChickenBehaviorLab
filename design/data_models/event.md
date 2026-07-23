# Behavior Event Data Model

**Project:** ChickenBehaviorLab
**Document Type:** Canonical Data Model Specification
**Entity:** Behavior Event
**Version:** 1.0.0

---

# 1. Overview

The **Behavior Event** data model represents a continuous episode of the same behavior performed by an individual chicken over a period of time.

Unlike the Behavior model, which provides frame-level semantic predictions, the Behavior Event aggregates consecutive behavior predictions into meaningful temporal events.

Behavior Events are the primary units for long-term behavioral analysis, welfare assessment, anomaly detection, and mortality prediction.

---

# 2. Purpose

The Behavior Event model converts instantaneous behavior predictions into temporally coherent episodes.

This representation enables researchers to analyze:

* Behavior duration
* Behavior frequency
* Transition patterns
* Daily activity profiles
* Behavioral abnormalities

---

# 3. Relationships

```text
Track
    │
    └── Behavior Sequence
              │
              ▼
      Behavior Event
              │
              ▼
      Flock Analytics
              │
              ▼
    Mortality Prediction
```

Each Behavior Event is generated from a sequence of consecutive Behavior objects belonging to the same Track.

---

# 4. Required Attributes

| Field          | Type    | Description                |
| -------------- | ------- | -------------------------- |
| event_id       | String  | Unique event identifier    |
| track_id       | String  | Associated chicken track   |
| behavior_class | String  | Event behavior category    |
| start_frame    | Integer | First frame of the event   |
| end_frame      | Integer | Last frame of the event    |
| start_time     | Float   | Event start time (seconds) |
| end_time       | Float   | Event end time (seconds)   |
| duration       | Float   | Event duration (seconds)   |

---

# 5. Derived Statistics

Each event may include:

| Field              | Type    | Description                             |
| ------------------ | ------- | --------------------------------------- |
| average_confidence | Float   | Mean prediction confidence              |
| frame_count        | Integer | Number of frames                        |
| mean_speed         | Float   | Average movement speed during the event |
| distance_traveled  | Float   | Total distance covered                  |
| dominant_pose      | String  | Most frequent pose configuration        |

---

# 6. Event Categories

Behavior Event categories follow the Chicken Behavior Ontology (CBO).

Examples include:

### Feeding Events

Continuous feeding activity.

### Drinking Events

Continuous drinking activity.

### Walking Events

Continuous locomotion.

### Resting Events

Continuous inactivity or resting.

### Social Interaction Events

Interactions with one or more neighboring chickens.

### Abnormal Events

Examples include:

* Prolonged inactivity
* Persistent limping
* Repeated aggressive pecking
* Isolation from the flock
* Repetitive abnormal movements

---

# 7. Validation Rules

A valid Behavior Event shall satisfy:

* start_frame ≤ end_frame
* start_time ≤ end_time
* duration > 0
* behavior_class defined in CBO
* all behaviors belong to the same Track

---

# 8. Produced By

Behavior Event Generator

Input:

* Behavior Sequence

Output:

* Behavior Events

---

# 9. Consumed By

Behavior Events serve as input for:

* Daily behavior summaries
* Welfare assessment
* Health monitoring
* Temporal analytics
* Behavioral trend analysis
* Mortality prediction

---

# 10. Serialization Example

```json
{
  "event_id": "EVT_000045",
  "track_id": "TRK_00027",
  "behavior_class": "Feeding",
  "start_frame": 120,
  "end_frame": 215,
  "start_time": 4.00,
  "end_time": 7.17,
  "duration": 3.17
}
```

---

# 11. Design Decisions

Behavior Events intentionally aggregate multiple frame-level Behavior predictions.

This separation improves robustness against prediction noise and enables biologically meaningful temporal analysis.

Events should be generated using configurable temporal aggregation rules to accommodate different datasets and research objectives.

---

# 12. Future Extensions

Future versions may include:

* Composite events
* Hierarchical events
* Multi-animal interaction events
* Environmental context integration
* Event confidence intervals
* Causal event relationships
* Event graphs

---

# 13. Python Reference Model

Future implementations should provide an equivalent Python representation:

```python
BehaviorEvent(
    event_id: str,
    track_id: str,
    behavior_class: str,
    start_frame: int,
    end_frame: int,
    start_time: float,
    end_time: float,
    duration: float
)
```

This specification defines the canonical temporal representation of behavioral episodes used throughout ChickenBehaviorLab.
