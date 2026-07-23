# Flock Analytics Data Model

**Project:** ChickenBehaviorLab
**Document Type:** Canonical Data Model Specification
**Entity:** Flock Analytics
**Version:** 1.0.0

---

# 1. Overview

The **Flock Analytics** data model represents the aggregated behavioral state of a poultry flock over a defined time interval.

Unlike previous data models that describe individual chickens, this model summarizes information across the entire flock and provides indicators for welfare assessment, health monitoring, and farm management.

---

# 2. Purpose

The Flock Analytics model aggregates individual Behavior Events into population-level metrics.

It enables researchers and farm managers to monitor collective behavior, detect emerging problems, and quantify long-term trends.

---

# 3. Relationships

```text
Behavior Events
        │
        ▼
Individual Statistics
        │
        ▼
Flock Analytics
        │
        ▼
Mortality Prediction
```

Each Flock Analytics object summarizes all relevant events within a configurable observation window.

---

# 4. Required Attributes

| Field             | Type    | Description                 |
| ----------------- | ------- | --------------------------- |
| analytics_id      | String  | Unique analytics identifier |
| farm_id           | String  | Farm identifier             |
| house_id          | String  | Poultry house identifier    |
| observation_start | Float   | Start time (seconds)        |
| observation_end   | Float   | End time (seconds)          |
| flock_size        | Integer | Number of observed chickens |

---

# 5. Core Behavioral Metrics

The analytics object may include metrics such as:

### Activity

* Activity Index
* Mean Movement Speed
* Total Distance Traveled

### Feeding

* Feeding Frequency
* Mean Feeding Duration
* Total Feeding Time

### Drinking

* Drinking Frequency
* Drinking Duration

### Resting

* Resting Time
* Resting Ratio

### Social Behavior

* Interaction Frequency
* Aggressive Interaction Count
* Isolation Ratio

### Abnormal Behavior

* Lameness Incidents
* Prolonged Inactivity
* Abnormal Motion Count

---

# 6. Spatial Metrics

Examples include:

* Occupancy Heatmap
* Zone Utilization
* Crowd Density
* Spatial Dispersion
* Preferred Areas

---

# 7. Temporal Metrics

Examples include:

* Hourly Activity Profile
* Circadian Rhythm
* Behavior Transition Matrix
* Event Frequency Timeline

---

# 8. Validation Rules

A valid analytics object shall satisfy:

* observation_start < observation_end
* flock_size > 0
* Metrics computed from validated Behavior Events
* Consistent farm and house identifiers

---

# 9. Produced By

The Flock Analytics module aggregates:

* Behavior Events
* Track Statistics
* Spatial Information
* Motion Statistics

---

# 10. Consumed By

Outputs may be used for:

* Farm Dashboards
* Welfare Monitoring
* Health Assessment
* Early Warning Systems
* Mortality Prediction
* Scientific Analysis

---

# 11. Serialization Example

```json
{
  "analytics_id": "FA_2026_07_23_001",
  "farm_id": "Farm01",
  "house_id": "HouseA",
  "observation_start": 0.0,
  "observation_end": 3600.0,
  "flock_size": 1280,
  "feeding_frequency": 642,
  "mean_activity_index": 0.73,
  "aggressive_interactions": 14,
  "abnormal_events": 3
}
```

---

# 12. Design Decisions

The Flock Analytics model intentionally stores aggregated indicators rather than individual detections or behaviors.

This separation:

* reduces data volume,
* simplifies downstream analysis,
* improves privacy where required,
* enables direct comparison across farms and time periods.

---

# 13. Future Extensions

Future versions may include:

* Environmental sensor integration
* Feed and water consumption records
* Climate measurements
* Knowledge graph integration
* Cross-house comparisons
* Longitudinal flock health profiles

---

# 14. Python Reference Model

Future implementations should provide an equivalent Python representation:

```python
FlockAnalytics(
    analytics_id: str,
    farm_id: str,
    house_id: str,
    observation_start: float,
    observation_end: float,
    flock_size: int,
    metrics: dict
)
```

This specification defines the canonical population-level representation used throughout ChickenBehaviorLab.
