# Mortality Prediction Data Model

**Project:** ChickenBehaviorLab
**Document Type:** Canonical Data Model Specification
**Entity:** Mortality Prediction
**Version:** 1.0.0

---

# 1. Overview

The **Mortality Prediction** data model represents the estimated mortality risk of a poultry flock over a predefined prediction horizon.

Rather than reporting observed mortality, this model captures the output of predictive models that integrate behavioral, temporal, and contextual information to estimate future risk.

---

# 2. Purpose

The Mortality Prediction model provides an early-warning representation that supports proactive farm management.

It allows researchers and farm operators to identify elevated mortality risk before significant losses occur.

---

# 3. Relationships

```text
Behavior Events
        │
        ▼
Flock Analytics
        │
        ├───────────────┐
        ▼               │
Environmental Data      │
        │               │
        └───────────────┤
                        ▼
             Mortality Prediction
```

The prediction model may combine behavioral analytics with additional contextual information.

---

# 4. Required Attributes

| Field              | Type     | Description                          |
| ------------------ | -------- | ------------------------------------ |
| prediction_id      | String   | Unique prediction identifier         |
| farm_id            | String   | Farm identifier                      |
| house_id           | String   | Poultry house identifier             |
| prediction_time    | DateTime | Time when prediction was generated   |
| prediction_horizon | String   | Forecast window (e.g., 24h, 48h, 7d) |
| mortality_risk     | Float    | Predicted probability (0–1)          |
| risk_level         | String   | Risk category (Low, Moderate, High)  |
| model_version      | String   | Predictive model version             |

---

# 5. Supporting Indicators

The prediction may reference contributing indicators such as:

* Activity Index
* Feeding Duration
* Drinking Frequency
* Resting Ratio
* Abnormal Event Count
* Social Interaction Metrics
* Movement Statistics

These indicators improve interpretability and facilitate scientific analysis.

---

# 6. Optional Context

Future versions may incorporate:

* Temperature
* Humidity
* Ammonia concentration
* CO₂ level
* Feed consumption
* Water consumption
* Bird age
* Breed
* Stocking density
* Vaccination schedule

These variables are optional and should not be required for the core data model.

---

# 7. Validation Rules

A valid Mortality Prediction shall satisfy:

* mortality_risk ∈ [0,1]
* prediction_horizon is defined
* risk_level matches configured thresholds
* model_version is recorded
* prediction_time is valid

---

# 8. Produced By

Potential prediction models include:

* Gradient Boosting Models
* Random Forest
* Temporal Neural Networks
* Graph Neural Networks
* Transformer-based Predictors
* Hybrid Machine Learning Systems

The prediction algorithm is implementation-specific and independent of this data model.

---

# 9. Consumed By

Outputs may be used for:

* Farm dashboards
* Decision support systems
* Alert generation
* Scientific studies
* Veterinary review
* Longitudinal monitoring

---

# 10. Serialization Example

```json
{
  "prediction_id": "MP_2026_07_23_001",
  "farm_id": "Farm01",
  "house_id": "HouseA",
  "prediction_time": "2026-07-23T18:00:00Z",
  "prediction_horizon": "24h",
  "mortality_risk": 0.82,
  "risk_level": "High",
  "model_version": "v1.0"
}
```

---

# 11. Design Decisions

The Mortality Prediction model represents an estimate rather than an observed outcome.

Observed mortality records should be stored separately and used for validation, benchmarking, and model evaluation.

This separation ensures a clear distinction between predictive outputs and ground-truth observations.

---

# 12. Future Extensions

Future versions may support:

* Explainable AI outputs
* Confidence intervals
* Calibration metrics
* Survival analysis
* Multi-horizon forecasting
* Cross-farm benchmarking
* Ensemble prediction summaries

---

# 13. Python Reference Model

Future implementations should provide an equivalent Python representation:

```python
MortalityPrediction(
    prediction_id: str,
    farm_id: str,
    house_id: str,
    prediction_time: datetime,
    prediction_horizon: str,
    mortality_risk: float,
    risk_level: str,
    model_version: str
)
```

This specification defines the canonical predictive output used throughout ChickenBehaviorLab.
