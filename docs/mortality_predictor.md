
# Mortality Prediction Interface

## Purpose

This module wraps a separately trained binary probabilistic estimator.
It does not train a model or infer mortality risk from behavior counts alone.

## File locations

- `src/chicken_behavior_lab/prediction/mortality_predictor.py`
- `src/chicken_behavior_lab/prediction/__init__.py`
- `tests/prediction/test_mortality_predictor.py`

## Estimator requirements

The estimator must be fitted and expose:

- `classes_`: exactly two class labels
- `predict_proba(X)`: class probabilities for each input row

The `positive_label` argument determines which probability column
is returned as `risk_probability`.

## Feature validation

The caller must provide the exact feature names declared by
`feature_names`. Feature order is controlled by that declaration.

Missing features, unexpected features, non-numeric values,
NaN, and infinite values are rejected.

## Example

```python
from chicken_behavior_lab.prediction import MortalityPredictor

# Supply a separately trained and validated estimator.
predictor = MortalityPredictor(
    estimator=fitted_estimator,
    feature_names=["activity", "feed"],
    positive_label=1,
    threshold=0.5,
)

predictions = predictor.predict(
    [{"activity": 0.4, "feed": 2.1}],
    sample_ids=["bird-001"],
)

for prediction in predictions:
    print(prediction)
```

## Testing

Run the module tests from the repository root:

```bash
python -m unittest discover -s tests/prediction -v
python -m compileall -q src tests
```

## Limitations

The returned probability and threshold-based label are model outputs,
not a diagnosis or causal explanation.

Predictive validity, calibration, threshold selection, and operational
use require evaluation on representative, independently held-out data.
The data must include a clearly defined target and appropriate labels.
