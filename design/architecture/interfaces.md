# Module Interfaces

**Project:** ChickenBehaviorLab

**Document Type:** Interface Specification

**Version:** 1.0.0

---

# 1. Overview

This document defines the public interfaces between software modules in the ChickenBehaviorLab framework.

Interfaces specify the contracts that enable independent development, testing, and replacement of individual modules while preserving overall system compatibility.

Implementations may vary, but interface contracts must remain stable.

---

# 2. Design Goals

Interfaces are designed to be:

* Stable
* Minimal
* Explicit
* Framework-independent
* Type-safe
* Easily testable

---

# 3. General Interface Rules

Every module shall:

* Accept canonical data models as input.
* Return canonical data models as output.
* Avoid modifying input objects.
* Validate inputs before processing.
* Report processing failures explicitly.
* Record processing metadata when appropriate.

---

# 4. Detector Interface

### Responsibility

Detect chickens in a single frame.

### Input

Frame

### Output

List[Detection]

### Contract

```python
detect(frame: Frame) -> list[Detection]
```

### Preconditions

* Valid Frame object.
* Image data available.
* Image dimensions are known.

### Postconditions

* Every Detection references the input frame.
* Confidence values are within [0,1].
* Bounding boxes are valid.

---

# 5. Tracker Interface

### Responsibility

Associate detections across frames.

### Input

List[Detection]

### Output

List[Track]

### Contract

```python
track(detections: list[Detection]) -> list[Track]
```

### Preconditions

* Valid detections.
* Consistent frame ordering.

### Postconditions

* Every Track contains valid detections.
* Track identifiers are unique.

---

# 6. Pose Estimator Interface

### Responsibility

Estimate anatomical landmarks.

### Input

Track

### Output

Pose

### Contract

```python
predict(track: Track) -> Pose
```

### Preconditions

* Valid Track object.
* Referenced detections available.

### Postconditions

* Pose references the input track.
* Required landmarks are represented or explicitly marked as unavailable.

---

# 7. Skeleton Builder Interface

### Responsibility

Build a graph representation from a pose.

### Input

Pose

### Output

Skeleton

### Contract

```python
build(pose: Pose) -> Skeleton
```

### Preconditions

* Pose satisfies validation rules.
* Required keypoints are present when available.

### Postconditions

* Skeleton graph is structurally valid.
* Nodes and edges follow the CBAS topology.

---

# 8. Behavior Recognizer Interface

### Responsibility

Predict semantic behaviors.

### Input

Sequence[ Skeleton ]

### Output

Behavior

### Contract

```python
predict(sequence: Sequence[Skeleton]) -> Behavior
```

### Preconditions

* Ordered skeleton sequence.
* Minimum sequence length defined by the implementation.

### Postconditions

* Behavior label conforms to CBO.
* Confidence score is reported.

---

# 9. Event Generator Interface

### Responsibility

Aggregate behaviors into temporal events.

### Input

Sequence[Behavior]

### Output

List[BehaviorEvent]

### Contract

```python
generate(sequence: Sequence[Behavior]) -> list[BehaviorEvent]
```

### Preconditions

* Behaviors belong to the same track.
* Sequence is temporally ordered.

### Postconditions

* Events have valid start and end times.
* Durations are positive.

---

# 10. Flock Analytics Interface

### Responsibility

Compute flock-level indicators.

### Input

List[BehaviorEvent]

### Output

FlockAnalytics

### Contract

```python
compute(events: list[BehaviorEvent]) -> FlockAnalytics
```

### Preconditions

* Events are validated.
* Observation window is defined.

### Postconditions

* Aggregate metrics are internally consistent.
* Metadata identifies the observation period.

---

# 11. Mortality Predictor Interface

### Responsibility

Estimate future mortality risk.

### Input

FlockAnalytics

### Optional Input

Environmental data

### Output

MortalityPrediction

### Contract

```python
predict(
    analytics: FlockAnalytics,
    context: EnvironmentalContext | None = None
) -> MortalityPrediction
```

### Preconditions

* Analytics object is valid.
* Required predictive features are available.

### Postconditions

* Risk probability lies within [0,1].
* Prediction horizon is recorded.
* Model version is recorded.

---

# 12. Error Contract

Modules should classify failures consistently.

Suggested categories include:

* ValidationError
* DataFormatError
* ModelInferenceError
* ResourceError
* ConfigurationError
* UnsupportedFeatureError

Errors should be descriptive and traceable.

---

# 13. Compatibility Rules

An implementation is considered interface-compatible if:

* Function signatures remain unchanged.
* Canonical data models are preserved.
* Validation rules are respected.
* Required metadata is maintained.

---

# 14. Future Extensions

Future interface definitions may include:

* Streaming interfaces
* Asynchronous execution
* Batch processing
* Distributed inference
* Remote procedure calls (RPC)
* REST and gRPC APIs

---

# 15. Summary

Stable interfaces decouple software modules from algorithmic implementations.

This enables reproducible experimentation, interchangeable components, and long-term maintainability across the ChickenBehaviorLab ecosystem.
