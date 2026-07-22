# Data Models

## Overview

This directory defines the canonical data models used throughout the ChickenBehaviorLab framework.

Each data model represents a core entity exchanged between modules in the computer vision and behavior analysis pipeline.

The purpose of these models is to provide a standardized representation of information independent of any specific implementation.

These specifications act as the contract between software modules and ensure consistency across datasets, algorithms, experiments, and future extensions.

---

## Design Principles

All data models should:

* Represent a single logical entity.
* Be implementation-independent.
* Support serialization.
* Be versioned when modified.
* Be compatible with CBAS and CBO standards.
* Be extensible without breaking backward compatibility.

---

## Data Model Hierarchy

```text
Video
   ↓
Frame
   ↓
Detection
   ↓
Track
   ↓
Pose
   ↓
Skeleton
   ↓
Behavior
   ↓
Behavior Event
   ↓
Flock Statistics
   ↓
Mortality Prediction
```

Each stage consumes one or more data models and produces new standardized outputs for the next stage.

---

## Relationship with the Pipeline

These models define the information exchanged between:

* Detection
* Tracking
* Pose Estimation
* Skeleton Processing
* Behavior Recognition
* Event Generation
* Flock Analytics
* Mortality Prediction

The software implementation should follow these specifications as closely as possible.
