# Dataset Documentation

## Overview

The ChickenBehaviorLab dataset is designed for skeleton-based chicken behavior analysis in commercial poultry farms.

The dataset supports multiple research tasks, including:

- Chicken detection
- Multi-object tracking
- Pose estimation
- Skeleton extraction
- Behavior recognition
- Abnormal behavior detection
- Flock-level behavioral analysis
- Mortality risk prediction

---

# Data Sources

The dataset may include videos collected from:

- Commercial poultry farms
- Fixed surveillance cameras
- Overhead cameras
- Side-view cameras
- Public benchmark datasets
- Custom annotated datasets

---

# Dataset Organization

datasets/

    raw/
    processed/
    annotations/
    skeletons/
    behaviors/
    metadata/

---

# Raw Videos

Raw videos are stored without modification.

Examples:

raw/

    farm01/
    farm02/
    farm03/

---

# Processed Videos

Processed videos may include:

- resized videos
- stabilized videos
- cropped videos
- denoised videos

---

# Annotation Types

The framework supports several annotation levels.

## Detection

Bounding boxes

## Tracking

Unique Chicken IDs

## Pose Estimation

Skeleton keypoints

## Behavior Labels

Behavior classes

## Anomaly Labels

Abnormal behavior annotations

## Mortality Labels

Farm-level mortality records

---

# Skeleton Format

Each chicken skeleton consists of predefined anatomical keypoints.

Example:

- Head
- Beak
- Neck
- Body Center
- Left Wing
- Right Wing
- Tail
- Left Leg
- Right Leg
- Left Foot
- Right Foot

---

# Supported Behavior Classes

Examples include:

- Walking
- Running
- Standing
- Sitting
- Eating
- Drinking
- Pecking
- Preening
- Wing Flapping
- Dust Bathing
- Aggression
- Crowding
- Isolation
- Lameness

The list will expand over time.

---

# Dataset Split

The framework supports:

- Training
- Validation
- Testing

The splitting strategy depends on the experimental protocol.

---

# Metadata

Metadata may include:

- Farm ID
- Camera ID
- Timestamp
- Lighting condition
- Bird age
- Breed
- Environmental conditions

---

# Mortality Records

Mortality records are stored separately from behavioral annotations.

Typical information includes:

- Farm
- Date
- Number of deaths
- Age of flock

These records are used for mortality prediction and validation.

---

# Future Extensions

Future versions of the dataset may include:

- Thermal images
- Depth cameras
- Audio recordings
- Environmental sensors
- Feed consumption
- Water consumption
- CO₂ concentration
- Ammonia level
- Temperature
- Humidity
