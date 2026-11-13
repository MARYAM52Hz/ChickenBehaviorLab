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

The framework supports training, validation, and testing splits. For behavior
learning, related samples must remain in the same split:

- Video-level splitting is the default and keeps all samples from one source
  video together.
- Track-level splitting is optional and uses the pair (video_id, track_id),
  because track IDs may restart in each video.
- A sample-level random split is only suitable when samples are genuinely
  independent and do not share source videos or tracks.

The baseline scripts/train.py uses a group-aware train/validation split.
The validation fraction is applied to the number of groups, not the number of
samples. If a behavior class appears in validation but not in training, the
script stops with an explicit error rather than reporting an invalid score.
Stratified group assignment is not implemented yet; rare classes may require
a different seed or additional source groups.

## Graph dataset adapters

PyGGraphDataset preserves the temporal dimension of graph samples and is
kept for existing temporal consumers. FrameGraphDataset is a separate
adapter for the spatial ChickenBehaviorGNN baseline: it accepts exactly one
frame per sample and exposes node and edge features without the singleton
temporal dimension. Multi-frame samples must use the temporal pipeline rather
than being silently reduced to one frame.

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
