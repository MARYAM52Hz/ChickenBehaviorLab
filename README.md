# ChickenBehaviorLab

An open-source research framework for skeleton-based chicken behavior analysis using computer vision, graph neural networks, and temporal deep learning.

## Overview

ChickenBehaviorLab is designed to transform raw poultry farm videos into meaningful behavioral insights. The framework integrates chicken detection, multi-object tracking, pose estimation, skeleton-based behavior recognition, anomaly detection, and mortality risk prediction into a unified modular pipeline.

## Key Features

- Skeleton-based chicken pose estimation
- Multi-object tracking
- Graph neural network-based behavior recognition
- Abnormal behavior detection
- Flock-level behavioral analytics
- Mortality risk prediction
- Modular research framework
- Reproducible experiments

## Project Goals

- Build a complete skeleton-based behavior analysis pipeline.
- Support multiple graph neural network architectures.
- Benchmark state-of-the-art behavior recognition methods.
- Detect abnormal behaviors automatically.
- Predict mortality risk using behavioral statistics.
- Provide an extensible framework for precision poultry farming research.

## Planned Pipeline

Video
→ Detection
→ Tracking
→ Pose Estimation
→ Skeleton Generation
→ Graph Representation
→ Behavior Recognition
→ Anomaly Detection
→ Mortality Prediction

## Installation

ChickenBehaviorLab requires Python 3.11 or newer.

Install the core package:

```bash
python -m pip install -e .
```

Optional dependencies are grouped by capability. For example:

```bash
python -m pip install -e ".[vision,analysis]"
```

See [Installation Guide](docs/installation.md) for the available extras and PyTorch setup.

## Repository Structure

- `src/chicken_behavior_lab/`: framework source code
- `tests/`: automated tests
- `docs/`: project and workflow documentation
- `design/`: architecture and research design documents

## Tests

```bash
python -m pytest -q tests
```

## License

MIT License
