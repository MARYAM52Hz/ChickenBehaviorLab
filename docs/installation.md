# Installation

ChickenBehaviorLab requires Python 3.11 or newer.

## Install the core package

From the repository root:

```bash
python -m pip install -e .
```

The core installation includes NumPy and the package metadata. Optional dependencies are separated by workflow so a user does not need to install the full computer-vision and deep-learning stack for every task.

## Optional capabilities

Install only the extras required by the modules you plan to use:

```bash
python -m pip install -e ".[vision]"
python -m pip install -e ".[learning]"
python -m pip install -e ".[analysis]"
python -m pip install -e ".[dev]"
```

- `vision`: OpenCV and Pillow for image/video workflows.
- `learning`: PyTorch and PyTorch Geometric for neural-network and graph-learning workflows.
- `analysis`: SciPy, scikit-learn, pandas, matplotlib, and PyYAML for analysis and configuration workflows.
- `dev`: pytest for running tests.

For a CUDA-specific setup, install the PyTorch build that matches the local hardware and driver first, then install the `learning` extra. The CPU-only command used by CI is:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[learning]"
```

## Run tests

Install the optional dependencies needed by the test modules, then run:

```bash
python -m pytest -q tests
```

The full CI workflow installs the project in editable mode and runs the test directory without relying on `PYTHONPATH=src`. This checks that package discovery and imports work through the installation configuration.
