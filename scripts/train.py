from __future__ import annotations

import torch

from torch_geometric.loader import (
    DataLoader,
)

from chicken_behavior_lab.dataset import (
    PyGGraphDataset,
    train_validation_split,
)

from chicken_behavior_lab.models import (
    ChickenBehaviorGNN,
)

from chicken_behavior_lab.training import (
    TrainingConfig,
    Trainer,
)


def choose_device() -> torch.device:
    """
    Select best available training device.
    """

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    if (
        hasattr(
            torch.backends,
            "mps",
        )
        and torch.backends.mps.is_available()
    ):
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


def main() -> None:

    config = TrainingConfig()

    config.validate()

    device = choose_device()

    print(
        f"Using device: {device}"
    )

    # =====================================================
    # Dataset
    # =====================================================
    #
    # In the next integration step this will be replaced
    # by:
    #
    # AnnotationLoader
    #       ↓
    # FeatureStore
    #       ↓
    # DatasetBuilder
    #       ↓
    # GraphDataset
    #       ↓
    # PyGGraphDataset
    #
    # For now, the training script expects samples to
    # already have been prepared.
    # =====================================================

    raise RuntimeError(
        "Dataset construction has not yet been "
        "connected to the training entry point. "
        "Use the end-to-end dataset pipeline "
        "in the next integration step."
    )


if __name__ == "__main__":
    main()
