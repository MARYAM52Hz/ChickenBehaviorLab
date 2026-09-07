from __future__ import annotations

import torch
from torch import Tensor
from torch import nn


class BehaviorClassificationLoss:
    """
    Cross-entropy loss for behavior classification.
    """

    def __init__(
        self,
        class_weights: Tensor | None = None,
    ) -> None:

        self.loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
        )

    def __call__(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:

        if logits.ndim != 2:
            raise ValueError(
                "logits must have shape "
                "(B, C)."
            )

        if targets.ndim != 1:
            raise ValueError(
                "targets must have shape "
                "(B,)."
            )

        return self.loss_fn(
            logits,
            targets,
        )
