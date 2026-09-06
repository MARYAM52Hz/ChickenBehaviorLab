from __future__ import annotations

import numpy as np


class TemporalEdgeBuilder:
    """
    Build temporal connections between identical keypoints
    in consecutive frames.
    """

    def __init__(
        self,
        bidirectional: bool = True,
    ) -> None:

        self.bidirectional = (
            bidirectional
        )

    def build(
        self,
        num_frames: int,
        num_keypoints: int,
    ) -> np.ndarray:
        """
        Create temporal graph edges.

        For every keypoint v:

            (t, v) -> (t + 1, v)

        If bidirectional=True:

            (t + 1, v) -> (t, v)

        is also added.

        Returns
        -------
        np.ndarray
            Shape:

                (2, E_temporal)
        """

        if num_frames <= 0:
            raise ValueError(
                "num_frames must be positive."
            )

        if num_keypoints <= 0:
            raise ValueError(
                "num_keypoints must be positive."
            )

        if num_frames == 1:
            return np.empty(
                (2, 0),
                dtype=np.int64,
            )

        sources: list[int] = []
        targets: list[int] = []

        for frame_index in range(
            num_frames - 1
        ):

            current_offset = (
                frame_index
                * num_keypoints
            )

            next_offset = (
                (frame_index + 1)
                * num_keypoints
            )

            for keypoint_index in range(
                num_keypoints
            ):

                current_node = (
                    current_offset
                    + keypoint_index
                )

                next_node = (
                    next_offset
                    + keypoint_index
                )

                sources.append(
                    current_node
                )

                targets.append(
                    next_node
                )

                if self.bidirectional:
                    sources.append(
                        next_node
                    )

                    targets.append(
                        current_node
                    )

        return np.asarray(
            [
                sources,
                targets,
            ],
            dtype=np.int64,
        )
