from __future__ import annotations

import numpy as np


class SpatialEdgeExpander:
    """
    Expand skeleton connectivity across all frames.
    """

    def build(
        self,
        skeleton_edge_index: np.ndarray,
        num_frames: int,
        num_keypoints: int,
    ) -> np.ndarray:
        """
        Repeat spatial skeleton edges for each frame.
        """

        skeleton_edge_index = np.asarray(
            skeleton_edge_index,
            dtype=np.int64,
        )

        if skeleton_edge_index.ndim != 2:
            raise ValueError(
                "skeleton_edge_index must have "
                "shape (2, E)."
            )

        if skeleton_edge_index.shape[0] != 2:
            raise ValueError(
                "skeleton_edge_index must have "
                "shape (2, E)."
            )

        if num_frames <= 0:
            raise ValueError(
                "num_frames must be positive."
            )

        if num_keypoints <= 0:
            raise ValueError(
                "num_keypoints must be positive."
            )

        if skeleton_edge_index.size > 0:

            if np.any(
                skeleton_edge_index < 0
            ):
                raise ValueError(
                    "Skeleton edge indices "
                    "cannot be negative."
                )

            if np.any(
                skeleton_edge_index
                >= num_keypoints
            ):
                raise ValueError(
                    "Skeleton edge index exceeds "
                    "number of keypoints."
                )

        expanded_edges: list[
            np.ndarray
        ] = []

        for frame_index in range(
            num_frames
        ):

            offset = (
                frame_index
                * num_keypoints
            )

            frame_edges = (
                skeleton_edge_index
                + offset
            )

            expanded_edges.append(
                frame_edges
            )

        if not expanded_edges:
            return np.empty(
                (2, 0),
                dtype=np.int64,
            )

        return np.concatenate(
            expanded_edges,
            axis=1,
        )
