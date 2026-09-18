from __future__ import annotations

from typing import Any

import numpy as np


class GraphFeatureDiagnosticExtractor:
    """
    Extract diagnostic statistics from a temporal skeleton graph.

    The extractor is intentionally defensive because different
    graph representations may expose features with different
    temporal layouts.
    """

    def extract(
        self,
        graph: Any,
    ) -> dict[str, float | None]:
        graph.validate()

        node_features = np.asarray(
            graph.node_features,
            dtype=np.float32,
        )

        edge_features = getattr(
            graph,
            "edge_features",
            None,
        )

        if edge_features is not None:
            edge_features = np.asarray(
                edge_features,
                dtype=np.float32,
            )

        return {
            "mean_motion_energy": (
                self._extract_motion_energy(
                    graph,
                    node_features,
                )
            ),
            "mean_node_feature_magnitude": (
                self._mean_magnitude(
                    node_features
                )
            ),
            "mean_edge_feature_magnitude": (
                self._mean_magnitude(
                    edge_features
                )
                if edge_features is not None
                else 0.0
            ),
            "valid_keypoint_ratio": (
                self._extract_valid_keypoint_ratio(
                    graph,
                    node_features,
                )
            ),
            "valid_node_ratio": (
                self._extract_valid_node_ratio(
                    node_features
                )
            ),
            "mean_velocity": (
                self._extract_optional_scalar(
                    graph,
                    "velocity",
                )
            ),
            "mean_acceleration": (
                self._extract_optional_scalar(
                    graph,
                    "acceleration",
                )
            ),
            "mean_joint_angle": (
                self._extract_optional_scalar(
                    graph,
                    "joint_angles",
                )
            ),
            "std_joint_angle": (
                self._extract_optional_std(
                    graph,
                    "joint_angles",
                )
            ),
            "mean_body_orientation": (
                self._extract_optional_scalar(
                    graph,
                    "body_orientation",
                )
            ),
            "std_body_orientation": (
                self._extract_optional_std(
                    graph,
                    "body_orientation",
                )
            ),
        }

    @staticmethod
    def _mean_magnitude(
        values: np.ndarray | None,
    ) -> float:
        if values is None:
            return 0.0

        if values.size == 0:
            return 0.0

        finite_values = values[
            np.isfinite(values)
        ]

        if finite_values.size == 0:
            return 0.0

        if finite_values.ndim == 1:
            return float(
                np.mean(
                    np.abs(
                        finite_values
                    )
                )
            )

        flattened = finite_values.reshape(
            finite_values.shape[0],
            -1,
        )

        magnitudes = np.linalg.norm(
            flattened,
            axis=-1,
        )

        return float(
            np.mean(magnitudes)
        )

    def _extract_motion_energy(
        self,
        graph: Any,
        node_features: np.ndarray,
    ) -> float:
        motion_energy = getattr(
            graph,
            "motion_energy",
            None,
        )

        if motion_energy is not None:
            return self._scalar_mean(
                motion_energy
            )

        velocity = getattr(
            graph,
            "velocity",
            None,
        )

        if velocity is not None:
            velocity_array = np.asarray(
                velocity,
                dtype=np.float32,
            )

            if velocity_array.size == 0:
                return 0.0

            if velocity_array.ndim == 1:
                return float(
                    np.mean(
                        np.abs(
                            velocity_array
                        )
                    )
                )

            flattened = velocity_array.reshape(
                velocity_array.shape[0],
                -1,
            )

            magnitudes = np.linalg.norm(
                flattened,
                axis=-1,
            )

            return float(
                np.mean(
                    magnitudes ** 2
                )
            )

        # Fallback diagnostic:
        # estimate variation across the first axis.
        if node_features.ndim >= 3:
            temporal_difference = np.diff(
                node_features,
                axis=0,
            )

            if temporal_difference.size:
                return float(
                    np.mean(
                        temporal_difference
                        ** 2
                    )
                )

        return 0.0

    @staticmethod
    def _extract_valid_keypoint_ratio(
        graph: Any,
        node_features: np.ndarray,
    ) -> float:
        visibility = getattr(
            graph,
            "visibility",
            None,
        )

        if visibility is not None:
            visibility_array = np.asarray(
                visibility,
                dtype=np.float32,
            )

            if visibility_array.size == 0:
                return 0.0

            finite = visibility_array[
                np.isfinite(
                    visibility_array
                )
            ]

            if finite.size == 0:
                return 0.0

            return float(
                np.mean(
                    finite > 0
                )
            )

        # Try NaN-based missing-keypoint detection.
        if np.issubdtype(
            node_features.dtype,
            np.floating,
        ):
            return float(
                np.mean(
                    np.isfinite(
                        node_features
                    )
                )
            )

        return 1.0

    @staticmethod
    def _extract_valid_node_ratio(
        node_features: np.ndarray,
    ) -> float:
        if node_features.size == 0:
            return 0.0

        if node_features.ndim == 1:
            return float(
                np.mean(
                    np.isfinite(
                        node_features
                    )
                )
            )

        if node_features.ndim == 2:
            valid_nodes = np.all(
                np.isfinite(
                    node_features
                ),
                axis=-1,
            )

            return float(
                np.mean(valid_nodes)
            )

        valid_nodes = np.all(
            np.isfinite(
                node_features
            ),
            axis=-1,
        )

        return float(
            np.mean(valid_nodes)
        )

    @staticmethod
    def _scalar_mean(
        values: Any,
    ) -> float:
        array = np.asarray(
            values,
            dtype=np.float32,
        )

        if array.size == 0:
            return 0.0

        finite = array[
            np.isfinite(array)
        ]

        if finite.size == 0:
            return 0.0

        return float(
            np.mean(
                np.abs(finite)
            )
        )

    @staticmethod
    def _extract_optional_scalar(
        graph: Any,
        attribute_name: str,
    ) -> float | None:
        values = getattr(
            graph,
            attribute_name,
            None,
        )

        if values is None:
            return None

        return GraphFeatureDiagnosticExtractor._scalar_mean(
            values
        )

    @staticmethod
    def _extract_optional_std(
        graph: Any,
        attribute_name: str,
    ) -> float | None:
        values = getattr(
            graph,
            attribute_name,
            None,
        )

        if values is None:
            return None

        array = np.asarray(
            values,
            dtype=np.float32,
        )

        if array.size == 0:
            return None

        finite = array[
            np.isfinite(array)
        ]

        if finite.size == 0:
            return None

        return float(
            np.std(finite)
        )
