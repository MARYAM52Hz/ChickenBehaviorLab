
"""Integration tests for the pose-to-graph processing path."""

import unittest

from chicken_behavior_lab.pipeline.contracts import (
    ContractValidationError,
    Keypoint,
)
from chicken_behavior_lab.pipeline.keypoint_schema import KeypointSchema
from chicken_behavior_lab.pipeline.pose_adapter import PoseAdapter
from chicken_behavior_lab.pipeline.skeleton_adapter import (
    PoseToSkeletonAdapter,
)
from chicken_behavior_lab.pipeline.skeleton_graph_data import (
    SkeletonGraphEncoder,
)


def make_schema():
    """Create a small test schema, not the production chicken schema."""
    return KeypointSchema(
        names=("point_a", "point_b", "point_c"),
        edges=(
            ("point_a", "point_b"),
            ("point_b", "point_c"),
        ),
    )


class PoseGraphIntegrationTests(unittest.TestCase):

    def setUp(self):
        self.schema = make_schema()
        self.pose_adapter = PoseAdapter(
            self.schema,
            confidence_threshold=0.5,
            missing_policy="mask",
        )
        self.skeleton_adapter = PoseToSkeletonAdapter(self.schema)
        self.graph_encoder = SkeletonGraphEncoder(self.schema)

    def test_valid_pose_reaches_graph_encoder(self):
        pose = self.pose_adapter.adapt(
            track_id=7,
            frame_index=12,
            keypoints={
                # Deliberately shuffled to verify canonical ordering.
                "point_c": Keypoint(30.0, 30.0, 0.95),
                "point_a": Keypoint(10.0, 10.0, 0.90),
                "point_b": Keypoint(20.0, 20.0, 0.85),
            },
        )

        skeleton = self.skeleton_adapter.adapt(pose)
        graph = self.graph_encoder.encode(skeleton)

        self.assertEqual(graph.track_id, "track_7")
        self.assertEqual(graph.frame_index, 12)
        self.assertEqual(
            graph.node_names,
            ("point_a", "point_b", "point_c"),
        )
        self.assertEqual(
            graph.node_features,
            (
                (10.0, 10.0, 0.90, 1.0),
                (20.0, 20.0, 0.85, 1.0),
                (30.0, 30.0, 0.95, 1.0),
            ),
        )
        self.assertEqual(
            graph.edge_index,
            ((0, 1), (1, 2)),
        )

    def test_missing_keypoint_is_masked_end_to_end(self):
        pose = self.pose_adapter.adapt(
            track_id=3,
            frame_index=25,
            keypoints={
                "point_a": Keypoint(10.0, 10.0, 0.9),
                "point_b": None,
                "point_c": Keypoint(30.0, 30.0, 0.9),
            },
        )

        skeleton = self.skeleton_adapter.adapt(pose)
        graph = self.graph_encoder.encode(skeleton)

        self.assertEqual(len(graph.node_features), 3)
        self.assertEqual(
            graph.node_features[1],
            (0.0, 0.0, 0.0, 0.0),
        )

    def test_low_confidence_keypoint_is_masked_end_to_end(self):
        pose = self.pose_adapter.adapt(
            track_id=3,
            frame_index=26,
            keypoints={
                "point_a": Keypoint(10.0, 10.0, 0.9),
                "point_b": Keypoint(20.0, 20.0, 0.2),
                "point_c": Keypoint(30.0, 30.0, 0.9),
            },
        )

        skeleton = self.skeleton_adapter.adapt(pose)
        graph = self.graph_encoder.encode(skeleton)

        # The coordinates and confidence remain available, but visibility
        # is zero. Downstream features must respect this mask.
        self.assertEqual(
            graph.node_features[1],
            (20.0, 20.0, 0.2, 0.0),
        )

    def test_unknown_keypoint_is_rejected(self):
        with self.assertRaises(ContractValidationError):
            self.pose_adapter.adapt(
                track_id=1,
                frame_index=0,
                keypoints={
                    "point_a": Keypoint(10.0, 10.0, 0.9),
                    "point_b": Keypoint(20.0, 20.0, 0.9),
                    "point_c": Keypoint(30.0, 30.0, 0.9),
                    "unexpected_point": Keypoint(40.0, 40.0, 0.9),
                },
            )

    def test_missing_keypoint_is_rejected_in_strict_mode(self):
        strict_adapter = PoseAdapter(
            self.schema,
            confidence_threshold=0.5,
            missing_policy="raise",
        )

        with self.assertRaises(ContractValidationError):
            strict_adapter.adapt(
                track_id=1,
                frame_index=0,
                keypoints={
                    "point_a": Keypoint(10.0, 10.0, 0.9),
                    "point_b": None,
                    "point_c": Keypoint(30.0, 30.0, 0.9),
                },
            )


if __name__ == "__main__":
    unittest.main()
