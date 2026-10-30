import unittest

from chicken_behavior_lab.pipeline.contracts import (
    ContractValidationError,
    Keypoint,
    PoseEstimate,
)
from chicken_behavior_lab.pipeline.keypoint_schema import KeypointSchema
from chicken_behavior_lab.pipeline.skeleton_adapter import (
    PoseToSkeletonAdapter,
)


def make_schema():
    # Example schema only; replace with the actual 13-keypoint definition.
    return KeypointSchema(
        names=("point_a", "point_b", "point_c"),
        edges=(
            ("point_a", "point_b"),
            ("point_b", "point_c"),
        ),
    )


def make_pose():
    return PoseEstimate(
        track_id=7,
        frame_index=12,
        keypoints={
            # Deliberately not in canonical order.
            "point_c": Keypoint(30.0, 30.0, 0.9),
            "point_a": Keypoint(10.0, 10.0, 0.9),
            "point_b": Keypoint(20.0, 20.0, 0.9),
        },
    )


class PoseToSkeletonAdapterTests(unittest.TestCase):

    def test_preserves_canonical_node_order(self):
        adapter = PoseToSkeletonAdapter(make_schema())

        skeleton = adapter.adapt(make_pose())

        self.assertEqual(
            tuple(skeleton.nodes),
            ("point_a", "point_b", "point_c"),
        )

    def test_preserves_schema_edges(self):
        schema = make_schema()
        adapter = PoseToSkeletonAdapter(schema)

        skeleton = adapter.adapt(make_pose())

        self.assertEqual(skeleton.edges, schema.edges)

    def test_preserves_track_and_frame_ids(self):
        skeleton = PoseToSkeletonAdapter(make_schema()).adapt(
            make_pose()
        )

        self.assertEqual(skeleton.track_id, 7)
        self.assertEqual(skeleton.frame_index, 12)

    def test_preserves_invisible_keypoint(self):
        pose = PoseEstimate(
            track_id=7,
            frame_index=12,
            keypoints={
                "point_a": Keypoint(10.0, 10.0, 0.9),
                "point_b": Keypoint(
                    0.0, 0.0, 0.0, visible=False
                ),
                "point_c": Keypoint(30.0, 30.0, 0.9),
            },
        )

        skeleton = PoseToSkeletonAdapter(make_schema()).adapt(pose)

        self.assertIn("point_b", skeleton.nodes)
        self.assertFalse(skeleton.nodes["point_b"].visible)
        self.assertEqual(skeleton.nodes["point_b"].confidence, 0.0)

    def test_rejects_missing_schema_keypoint(self):
        pose = PoseEstimate(
            track_id=7,
            frame_index=12,
            keypoints={
                "point_a": Keypoint(10.0, 10.0, 0.9),
                "point_b": Keypoint(20.0, 20.0, 0.9),
            },
        )

        adapter = PoseToSkeletonAdapter(make_schema())

        with self.assertRaises(ContractValidationError):
            adapter.adapt(pose)

    def test_rejects_extra_keypoint(self):
        pose = PoseEstimate(
            track_id=7,
            frame_index=12,
            keypoints={
                "point_a": Keypoint(10.0, 10.0, 0.9),
                "point_b": Keypoint(20.0, 20.0, 0.9),
                "point_c": Keypoint(30.0, 30.0, 0.9),
                "unexpected": Keypoint(40.0, 40.0, 0.9),
            },
        )

        adapter = PoseToSkeletonAdapter(make_schema())

        with self.assertRaises(ContractValidationError):
            adapter.adapt(pose)


if __name__ == "__main__":
    unittest.main()
