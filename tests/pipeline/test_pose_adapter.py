import unittest

from chicken_behavior_lab.pipeline.contracts import (
    ContractValidationError,
    Keypoint,
)
from chicken_behavior_lab.pipeline.keypoint_schema import KeypointSchema
from chicken_behavior_lab.pipeline.pose_adapter import PoseAdapter


def make_schema():
    # Example names only; replace with the project's actual 13 keypoints.
    return KeypointSchema(
        names=("point_a", "point_b", "point_c"),
        edges=(
            ("point_a", "point_b"),
            ("point_b", "point_c"),
        ),
    )


class PoseAdapterTests(unittest.TestCase):

    def test_keypoints_follow_canonical_order(self):
        schema = make_schema()
        adapter = PoseAdapter(schema)

        # Input order deliberately differs from schema order.
        pose = adapter.adapt(
            track_id=2,
            frame_index=10,
            keypoints={
                "point_c": Keypoint(30, 30, 0.9),
                "point_a": Keypoint(10, 10, 0.9),
                "point_b": Keypoint(20, 20, 0.9),
            },
        )

        self.assertEqual(
            tuple(pose.keypoints),
            ("point_a", "point_b", "point_c"),
        )

    def test_missing_keypoint_raises_by_default(self):
        adapter = PoseAdapter(make_schema())

        with self.assertRaises(ContractValidationError):
            adapter.adapt(
                track_id=2,
                frame_index=10,
                keypoints={
                    "point_a": Keypoint(10, 10, 0.9),
                    "point_b": Keypoint(20, 20, 0.9),
                },
            )

    def test_missing_keypoint_can_be_masked(self):
        adapter = PoseAdapter(
            make_schema(),
            missing_policy="mask",
        )

        pose = adapter.adapt(
            track_id=2,
            frame_index=10,
            keypoints={
                "point_a": Keypoint(10, 10, 0.9),
                "point_b": None,
                "point_c": Keypoint(30, 30, 0.9),
            },
        )

        missing = pose.keypoints["point_b"]

        self.assertFalse(missing.visible)
        self.assertEqual(missing.confidence, 0.0)
        self.assertEqual((missing.x, missing.y), (0.0, 0.0))

    def test_low_confidence_point_is_marked_invisible(self):
        adapter = PoseAdapter(
            make_schema(),
            confidence_threshold=0.5,
        )

        pose = adapter.adapt(
            track_id=2,
            frame_index=10,
            keypoints={
                "point_a": Keypoint(10, 10, 0.9),
                "point_b": Keypoint(20, 20, 0.2),
                "point_c": Keypoint(30, 30, 0.9),
            },
        )

        self.assertFalse(pose.keypoints["point_b"].visible)
        self.assertEqual(
            pose.keypoints["point_b"].confidence,
            0.2,
        )

    def test_unexpected_keypoint_is_rejected(self):
        adapter = PoseAdapter(make_schema())

        with self.assertRaises(ContractValidationError):
            adapter.adapt(
                track_id=2,
                frame_index=10,
                keypoints={
                    "point_a": Keypoint(10, 10, 0.9),
                    "point_b": Keypoint(20, 20, 0.9),
                    "point_c": Keypoint(30, 30, 0.9),
                    "unexpected": Keypoint(40, 40, 0.9),
                },
            )

    def test_invalid_confidence_threshold_is_rejected(self):
        with self.assertRaises(ContractValidationError):
            PoseAdapter(
                make_schema(),
                confidence_threshold=1.5,
            )

    def test_schema_rejects_unknown_edge_endpoint(self):
        with self.assertRaises(ContractValidationError):
            KeypointSchema(
                names=("point_a", "point_b"),
                edges=(("point_a", "not_defined"),),
            )


if __name__ == "__main__":
    unittest.main()
