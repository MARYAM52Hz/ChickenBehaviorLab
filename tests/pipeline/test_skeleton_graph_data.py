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
from chicken_behavior_lab.pipeline.skeleton_graph_data import (
    SkeletonGraphEncoder,
    SkeletonGraphData,
)


def make_schema():
    # Illustrative names only; replace with the actual project schema.
    return KeypointSchema(
        names=("point_a", "point_b", "point_c"),
        edges=(
            ("point_a", "point_b"),
            ("point_b", "point_c"),
        ),
    )


def make_skeleton():
    schema = make_schema()
    pose = PoseEstimate(
        track_id=4,
        frame_index=20,
        keypoints={
            "point_c": Keypoint(30.0, 30.0, 0.8),
            "point_a": Keypoint(10.0, 10.0, 0.9),
            "point_b": Keypoint(20.0, 20.0, 0.7),
        },
    )
    return schema, PoseToSkeletonAdapter(schema).adapt(pose)


class SkeletonGraphEncoderTests(unittest.TestCase):

    def test_node_features_follow_schema_order(self):
        schema, skeleton = make_skeleton()

        encoded = SkeletonGraphEncoder(schema).encode(skeleton)

        self.assertEqual(
            encoded.node_names,
            ("point_a", "point_b", "point_c"),
        )
        self.assertEqual(
            encoded.node_features[0],
            (10.0, 10.0, 0.9, 1.0),
        )

    def test_edges_are_converted_to_integer_indices(self):
        schema, skeleton = make_skeleton()

        encoded = SkeletonGraphEncoder(schema).encode(skeleton)

        self.assertEqual(
            encoded.edge_index,
            ((0, 1), (1, 2)),
        )

    def test_track_and_frame_ids_are_preserved(self):
        schema, skeleton = make_skeleton()

        encoded = SkeletonGraphEncoder(schema).encode(skeleton)

        self.assertEqual(encoded.track_id, 4)
        self.assertEqual(encoded.frame_index, 20)

    def test_invisible_keypoint_keeps_visibility_mask(self):
        schema = make_schema()
        pose = PoseEstimate(
            track_id=4,
            frame_index=20,
            keypoints={
                "point_a": Keypoint(10.0, 10.0, 0.9),
                "point_b": Keypoint(
                    0.0, 0.0, 0.0, visible=False
                ),
                "point_c": Keypoint(30.0, 30.0, 0.8),
            },
        )
        skeleton = PoseToSkeletonAdapter(schema).adapt(pose)

        encoded = SkeletonGraphEncoder(schema).encode(skeleton)

        self.assertEqual(
            encoded.node_features[1],
            (0.0, 0.0, 0.0, 0.0),
        )

    def test_rejects_skeleton_with_different_edges(self):
        schema, skeleton = make_skeleton()

        altered = type(skeleton)(
            track_id=skeleton.track_id,
            frame_index=skeleton.frame_index,
            nodes=skeleton.nodes,
            edges=(("point_a", "point_c"),),
        )

        with self.assertRaises(ContractValidationError):
            SkeletonGraphEncoder(schema).encode(altered)

    def test_rejects_invalid_edge_indices(self):
        with self.assertRaises(ContractValidationError):
            SkeletonGraphData(
                track_id=1,
                frame_index=0,
                node_names=("point_a", "point_b"),
                node_features=(
                    (1.0, 2.0, 0.9, 1.0),
                    (3.0, 4.0, 0.8, 1.0),
                ),
                edge_index=((0,), (2,)),
            )


if __name__ == "__main__":
    unittest.main()
