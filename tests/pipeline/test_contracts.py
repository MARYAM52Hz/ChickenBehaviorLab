
import unittest

from chicken_behavior_lab.pipeline.contracts import (
    ContractValidationError,
    Detection,
    Keypoint,
    PoseEstimate,
    SkeletonFrame,
    TrackObservation,
    validate_pose_matches_track,
    validate_skeleton_matches_pose,
)


def make_detection(frame_index=5):
    return Detection(
        frame_index=frame_index,
        bbox_xyxy=(10.0, 20.0, 80.0, 100.0),
        confidence=0.95,
        class_id=0,
        class_name="chicken",
    )


def make_pose(track_id=3, frame_index=5):
    return PoseEstimate(
        track_id=track_id,
        frame_index=frame_index,
        keypoints={
            "head": Keypoint(30.0, 35.0, 0.98),
            "left_foot": Keypoint(40.0, 90.0, 0.85),
        },
    )


def make_skeleton(track_id=3, frame_index=5):
    return SkeletonFrame(
        track_id=track_id,
        frame_index=frame_index,
        nodes={
            "head": Keypoint(30.0, 35.0, 0.98),
            "left_foot": Keypoint(40.0, 90.0, 0.85),
        },
        edges=(("head", "left_foot"),),
    )


class PipelineContractTests(unittest.TestCase):

    def test_valid_detection_is_accepted(self):
        detection = make_detection()

        self.assertEqual(detection.frame_index, 5)
        self.assertEqual(
            detection.bbox_xyxy,
            (10.0, 20.0, 80.0, 100.0),
        )

    def test_invalid_bounding_box_is_rejected(self):
        with self.assertRaises(ContractValidationError):
            Detection(
                frame_index=0,
                bbox_xyxy=(20.0, 20.0, 10.0, 40.0),
                confidence=0.9,
                class_id=0,
            )

    def test_invalid_confidence_is_rejected(self):
        with self.assertRaises(ContractValidationError):
            Detection(
                frame_index=0,
                bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
                confidence=1.5,
                class_id=0,
            )

    def test_pose_must_match_track_and_frame(self):
        track = TrackObservation(
            track_id=3,
            detection=make_detection(frame_index=5),
        )
        pose = make_pose(track_id=3, frame_index=5)

        validate_pose_matches_track(track, pose)

        wrong_track = make_pose(track_id=4, frame_index=5)

        with self.assertRaises(ContractValidationError):
            validate_pose_matches_track(track, wrong_track)

        wrong_frame = make_pose(track_id=3, frame_index=6)

        with self.assertRaises(ContractValidationError):
            validate_pose_matches_track(track, wrong_frame)

    def test_skeleton_edges_must_reference_existing_nodes(self):
        with self.assertRaises(ContractValidationError):
            SkeletonFrame(
                track_id=3,
                frame_index=5,
                nodes={
                    "head": Keypoint(10.0, 10.0, 0.9),
                },
                edges=(("head", "missing_node"),),
            )

    def test_skeleton_must_match_pose(self):
        pose = make_pose()
        skeleton = make_skeleton()

        validate_skeleton_matches_pose(pose, skeleton)

        wrong_skeleton = make_skeleton(track_id=9)

        with self.assertRaises(ContractValidationError):
            validate_skeleton_matches_pose(pose, wrong_skeleton)

    def test_pose_mapping_cannot_be_modified_after_creation(self):
        pose = make_pose()

        with self.assertRaises(TypeError):
            pose.keypoints["new_node"] = Keypoint(
                1.0,
                2.0,
                0.5,
            )


if __name__ == "__main__":
    unittest.main()
