
"""End-to-end orchestration tests using lightweight fake components."""

import unittest

from chicken_behavior_lab.pipeline import (
    ChickenBehaviorPipeline,
    PipelineStage,
    PipelineStageError,
)


class FakeDetector:
    def detect(self, video):
        return [{"frame": 1, "bbox": [0, 0, 10, 10]}]


class FakeTracker:
    def track(self, detections):
        return [{"track_id": 7, **detections[0]}]


class FakePoseEstimator:
    def predict(self, tracks):
        return [
            {
                "track_id": tracks[0]["track_id"],
                "keypoints": [(2, 3)],
            }
        ]


class FakeSkeletonBuilder:
    def build(self, poses):
        return [
            {
                "track_id": poses[0]["track_id"],
                "nodes": ["head"],
            }
        ]


class FakeBehaviorRecognizer:
    def predict(self, skeletons):
        return [
            {
                "track_id": skeletons[0]["track_id"],
                "label": "pecking",
            }
        ]


class FakeEventGenerator:
    def generate(self, behaviors):
        return [
            {
                "track_id": behaviors[0]["track_id"],
                "event": "pecking_started",
            }
        ]


class FakeAnalytics:
    def compute(self, events):
        return {
            "event_count": len(events),
            "event_types": [event["event"] for event in events],
        }


class EndToEndPipelineTests(unittest.TestCase):

    def setUp(self):
        detector = FakeDetector()
        tracker = FakeTracker()
        pose_estimator = FakePoseEstimator()
        skeleton_builder = FakeSkeletonBuilder()
        behavior_recognizer = FakeBehaviorRecognizer()
        event_generator = FakeEventGenerator()
        analytics = FakeAnalytics()

        self.pipeline = ChickenBehaviorPipeline([
            PipelineStage(
                "detection",
                lambda c: detector.detect(c.video),
            ),
            PipelineStage(
                "tracking",
                lambda c: tracker.track(
                    c.get_output("detection")
                ),
                requires=("detection",),
            ),
            PipelineStage(
                "pose",
                lambda c: pose_estimator.predict(
                    c.get_output("tracking")
                ),
                requires=("tracking",),
            ),
            PipelineStage(
                "skeleton",
                lambda c: skeleton_builder.build(
                    c.get_output("pose")
                ),
                requires=("pose",),
            ),
            PipelineStage(
                "behavior",
                lambda c: behavior_recognizer.predict(
                    c.get_output("skeleton")
                ),
                requires=("skeleton",),
            ),
            PipelineStage(
                "events",
                lambda c: event_generator.generate(
                    c.get_output("behavior")
                ),
                requires=("behavior",),
            ),
            PipelineStage(
                "analytics",
                lambda c: analytics.compute(
                    c.get_output("events")
                ),
                requires=("events",),
            ),
        ])

    def test_full_pipeline_produces_expected_analytics(self):
        result = self.pipeline.run(
            "synthetic-video.mp4",
            metadata={"video_id": "video-001"},
        )

        self.assertEqual(
            result.get("analytics"),
            {
                "event_count": 1,
                "event_types": ["pecking_started"],
            },
        )

        self.assertEqual(
            result.get("behavior")[0]["label"],
            "pecking",
        )

        self.assertEqual(
            result.get("tracking")[0]["track_id"],
            7,
        )

        self.assertEqual(
            tuple(result.outputs),
            self.pipeline.stage_names,
        )

        self.assertEqual(
            tuple(result.stage_durations_seconds),
            self.pipeline.stage_names,
        )

    def test_metadata_is_available_to_each_stage(self):
        seen = []

        pipeline = ChickenBehaviorPipeline([
            PipelineStage(
                "capture_metadata",
                lambda context: (
                    seen.append(context.metadata["video_id"])
                    or "ok"
                ),
            ),
        ])

        pipeline.run(
            "video.mp4",
            metadata={"video_id": "v-42"},
        )

        self.assertEqual(seen, ["v-42"])

    def test_failure_stops_downstream_stages(self):
        called = []

        def fail(context):
            called.append("failed")
            raise RuntimeError("synthetic component failure")

        pipeline = ChickenBehaviorPipeline([
            PipelineStage("first", fail),
            PipelineStage(
                "second",
                lambda context: called.append("second"),
            ),
        ])

        with self.assertRaises(PipelineStageError):
            pipeline.run("video.mp4")

        self.assertEqual(called, ["failed"])


if __name__ == "__main__":
    unittest.main()
