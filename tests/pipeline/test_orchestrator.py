
import unittest

from chicken_behavior_lab.pipeline import (
    ChickenBehaviorPipeline,
    PipelineConfigurationError,
    PipelineStage,
    PipelineStageError,
)


class PipelineOrchestratorTests(unittest.TestCase):

    def test_runs_stages_in_order_and_passes_previous_outputs(self):
        observed = []

        def detect(context):
            observed.append("detection")
            self.assertEqual(context.video, "sample.mp4")
            return ["detections"]

        def track(context):
            observed.append("tracking")
            return context.get_output("detection") + ["tracks"]

        def behavior(context):
            observed.append("behavior")
            return {
                "track_count": len(
                    context.get_output("tracking")
                )
            }

        pipeline = ChickenBehaviorPipeline([
            PipelineStage("detection", detect),
            PipelineStage(
                "tracking",
                track,
                requires=("detection",),
            ),
            PipelineStage(
                "behavior",
                behavior,
                requires=("tracking",),
            ),
        ])

        result = pipeline.run(
            "sample.mp4",
            metadata={"camera": "side"},
        )

        self.assertEqual(
            observed,
            ["detection", "tracking", "behavior"],
        )
        self.assertEqual(
            result.get("behavior"),
            {"track_count": 2},
        )
        self.assertEqual(
            tuple(result.stage_durations_seconds),
            ("detection", "tracking", "behavior"),
        )
        self.assertTrue(
            all(
                value >= 0
                for value in result.stage_durations_seconds.values()
            )
        )

    def test_rejects_duplicate_stage_names(self):
        stage = PipelineStage(
            "detection",
            lambda context: None,
        )

        with self.assertRaises(PipelineConfigurationError):
            ChickenBehaviorPipeline([stage, stage])

    def test_rejects_dependency_on_future_or_missing_stage(self):
        with self.assertRaises(PipelineConfigurationError):
            ChickenBehaviorPipeline([
                PipelineStage(
                    "tracking",
                    lambda context: None,
                    requires=("detection",),
                ),
                PipelineStage(
                    "detection",
                    lambda context: None,
                ),
            ])

    def test_wraps_stage_error_and_preserves_original_cause(self):
        def broken_stage(context):
            raise ValueError("synthetic failure")

        pipeline = ChickenBehaviorPipeline([
            PipelineStage("pose", broken_stage),
        ])

        with self.assertRaises(PipelineStageError) as raised:
            pipeline.run("sample.mp4")

        self.assertEqual(raised.exception.stage_name, "pose")
        self.assertIsInstance(
            raised.exception.__cause__,
            ValueError,
        )
        self.assertEqual(
            str(raised.exception.__cause__),
            "synthetic failure",
        )

    def test_empty_pipeline_is_rejected(self):
        with self.assertRaises(PipelineConfigurationError):
            ChickenBehaviorPipeline([])

    def test_stage_processor_must_be_callable(self):
        with self.assertRaises(PipelineConfigurationError):
            PipelineStage("detection", None)


if __name__ == "__main__":
    unittest.main()
