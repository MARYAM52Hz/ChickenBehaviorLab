"""Tests for quality diagnostics on the public annotation schema."""

import unittest

from chicken_behavior_lab.annotations.quality import analyze_annotation_quality
from chicken_behavior_lab.annotations.schema import AnnotationSet, BehaviorAnnotation


class AnnotationQualityTests(unittest.TestCase):
    def test_conflicting_overlap_is_reported(self):
        aset = AnnotationSet("video-1", (
            BehaviorAnnotation(1, "pecking", 10, 20, annotation_id="a"),
            BehaviorAnnotation("track_1", "walking", 18, 25, annotation_id="b"),
        ))
        report = analyze_annotation_quality(aset)
        self.assertTrue(report.has_conflicting_overlaps)
        issue = next(i for i in report.issues if i.code == "conflicting_overlap")
        self.assertEqual((issue.start_frame, issue.end_frame), (18, 20))
        self.assertEqual(issue.track_id, "track_1")

    def test_duplicate_interval_is_reported_across_legacy_id_aliases(self):
        aset = AnnotationSet("video-1", (
            BehaviorAnnotation(1, "pecking", 2, 4, annotation_id="a"),
            BehaviorAnnotation("track_1", "pecking", 2, 4, annotation_id="b"),
        ))
        report = analyze_annotation_quality(aset)
        self.assertIn("duplicate_interval", report.issue_counts)
        self.assertEqual(report.track_count, 1)

    def test_annotator_disagreement_requires_different_annotators(self):
        aset = AnnotationSet("video-1", (
            BehaviorAnnotation(1, "pecking", 5, 8, annotation_id="a", annotator_id="ann1"),
            BehaviorAnnotation("track_1", "walking", 5, 8, annotation_id="b", annotator_id="ann2"),
        ))
        report = analyze_annotation_quality(aset)
        self.assertIn("annotator_disagreement", report.issue_counts)

    def test_uncovered_frames_and_empty_expected_track(self):
        aset = AnnotationSet("video-1", (
            BehaviorAnnotation(1, "pecking", 2, 4),
            BehaviorAnnotation("track_1", "walking", 7, 8),
        ))
        report = analyze_annotation_quality(aset, video_length_frames=10, expected_track_ids=[2])
        self.assertEqual(report.uncovered_frames_by_track["track_1"], 5)
        self.assertEqual(report.uncovered_frames_by_track["track_2"], 10)
        self.assertIn("track_without_annotations", report.issue_counts)

    def test_overlapping_same_behavior_counts_union_not_sum(self):
        aset = AnnotationSet("video-1", (
            BehaviorAnnotation(1, "pecking", 0, 5, annotation_id="a"),
            BehaviorAnnotation("track_1", "pecking", 3, 8, annotation_id="b"),
        ))
        report = analyze_annotation_quality(aset, video_length_frames=10)
        self.assertEqual(report.uncovered_frames_by_track["track_1"], 1)

    def test_invalid_video_length_rejected(self):
        aset = AnnotationSet("video-1", ())
        with self.assertRaises(ValueError):
            analyze_annotation_quality(aset, video_length_frames=0)


if __name__ == "__main__":
    unittest.main()
