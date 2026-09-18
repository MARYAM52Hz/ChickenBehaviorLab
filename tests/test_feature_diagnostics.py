from chicken_behavior_lab.analysis import (
    FeatureDiagnosticAnalyzer,
    FeatureDiagnosticRecord,
)


def make_records():
    return [
        FeatureDiagnosticRecord(
            sample_id="sample_001",
            video_id="video_001",
            track_id=1,
            start_frame=100,
            end_frame=180,
            true_behavior="feeding",
            predicted_behavior="feeding",
            confidence=0.94,
            is_error=False,
            mean_motion_energy=0.40,
            mean_node_feature_magnitude=1.20,
            mean_edge_feature_magnitude=0.80,
            valid_keypoint_ratio=0.95,
            valid_node_ratio=1.0,
            mean_velocity=0.30,
            mean_acceleration=0.20,
            mean_joint_angle=45.0,
            std_joint_angle=8.0,
            mean_body_orientation=10.0,
            std_body_orientation=3.0,
        ),
        FeatureDiagnosticRecord(
            sample_id="sample_002",
            video_id="video_001",
            track_id=1,
            start_frame=181,
            end_frame=230,
            true_behavior="feeding",
            predicted_behavior="standing",
            confidence=0.61,
            is_error=True,
            mean_motion_energy=0.05,
            mean_node_feature_magnitude=0.80,
            mean_edge_feature_magnitude=0.50,
            valid_keypoint_ratio=0.50,
            valid_node_ratio=0.60,
            mean_velocity=0.04,
            mean_acceleration=0.03,
            mean_joint_angle=42.0,
            std_joint_angle=5.0,
            mean_body_orientation=8.0,
            std_body_orientation=2.0,
        ),
        FeatureDiagnosticRecord(
            sample_id="sample_003",
            video_id="video_001",
            track_id=2,
            start_frame=100,
            end_frame=160,
            true_behavior="walking",
            predicted_behavior="standing",
            confidence=0.87,
            is_error=True,
            mean_motion_energy=0.35,
            mean_node_feature_magnitude=1.10,
            mean_edge_feature_magnitude=0.70,
            valid_keypoint_ratio=0.90,
            valid_node_ratio=1.0,
            mean_velocity=0.28,
            mean_acceleration=0.15,
            mean_joint_angle=60.0,
            std_joint_angle=10.0,
            mean_body_orientation=15.0,
            std_body_orientation=4.0,
        ),
        FeatureDiagnosticRecord(
            sample_id="sample_004",
            video_id="video_002",
            track_id=4,
            start_frame=50,
            end_frame=90,
            true_behavior="standing",
            predicted_behavior="standing",
            confidence=0.91,
            is_error=False,
            mean_motion_energy=0.08,
            mean_node_feature_magnitude=0.70,
            mean_edge_feature_magnitude=0.45,
            valid_keypoint_ratio=0.92,
            valid_node_ratio=1.0,
            mean_velocity=0.02,
            mean_acceleration=0.01,
            mean_joint_angle=40.0,
            std_joint_angle=4.0,
            mean_body_orientation=7.0,
            std_body_orientation=2.0,
        ),
    ]


def test_error_and_correct_counts():
    analyzer = FeatureDiagnosticAnalyzer(
        make_records()
    )

    assert len(analyzer) == 4
    assert len(analyzer.errors) == 2
    assert len(analyzer.correct) == 2


def test_error_vs_correct_comparison():
    analyzer = FeatureDiagnosticAnalyzer(
        make_records()
    )

    result = (
        analyzer.compare_error_vs_correct()
    )

    motion = result["features"][
        "mean_motion_energy"
    ]

    assert motion["error_mean"] == (
        (0.05 + 0.35) / 2
    )

    assert motion["correct_mean"] == (
        (0.40 + 0.08) / 2
    )


def test_behavior_level_comparison():
    analyzer = FeatureDiagnosticAnalyzer(
        make_records()
    )

    results = (
        analyzer.compare_by_true_behavior()
    )

    feeding = next(
        item
        for item in results
        if item["behavior"] == "feeding"
    )

    assert feeding["total"] == 2
    assert feeding["errors"] == 1
    assert feeding["correct"] == 1
    assert feeding["error_rate"] == 0.5


def test_suspicious_error_patterns():
    analyzer = FeatureDiagnosticAnalyzer(
        make_records()
    )

    suspicious = (
        analyzer.suspicious_error_patterns(
            motion_threshold=0.10,
            keypoint_ratio_threshold=0.60,
        )
    )

    assert len(suspicious) == 1

    assert (
        suspicious[0]["sample_id"]
        == "sample_002"
    )

    assert (
        "low_motion_energy"
        in suspicious[0]["reasons"]
    )

    assert (
        "low_keypoint_visibility"
        in suspicious[0]["reasons"]
    )


def test_save_and_load_json(tmp_path):
    analyzer = FeatureDiagnosticAnalyzer(
        make_records()
    )

    path = (
        tmp_path
        / "error_diagnostics.json"
    )

    analyzer.save_json(path)

    loaded = (
        FeatureDiagnosticAnalyzer.load_json(
            path
        )
    )

    assert len(loaded) == 4
    assert len(loaded.errors) == 2
