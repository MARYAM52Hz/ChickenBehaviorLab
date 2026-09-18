import numpy as np

from chicken_behavior_lab.analysis import (
    GraphFeatureDiagnosticExtractor,
)


class MockGraph:
    def __init__(self):
        self.node_features = np.array(
            [
                [
                    [1.0, 2.0],
                    [2.0, 3.0],
                ],
                [
                    [1.1, 2.1],
                    [2.2, 3.2],
                ],
                [
                    [1.3, 2.3],
                    [2.4, 3.4],
                ],
            ],
            dtype=np.float32,
        )

        self.edge_features = np.array(
            [
                [1.0, 0.5],
                [0.8, 0.4],
            ],
            dtype=np.float32,
        )

        self.velocity = np.array(
            [
                [0.1, 0.2],
                [0.2, 0.3],
            ],
            dtype=np.float32,
        )

        self.acceleration = np.array(
            [
                [0.05, 0.10],
                [0.10, 0.15],
            ],
            dtype=np.float32,
        )

        self.motion_energy = np.array(
            [0.10, 0.20, 0.30],
            dtype=np.float32,
        )

        self.visibility = np.array(
            [
                [1.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )

        self.joint_angles = np.array(
            [40.0, 45.0, 50.0],
            dtype=np.float32,
        )

        self.body_orientation = np.array(
            [5.0, 10.0, 15.0],
            dtype=np.float32,
        )

    def validate(self):
        return None


def test_feature_extractor():
    extractor = (
        GraphFeatureDiagnosticExtractor()
    )

    result = extractor.extract(
        MockGraph()
    )

    assert (
        result["mean_motion_energy"]
        == 0.20
    )

    assert (
        result["mean_velocity"]
        is not None
    )

    assert (
        result["mean_acceleration"]
        is not None
    )

    assert (
        result["mean_joint_angle"]
        is not None
    )

    assert (
        result["mean_body_orientation"]
        is not None
    )

    assert (
        0.0
        <= result["valid_keypoint_ratio"]
        <= 1.0
    )
