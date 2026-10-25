
import unittest

from chicken_behavior_lab.prediction import (
    MortalityPredictionError,
    MortalityPredictor,
)


class FakeEstimator:
    classes_ = [0, 1]

    def predict_proba(self, rows):
        return [
            [0.8, 0.2] if row[0] < 1 else [0.1, 0.9]
            for row in rows
        ]


class ReversedEstimator:
    # The positive class is the first column in this estimator.
    classes_ = [1, 0]

    def predict_proba(self, rows):
        return [[0.9, 0.1] for _ in rows]


class InvalidEstimator:
    classes_ = [0, 1]

    def predict_proba(self, rows):
        return [[0.2, 1.2] for _ in rows]


class MortalityPredictorTests(unittest.TestCase):

    def setUp(self):
        self.predictor = MortalityPredictor(
            FakeEstimator(),
            feature_names=["activity", "feed"],
            threshold=0.5,
        )

    def test_feature_order_and_sample_id(self):
        result = self.predictor.predict(
            [{"feed": 3.0, "activity": 2.0}],
            sample_ids=["bird-1"],
        )[0]

        self.assertEqual(result.sample_id, "bird-1")
        self.assertAlmostEqual(result.risk_probability, 0.9)
        self.assertEqual(
            result.risk_label,
            "higher_modelled_risk",
        )

    def test_low_probability(self):
        result = self.predictor.predict(
            [{"activity": 0.0, "feed": 3.0}]
        )[0]

        self.assertEqual(
            result.risk_label,
            "lower_modelled_risk",
        )

    def test_positive_class_is_not_assumed_second(self):
        predictor = MortalityPredictor(
            ReversedEstimator(),
            feature_names=["activity", "feed"],
            positive_label=1,
        )

        result = predictor.predict(
            [{"activity": 2.0, "feed": 3.0}]
        )[0]

        self.assertAlmostEqual(result.risk_probability, 0.9)

    def test_missing_or_extra_features(self):
        with self.assertRaises(MortalityPredictionError):
            self.predictor.predict([{"activity": 1.0}])

        with self.assertRaises(MortalityPredictionError):
            self.predictor.predict([
                {
                    "activity": 1.0,
                    "feed": 2.0,
                    "unknown": 3.0,
                }
            ])

    def test_non_finite_feature(self):
        with self.assertRaises(MortalityPredictionError):
            self.predictor.predict([
                {
                    "activity": float("nan"),
                    "feed": 2.0,
                }
            ])

    def test_sample_id_count(self):
        with self.assertRaises(MortalityPredictionError):
            self.predictor.predict(
                [{"activity": 1.0, "feed": 2.0}],
                sample_ids=["bird-1", "bird-2"],
            )

    def test_invalid_probability(self):
        predictor = MortalityPredictor(
            InvalidEstimator(),
            feature_names=["activity", "feed"],
        )

        with self.assertRaises(MortalityPredictionError):
            predictor.predict([
                {"activity": 1.0, "feed": 2.0}
            ])


if __name__ == "__main__":
    unittest.main()
