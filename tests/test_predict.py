import unittest

from predict import risk_band, score_applicant, typical_applicant
from src.data import add_features, column_types
from src.models import build_models
from tests.helpers import make_fake_credit_data


class PredictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_fake_credit_data(300)
        cls.raw = X
        features = add_features(X)
        numeric, categorical = column_types(features)
        model = build_models(numeric, categorical, seed=1)["Logistic Regression"]
        cls.model = model.fit(features, y)

    def applicant(self):
        return self.raw.iloc[0].to_dict()

    def test_score_is_between_0_and_1(self):
        score = score_applicant(self.model, self.applicant())
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_missing_detail_gives_clear_error(self):
        applicant = self.applicant()
        del applicant["housing"]
        with self.assertRaises(ValueError):
            score_applicant(self.model, applicant)

    def test_zero_duration_is_rejected(self):
        applicant = self.applicant()
        applicant["duration"] = 0
        with self.assertRaises(ValueError):
            score_applicant(self.model, applicant)

    def test_risk_bands(self):
        self.assertEqual(risk_band(0.10)["tone"], "low")
        self.assertEqual(risk_band(0.50)["tone"], "medium")
        self.assertEqual(risk_band(0.90)["tone"], "high")

    def test_typical_applicant_uses_options(self):
        options = {"columns": {
            "housing": {"type": "categorical", "choices": ["own", "rent"]},
            "age": {"type": "numeric", "min": 19, "max": 75, "median": 33.0, "integer": True},
        }}
        self.assertEqual(typical_applicant(options), {"housing": "own", "age": 33})


if __name__ == "__main__":
    unittest.main()
