"""Runs the whole training script on FAKE data to prove the pipeline works.
The numbers it produces are meaningless and are thrown away.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import train_model
from predict import load_model, score_applicant, typical_applicant
from tests.helpers import make_fake_credit_data


class TrainSmokeTest(unittest.TestCase):
    def test_training_saves_files_and_model_can_score(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(train_model, "MODELS_DIR", Path(folder)), \
                 mock.patch.object(train_model, "load_credit_data",
                                   lambda: make_fake_credit_data(400, seed=3)):
                train_model.main([])

            for name in ["credit_model.joblib", "metrics.json",
                         "classification_report.txt", "feature_options.json"]:
                self.assertTrue((Path(folder) / name).exists(), name)

            metrics = json.loads((Path(folder) / "metrics.json").read_text())
            self.assertIn(metrics["selected_model"], metrics["cv_comparison"])
            self.assertEqual(set(metrics["cv_comparison"]),
                             {"Logistic Regression", "Decision Tree", "Random Forest"})

            options = json.loads((Path(folder) / "feature_options.json").read_text())
            model = load_model(Path(folder) / "credit_model.joblib")
            score = score_applicant(model, typical_applicant(options))
            self.assertTrue(0.0 <= score <= 1.0)


if __name__ == "__main__":
    unittest.main()
