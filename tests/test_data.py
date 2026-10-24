import unittest

import pandas as pd

from src.data import add_features, check_columns, clean_types, column_types, split_data
from tests.helpers import make_fake_credit_data


class DataTests(unittest.TestCase):
    def test_add_features_values(self):
        df = pd.DataFrame({"duration": [10, 30], "credit_amount": [1000, 3000]})
        result = add_features(df)
        self.assertEqual(result["monthly_payment"].tolist(), [100.0, 100.0])
        self.assertEqual(result["long_term_loan"].tolist(), [0, 1])
        self.assertIn("log_credit_amount", result.columns)

    def test_add_features_does_not_change_input(self):
        df = pd.DataFrame({"duration": [10], "credit_amount": [1000]})
        add_features(df)
        self.assertNotIn("monthly_payment", df.columns)

    def test_clean_types_turns_categories_into_text(self):
        df = pd.DataFrame({"a": pd.Categorical(["x", "y"]), "b": [1, 2]})
        cleaned = clean_types(df)
        self.assertTrue(all(isinstance(v, str) for v in cleaned["a"]))
        self.assertEqual(cleaned["b"].tolist(), [1, 2])

    def test_check_columns_reports_missing(self):
        with self.assertRaises(ValueError):
            check_columns(pd.DataFrame({"age": [30]}))

    def test_split_keeps_class_ratio(self):
        X, y = make_fake_credit_data(400)
        _, _, y_train, y_test = split_data(X, y, seed=1, test_size=0.25)
        self.assertEqual(len(y_test), 100)
        self.assertAlmostEqual(y_train.mean(), y_test.mean(), delta=0.03)

    def test_column_types(self):
        X, _ = make_fake_credit_data(50)
        numeric, categorical = column_types(add_features(X))
        self.assertIn("credit_amount", numeric)
        self.assertIn("housing", categorical)


if __name__ == "__main__":
    unittest.main()
