"""The three algorithms we compare. Each one is a full pipeline:
cleaning (missing values, scaling, encoding) followed by the classifier.
Putting cleaning inside the pipeline means it is learned from training data only.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def make_preprocessor(numeric_columns, categorical_columns):
    """Fill gaps, scale numbers, and turn text categories into 0/1 columns."""
    numbers = Pipeline(
        [("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categories = Pipeline(
        [
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numbers", numbers, numeric_columns), ("categories", categories, categorical_columns)]
    )


def build_models(numeric_columns, categorical_columns, seed=42):
    """Return {name: untrained pipeline}.

    class_weight="balanced" because only about 30% of loans are 'bad'.
    """

    def pipeline(classifier):
        return Pipeline(
            [
                ("prepare", make_preprocessor(numeric_columns, categorical_columns)),
                ("classifier", classifier),
            ]
        )

    return {
        "Logistic Regression": pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "Decision Tree": pipeline(
            DecisionTreeClassifier(
                max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=seed
            )
        ),
        "Random Forest": pipeline(
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
        ),
    }
