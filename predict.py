"""Load the trained model and score one applicant.

The score is the model's estimate of credit RISK (0 = low, 1 = high).
It is a model score, not a guaranteed chance of default.

Usage:  python predict.py     (scores a "typical" applicant as a demo)
"""
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data import add_features

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "credit_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
OPTIONS_PATH = MODELS_DIR / "feature_options.json"

# Illustrative display bands, not calibrated probabilities
LOWER_LIMIT = 0.35
HIGHER_LIMIT = 0.60


def load_model(path=MODEL_PATH):
    """Load the saved model, with a clear message if it is missing."""
    if not Path(path).exists():
        raise FileNotFoundError("Trained model not found. Run 'python train_model.py' first.")
    return joblib.load(path)


def load_json(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"{Path(path).name} not found. Run 'python train_model.py' first.")
    return json.loads(Path(path).read_text())


def score_applicant(model, applicant):
    """Return a risk score between 0 and 1 for a dict of applicant details."""
    row = pd.DataFrame([applicant])
    if "duration" in row and float(row["duration"].iloc[0]) <= 0:
        raise ValueError("Loan duration must be at least 1 month.")
    if "credit_amount" in row and float(row["credit_amount"].iloc[0]) <= 0:
        raise ValueError("Credit amount must be greater than zero.")
    try:
        row = add_features(row)
    except KeyError as error:
        raise ValueError(f"Missing applicant detail: {error}") from error

    expected = list(getattr(model, "feature_names_in_", row.columns))
    missing = [name for name in expected if name not in row.columns]
    if missing:
        raise ValueError(f"Missing applicant details: {missing}")
    return float(model.predict_proba(row[expected])[0, 1])


def risk_band(score):
    """Turn a score into a label and a short, careful message."""
    if score < LOWER_LIMIT:
        return {"tone": "low", "label": "Lower risk",
                "message": "The model sees this profile as similar to loans that were repaid."}
    if score < HIGHER_LIMIT:
        return {"tone": "medium", "label": "Moderate risk",
                "message": "Mixed signals. A human reviewer would want more information."}
    return {"tone": "high", "label": "Higher risk",
            "message": "The model sees this profile as similar to loans that went wrong."}


def typical_applicant(options):
    """Most common choice / median value for every field (used for the demo)."""
    applicant = {}
    for name, spec in options["columns"].items():
        if spec["type"] == "categorical":
            applicant[name] = spec["choices"][0]
        else:
            value = spec["median"]
            applicant[name] = int(round(value)) if spec["integer"] else value
    return applicant


def main():
    try:
        model = load_model()
        applicant = typical_applicant(load_json(OPTIONS_PATH))
        score = score_applicant(model, applicant)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    band = risk_band(score)
    print(f"Typical applicant: risk score {score:.1%} ({band['label']})")
    print("Educational demo only. Not a lending decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
