"""Compare three models, pick the best, test it once, and save everything.

Usage:  python train_model.py
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report

from src.data import add_features, column_types, load_credit_data, split_data
from src.evaluate import cross_validate_model, evaluate_on, top_features
from src.models import build_models

SEED = 42
THRESHOLD = 0.5
MODELS_DIR = Path(__file__).resolve().parent / "models"


def describe_inputs(raw_train):
    """What the web form needs: choices for text fields, ranges for numbers."""
    columns = {}
    for name in raw_train.columns:
        column = raw_train[name]
        if pd.api.types.is_numeric_dtype(column):
            columns[name] = {
                "type": "numeric",
                "min": float(column.min()),
                "max": float(column.max()),
                "median": float(column.median()),
                "integer": bool((column.dropna() % 1 == 0).all()),
            }
        else:
            columns[name] = {
                "type": "categorical",
                "choices": column.value_counts().index.astype(str).tolist(),
            }
    return {"order": list(raw_train.columns), "columns": columns}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the credit risk model.")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args(argv)

    print("Loading the German Credit dataset (first run downloads it)...")
    raw_features, target = load_credit_data()
    raw_columns = list(raw_features.columns)
    features = add_features(raw_features)
    print(f"Rows: {len(features)}   Share of bad risks: {target.mean():.1%}")

    X_train, X_test, y_train, y_test = split_data(features, target, SEED, args.test_size)
    numeric, categorical = column_types(X_train)
    models = build_models(numeric, categorical, SEED)

    # Choose a model with cross-validation on the training data only
    comparison = {}
    for name, model in models.items():
        comparison[name] = cross_validate_model(model, X_train, y_train, SEED)
        scores = comparison[name]
        print(
            f"{name:20s} ROC-AUC {scores['roc_auc']['mean']:.3f}  "
            f"precision {scores['precision']['mean']:.3f}  "
            f"recall {scores['recall']['mean']:.3f}  F1 {scores['f1']['mean']:.3f}"
        )
    best_name = max(comparison, key=lambda name: comparison[name]["roc_auc"]["mean"])
    print(f"\nSelected model: {best_name}")

    # The test set is used once, after the choice is made
    best_model = models[best_name].fit(X_train, y_train)
    test_scores = evaluate_on(best_model, X_test, y_test, THRESHOLD)
    predicted = (best_model.predict_proba(X_test)[:, 1] >= THRESHOLD).astype(int)
    report = classification_report(
        y_test, predicted, target_names=["good risk", "bad risk"], digits=3
    )
    print("\nTest set results:")
    print(report)
    print(f"ROC-AUC: {test_scores['roc_auc']}")
    print(f"Confusion matrix [[true good, false bad], [missed bad, true bad]]: "
          f"{test_scores['confusion_matrix']}")

    metrics = {
        "dataset": {
            "name": "German Credit (UCI Statlog, via OpenML credit-g)",
            "rows": int(len(features)),
            "bad_risk_share": round(float(target.mean()), 4),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "seed": SEED,
        "threshold": THRESHOLD,
        "cv_comparison": comparison,
        "selected_model": best_name,
        "test": test_scores,
        "top_features": top_features(best_model, X_test, y_test, SEED),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "credit_model.joblib", compress=3)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODELS_DIR / "classification_report.txt").write_text(report)
    (MODELS_DIR / "feature_options.json").write_text(
        json.dumps(describe_inputs(X_train[raw_columns]), indent=2)
    )
    print(f"\nSaved model and results to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
