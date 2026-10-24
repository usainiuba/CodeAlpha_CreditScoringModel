"""Metrics: ROC-AUC, precision, recall and F1 for the 'bad risk' class."""
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

CV_SCORING = {
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


def cross_validate_model(model, X, y, seed=42, folds=5):
    """5-fold cross-validation on the TRAINING data. Returns mean and spread."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_validate(model, X, y, cv=splitter, scoring=CV_SCORING)
    return {
        name: {
            "mean": round(float(np.mean(scores[f"test_{name}"])), 4),
            "std": round(float(np.std(scores[f"test_{name}"])), 4),
        }
        for name in CV_SCORING
    }


def evaluate_on(model, X, y, threshold=0.5):
    """Score a fitted model. Used once, on the held-out test set."""
    scores = model.predict_proba(X)[:, 1]
    predicted = (scores >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
        "precision": round(float(precision_score(y, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y, predicted, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y, predicted)), 4),
        # [[true good, false bad], [missed bad, true bad]]
        "confusion_matrix": confusion_matrix(y, predicted).tolist(),
    }


def top_features(model, X, y, seed=42, count=8):
    """Which inputs matter most overall? Shuffle one column at a time and
    see how much the ROC-AUC drops. Bigger drop = more important."""
    result = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=10, random_state=seed
    )
    ranked = sorted(zip(X.columns, result.importances_mean), key=lambda p: -p[1])
    return [
        {"feature": name, "importance": round(float(value), 4)}
        for name, value in ranked[:count]
        if value > 0
    ]
