"""Load the German Credit dataset and prepare features."""
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGET = "class"

# Removed on purpose: sex/marital status and nationality should not drive a
# credit decision (it is unfair, and illegal in many places).
EXCLUDED_COLUMNS = ["personal_status", "foreign_worker"]
REQUIRED_COLUMNS = ["duration", "credit_amount", "installment_commitment"]


def load_credit_data():
    """Return features X and target y. y is 1 for a 'bad' risk, 0 for 'good'.

    The first call downloads the data (about 1000 rows) into the data/ folder.
    """
    try:
        dataset = fetch_openml(
            "credit-g", version=1, as_frame=True, data_home=str(DATA_DIR)
        )
    except Exception as error:  # network or server problems
        raise RuntimeError(
            "Could not download the dataset. Check your internet connection and try again."
        ) from error

    data = dataset.frame
    y = (data[TARGET] == "bad").astype(int)
    X = data.drop(columns=[TARGET] + [c for c in EXCLUDED_COLUMNS if c in data.columns])
    X = clean_types(X)
    check_columns(X)
    return X, y


def clean_types(X):
    """Turn category columns into plain text so new values never crash the model."""
    X = X.copy()
    for column in X.select_dtypes(include=["category", "object"]).columns:
        X[column] = X[column].astype(str)
    return X


def check_columns(X):
    """Stop early, with a clear message, if the dataset layout is not as expected."""
    missing = [name for name in REQUIRED_COLUMNS if name not in X.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")


def add_features(df):
    """Add simple features built from the loan request.

    Each row is handled on its own, so no information leaks from the test set.
    """
    df = df.copy()
    df["monthly_payment"] = df["credit_amount"] / df["duration"]
    df["long_term_loan"] = (df["duration"] > 24).astype(int)
    df["log_credit_amount"] = np.log1p(df["credit_amount"])  # amounts are very skewed
    return df


def split_data(X, y, seed=42, test_size=0.2):
    """Hold out a test set. Keeps the good/bad ratio the same in both parts."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def column_types(X):
    """Return (numeric column names, text column names)."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [name for name in X.columns if name not in numeric]
    return numeric, categorical
