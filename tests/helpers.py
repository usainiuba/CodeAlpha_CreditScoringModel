"""Fake credit data with the same columns as the real dataset.
Used only so the tests run offline. Never used for real results.
"""
import numpy as np
import pandas as pd

CATEGORIES = {
    "checking_status": ["<0", "0<=X<200", ">=200", "no checking"],
    "credit_history": ["existing paid", "critical/other existing credit", "delayed previously"],
    "purpose": ["radio/tv", "new car", "furniture/equipment", "education"],
    "savings_status": ["<100", "100<=X<500", "no known savings"],
    "employment": ["<1", "1<=X<4", ">=7"],
    "other_parties": ["none", "guarantor"],
    "property_magnitude": ["real estate", "car", "no known property"],
    "other_payment_plans": ["none", "bank", "stores"],
    "housing": ["own", "rent", "for free"],
    "job": ["skilled", "unskilled resident", "high qualif/self emp/mgmt"],
    "own_telephone": ["none", "yes"],
}


def make_fake_credit_data(rows=300, seed=0):
    """Return (X, y) shaped like the real data, with a little signal in it."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({name: rng.choice(values, rows) for name, values in CATEGORIES.items()})
    X["duration"] = rng.integers(4, 73, rows)
    X["credit_amount"] = rng.integers(250, 18000, rows)
    X["installment_commitment"] = rng.integers(1, 5, rows)
    X["residence_since"] = rng.integers(1, 5, rows)
    X["age"] = rng.integers(19, 76, rows)
    X["existing_credits"] = rng.integers(1, 5, rows)
    X["num_dependents"] = rng.integers(1, 3, rows)
    risk = (
        0.03 * X["duration"]
        + X["credit_amount"] / 6000
        + (X["checking_status"] == "<0") * 1.2
        + rng.normal(0, 1, rows)
    )
    y = pd.Series((risk > np.quantile(risk, 0.7)).astype(int))
    return X, y
