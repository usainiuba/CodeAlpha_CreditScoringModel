"""Readable names and form layout for the credit risk app."""

FIELD_LABELS = {
    "checking_status": "Checking account status",
    "duration": "Loan duration (months)",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "credit_amount": "Credit amount",
    "savings_status": "Savings account",
    "employment": "Time in current job",
    "installment_commitment": "Installment rate (1 to 4, % of income)",
    "other_parties": "Other debtors or guarantors",
    "residence_since": "Years at current residence",
    "property_magnitude": "Main property owned",
    "age": "Age (years)",
    "other_payment_plans": "Other payment plans",
    "housing": "Housing",
    "existing_credits": "Existing credits at this bank",
    "job": "Job type",
    "num_dependents": "People financially dependent",
    "own_telephone": "Telephone",
    # Features the code builds from the loan request
    "monthly_payment": "Monthly payment (amount / duration)",
    "long_term_loan": "Long-term loan (over 24 months)",
    "log_credit_amount": "Credit amount (log scale)",
}

# (section title, fields). Any field not listed here goes in "Other details".
FIELD_GROUPS = [
    ("Loan request", ["credit_amount", "duration", "purpose", "installment_commitment"]),
    ("Financial profile", ["checking_status", "savings_status", "credit_history",
                           "employment", "other_payment_plans", "existing_credits"]),
    ("Personal and housing", ["age", "housing", "job", "residence_since",
                              "num_dependents", "property_magnitude",
                              "other_parties", "own_telephone"]),
]


def field_label(name):
    """Friendly label, or a cleaned-up version of the column name."""
    return FIELD_LABELS.get(name, name.replace("_", " ").capitalize())
