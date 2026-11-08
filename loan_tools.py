"""Loan application lookup and escalation scoring."""

from math import ceil
from dataset import LOAN_APPLICATIONS


def _percentile_nearest_rank(values: list[int], percentile: float) -> int:
    """Return a simple nearest-rank percentile for a small deterministic dataset."""
    if not values:
        raise ValueError("values cannot be empty")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be between 0 and 1")

    ordered = sorted(values)
    index = max(0, ceil(percentile * len(ordered)) - 1)
    return ordered[index]


# Fraud review is the main risk signal. Application age contributes a smaller
# urgency signal so two fraud-flagged records can still receive different scores.
FRAUD_WEIGHT = 0.70
AGE_WEIGHT = 0.30

DAYS_P75 = _percentile_nearest_rank(
    [row["days_since_created"] for row in LOAN_APPLICATIONS],
    0.75,
)

# A fraud-flagged application at the 75th percentile of application age is the
# point at which this demo recommends escalation.
ESCALATION_THRESHOLD = round(
    FRAUD_WEIGHT + AGE_WEIGHT * (DAYS_P75 / 30),
    3,
)


def check_loan_application_status(record_id: str) -> dict:
    normalized_id = record_id.upper().strip()

    application = next(
        (
            row
            for row in LOAN_APPLICATIONS
            if row["record_id"] == normalized_id
        ),
        None,
    )

    if application is None:
        return {
            "found": False,
            "record_id": normalized_id,
            "message": "Application not found",
        }

    age_signal = application["days_since_created"] / 30
    fraud_signal = float(application["flagged_for_fraud_review"])

    escalation_score = round(
        FRAUD_WEIGHT * fraud_signal + AGE_WEIGHT * age_signal,
        3,
    )

    return {
        "found": True,
        "record_id": application["record_id"],
        "status": application["status"],
        "loan_amount_inr": application["loan_amount_inr"],
        "escalation_score": escalation_score,
        "recommend_escalation": escalation_score >= ESCALATION_THRESHOLD,
    }
