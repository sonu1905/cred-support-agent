"""Task 6 demonstration for the loan lookup tool."""

from dataset import LOAN_APPLICATIONS
from loan_tools import (
    DAYS_P75,
    ESCALATION_THRESHOLD,
    FRAUD_WEIGHT,
    AGE_WEIGHT,
    check_loan_application_status,
)


def score_for(row):
    age_signal = row["days_since_created"] / 30
    fraud_signal = float(row["flagged_for_fraud_review"])
    return round(FRAUD_WEIGHT * fraud_signal + AGE_WEIGHT * age_signal, 3)


if __name__ == "__main__":
    scored = [
        (row["record_id"], row["days_since_created"], row["flagged_for_fraud_review"], score_for(row))
        for row in LOAN_APPLICATIONS
    ]

    highest = max(scored, key=lambda item: item[3])
    normal = next(item for item in scored if not item[2])

    print("Escalation formula:")
    print("score = 0.70 * fraud_flag + 0.30 * (days_since_created / 30)")
    print()
    print(f"75th percentile of days_since_created: {DAYS_P75} days")
    print(f"Escalation threshold: {ESCALATION_THRESHOLD}")
    print(
        "Reason: the threshold equals the score of a fraud-flagged application "
        f"at the dataset's 75th-percentile age ({DAYS_P75} days)."
    )

    print("\nHigh-risk example:")
    print(check_loan_application_status(highest[0]))

    print("\nNormal example:")
    print(check_loan_application_status(normal[0]))

    print("\nMissing-record example:")
    print(check_loan_application_status("CRD-9999"))
