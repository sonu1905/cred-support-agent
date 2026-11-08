"""Synthetic loan applications used throughout the demo."""
import random
from collections import Counter

SEED = 27
CATEGORY_WEIGHTS = {
    'Personal Loan': 28, 'Home Loan': 22, 'Auto Loan': 20,
    'Education Loan': 16, 'Business Loan': 14,
}
STATUS_WEIGHTS = {
    'Submitted': 22, 'Under Review': 30, 'Approved': 22,
    'Rejected': 12, 'Disbursed': 14,
}
AMOUNT_RANGE = (50_000, 5_000_000)


def generate_applications(count=60, seed=SEED):
    if count < 40:
        raise ValueError('At least 40 applications are required')
    rng = random.Random(seed)
    categories = list(CATEGORY_WEIGHTS)
    statuses = list(STATUS_WEIGHTS)
    # Guarantee vocabulary coverage without changing individual records later.
    category_draws = categories * 3 + rng.choices(categories, weights=CATEGORY_WEIGHTS.values(), k=count - 15)
    status_draws = statuses + rng.choices(statuses, weights=STATUS_WEIGHTS.values(), k=count - 5)
    rng.shuffle(category_draws)
    rng.shuffle(status_draws)
    records = []
    for i in range(count):
        records.append({
            'record_id': f'CRD-{i + 1:04}',
            'category': category_draws[i],
            'status': status_draws[i],
            'loan_amount_inr': rng.randrange(AMOUNT_RANGE[0], AMOUNT_RANGE[1] + 1, 10_000),
            'days_since_created': rng.randint(0, 30),
            'flagged_for_fraud_review': rng.random() < 0.19,
        })
    return records


LOAN_APPLICATIONS = generate_applications()


def validate(records):
    categories = Counter(r['category'] for r in records)
    statuses = Counter(r['status'] for r in records)
    flagged_pct = 100 * sum(r['flagged_for_fraud_review'] for r in records) / len(records)
    assert len(records) >= 40
    assert all(categories[c] >= 3 for c in CATEGORY_WEIGHTS)
    assert all(statuses[s] >= 1 for s in STATUS_WEIGHTS)
    assert 10 <= flagged_pct <= 30, f'Fraud percentage {flagged_pct:.1f}% is outside allowed range'
    assert all(0 <= r['days_since_created'] <= 30 for r in records)
    assert len({r['record_id'] for r in records}) == len(records)
    return {'total': len(records), 'categories': dict(categories), 'statuses': dict(statuses), 'fraud_pct': round(flagged_pct, 2)}


if __name__ == '__main__':
    import json
    print(json.dumps(validate(LOAN_APPLICATIONS), indent=2))
