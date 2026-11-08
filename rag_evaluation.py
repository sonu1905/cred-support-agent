"""Compare chunking strategies and calibrate the groundedness threshold."""

from rag_engine import PolicySearch

CASES = [
    ("What are the requirements for a personal loan?", "personal_eligibility"),
    ("What documents are needed for KYC?", "kyc"),
    ("How is EMI calculated?", "emi"),
    ("Can a loan have a prepayment penalty?", "prepayment"),
    ("What affects a person's credit score?", "credit_score"),
]

OUT_OF_SCOPE = [
    "How do I replace the battery in my bicycle light?",
    "What is the best soil for growing lavender?",
]


def unique_parent_documents(hits):
    return list(dict.fromkeys(hit["doc_id"] for hit in hits))


def evaluate(engine, k=3):
    summary = {}

    for strategy in ("fixed", "sentence"):
        print(f"\n{strategy.upper()} retrieval (top {k})")

        precisions = []
        recalls = []

        for query, relevant_id in CASES:
            hits = engine.search(query, strategy=strategy, k=k)
            retrieved = unique_parent_documents(hits)

            true_positives = int(relevant_id in retrieved)
            precision = true_positives / len(retrieved) if retrieved else 0.0
            recall = true_positives / 1

            precisions.append(precision)
            recalls.append(recall)

            print(query)
            print(f"  relevant={{'{relevant_id}'}}")
            print(f"  retrieved={retrieved}")
            print(
                f"  precision={true_positives}/{len(retrieved)}={precision:.3f}; "
                f"recall={true_positives}/1={recall:.3f}"
            )

        mean_precision = sum(precisions) / len(precisions)
        mean_recall = sum(recalls) / len(recalls)
        summary[strategy] = (mean_precision, mean_recall)

        print(f"Mean precision={mean_precision:.3f}")
        print(f"Mean recall={mean_recall:.3f}")

    return summary


def calibrate(engine, strategy="sentence"):
    print(f"\nTop-1 similarity calibration using {strategy} chunks")

    in_scope_scores = []
    out_scope_scores = []

    for query, _ in CASES:
        score = engine.search(query, strategy=strategy, k=1)[0]["similarity"]
        in_scope_scores.append(score)
        print(f"IN  {score:.4f}  {query}")

    for query in OUT_OF_SCOPE:
        score = engine.search(query, strategy=strategy, k=1)[0]["similarity"]
        out_scope_scores.append(score)
        print(f"OUT {score:.4f}  {query}")

    lowest_in_scope = min(in_scope_scores)
    highest_out_scope = max(out_scope_scores)

    print(f"Lowest in-scope similarity: {lowest_in_scope:.4f}")
    print(f"Highest out-of-scope similarity: {highest_out_scope:.4f}")

    if lowest_in_scope <= highest_out_scope:
        print("Score ranges overlap. Revise the retrieval setup before choosing a threshold.")
        return None

    threshold = round((lowest_in_scope + highest_out_scope) / 2, 4)
    print(f"Candidate threshold: {threshold:.4f}")
    return threshold


if __name__ == "__main__":
    engine = PolicySearch(auto_index=True)
    results = evaluate(engine)

    print("\nStrategy summary")
    for name, (precision, recall) in results.items():
        print(f"{name}: mean precision={precision:.3f}, mean recall={recall:.3f}")

    fixed_precision, fixed_recall = results["fixed"]
    sentence_precision, sentence_recall = results["sentence"]

    print("\nRecommendation")
    if (
        sentence_recall >= fixed_recall
        and sentence_precision >= fixed_precision
    ):
        print(
            "Use sentence-based chunking. It preserves at least the same recall "
            "and gives equal or better document-level precision on this test set."
        )
    else:
        print(
            "Use the strategy with the stronger measured precision/recall trade-off "
            "shown above."
        )

    threshold = calibrate(engine, strategy="sentence")
    if threshold is not None:
        print(f"\nMeasured groundedness threshold to use: {threshold:.4f}")
