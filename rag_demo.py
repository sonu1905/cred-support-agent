"""Task 4 demo: five in-scope questions and one out-of-scope question."""

from rag_engine import PolicySearch

# Replace this only if rag_evaluation.py produces a different measured threshold
# after the title-aware retrieval change.
THRESHOLD = 0.2211

QUERIES = [
    "What are the requirements for a personal loan?",
    "What documents are needed for KYC?",
    "How is EMI calculated?",
    "Can a loan have a prepayment penalty?",
    "What affects a person's credit score?",
    "How do I replace the battery in my bicycle light?",
]


if __name__ == "__main__":
    engine = PolicySearch(auto_index=True)

    for number, query in enumerate(QUERIES, start=1):
        result = engine.answer(
            query,
            threshold=THRESHOLD,
            strategy="sentence",
            k=3,
        )

        print(f"\nQuery {number}: {query}")
        print(f"Supported: {result['supported']}")
        print(f"Top similarity: {result['top_similarity']}")
        print(f"Sources: {result['sources']}")
        print(f"Answer: {result['answer']}")
