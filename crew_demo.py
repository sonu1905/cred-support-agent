"""Task 7 and Task 9 demonstration."""

from crew_service import run_support_query
from crew_tools import TOOL_CALL_LOG


def show(label: str, query: str) -> None:
    before = len(TOOL_CALL_LOG)
    response = run_support_query(query)
    new_calls = TOOL_CALL_LOG[before:]

    print(f"\n--- {label} ---")
    print("Query:", query)
    print("Validated response:", response.model_dump())
    print("Tool calls made by this kickoff:", new_calls)


if __name__ == "__main__":
    show(
        "Policy query - RAG tool should run",
        "What documents are needed for KYC?",
    )

    show(
        "Application query - lookup tool should run",
        "What is the status of application CRD-0001?",
    )
