import asyncio

from autogen_review import review_answer
from cache_layer import CachedSupportService, normalize_query
from governance import (
    RiskLevel,
    RuntimeBudget,
    classify_risk,
    evaluate_governance,
    inspect_tool_permissions,
)


def test_autogen_approves_supported_answer():
    grounded = "A fee may apply when the signed agreement specifies one."

    verdict, _ = asyncio.run(
        review_answer(
            draft_answer=grounded,
            grounded_answer=grounded,
            evidence="Prepayment fees depend on the agreement.",
        )
    )

    assert verdict.approved is True
    assert verdict.final_answer == grounded


def test_autogen_revises_injected_bad_claim():
    grounded = "A fee may apply when the signed agreement specifies one."

    verdict, _ = asyncio.run(
        review_answer(
            draft_answer=grounded + " Approval is guaranteed.",
            grounded_answer=grounded,
            evidence="Prepayment fees depend on the agreement.",
        )
    )

    assert verdict.approved is False
    assert verdict.final_answer == grounded
    assert "guaranteed" not in verdict.final_answer.lower()


def test_only_lookup_agent_has_record_lookup_tool():
    permissions = inspect_tool_permissions()

    holders = [
        role
        for role, tools in permissions.items()
        if "check_loan_application_status" in tools
    ]

    assert holders == ["Loan Application Lookup Agent"]


def test_financial_customer_data_is_high_risk():
    assert (
        classify_risk("What is the status of application CRD-0001?")
        == RiskLevel.HIGH
    )


def test_oversized_query_is_rejected_by_runtime_budget():
    query = "Explain the banking policy in detail. " * 250
    decision = evaluate_governance(query)

    assert decision.allowed is False
    assert (
        decision.estimated_tokens > RuntimeBudget.max_total_tokens
        or decision.estimated_cost_usd > RuntimeBudget.max_estimated_cost_usd
    )


def test_cache_normalizes_and_hits_without_extra_backend_call():
    service = CachedSupportService()

    first = service.ask("What is the status of application CRD-0001?")
    second = service.ask("  WHAT is the status   of application CRD-0001? ")

    assert normalize_query(
        "What is the status of application CRD-0001?"
    ) == normalize_query(
        "  WHAT is the status   of application CRD-0001? "
    )
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.backend_call_count == 1
    assert first.response.answer == second.response.answer
