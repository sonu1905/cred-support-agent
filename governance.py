"""Task 15: governance controls without loading CrewAI at demo startup."""

import ast
import math
import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from guardrails import apply_input_guardrails


ROOT = Path(__file__).resolve().parent
CREW_SERVICE_FILE = ROOT / "crew_service.py"


class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class GovernanceDecision(BaseModel):
    allowed: bool
    risk_level: RiskLevel
    masked_query: str
    injection_detected: bool
    estimated_tokens: int
    estimated_cost_usd: float
    reason: str


class RuntimeBudget:
    max_total_tokens = 1000
    max_estimated_cost_usd = 0.0010
    estimated_cost_per_1k_tokens = 0.0010
    fixed_orchestration_tokens = 450

    @classmethod
    def estimate(cls, query: str) -> tuple[int, float]:
        input_tokens = math.ceil(len(query) / 4)
        total_tokens = input_tokens + cls.fixed_orchestration_tokens
        estimated_cost = (
            total_tokens / 1000
        ) * cls.estimated_cost_per_1k_tokens
        return total_tokens, round(estimated_cost, 6)


def classify_risk(query: str) -> RiskLevel:
    lowered = query.lower()

    high_risk_patterns = [
        r"\bcrd-\d{4}\b",
        r"\bpan\b",
        r"\baadhaar\b",
        r"bank\s+account",
        r"account\s+number",
        r"loan\s+amount",
        r"application\s+status",
        r"transaction",
    ]

    if any(re.search(pattern, lowered, re.I) for pattern in high_risk_patterns):
        return RiskLevel.HIGH

    banking_terms = (
        "loan",
        "kyc",
        "emi",
        "credit",
        "interest",
        "prepayment",
        "account",
        "balance",
        "nri",
        "fraud",
    )

    if any(term in lowered for term in banking_terms):
        return RiskLevel.MEDIUM

    return RiskLevel.LOW


def _constructor_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
    return ""


def inspect_tool_permissions() -> dict:
    """
    Inspect the actual CrewAI agent definitions in crew_service.py without
    importing CrewAI or creating agents.
    """
    source = CREW_SERVICE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    class_to_tool_name = {
        "RagLookupTool": "rag_lookup",
        "LoanStatusTool": "check_loan_application_status",
    }

    permissions = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue

        call = node.value
        if _constructor_name(call) != "Agent":
            continue

        role = None
        tools = []

        for keyword in call.keywords:
            if keyword.arg == "role":
                if isinstance(keyword.value, ast.Constant):
                    role = str(keyword.value.value)

            if keyword.arg == "tools" and isinstance(keyword.value, ast.List):
                for item in keyword.value.elts:
                    class_name = _constructor_name(item)
                    if class_name:
                        tools.append(class_to_tool_name.get(class_name, class_name))

        if role:
            permissions[role] = tools

    expected_roles = {
        "Policy Retrieval Agent",
        "Loan Application Lookup Agent",
        "Response Composer",
    }

    if not expected_roles.issubset(permissions):
        raise RuntimeError(
            "Could not verify all three agent definitions in crew_service.py"
        )

    holders = [
        role
        for role, tools in permissions.items()
        if "check_loan_application_status" in tools
    ]

    if holders != ["Loan Application Lookup Agent"]:
        raise RuntimeError(
            "Least-privilege violation: lookup tool is assigned incorrectly"
        )

    return permissions


def evaluate_governance(query: str) -> GovernanceDecision:
    guarded = apply_input_guardrails(query)
    risk = classify_risk(query)

    estimated_tokens, estimated_cost = RuntimeBudget.estimate(guarded.text)

    if guarded.injection_detected:
        return GovernanceDecision(
            allowed=False,
            risk_level=risk,
            masked_query=guarded.text,
            injection_detected=True,
            estimated_tokens=estimated_tokens,
            estimated_cost_usd=estimated_cost,
            reason="Rejected by input guardrail.",
        )

    if estimated_tokens > RuntimeBudget.max_total_tokens:
        return GovernanceDecision(
            allowed=False,
            risk_level=risk,
            masked_query=guarded.text,
            injection_detected=False,
            estimated_tokens=estimated_tokens,
            estimated_cost_usd=estimated_cost,
            reason="Rejected because the runtime token budget would be exceeded.",
        )

    if estimated_cost > RuntimeBudget.max_estimated_cost_usd:
        return GovernanceDecision(
            allowed=False,
            risk_level=risk,
            masked_query=guarded.text,
            injection_detected=False,
            estimated_tokens=estimated_tokens,
            estimated_cost_usd=estimated_cost,
            reason="Rejected because the runtime cost budget would be exceeded.",
        )

    return GovernanceDecision(
        allowed=True,
        risk_level=risk,
        masked_query=guarded.text,
        injection_detected=False,
        estimated_tokens=estimated_tokens,
        estimated_cost_usd=estimated_cost,
        reason="Passed all governance checks.",
    )
