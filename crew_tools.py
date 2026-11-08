"""CrewAI tools for policy retrieval and loan lookup."""

import json
import threading
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from loan_tools import check_loan_application_status
from rag_engine import PolicySearch


GROUNDING_THRESHOLD = 0.3581
TOOL_CALL_LOG: list[dict] = []

_RAG_ENGINE: PolicySearch | None = None
_RAG_LOCK = threading.Lock()


def get_shared_rag_engine() -> PolicySearch:
    global _RAG_ENGINE

    if _RAG_ENGINE is None:
        with _RAG_LOCK:
            if _RAG_ENGINE is None:
                _RAG_ENGINE = PolicySearch(auto_index=True)

    return _RAG_ENGINE


def reset_shared_rag_engine() -> None:
    global _RAG_ENGINE
    with _RAG_LOCK:
        _RAG_ENGINE = None


class RagLookupInput(BaseModel):
    query: str = Field(..., description="Banking policy question to retrieve evidence for")


class LoanStatusInput(BaseModel):
    record_id: str = Field(..., description="Loan application ID such as CRD-0001")


class RagLookupTool(BaseTool):
    name: str = "rag_lookup"
    description: str = (
        "Retrieves grounded banking-policy evidence from the local knowledge base. "
        "Input schema has one field named query."
    )
    args_schema: Type[BaseModel] = RagLookupInput

    def _run(self, query: str) -> str:
        engine = get_shared_rag_engine()
        result = engine.answer(
            query=query,
            threshold=GROUNDING_THRESHOLD,
            strategy="sentence",
            k=3,
        )

        TOOL_CALL_LOG.append({"tool": self.name, "query": query})
        return "RAG_RESULT_JSON=" + json.dumps(result, separators=(",", ":"))


class LoanStatusTool(BaseTool):
    name: str = "check_loan_application_status"
    description: str = (
        "Looks up one fabricated loan application. "
        "Input schema has one field named record_id."
    )
    args_schema: Type[BaseModel] = LoanStatusInput

    def _run(self, record_id: str) -> str:
        result = check_loan_application_status(record_id)
        TOOL_CALL_LOG.append({"tool": self.name, "record_id": record_id})
        return "LOOKUP_RESULT_JSON=" + json.dumps(result, separators=(",", ":"))
