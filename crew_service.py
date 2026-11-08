"""CrewAI orchestration layer for the Cred support agent."""

import json
import os
from typing import Literal

# Zero-network grading mode.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field, ValidationError

from crew_tools import LoanStatusTool, RagLookupTool
from guardrails import apply_input_guardrails
from mock_llm import MockCrewLLM


class CrewResponse(BaseModel):
    query_type: Literal["policy", "application_status", "mixed", "guardrail"]
    answer: str = Field(min_length=1)
    sources: list[str] = Field(default_factory=list)
    record_id: str | None = None
    escalation_score: float | None = Field(default=None, ge=0.0, le=1.0)
    recommend_escalation: bool = False
    grounded: bool
    pii_was_masked: bool = False


def _guardrail_response(message: str, pii_was_masked: bool = False) -> CrewResponse:
    return CrewResponse(
        query_type="guardrail",
        answer=message,
        grounded=False,
        pii_was_masked=pii_was_masked,
    )


def build_crew(user_query: str) -> Crew:
    mock_llm = MockCrewLLM()

    retrieval_agent = Agent(
        role="Policy Retrieval Agent",
        goal="Retrieve only evidence relevant to banking policy questions",
        backstory=(
            "You are a narrow retrieval worker. You only use the local RAG tool. "
            "You do not look up application records."
        ),
        tools=[RagLookupTool()],
        llm=mock_llm,
        verbose=False,
        allow_delegation=False,
    )

    lookup_agent = Agent(
        role="Loan Application Lookup Agent",
        goal="Look up a specific fabricated loan application when an ID is supplied",
        backstory=(
            "You are the only agent allowed to use the loan-status lookup tool. "
            "You never invent application records."
        ),
        tools=[LoanStatusTool()],
        llm=mock_llm,
        verbose=False,
        allow_delegation=False,
    )

    composer_agent = Agent(
        role="Response Composer",
        goal="Combine grounded policy evidence and lookup results into one response",
        backstory=(
            "You do not have direct access to either tool. "
            "You compose only from outputs supplied by the specialist agents."
        ),
        tools=[],
        llm=mock_llm,
        verbose=False,
        allow_delegation=False,
    )

    retrieval_task = Task(
        description=(
            "POLICY_RETRIEVAL_TASK\n"
            "USER_QUERY_BEGIN\n{user_query}\nUSER_QUERY_END\n"
            "If this is a banking-policy question, use your declared RAG tool. "
            "If it is only an application-status request, return NOT_APPLICABLE."
        ),
        expected_output="A RAG_RESULT_JSON line or NOT_APPLICABLE.",
        agent=retrieval_agent,
    )

    lookup_task = Task(
        description=(
            "APPLICATION_LOOKUP_TASK\n"
            "USER_QUERY_BEGIN\n{user_query}\nUSER_QUERY_END\n"
            "If the user supplied a CRD-#### application ID, use your declared "
            "record lookup tool. Otherwise return NOT_APPLICABLE."
        ),
        expected_output="A LOOKUP_RESULT_JSON line or NOT_APPLICABLE.",
        agent=lookup_agent,
    )

    composer_task = Task(
        description=(
            "RESPONSE_COMPOSER_TASK\n"
            "USER_QUERY_BEGIN\n{user_query}\nUSER_QUERY_END\n"
            "Use only the two prior task outputs. Return one JSON object matching "
            "the CrewResponse schema. Never add facts that are absent from them."
        ),
        expected_output="A valid CrewResponse JSON object.",
        agent=composer_agent,
        context=[retrieval_task, lookup_task],
        output_pydantic=CrewResponse,
    )

    return Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent],
        tasks=[retrieval_task, lookup_task, composer_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
        tracing=False,
    )


def _validate_crew_output(result) -> CrewResponse:
    if getattr(result, "pydantic", None) is not None:
        return CrewResponse.model_validate(result.pydantic)

    raw = getattr(result, "raw", str(result))
    try:
        return CrewResponse.model_validate_json(raw)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Crew returned invalid structured output: {raw}") from exc


def run_support_query(user_query: str) -> CrewResponse:
    guarded = apply_input_guardrails(user_query)

    if guarded.injection_detected:
        return _guardrail_response(
            "Request blocked because it contains a prompt-injection pattern.",
            pii_was_masked=guarded.pii_masked,
        )

    crew = build_crew(guarded.text)
    result = crew.kickoff(inputs={"user_query": guarded.text})
    response = _validate_crew_output(result)
    response.pii_was_masked = guarded.pii_masked

    # Output-side groundedness guardrail.
    if response.query_type in {"policy", "mixed"} and not response.grounded:
        response.answer = "I don't know based on the available policy documents."
        response.sources = []

    return response
