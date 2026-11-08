"""Deterministic zero-network CrewAI model used for grading transcripts."""

import json
import re
from typing import Any, Dict, List, Optional, Union

from crewai import BaseLLM


class MockCrewLLM(BaseLLM):
    """
    Small deterministic CrewAI LLM used for MOCK_LLM mode.

    Important implementation choices:
    - Uses CrewAI's text/ReAct tool path instead of native function calling.
    - Accepts the current CrewAI call metadata arguments.
    - Chooses tool arguments from each tool's declared args_schema.
    - Never searches the system prompt for the generic word "Observation:".
      Tool results are recognized only by our unique result markers.
    """

    def __init__(self):
        super().__init__(model="local/mock-crewai", temperature=0.0)

    def supports_function_calling(self) -> bool:
        # False is intentional. CrewAI then executes the normal ReAct loop:
        # Thought / Action / Action Input -> actual tool execution -> Final Answer.
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 12000

    @staticmethod
    def _flatten(messages: Union[str, List[Dict[str, Any]]]) -> str:
        if isinstance(messages, str):
            return messages

        parts = []
        for message in messages:
            content = message.get("content", "")
            if content:
                parts.append(str(content))
        return "\n".join(parts)

    @staticmethod
    def _user_query(text: str) -> str:
        match = re.search(
            r"USER_QUERY_BEGIN\s*(.*?)\s*USER_QUERY_END",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip() if match else text.strip()

    @staticmethod
    def _record_id(text: str) -> Optional[str]:
        match = re.search(r"\bCRD-\d{4}\b", text, re.IGNORECASE)
        return match.group(0).upper() if match else None

    @staticmethod
    def _looks_like_status_only(query: str) -> bool:
        lowered = query.lower()
        has_record = bool(re.search(r"\bcrd-\d{4}\b", lowered))
        status_words = any(
            word in lowered
            for word in (
                "status",
                "application",
                "approved",
                "rejected",
                "review",
                "disbursed",
                "submitted",
            )
        )
        policy_words = any(
            word in lowered
            for word in (
                "policy",
                "kyc",
                "emi",
                "eligib",
                "interest",
                "prepayment",
                "fee",
                "balance",
                "credit score",
                "joint",
                "nri",
                "fraud",
                "closure",
                "document",
            )
        )
        return has_record and status_words and not policy_words

    @staticmethod
    def _declared_tool_fields(tool: Any) -> set[str]:
        """
        Read the tool's own Pydantic argument schema.

        This is intentionally schema-based. We do not decide behavior by
        checking whether a word such as "lookup" appears in the tool name.
        """
        schema = getattr(tool, "args_schema", None)
        fields = getattr(schema, "model_fields", None)
        return set(fields.keys()) if fields else set()

    @staticmethod
    def _tool_for_field(from_agent: Any, field_name: str) -> Any | None:
        if from_agent is None:
            return None

        for tool in getattr(from_agent, "tools", []) or []:
            if field_name in MockCrewLLM._declared_tool_fields(tool):
                return tool
        return None

    @staticmethod
    def _all_marker_payloads(text: str, marker: str) -> list[dict]:
        payloads = []
        pattern = re.compile(re.escape(marker) + r"(\{[^\n]*\})")

        for match in pattern.finditer(text):
            try:
                payloads.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                continue

        return payloads

    @staticmethod
    def _latest_marker_payload(
        messages: Union[str, List[Dict[str, Any]]],
        marker: str,
    ) -> Optional[dict]:
        """
        Inspect actual conversation messages from newest to oldest.

        We do NOT scan for generic "Observation:" because CrewAI's own system
        template contains a fake example using that word.
        """
        if isinstance(messages, str):
            payloads = MockCrewLLM._all_marker_payloads(messages, marker)
            return payloads[-1] if payloads else None

        for message in reversed(messages):
            content = str(message.get("content", ""))
            payloads = MockCrewLLM._all_marker_payloads(content, marker)
            if payloads:
                return payloads[-1]

        return None

    @staticmethod
    def _action(tool: Any, arguments: dict) -> str:
        return (
            "Thought: I need to use the declared tool for this task.\n"
            f"Action: {tool.name}\n"
            f"Action Input: {json.dumps(arguments)}"
        )

    @staticmethod
    def _final(text: str) -> str:
        return (
            "Thought: I now have the result needed for this task.\n"
            f"Final Answer: {text}"
        )

    @staticmethod
    def _compose(text: str, query: str) -> str:
        rag_payloads = MockCrewLLM._all_marker_payloads(
            text,
            "RAG_RESULT_JSON=",
        )
        lookup_payloads = MockCrewLLM._all_marker_payloads(
            text,
            "LOOKUP_RESULT_JSON=",
        )

        rag = rag_payloads[-1] if rag_payloads else None
        lookup = lookup_payloads[-1] if lookup_payloads else None

        if rag and lookup and lookup.get("found"):
            answer = (
                f"{rag['answer']} Application {lookup['record_id']} is "
                f"{lookup['status']} with a loan amount of INR "
                f"{lookup['loan_amount_inr']:,}."
            )
            return json.dumps(
                {
                    "query_type": "mixed",
                    "answer": answer,
                    "sources": rag.get("sources", []),
                    "record_id": lookup["record_id"],
                    "escalation_score": lookup["escalation_score"],
                    "recommend_escalation": lookup["recommend_escalation"],
                    "grounded": bool(rag.get("supported")),
                    "pii_was_masked": False,
                }
            )

        if lookup:
            if lookup.get("found"):
                escalation_text = (
                    "Escalation is recommended."
                    if lookup["recommend_escalation"]
                    else "Escalation is not recommended."
                )
                return json.dumps(
                    {
                        "query_type": "application_status",
                        "answer": (
                            f"Application {lookup['record_id']} is "
                            f"{lookup['status']}. Loan amount: INR "
                            f"{lookup['loan_amount_inr']:,}. Escalation score: "
                            f"{lookup['escalation_score']:.3f}. "
                            f"{escalation_text}"
                        ),
                        "sources": [],
                        "record_id": lookup["record_id"],
                        "escalation_score": lookup["escalation_score"],
                        "recommend_escalation": lookup["recommend_escalation"],
                        "grounded": True,
                        "pii_was_masked": False,
                    }
                )

            return json.dumps(
                {
                    "query_type": "application_status",
                    "answer": (
                        "I could not find that application in the synthetic dataset."
                    ),
                    "sources": [],
                    "record_id": MockCrewLLM._record_id(query),
                    "escalation_score": None,
                    "recommend_escalation": False,
                    "grounded": True,
                    "pii_was_masked": False,
                }
            )

        if rag:
            return json.dumps(
                {
                    "query_type": "policy",
                    "answer": rag["answer"],
                    "sources": rag.get("sources", []),
                    "record_id": None,
                    "escalation_score": None,
                    "recommend_escalation": False,
                    "grounded": bool(rag.get("supported")),
                    "pii_was_masked": False,
                }
            )

        return json.dumps(
            {
                "query_type": "policy",
                "answer": (
                    "I don't know based on the available policy documents."
                ),
                "sources": [],
                "record_id": None,
                "escalation_score": None,
                "recommend_escalation": False,
                "grounded": False,
                "pii_was_masked": False,
            }
        )

    def call(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
        **kwargs: Any,
    ) -> str:
        text = self._flatten(messages)
        query = self._user_query(text)

        task_description = str(getattr(from_task, "description", "") or "")

        # Composer has no tools. Because output_pydantic is configured, CrewAI
        # expects valid JSON here, not a ReAct wrapper.
        if (
            "RESPONSE_COMPOSER_TASK" in task_description
            or "RESPONSE_COMPOSER_TASK" in text
        ):
            return self._compose(text, query)

        # If a real tool already ran, finish this specialist task using only the
        # uniquely marked result. This avoids the fake "Observation:" example
        # inside CrewAI's own prompt template.
        rag_result = self._latest_marker_payload(
            messages,
            "RAG_RESULT_JSON=",
        )
        if rag_result is not None:
            marker_text = (
                "RAG_RESULT_JSON="
                + json.dumps(rag_result, separators=(",", ":"))
            )
            return self._final(marker_text)

        lookup_result = self._latest_marker_payload(
            messages,
            "LOOKUP_RESULT_JSON=",
        )
        if lookup_result is not None:
            marker_text = (
                "LOOKUP_RESULT_JSON="
                + json.dumps(lookup_result, separators=(",", ":"))
            )
            return self._final(marker_text)

        # Decide which real CrewAI tool to request by inspecting its declared
        # argument schema, not by matching the tool name.
        record_tool = self._tool_for_field(from_agent, "record_id")
        query_tool = self._tool_for_field(from_agent, "query")

        if record_tool is not None:
            record_id = self._record_id(query)
            if record_id:
                return self._action(
                    record_tool,
                    {"record_id": record_id},
                )
            return self._final("NOT_APPLICABLE")

        if query_tool is not None:
            if self._looks_like_status_only(query):
                return self._final("NOT_APPLICABLE")
            return self._action(
                query_tool,
                {"query": query},
            )

        return self._final("NOT_APPLICABLE")

    async def acall(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
        **kwargs: Any,
    ) -> str:
        return self.call(
            messages=messages,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
            **kwargs,
        )
