Tasks 7-10 notes

CrewAI
- Uses a custom MockCrewLLM subclass of CrewAI BaseLLM.
- CREWAI_DISABLE_TELEMETRY and OTEL_SDK_DISABLED are set before kickoff.
- Retrieval Agent alone owns RagLookupTool.
- Lookup Agent alone owns LoanStatusTool.
- Composer has no tools.
- MOCK_LLM chooses tool arguments from declared tool schemas, not tool-name substrings.
- Unique RAG_RESULT_JSON / LOOKUP_RESULT_JSON markers are used instead of searching
  CrewAI's system prompt for generic "Observation:" text.

Memory
- Uses InMemoryChatMessageHistory + RunnableWithMessageHistory.
- Session memory is process-local and intentionally disappears on restart.

Structured output
- Every final crew response is validated as CrewResponse (Pydantic).

Guardrails
- Input: fixed-format PAN, Aadhaar and context-labelled bank-account masking.
- Input: deterministic prompt-injection patterns.
- Output: unsupported policy queries fall back to "I don't know".
