Tasks 14-16

Task 14 - AutoGen
- Uses Microsoft AutoGen AgentChat.
- Two custom BaseChatAgent agents run inside RoundRobinGroupChat.
- max_turns=2.
- The team registers ReviewNoteMessage and FinalVerdictMessage as custom message types.
- Final Editor creates a Pydantic FinalVerdict with:
  approved, final_answer, reason.
- autogen_demo.py demonstrates:
  1) supported answer approved unchanged,
  2) deliberately appended unsupported claim removed.

Task 15 - Four governance layers
1. Input safety: PII masking + prompt-injection detection.
2. Risk classification: customer-specific financial data is High risk.
3. Least autonomy: only Loan Application Lookup Agent owns the lookup tool.
4. Runtime budget: token and estimated-cost caps reject oversized input.

Task 16 - Cache
- Process-local in-memory dictionary.
- Cache key is normalized with strip + lowercase + whitespace collapse.
- Demo prints cache hit/miss, backend call counter and timing.
