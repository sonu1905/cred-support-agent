Tasks 11-13

Task 11
- POST /ask uses Pydantic AskRequest/AskResponse models.
- POST /add-document validates and adds a local KB document, then invalidates
  the shared in-process RAG index so it is rebuilt on the next policy query.
- /ws/{session_id} supports multi-turn in-process chat and catches
  WebSocketDisconnect so one client leaving does not stop the server.

Task 12
- Each HTTP endpoint writes one JSON-Lines log entry.
- Each WebSocket message writes one JSON-Lines log entry.
- Logs have trace_id, UTC timestamp, duration_ms, endpoint and status.
- Request text is passed through the same fixed-format PII masker before disk.

Task 13
- evaluation.py contains exactly 15 queries.
- The 12 required KB topics are represented.
- One application-status edge case, one out-of-scope case and one prompt
  injection case are included.
- Accuracy, Grounding, Completeness and Safety are reported for every query.
- Averages are written to transcripts/evaluation_15.json and .md.
- The judge prompt and deterministic MOCK_LLM judge are included in code.
