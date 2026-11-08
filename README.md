# Cred Domain Support Agent

This project is a local banking-support prototype built for the Agentic AI capstone. It combines a deterministic synthetic loan dataset, retrieval-augmented generation, CrewAI agents, session memory, guardrails, FastAPI, AutoGen review, governance controls and a response cache.

The project is designed to run in `MOCK_LLM` mode without an external LLM API key.

## What the project does

A user can ask banking-policy questions such as KYC, EMI, eligibility, prepayment or credit-score questions. The system retrieves relevant policy text from a local ChromaDB collection and answers only when the retrieval score is above the measured grounding threshold.

A user can also check a synthetic loan application such as `CRD-0001`. The lookup returns the application status, loan amount and an escalation score.

The main agent flow has three CrewAI agents:

| Agent | Responsibility | Tools |
|---|---|---|
| Policy Retrieval Agent | Retrieves local policy evidence | `rag_lookup` |
| Loan Application Lookup Agent | Looks up synthetic application records | `check_loan_application_status` |
| Response Composer | Produces the final structured answer | None |

The Composer intentionally has no direct tool access.

## Environment

The project was tested on:

- Python 3.11.9
- Windows 11 / PowerShell
- CrewAI 1.15.35
- AutoGen AgentChat 0.7.5
- ChromaDB 1.5.9
- SentenceTransformers 5.7.0
- FastAPI 0.162.2

Python 3.11 is recommended for this project because the Windows Chroma HNSW dependency was stable in this environment.

## Setup

Create and activate a Python 3.11 virtual environment:

```powershell
py -3.11 -m venv .venv311
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv311\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

No external LLM API key is required for the graded path.

## Dataset

`dataset.py` creates a deterministic synthetic loan-application dataset.

The final generated dataset contains 60 records across:

- Personal Loan
- Home Loan
- Auto Loan
- Education Loan
- Business Loan

Statuses include Submitted, Under Review, Approved, Rejected and Disbursed.

The observed fraud-review rate is 21.67%.

The data is synthetic and is not intended to represent real customers.

## Knowledge base and RAG

The local knowledge base is stored in `kb/policies.json`.

It covers the required banking topics, including loan eligibility, EMI rules, card fees, KYC, fraud disputes, account closure, interest rates, prepayment, minimum balance, credit score, joint accounts and NRI accounts.

Two chunking strategies are implemented:

| Strategy | Mean precision | Mean recall |
|---|---:|---:|
| Fixed-size with overlap | 0.400 | 1.000 |
| Sentence-based | 0.500 | 1.000 |

Sentence-based chunking was selected because it preserved full recall while improving document-level precision on the evaluation set.

The grounding threshold was calibrated using five in-scope and two out-of-scope questions.

- Lowest in-scope top-1 similarity: `0.6286`
- Highest out-of-scope top-1 similarity: `0.0876`
- Selected midpoint threshold: `0.3581`

An unsupported question falls back to:

```text
I don't know based on the available policy documents.
```

Run the RAG demo:

```powershell
python -u rag_demo.py
```

Run the retrieval evaluation:

```powershell
python -u rag_evaluation.py
```

## Loan lookup and escalation

The status tool is:

```text
check_loan_application_status(record_id)
```

The escalation formula is:

```text
score = 0.70 * fraud_flag + 0.30 * (days_since_created / 30)
```

The dataset's 75th percentile for application age is 23 days, giving an escalation threshold of `0.93`.

Run:

```powershell
python -u task6_demo.py
```

## CrewAI orchestration

The CrewAI workflow uses a custom `MockCrewLLM` subclass of `BaseLLM`.

It uses CrewAI's normal text/ReAct tool loop and avoids two common mock-model pitfalls:

1. Tool selection is based on each tool's declared argument schema rather than matching substrings in tool names.
2. Tool results are recognized with unique `RAG_RESULT_JSON=` and `LOOKUP_RESULT_JSON=` markers instead of searching the prompt for the generic word `Observation:`.

CrewAI telemetry/tracing is disabled for the local graded path.

Run:

```powershell
python -u crew_demo.py
```

## Session memory

Session memory uses:

```text
InMemoryChatMessageHistory
RunnableWithMessageHistory
```

One session can remember an application ID while a fresh session starts without that context.

Run:

```powershell
python -u memory_demo.py
```

The current LangChain version prints a deprecation warning for these classes. They are kept intentionally because the capstone specifically requires them.

## Structured output and guardrails

Final CrewAI responses are validated with the `CrewResponse` Pydantic model.

Input guardrails provide deterministic masking for:

- PAN
- Aadhaar
- context-labelled bank account numbers

Prompt-injection patterns are blocked before the crew runs.

Unsupported policy responses are also checked on output and replaced with the grounded fallback.

Run:

```powershell
python -u guardrail_demo.py
```

## FastAPI and WebSocket

The FastAPI deployment includes:

| Route | Purpose |
|---|---|
| `GET /health` | Health check |
| `POST /ask` | Ask the support agent |
| `POST /add-document` | Add a validated local KB document |
| `WS /ws/{session_id}` | Session-aware WebSocket chat |

`WebSocketDisconnect` is handled so a client disconnect does not terminate the server.

Run the API demo:

```powershell
python -u api_demo.py
```

Or start the local server:

```powershell
python -u api.py
```

## Structured logging

Requests are written as JSON Lines to:

```text
logs/requests.jsonl
```

Each entry includes a trace ID, endpoint, method, UTC timestamp, status and duration.

PII masking is applied before request text is written to disk.

## Fifteen-query evaluation

`evaluation.py` runs 15 local evaluation cases and uses a deterministic `MOCK_LLM` judge.

The final measured averages were:

| Metric | Average |
|---|---:|
| Accuracy | 0.933 |
| Grounding | 0.933 |
| Completeness | 0.789 |
| Safety | 1.000 |

Evaluation artifacts are written to:

```text
transcripts/evaluation_15.json
transcripts/evaluation_15.md
transcripts/evaluation_judge_prompt.txt
```

Run:

```powershell
python -u evaluation.py
```

## AutoGen review

The review workflow uses two custom AutoGen agents in a `RoundRobinGroupChat` with `max_turns=2`.

The first agent checks whether the draft matches the grounded answer. The Final Editor returns:

```text
approved
final_answer
reason
```

Two custom message types are registered with the team.

The demo shows both required outcomes:

- a supported answer is approved unchanged
- a deliberately injected unsupported claim is removed

Run:

```powershell
python -u autogen_demo.py
```

## Governance

The project demonstrates four governance layers:

1. Input safety through PII masking and prompt-injection detection.
2. Risk classification where customer-specific financial data is classified as High.
3. Least-autonomy tool assignment: only the Loan Application Lookup Agent has the application lookup tool.
4. Runtime token and estimated-cost caps that reject oversized input.

Run:

```powershell
python -u governance_demo.py
```

## Response cache

The in-memory cache uses a normalized query key:

```text
strip -> lowercase -> collapse whitespace
```

A repeated normalized query is returned from memory without another backend call.

Observed demo behavior:

```text
first call:  cache_hit=False, backend_call_count=1
second call: cache_hit=True,  backend_call_count=1
```

Run:

```powershell
python -u cache_demo.py
```

## Tests

Run the complete suite:

```powershell
python -m pytest -q
```

Final local result:

```text
22 passed
```

Deprecation warnings from LangChain and CrewAI may appear. They do not affect the passing tests.

## Main project files

```text
dataset.py
rag_engine.py
rag_evaluation.py
rag_demo.py
loan_tools.py
task6_demo.py
guardrails.py
crew_tools.py
mock_llm.py
crew_service.py
crew_demo.py
memory_layer.py
memory_demo.py
guardrail_demo.py
api.py
api_demo.py
structured_logging.py
evaluation.py
autogen_review.py
autogen_demo.py
governance.py
governance_demo.py
cache_layer.py
cache_demo.py
kb/
tests/
transcripts/
```

## Notes

This project is an educational prototype. The loan records and policy content are synthetic and should not be used for real financial decisions.
