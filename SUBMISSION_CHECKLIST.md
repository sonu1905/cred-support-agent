# Final Submission Checklist

Use this checklist before submitting the public GitHub repository.

## Repository

- [ ] Repository is public.
- [ ] Repository opens without permission errors.
- [ ] `.venv`, `.venv311`, model caches and local logs are not committed.
- [ ] No API keys, tokens, passwords or personal credentials are present.
- [ ] `README.md` is visible on the repository home page.
- [ ] `requirements.txt` is committed.
- [ ] `kb/policies.json` is committed.
- [ ] `tests/` is committed.
- [ ] `transcripts/evaluation_15.json` is committed.
- [ ] `transcripts/evaluation_15.md` is committed.
- [ ] `transcripts/evaluation_judge_prompt.txt` is committed.

## Final verification commands

Run from the activated Python 3.11 environment:

```powershell
python -m pip check
python -m pytest -q
python -u rag_demo.py
python -u task6_demo.py
python -u crew_demo.py
python -u memory_demo.py
python -u guardrail_demo.py
python -u api_demo.py
python -u evaluation.py
python -u autogen_demo.py
python -u governance_demo.py
python -u cache_demo.py
```

Expected final test result:

```text
22 passed
```

## Evidence to keep in README / transcripts

| Requirement | Evidence |
|---|---|
| Deterministic dataset | `python dataset.py` |
| Two RAG chunking strategies | `rag_evaluation.py` |
| Grounding threshold | `0.3581` calibration output |
| OOS fallback | bicycle-light query in `rag_demo.py` |
| Escalation score | `task6_demo.py` |
| Three CrewAI agents | `crew_service.py` and `crew_demo.py` |
| Both CrewAI tools used | `crew_demo.py` tool-call log |
| Session memory and reset | `memory_demo.py` |
| Pydantic output | `CrewResponse` |
| Input/output guardrails | `guardrail_demo.py` |
| FastAPI + WebSocket | `api.py` and `api_demo.py` |
| JSONL request logging | `logs/requests.jsonl` demo |
| 15-query evaluation | `transcripts/evaluation_15.*` |
| AutoGen review | `autogen_demo.py` |
| Four governance layers | `governance_demo.py` |
| Cache hit + call counter | `cache_demo.py` |
| Automated tests | `22 passed` |

## Git commands

After reviewing the files:

```powershell
git status
git add .
git commit -m "Complete Cred domain support agent capstone"
git push
```

Then open the GitHub repository in a browser and confirm that the README, source files, tests and transcripts are visible.
