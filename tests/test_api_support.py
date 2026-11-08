import json
import time

import structured_logging
from api import app
from evaluation import CASES


def test_required_api_routes_exist():
    paths = {route.path for route in app.routes}
    assert "/ask" in paths
    assert "/add-document" in paths
    assert "/ws/{session_id}" in paths


def test_evaluation_has_15_cases_and_edge_cases():
    assert len(CASES) == 15
    edge_count = sum(case.kind in {"refusal", "injection"} for case in CASES)
    assert edge_count >= 2


def test_jsonl_logger_masks_fixed_format_pii(tmp_path, monkeypatch):
    log_file = tmp_path / "requests.jsonl"
    monkeypatch.setattr(structured_logging, "LOG_FILE", log_file)

    structured_logging.log_request(
        trace_id="trace-test",
        endpoint="/ask",
        request_text=(
            "PAN ABCDE1234F Aadhaar 1234 5678 9012 "
            "bank account number 1234567890123456"
        ),
        started_at=time.perf_counter(),
        status="ok",
        method="POST",
    )

    line = log_file.read_text(encoding="utf-8").strip()
    payload = json.loads(line)

    assert payload["trace_id"] == "trace-test"
    assert "ABCDE1234F" not in payload["request_text"]
    assert "1234 5678 9012" not in payload["request_text"]
    assert "1234567890123456" not in payload["request_text"]
    assert "[MASKED_PAN]" in payload["request_text"]
