"""JSON-Lines request logging with PII-safe request text."""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from guardrails import mask_pii


LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "requests.jsonl"
_LOG_LOCK = threading.Lock()


def log_request(
    *,
    trace_id: str,
    endpoint: str,
    request_text: str,
    started_at: float,
    status: str,
    method: str,
    extra: dict[str, Any] | None = None,
) -> None:
    masked_text = mask_pii(request_text)

    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "endpoint": endpoint,
        "method": method,
        "status": status,
        "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "request_text": masked_text,
    }

    if extra:
        entry.update(extra)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with _LOG_LOCK:
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
