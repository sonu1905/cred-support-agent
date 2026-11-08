"""Small Task 11/12 transcript using FastAPI's TestClient."""

from pathlib import Path

from fastapi.testclient import TestClient

from api import KB_FILE, app
from crew_tools import reset_shared_rag_engine


client = TestClient(app)


if __name__ == "__main__":
    print("GET /health")
    print(client.get("/health").json())

    print("\nPOST /ask")
    response = client.post(
        "/ask",
        json={"query": "What is the status of application CRD-0001?"},
    )
    print("status:", response.status_code)
    print(response.json())

    print("\nPOST /add-document")
    original_kb = KB_FILE.read_bytes()

    try:
        response = client.post(
            "/add-document",
            json={
                "doc_id": "api_demo_policy",
                "title": "Demo service acknowledgement",
                "text": (
                    "Support requests should receive an acknowledgement before "
                    "the case is investigated. This demo policy exists only to "
                    "prove the add-document endpoint and is removed afterwards."
                ),
            },
        )
        print("status:", response.status_code)
        print(response.json())
    finally:
        KB_FILE.write_bytes(original_kb)
        reset_shared_rag_engine()

    print("\nWebSocket /ws/demo-session")
    with client.websocket_connect("/ws/demo-session") as websocket:
        websocket.send_text("What is the status of application CRD-0001?")
        print(websocket.receive_json())

    print("\nWebSocket client closed normally; server remained available.")
    print("health after disconnect:", client.get("/health").json())

    log_file = Path(__file__).resolve().parent / "logs" / "requests.jsonl"
    print("\nJSONL log file:", log_file)
    if log_file.exists():
        print("last log line:")
        print(log_file.read_text(encoding="utf-8").splitlines()[-1])
