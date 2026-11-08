"""FastAPI deployment for the Cred support agent."""

import asyncio
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from crew_service import CrewResponse, run_support_query
from crew_tools import reset_shared_rag_engine
from guardrails import apply_input_guardrails
from memory_layer import chat
from structured_logging import log_request


ROOT = Path(__file__).resolve().parent
KB_FILE = ROOT / "kb" / "policies.json"

app = FastAPI(
    title="Cred Domain Support Agent",
    version="1.0.0",
    description="Local MOCK_LLM deployment for the capstone project.",
)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


class AskResponse(BaseModel):
    trace_id: str
    response: CrewResponse


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=2, max_length=80)
    title: str = Field(min_length=3, max_length=160)
    text: str = Field(min_length=20, max_length=5000)


class AddDocumentResponse(BaseModel):
    trace_id: str
    status: str
    doc_id: str


class HealthResponse(BaseModel):
    status: str
    mock_llm: bool


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    started = time.perf_counter()
    trace_id = uuid.uuid4().hex
    log_request(
        trace_id=trace_id,
        endpoint="/health",
        request_text="",
        started_at=started,
        status="ok",
        method="GET",
    )
    return HealthResponse(status="ok", mock_llm=True)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    started = time.perf_counter()
    trace_id = uuid.uuid4().hex
    status = "ok"

    try:
        response = await asyncio.to_thread(run_support_query, request.query)
        return AskResponse(trace_id=trace_id, response=response)
    except Exception:
        status = "error"
        raise
    finally:
        log_request(
            trace_id=trace_id,
            endpoint="/ask",
            request_text=request.query,
            started_at=started,
            status=status,
            method="POST",
        )


@app.post("/add-document", response_model=AddDocumentResponse)
async def add_document(request: AddDocumentRequest) -> AddDocumentResponse:
    started = time.perf_counter()
    trace_id = uuid.uuid4().hex
    status = "ok"

    guarded = apply_input_guardrails(request.text)

    try:
        if guarded.injection_detected:
            status = "blocked"
            raise HTTPException(
                status_code=400,
                detail="Document rejected by prompt-injection guardrail.",
            )

        with KB_FILE.open(encoding="utf-8") as handle:
            documents = json.load(handle)

        if any(document["id"] == request.doc_id for document in documents):
            status = "duplicate"
            raise HTTPException(
                status_code=409,
                detail=f"Document id '{request.doc_id}' already exists.",
            )

        documents.append(
            {
                "id": request.doc_id,
                "title": request.title,
                "text": guarded.text,
            }
        )

        with KB_FILE.open("w", encoding="utf-8") as handle:
            json.dump(documents, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        # The next policy request rebuilds the in-process index with this document.
        reset_shared_rag_engine()

        return AddDocumentResponse(
            trace_id=trace_id,
            status="added",
            doc_id=request.doc_id,
        )
    except HTTPException:
        raise
    except Exception:
        status = "error"
        raise
    finally:
        log_request(
            trace_id=trace_id,
            endpoint="/add-document",
            request_text=f"{request.title}: {request.text}",
            started_at=started,
            status=status,
            method="POST",
        )


@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()

    try:
        while True:
            raw_text = await websocket.receive_text()
            started = time.perf_counter()
            trace_id = uuid.uuid4().hex
            status = "ok"

            guarded = apply_input_guardrails(raw_text)

            try:
                if guarded.injection_detected:
                    reply = "Request blocked because it contains a prompt-injection pattern."
                else:
                    reply = await asyncio.to_thread(chat, session_id, guarded.text)

                await websocket.send_json(
                    {
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "answer": reply,
                    }
                )
            except Exception:
                status = "error"
                raise
            finally:
                log_request(
                    trace_id=trace_id,
                    endpoint="/ws/{session_id}",
                    request_text=raw_text,
                    started_at=started,
                    status=status,
                    method="WEBSOCKET_MESSAGE",
                    extra={"session_id": session_id},
                )

    except WebSocketDisconnect:
        # Client disconnects are expected. The exception is contained so the
        # server continues serving other clients.
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
