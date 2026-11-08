"""LangChain in-process session memory for Task 8."""

import re

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from crew_service import run_support_query


_HISTORY: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _HISTORY:
        _HISTORY[session_id] = InMemoryChatMessageHistory()
    return _HISTORY[session_id]


def _latest_record_id(messages: list[BaseMessage]) -> str | None:
    for message in reversed(messages):
        match = re.search(r"\bCRD-\d{4}\b", message.content, re.IGNORECASE)
        if match:
            return match.group(0).upper()
    return None


def _memory_backend(payload: dict) -> AIMessage:
    messages = payload["messages"]
    current = messages[-1].content
    current_id = _latest_record_id([messages[-1]])

    if "remember" in current.lower() and current_id:
        return AIMessage(content=f"I will keep {current_id} in this session.")

    if current_id:
        response = run_support_query(current)
        return AIMessage(content=response.answer)

    if any(word in current.lower() for word in ("its status", "that application", "the application")):
        remembered_id = _latest_record_id(messages[:-1])
        if remembered_id:
            response = run_support_query(
                f"What is the status of application {remembered_id}?"
            )
            return AIMessage(content=response.answer)
        return AIMessage(content="I don't have an application ID in this session.")

    response = run_support_query(current)
    return AIMessage(content=response.answer)


_core = RunnableLambda(_memory_backend)
chat_with_history = RunnableWithMessageHistory(
    _core,
    get_session_history,
    input_messages_key="messages",
)


def chat(session_id: str, message: str) -> str:
    result = chat_with_history.invoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"session_id": session_id}},
    )
    return result.content
