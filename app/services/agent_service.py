"""Shared LangGraph agent service with persistent checkpointing."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.config import CHECKPOINT_DB, UPLOAD_DIR
from app.graph.builder import build_graph
from app.storage.patient_store import init_db

_graph = None
_checkpointer = None
_conn = None


def _ensure_dirs() -> None:
    Path(CHECKPOINT_DB).parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    init_db()


def get_graph():
    global _graph, _checkpointer, _conn
    if _graph is None:
        _ensure_dirs()
        _conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
        _checkpointer = SqliteSaver(_conn)
        _graph = build_graph(checkpointer=_checkpointer)
    return _graph


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _extract_interrupt(event: dict) -> dict | None:
    if "__interrupt__" in event:
        interrupts = event["__interrupt__"]
        if interrupts:
            payload = interrupts[0].value
            return payload if isinstance(payload, dict) else {"message": str(payload)}
    return None


def run_agent(
    message: str,
    *,
    thread_id: str | None = None,
    image_path: str | None = None,
    image_type: str | None = None,
) -> dict[str, Any]:
    graph = get_graph()
    thread_id = thread_id or str(uuid.uuid4())

    inputs: dict = {"messages": [HumanMessage(content=message)]}
    if image_path:
        inputs["image_path"] = image_path
    if image_type:
        inputs["image_type"] = image_type

    interrupt_payload = None
    for event in graph.stream(inputs, config=_config(thread_id), stream_mode="updates"):
        hit = _extract_interrupt(event)
        if hit:
            interrupt_payload = hit
            break

    state = graph.get_state(_config(thread_id))
    values = state.values

    if interrupt_payload:
        return {
            "thread_id": thread_id,
            "status": "awaiting_approval",
            "interrupt": interrupt_payload,
            "partial_state": {
                "intent": values.get("intent"),
                "image_type": values.get("image_type"),
                "detected_disease": values.get("detected_disease"),
                "patient_info": values.get("patient_info"),
            },
        }

    return {
        "thread_id": thread_id,
        "status": "completed",
        "response": values.get("final_response"),
        "detected_disease": values.get("detected_disease"),
        "confidence": values.get("confidence"),
        "prescription_draft": values.get("prescription_draft"),
        "risk_level": values.get("risk_level"),
    }


def resume_agent(
    thread_id: str,
    *,
    action: str = "approve",
    note: str | None = None,
) -> dict[str, Any]:
    graph = get_graph()
    feedback = {"action": action, "note": note}

    interrupt_payload = None
    for event in graph.stream(
        Command(resume=feedback),
        config=_config(thread_id),
        stream_mode="updates",
    ):
        hit = _extract_interrupt(event)
        if hit:
            interrupt_payload = hit
            break

    state = graph.get_state(_config(thread_id))
    values = state.values

    if interrupt_payload:
        return {
            "thread_id": thread_id,
            "status": "awaiting_approval",
            "interrupt": interrupt_payload,
            "partial_state": {
                "detected_disease": values.get("detected_disease"),
                "prescription_draft": values.get("prescription_draft"),
            },
        }

    return {
        "thread_id": thread_id,
        "status": "completed",
        "response": values.get("final_response"),
        "detected_disease": values.get("detected_disease"),
        "confidence": values.get("confidence"),
        "prescription_draft": values.get("prescription_draft"),
        "risk_level": values.get("risk_level"),
    }


def get_thread_state(thread_id: str) -> dict[str, Any]:
    graph = get_graph()
    state = graph.get_state(_config(thread_id))
    values = state.values
    messages = []
    for msg in values.get("messages", []):
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    return {
        "thread_id": thread_id,
        "messages": messages,
        "detected_disease": values.get("detected_disease"),
        "confidence": values.get("confidence"),
        "prescription_draft": values.get("prescription_draft"),
        "risk_level": values.get("risk_level"),
        "final_response": values.get("final_response"),
    }
