"""Info tool — extracts and structures patient information."""

import json
import re

from app.graph.state import MedAIState
from app.models.llm import chat

EXTRACT_SYSTEM = """Extract patient information from the message into JSON with these fields:
name, age, gender, symptoms (list), allergies (list), medical_history (list), medications (list).
Use null for missing fields. Return ONLY valid JSON, no markdown."""


def _last_user_message(state: MedAIState) -> str:
    for msg in reversed(state.get("messages", [])):
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
        if content and not content.startswith("[router]"):
            return content
    return ""


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_notes": raw}


def info_tool_node(state: MedAIState) -> dict:
    existing = state.get("patient_info") or {}
    if existing:
        return {"patient_info": existing, "hitl_pending": True}

    user_text = _last_user_message(state)
    raw = chat(EXTRACT_SYSTEM, user_text, max_tokens=256, temperature=0.0)
    patient_info = _parse_json(raw)

    return {
        "patient_info": patient_info,
        "hitl_pending": True,
    }
