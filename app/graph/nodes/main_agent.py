"""Main LLM agent — classifies intent and routes to tools."""

from langchain_core.messages import AIMessage, HumanMessage

from app.graph.state import MedAIState, Intent
from app.models.llm import chat

ROUTER_SYSTEM = """You are a medical assistant router.
Classify the user request into exactly one intent:
- image: user uploaded or mentions a medical image (X-ray, MRI, skin, retina)
- info: user wants to save or update patient information
- rag: user asks about disease, symptoms, treatment, or drugs
- direct: general medical chat that needs no tool

Reply with ONLY the intent word: image, info, rag, or direct."""


def _last_user_message(state: MedAIState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg["content"]
    return ""


def _parse_intent(raw: str) -> Intent:
    text = raw.strip().lower()
    for candidate in ("image", "info", "rag", "direct"):
        if candidate in text:
            return candidate  # type: ignore[return-value]
    return "rag"


def main_agent_node(state: MedAIState) -> dict:
    user_text = _last_user_message(state)
    raw_intent = chat(ROUTER_SYSTEM, user_text, max_tokens=16, temperature=0.0)
    intent = _parse_intent(raw_intent)

    return {
        "intent": intent,
        "messages": [AIMessage(content=f"[router] intent={intent}")],
    }
