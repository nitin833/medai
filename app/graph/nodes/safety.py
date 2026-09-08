"""Safety check — flags high-risk responses for HITL review."""

from app.graph.state import MedAIState, RiskLevel

EMERGENCY_KEYWORDS = (
    "chest pain",
    "heart attack",
    "stroke",
    "suicide",
    "overdose",
    "unconscious",
    "severe bleeding",
)


def _last_user_text(state: MedAIState) -> str:
    for msg in reversed(state.get("messages", [])):
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
        if content and not content.startswith("[router]"):
            return content.lower()
    return ""


def safety_check_node(state: MedAIState) -> dict:
    flags: list[str] = []
    user_text = _last_user_text(state)

    for kw in EMERGENCY_KEYWORDS:
        if kw in user_text:
            flags.append(f"emergency_keyword:{kw}")

    confidence = state.get("confidence") or 1.0
    if state.get("detected_disease") and confidence < 0.7:
        flags.append("low_confidence_diagnosis")

    if state.get("prescription_draft"):
        flags.append("prescription_present")

    if any(f.startswith("emergency_keyword") for f in flags) or "prescription_present" in flags:
        risk: RiskLevel = "high"
    elif flags:
        risk = "medium"
    else:
        risk = "low"

    return {
        "risk_level": risk,
        "safety_flags": flags,
        "hitl_pending": risk == "high",
    }
