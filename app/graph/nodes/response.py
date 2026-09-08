"""Final response generation."""

from langchain_core.messages import AIMessage

from app.graph.state import MedAIState
from app.models.llm import chat

RESPONSE_SYSTEM = """You are an expert biomedical assistant.
Use the provided context to answer the user.
Always include a disclaimer that this is not a substitute for professional medical advice.
If prescribing, note that recommendations require clinician approval."""


def generate_response_node(state: MedAIState) -> dict:
    if state.get("final_response"):
        return {}

    if state.get("hitl_action") == "reject":
        return {
            "final_response": state.get("final_response")
            or "Request declined. Please consult a licensed healthcare provider.",
            "messages": [AIMessage(content="Request declined pending clinical review.")],
        }

    user_question = ""
    for msg in reversed(state.get("messages", [])):
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
        if content and not content.startswith("[router]"):
            user_question = content
            break

    parts = []
    if state.get("detected_disease"):
        parts.append(f"Diagnosis: {state['detected_disease']} (confidence: {state.get('confidence', 'N/A')})")
    if state.get("rag_context"):
        parts.append(f"Retrieved context:\n{state['rag_context']}")
    if state.get("prescription_draft"):
        parts.append(f"Prescription draft:\n{state['prescription_draft']}")
    if state.get("human_feedback"):
        parts.append(f"Clinician feedback: {state['human_feedback']}")

    prompt = f"Context:\n{chr(10).join(parts)}\n\nQuestion:\n{user_question}\n\nAnswer:"
    answer = chat(RESPONSE_SYSTEM, prompt)

    return {
        "final_response": answer,
        "messages": [AIMessage(content=answer)],
        "hitl_pending": False,
    }
